"""
Vector backend abstraction for agent-memory.

Provides a uniform interface over ChromaDB (default) and sqlite-vec (lightweight
alternative, suitable for Raspberry Pi / low-power hardware).

Usage:
    from agent_memory.vector_backend import build_backend

    # Default (ChromaDB):
    vec = build_backend("chromadb", conn, "/path/to/chroma_dir")

    # Lightweight (sqlite-vec, pip install 'agent-memory[sqlite-vec]'):
    vec = build_backend("sqlite-vec", conn, "")
"""

from __future__ import annotations

import sqlite3
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorBackend(Protocol):
    """Common interface for vector storage backends."""

    def add(self, episode_id: int, embedding: np.ndarray) -> None:
        """Store an embedding keyed by integer episode_id."""
        ...

    def query(self, embedding: np.ndarray, n_results: int) -> list[tuple[int, float]]:
        """Return [(episode_id, distance), ...] sorted ascending by distance."""
        ...

    def get_embeddings(self, episode_ids: list[int]) -> dict[int, np.ndarray]:
        """Fetch embeddings for a list of episode IDs."""
        ...

    def delete(self, episode_id: int) -> None:
        """Remove the embedding for an episode."""
        ...

    def count(self) -> int:
        """Return total number of stored embeddings."""
        ...


# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

class ChromaBackend:
    """Vector backend backed by ChromaDB (default)."""

    def __init__(self, vector_store_path: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for the default vector backend. "
                "Install it with: pip install chromadb"
            ) from exc

        from pathlib import Path
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=vector_store_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name="episodes",
            metadata={"description": "Agent episodic memories"},
        )

    def add(self, episode_id: int, embedding: np.ndarray) -> None:
        from datetime import datetime
        self._col.add(
            ids=[f"episode_{episode_id}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"episode_id": episode_id, "timestamp": datetime.now().isoformat()}],
            documents=[f"episode_{episode_id}"],
        )

    def query(self, embedding: np.ndarray, n_results: int) -> list[tuple[int, float]]:
        results = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
        )
        if not results["ids"][0]:
            return []
        return [
            (int(meta["episode_id"]), float(dist))
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]

    def get_embeddings(self, episode_ids: list[int]) -> dict[int, np.ndarray]:
        if not episode_ids:
            return {}
        ids = [f"episode_{ep_id}" for ep_id in episode_ids]
        results = self._col.get(ids=ids, include=["embeddings"])
        out: dict[int, np.ndarray] = {}
        for chroma_id, emb in zip(results["ids"], results["embeddings"]):
            ep_id = int(chroma_id.split("_", 1)[1])
            out[ep_id] = np.array(emb, dtype=np.float32)
        return out

    def delete(self, episode_id: int) -> None:
        try:
            self._col.delete(ids=[f"episode_{episode_id}"])
        except Exception:
            pass

    def count(self) -> int:
        return self._col.count()


# ---------------------------------------------------------------------------
# sqlite-vec backend
# ---------------------------------------------------------------------------

class SqliteVecBackend:
    """
    Vector backend backed by sqlite-vec.

    Stores embeddings as a virtual table in the same SQLite file used for
    structured data — no separate vector store directory needed.

    Requires: pip install 'agent-memory[sqlite-vec]'
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise ImportError(
                "sqlite-vec is required for the sqlite-vec backend. "
                "Install it with: pip install 'agent-memory[sqlite-vec]'"
            ) from exc

        self._conn = conn
        self._dim: int | None = None

        conn.enable_load_extension(True)
        import sqlite_vec as _sv
        _sv.load(conn)
        conn.enable_load_extension(False)

        # Detect existing dimension from an already-created table
        self._detect_dim()

    def _detect_dim(self) -> None:
        try:
            row = self._conn.execute(
                "SELECT embedding FROM vec_episodes LIMIT 1"
            ).fetchone()
            if row and row[0]:
                data = row[0]
                if isinstance(data, bytes):
                    self._dim = len(data) // 4  # float32 = 4 bytes each
        except Exception:
            pass  # Table doesn't exist yet

    def _ensure_table(self, dim: int) -> None:
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes "
            f"USING vec0(embedding float[{dim}])"
        )
        self._conn.commit()
        self._dim = dim

    def add(self, episode_id: int, embedding: np.ndarray) -> None:
        if self._dim is None:
            self._ensure_table(len(embedding))
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (episode_id, emb_bytes),
        )
        self._conn.commit()

    def query(self, embedding: np.ndarray, n_results: int) -> list[tuple[int, float]]:
        if self._dim is None:
            return []
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        rows = self._conn.execute(
            "SELECT rowid, distance FROM vec_episodes "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (emb_bytes, n_results),
        ).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def get_embeddings(self, episode_ids: list[int]) -> dict[int, np.ndarray]:
        if not episode_ids or self._dim is None:
            return {}
        placeholders = ",".join("?" * len(episode_ids))
        rows = self._conn.execute(
            f"SELECT rowid, embedding FROM vec_episodes WHERE rowid IN ({placeholders})",
            episode_ids,
        ).fetchall()
        return {
            int(row[0]): np.frombuffer(row[1], dtype=np.float32)
            for row in rows
        }

    def delete(self, episode_id: int) -> None:
        try:
            self._conn.execute(
                "DELETE FROM vec_episodes WHERE rowid = ?", (episode_id,)
            )
            self._conn.commit()
        except Exception:
            pass

    def count(self) -> int:
        if self._dim is None:
            return 0
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM vec_episodes"
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend(
    backend: str,
    conn: sqlite3.Connection,
    vector_store_path: str,
) -> VectorBackend:
    """
    Instantiate a vector backend by name.

    Args:
        backend: "chromadb" (default) or "sqlite-vec"
        conn: Active SQLite connection (used by sqlite-vec backend)
        vector_store_path: Directory for ChromaDB data (ignored by sqlite-vec)

    Returns:
        A VectorBackend instance
    """
    if backend == "chromadb":
        return ChromaBackend(vector_store_path)
    elif backend == "sqlite-vec":
        return SqliteVecBackend(conn)
    else:
        raise ValueError(
            f"Unknown vector backend: {backend!r}. "
            "Valid options are 'chromadb' or 'sqlite-vec'."
        )
