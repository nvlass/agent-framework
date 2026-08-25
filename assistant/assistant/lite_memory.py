"""Lightweight SQLite+FTS5 memory backend.

Drop-in replacement for agent_memory.MemoryTools on environments where
ChromaDB/HDBSCAN are unavailable (Pi Zero, minimal installs).

Implements the same duck-typed interface as MemoryTools:
  store_memory(context, action, outcome, tags)
  recall_similar(query, limit)   → _Result(data={'memories': [...]})
  reflect_on_recent(hours, focus) → _Result

Uses SQLite FTS5 for full-text search — zero native dependencies.
Tags are stored as plain text inside the FTS index so they are
searchable alongside the note content.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS lite_memories USING fts5(
    context,
    action,
    outcome,
    tags,
    created_at UNINDEXED
);
"""

_REFLECT_PROMPT = """\
Review the following memory notes and write a concise 2-3 sentence summary \
of the key themes and any actionable insights worth remembering.

{notes}"""


@dataclass
class _Result:
    success: bool
    data: dict[str, Any] | None = None
    message: str = ""
    error: str | None = None

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        return self.message or self.error or ""


class LiteMemory:
    """SQLite FTS5 memory — no native dependencies required."""

    def __init__(self, db_path, llm=None) -> None:
        self._db = str(db_path)
        self._llm = llm
        with self._conn() as conn:
            try:
                conn.executescript(_SCHEMA)
            except sqlite3.OperationalError as exc:
                if "fts5" in str(exc).lower():
                    raise RuntimeError(
                        "SQLite FTS5 extension not available. "
                        "Rebuild Python with a newer SQLite (>=3.20)."
                    ) from exc
                raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _rows_to_memories(self, rows) -> list[dict]:
        result = []
        for r in rows:
            result.append({
                "context": r["context"],
                "action": r["action"],
                "outcome": r["outcome"],
                "tags": r["tags"],
                "created_at": r["created_at"],
            })
        return result

    # ------------------------------------------------------------------
    # Public interface (mirrors MemoryTools)
    # ------------------------------------------------------------------

    def store_memory(
        self,
        context: str,
        action: str,
        outcome: str = "",
        tags: list | None = None,
        **_compat,  # accepts MemoryTools kwargs (importance, dedup, ...) — ignored
    ) -> _Result:
        tags_str = " ".join(tags) if isinstance(tags, list) else (tags or "")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO lite_memories "
                    "(context, action, outcome, tags, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (context, action, outcome, tags_str, now),
                )
            return _Result(success=True, message="Memory stored.")
        except Exception as exc:
            return _Result(success=False, error=str(exc))

    def recall_similar(self, query: str, limit: int = 5) -> _Result:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT context, action, outcome, tags, created_at "
                    "FROM lite_memories WHERE lite_memories MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (self._fts_escape(query), limit),
                ).fetchall()
            memories = self._rows_to_memories(rows)
            return _Result(
                success=True,
                data={"memories": memories, "count": len(memories)},
                message=f"Found {len(memories)} memories.",
            )
        except Exception:
            # FTS MATCH failed (special chars etc.) — fall back to LIKE
            return self._recall_like(query, limit)

    def reflect_on_recent(self, hours: int = 1, focus: str | None = None) -> _Result:
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT context, action, outcome, created_at "
                    "FROM lite_memories WHERE created_at >= ? "
                    "ORDER BY created_at DESC LIMIT 50",
                    (since,),
                ).fetchall()
        except Exception as exc:
            return _Result(success=False, error=str(exc))

        if not rows:
            return _Result(
                success=True,
                data={"count": 0},
                message="No recent memories to analyze.",
            )

        if not self._llm:
            return _Result(
                success=True,
                data={"count": len(rows)},
                message=f"Reviewed {len(rows)} recent memories (no LLM for synthesis).",
            )

        notes = "\n".join(
            f"- [{r['created_at']}] {r['outcome'] or r['action']}"
            for r in rows
        )
        if focus:
            notes = f"Focus: {focus}\n\n{notes}"

        try:
            from assistant.conversation import _call_llm_raw
            response = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": _REFLECT_PROMPT.format(notes=notes)}],
                max_tokens=256,
                temperature=0.4,
            )
            insight = response["choices"][0]["message"].get("content", "").strip()
            return _Result(success=True, data={"insight": insight}, message=insight)
        except Exception as exc:
            return _Result(success=False, error=f"Reflection LLM failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _recall_like(self, query: str, limit: int) -> _Result:
        pattern = f"%{query}%"
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT context, action, outcome, tags, created_at "
                    "FROM lite_memories "
                    "WHERE outcome LIKE ? OR context LIKE ? OR action LIKE ? OR tags LIKE ? "
                    "ORDER BY rowid DESC LIMIT ?",
                    (pattern, pattern, pattern, pattern, limit),
                ).fetchall()
            memories = self._rows_to_memories(rows)
            return _Result(
                success=True,
                data={"memories": memories, "count": len(memories)},
            )
        except Exception as exc:
            return _Result(success=False, error=str(exc))

    @staticmethod
    def _fts_escape(query: str) -> str:
        """Wrap query in double quotes for FTS5 phrase search, escaping internal quotes."""
        escaped = query.replace('"', '""')
        return f'"{escaped}"'
