"""Tests for deduplication: novelty gate + reinforcement.

Covers find_similar_episode / find_similar_reflection, reinforce_*,
store_reflection_if_novel, the dedup path in MemoryTools.store_memory,
and the schema migration for pre-dedup databases.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from agent_memory.memory_store import MemoryStore, Reflection
from agent_memory.memory_tools import MemoryTools


class MockEmbeddingGenerator:
    """Hash-seeded deterministic embeddings: same text → identical vector,
    different text → near-orthogonal."""

    def __init__(self):
        self.dimension = 384
        self._cache = {}

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        if use_cache and text in self._cache:
            return self._cache[text]
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        if use_cache:
            self._cache[text] = embedding
        return embedding

    def generate_embeddings_batch(self, texts, use_cache: bool = True):
        return [self.generate_embedding(t, use_cache) for t in texts]


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(
        db_path=str(tmp_path / "test.db"),
        vector_store_path=str(tmp_path / "vectors"),
        embedding_generator=MockEmbeddingGenerator(),
        backend="sqlite-vec" if _has_sqlite_vec() else "chromadb",
    )
    yield s
    s.close()


def _has_sqlite_vec() -> bool:
    try:
        import sqlite_vec  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Episode dedup
# =============================================================================

class TestEpisodeDedup:
    def test_exact_duplicate_detected(self, store):
        ep_id = store.store_episode(
            context="conversation", action="auto-extracted fact",
            outcome="User prefers Clojure for backend work.",
        )
        match = store.find_similar_episode(
            "conversation", "auto-extracted fact",
            "User prefers Clojure for backend work.",
        )
        assert match is not None
        assert match[0] == ep_id
        assert match[1] > 0.99

    def test_different_outcome_is_not_duplicate(self, store):
        store.store_episode(
            context="conversation", action="auto-extracted fact",
            outcome="User prefers Clojure for backend work.",
        )
        match = store.find_similar_episode(
            "conversation", "auto-extracted fact",
            "User is building a Raspberry Pi research agent.",
        )
        assert match is None

    def test_reinforce_bumps_count_and_timestamp(self, store):
        ep_id = store.store_episode(context="c", action="a", outcome="o")
        assert store.reinforce_episode(ep_id) == 2
        assert store.reinforce_episode(ep_id) == 3
        row = store.conn.execute(
            "SELECT occurrence_count, last_confirmed FROM episodes WHERE id = ?",
            (ep_id,),
        ).fetchone()
        assert row[0] == 3
        assert row[1] is not None

    def test_reinforce_missing_episode_returns_zero(self, store):
        assert store.reinforce_episode(99999) == 0

    def test_store_memory_dedup_reinforces(self, store):
        tools = MemoryTools(store=store)
        first = tools.store_memory(
            context="conversation", action="auto-extracted fact",
            outcome="User prefers Clojure for backend work.", dedup=True,
        )
        second = tools.store_memory(
            context="conversation", action="auto-extracted fact",
            outcome="User prefers Clojure for backend work.", dedup=True,
        )
        assert first.success and second.success
        assert second.data["reinforced"] is True
        assert second.data["episode_id"] == first.data["episode_id"]
        assert second.data["occurrence_count"] == 2
        n = store.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert n == 1

    def test_store_memory_without_dedup_duplicates(self, store):
        tools = MemoryTools(store=store)
        tools.store_memory(context="c", action="a", outcome="same")
        tools.store_memory(context="c", action="a", outcome="same")
        n = store.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert n == 2

    def test_atoms_with_same_context_action_stay_distinct(self, store):
        """Regression: outcome must be part of the embedded text, otherwise
        every atom/note looks identical in vector space."""
        tools = MemoryTools(store=store)
        r1 = tools.store_memory(
            context="conversation", action="auto-extracted fact",
            outcome="Fact one about Clojure.", dedup=True,
        )
        r2 = tools.store_memory(
            context="conversation", action="auto-extracted fact",
            outcome="A completely different fact about Raspberry Pi hardware.",
            dedup=True,
        )
        assert not r2.data.get("reinforced")
        assert r1.data["episode_id"] != r2.data["episode_id"]


# =============================================================================
# Reflection dedup
# =============================================================================

class TestReflectionDedup:
    def _reflection(self, insight: str) -> Reflection:
        return Reflection(reflection_type="pattern_discovery", insight=insight)

    def test_duplicate_insight_reinforced(self, store):
        insight = "Curiosity-driven research covers varied topics but lacks depth."
        first_id, novel = store.store_reflection_if_novel(self._reflection(insight))
        assert novel is True
        second_id, novel = store.store_reflection_if_novel(self._reflection(insight))
        assert novel is False
        assert second_id == first_id
        row = store.conn.execute(
            "SELECT COUNT(*), MAX(occurrence_count) FROM reflections"
        ).fetchone()
        assert row == (1, 2)

    def test_new_insight_is_novel(self, store):
        store.store_reflection_if_novel(
            self._reflection("Research lacks depth across topics."))
        _, novel = store.store_reflection_if_novel(
            self._reflection("Tool failures cluster around network timeouts."))
        assert novel is True
        n = store.conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        assert n == 2

    def test_type_scoping(self, store):
        """Same insight under a different reflection_type is not a duplicate."""
        insight = "Identical insight text."
        store.store_reflection_if_novel(self._reflection(insight))
        r = Reflection(reflection_type="failure_analysis", insight=insight)
        _, novel = store.store_reflection_if_novel(r)
        assert novel is True

    def test_no_embedding_generator_skips_gate(self, tmp_path):
        s = MemoryStore(
            db_path=str(tmp_path / "noembed.db"),
            vector_store_path=str(tmp_path / "vec"),
            embedding_generator=None,
        )
        try:
            insight = "Some insight."
            _, novel1 = s.store_reflection_if_novel(self._reflection(insight))
            _, novel2 = s.store_reflection_if_novel(self._reflection(insight))
            # Without embeddings the gate can't judge similarity — both stored
            assert novel1 is True and novel2 is True
        finally:
            s.close()


# =============================================================================
# Schema migration
# =============================================================================

class TestMigration:
    def test_old_db_gains_columns(self, tmp_path):
        """A database created before the dedup columns existed gets them
        added on open, with existing rows intact."""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT,
                success_score REAL,
                tags TEXT DEFAULT '[]',
                embedding_id TEXT,
                outcome_category TEXT,
                failure_reason TEXT
            );
            CREATE TABLE reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reflection_type TEXT NOT NULL,
                trigger_episode_id INTEGER,
                insight TEXT NOT NULL,
                causal_chain TEXT DEFAULT '[]',
                actionable_takeaway TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                embedding_id TEXT
            );
            INSERT INTO episodes (context, action, outcome) VALUES ('c', 'a', 'o');
        """)
        conn.commit()
        conn.close()

        s = MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(tmp_path / "vec"),
            embedding_generator=MockEmbeddingGenerator(),
        )
        try:
            cols = {r[1] for r in s.conn.execute("PRAGMA table_info(episodes)")}
            assert {"occurrence_count", "last_confirmed"} <= cols
            cols = {r[1] for r in s.conn.execute("PRAGMA table_info(reflections)")}
            assert {"occurrence_count", "last_confirmed"} <= cols
            # Pre-existing row still readable, count defaults sensibly
            assert s.reinforce_episode(1) == 2
        finally:
            s.close()
