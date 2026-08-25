"""Tests for associative memory features: tag aggregation and analogy recall."""

import numpy as np
import pytest

from agent_memory.memory_store import MemoryStore
from agent_memory.memory_tools import MemoryTools


class MockEmbeddingGenerator:
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
    )
    yield s
    s.close()


class TestTagCounts:
    def test_aggregates_and_sorts(self, store):
        store.store_episode(context="c", action="a", outcome="o1",
                            tags=["lua", "note"])
        store.store_episode(context="c", action="a", outcome="o2",
                            tags=["lua", "consciousness"])
        store.store_episode(context="c", action="a", outcome="o3",
                            tags=["consciousness"])
        counts = dict(store.get_tag_counts())
        assert counts["lua"] == 2
        assert counts["consciousness"] == 2
        assert counts["note"] == 1

    def test_skip_tags_excluded(self, store):
        store.store_episode(context="c", action="a", outcome="o",
                            tags=["note", "lua"])
        counts = dict(store.get_tag_counts(skip_tags={"note"}))
        assert "note" not in counts
        assert "lua" in counts

    def test_occurrence_count_weights_log_damped(self, store):
        ep_id = store.store_episode(context="c", action="a", outcome="o1",
                                    tags=["lua"])
        store.store_episode(context="c", action="a", outcome="o2",
                            tags=["scheme"])
        for _ in range(7):  # occurrence_count -> 8
            store.reinforce_episode(ep_id)
        counts = dict(store.get_tag_counts())
        assert counts["lua"] == 4  # 1 + log2(8)
        assert counts["scheme"] == 1

    def test_runaway_count_does_not_drown_index(self, store):
        ep_id = store.store_episode(context="c", action="a", outcome="o1",
                                    tags=["runaway"])
        store.conn.execute(
            "UPDATE episodes SET occurrence_count = 1800 WHERE id = ?", (ep_id,))
        store.conn.commit()
        for i in range(5):
            store.store_episode(context="c", action="a", outcome=f"o{i}",
                                tags=["organic"])
        counts = dict(store.get_tag_counts())
        assert counts["runaway"] == 11  # 1 + floor(log2(1800)), not 1800
        assert counts["organic"] == 5

    def test_empty_store(self, store):
        assert store.get_tag_counts() == []

    def test_tags_normalized_lowercase(self, store):
        store.store_episode(context="c", action="a", outcome="o1", tags=["Lua"])
        store.store_episode(context="c", action="a", outcome="o2", tags=["lua"])
        counts = dict(store.get_tag_counts())
        assert counts["lua"] == 2


class TestRecallAnalogies:
    def test_returns_toolresult_with_analogies_list(self, store):
        tools = MemoryTools(store=store)
        for i in range(5):
            store.store_episode(
                context=f"debugging situation {i}",
                action=f"isolated the failing component {i}",
                outcome=f"found root cause {i}",
                tags=["debugging", "python"],
            )
        result = tools.recall_analogies("hardware fault isolation")
        assert result.success
        assert "analogies" in result.data
        for a in result.data["analogies"]:
            assert {"episode_id", "memory", "similarity",
                    "domain_distance"} <= set(a)

    def test_empty_store_is_graceful(self, store):
        tools = MemoryTools(store=store)
        result = tools.recall_analogies("anything")
        assert result.success
        assert result.data["analogies"] == []

    def test_no_embedding_generator_errors_cleanly(self, tmp_path):
        s = MemoryStore(
            db_path=str(tmp_path / "noembed.db"),
            vector_store_path=str(tmp_path / "vec"),
            embedding_generator=None,
        )
        try:
            tools = MemoryTools(store=s)
            result = tools.recall_analogies("anything")
            assert not result.success
            assert "embedding" in (result.error or "").lower()
        finally:
            s.close()
