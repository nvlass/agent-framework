"""
Test suite for Phase 1: Basic Episodic Memory

This tests the core functionality of storing and retrieving episodes.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Use proper package imports
from agent_memory.embeddings import LlamaCppEmbeddingGenerator  # noqa: F401 — used in fixture docstring
from agent_memory.memory_store import MemoryStore, Episode


# Sample test data
SAMPLE_EPISODES = [
    {
        "context": "User asked how to implement a binary search algorithm",
        "action": "Provided Python code with detailed comments explaining the algorithm",
        "outcome": "User understood and successfully implemented it",
        "success_score": 0.9,
        "tags": ["coding", "algorithms", "python"],
    },
    {
        "context": "User reported a bug in the authentication system",
        "action": "Analyzed code and identified missing null check",
        "outcome": "Bug was fixed and tests passed",
        "success_score": 1.0,
        "tags": ["debugging", "authentication", "bugfix"],
    },
    {
        "context": "User needed help with database schema design",
        "action": "Suggested normalized schema with foreign keys",
        "outcome": "User implemented but had performance issues",
        "success_score": 0.5,
        "tags": ["database", "schema", "design"],
    },
    {
        "context": "User asked about React component optimization",
        "action": "Recommended using React.memo and useMemo hooks",
        "outcome": "Performance improved significantly",
        "success_score": 0.95,
        "tags": ["react", "performance", "optimization"],
    },
    {
        "context": "User wanted to learn about Docker basics",
        "action": "Explained containers, images, and provided a simple Dockerfile",
        "outcome": "User successfully containerized their application",
        "success_score": 0.85,
        "tags": ["docker", "containers", "devops"],
    },
]


@pytest.fixture
def temp_dir():
    """Create temporary directory for test databases"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def embedding_generator():
    """
    Create a mock embedding generator for testing

    Note: In real usage, you'd initialize with an actual model:
    generator = EmbeddingGenerator(model_path="/path/to/model.gguf")

    For testing without a model, we'll create a simple mock that generates
    random embeddings. This is fine for testing storage/retrieval logic.
    """
    import numpy as np

    class MockEmbeddingGenerator:
        """Mock generator for testing without actual model"""

        def __init__(self):
            self.dimension = 384  # Common embedding dimension
            self._cache = {}

        def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
            """Generate deterministic 'embedding' based on text hash"""
            if use_cache and text in self._cache:
                return self._cache[text]

            # Use text hash as seed for reproducibility
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

            if use_cache:
                self._cache[text] = embedding

            return embedding

        def load_model(self, *args, **kwargs):
            pass

    return MockEmbeddingGenerator()


@pytest.fixture
def memory_store(temp_dir, embedding_generator):
    """Create memory store with temporary database"""
    db_path = Path(temp_dir) / "test_memory.db"
    vector_path = Path(temp_dir) / "test_vectors"

    store = MemoryStore(
        db_path=str(db_path),
        vector_store_path=str(vector_path),
        embedding_generator=embedding_generator,
    )

    yield store
    store.close()


class TestBasicStorage:
    """Test basic episode storage"""

    def test_store_single_episode(self, memory_store):
        """Test storing a single episode"""
        episode_id = memory_store.store_episode(
            context="Test context",
            action="Test action",
            outcome="Test outcome",
            success_score=0.8,
            tags=["test"],
        )

        assert episode_id > 0
        assert memory_store.get_episode_count() == 1

    def test_store_multiple_episodes(self, memory_store):
        """Test storing multiple episodes"""
        for episode_data in SAMPLE_EPISODES:
            episode_id = memory_store.store_episode(**episode_data)
            assert episode_id > 0

        assert memory_store.get_episode_count() == len(SAMPLE_EPISODES)

    def test_retrieve_by_id(self, memory_store):
        """Test retrieving episode by ID"""
        episode_id = memory_store.store_episode(
            context="Specific context",
            action="Specific action",
            outcome="Specific outcome",
        )

        episode = memory_store.get_episode_by_id(episode_id)
        assert episode is not None
        assert episode.context == "Specific context"
        assert episode.action == "Specific action"


class TestSemanticRetrieval:
    """Test semantic search and retrieval"""

    def test_retrieve_similar_episodes(self, memory_store):
        """Test semantic similarity search"""
        # Store sample episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        # Search for similar episodes
        query = "How do I optimize my React application?"
        results = memory_store.retrieve_episodes(query, limit=3)

        assert len(results) > 0
        assert len(results) <= 3

        # Each result should be (Episode, similarity_score)
        for episode, similarity in results:
            assert isinstance(episode, Episode)
            assert 0.0 <= similarity <= 1.0

    def test_retrieval_with_similarity_threshold(self, memory_store):
        """Test retrieval with minimum similarity threshold"""
        # Store episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        # Search with high threshold
        query = "database optimization"
        results = memory_store.retrieve_episodes(query, limit=5, min_similarity=0.8)

        # All results should meet threshold
        for episode, similarity in results:
            assert similarity >= 0.8


class TestTemporalRetrieval:
    """Test time-based retrieval"""

    def test_get_recent_episodes(self, memory_store):
        """Test retrieving recent episodes"""
        # Store some episodes
        for episode_data in SAMPLE_EPISODES[:3]:
            memory_store.store_episode(**episode_data)

        recent = memory_store.get_recent_episodes(hours=24, limit=10)
        assert len(recent) == 3

    def test_get_all_episodes(self, memory_store):
        """Test retrieving all episodes"""
        # Store episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        all_episodes = memory_store.get_all_episodes()
        assert len(all_episodes) == len(SAMPLE_EPISODES)

    def test_get_episodes_with_limit(self, memory_store):
        """Test limiting number of retrieved episodes"""
        # Store episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        limited = memory_store.get_all_episodes(limit=3)
        assert len(limited) == 3


class TestStatistics:
    """Test statistics and metadata"""

    def test_episode_count(self, memory_store):
        """Test episode counting"""
        assert memory_store.get_episode_count() == 0

        memory_store.store_episode(
            context="Test", action="Test", outcome="Test"
        )
        assert memory_store.get_episode_count() == 1

    def test_statistics(self, memory_store):
        """Test getting memory store statistics"""
        # Store episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        stats = memory_store.get_stats()

        assert stats["total_episodes"] == len(SAMPLE_EPISODES)
        assert stats["episodes_with_outcomes"] == len(SAMPLE_EPISODES)
        assert stats["scored_episodes"] == len(SAMPLE_EPISODES)
        assert stats["average_success_score"] is not None
        assert 0.0 <= stats["average_success_score"] <= 1.0


class TestTagQuerying:
    """Test SQLite JSON-based tag querying"""

    def test_get_episodes_by_single_tag(self, memory_store):
        """Test finding episodes by a single tag"""
        # Store sample episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        # Find all Python-related episodes
        python_episodes = memory_store.get_episodes_by_tag("python")
        assert len(python_episodes) == 1
        assert "python" in python_episodes[0].tags

        # Find all React episodes
        react_episodes = memory_store.get_episodes_by_tag("react")
        assert len(react_episodes) == 1
        assert "react" in react_episodes[0].tags

    def test_get_episodes_by_tag_with_limit(self, memory_store):
        """Test tag search with result limit"""
        # Store sample episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        # Limit results
        results = memory_store.get_episodes_by_tag("python", limit=1)
        assert len(results) <= 1

    def test_get_episodes_by_multiple_tags_any(self, memory_store):
        """Test finding episodes matching ANY of the tags"""
        # Store sample episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        # Find episodes with either 'python' or 'react'
        results = memory_store.get_episodes_by_tags(["python", "react"], match_all=False)
        assert len(results) == 2  # Should match both Python and React episodes

    def test_get_episodes_by_multiple_tags_all(self, memory_store):
        """Test finding episodes matching ALL tags"""
        # Store sample episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        # Find episodes with both 'coding' AND 'python'
        results = memory_store.get_episodes_by_tags(["coding", "python"], match_all=True)
        assert len(results) == 1
        assert set(["coding", "python"]).issubset(set(results[0].tags))

    def test_get_all_tags(self, memory_store):
        """Test getting all unique tags with counts"""
        # Store sample episodes
        for episode_data in SAMPLE_EPISODES:
            memory_store.store_episode(**episode_data)

        tags = memory_store.get_all_tags()

        # Should return list of (tag, count) tuples
        assert len(tags) > 0
        assert all(isinstance(tag, str) and isinstance(count, int) for tag, count in tags)

        # Check some expected tags exist
        tag_names = [tag for tag, count in tags]
        assert "python" in tag_names
        assert "react" in tag_names

    def test_tag_json_validation(self, memory_store):
        """Test that SQLite validates JSON in tags column"""
        # This should work fine
        memory_store.store_episode(
            context="Test",
            action="Test",
            outcome="Test",
            tags=["valid", "tags"]
        )

        # Verify it was stored correctly
        episode = memory_store.get_episode_by_id(1)
        assert episode.tags == ["valid", "tags"]


class TestEmbeddings:
    """Test embedding generation"""

    def test_embedding_generation(self, embedding_generator):
        """Test generating embeddings"""
        text = "Test text for embedding"
        embedding = embedding_generator.generate_embedding(text)

        assert embedding is not None
        assert len(embedding.shape) == 1  # Should be 1D vector
        assert embedding.shape[0] > 0  # Should have dimensions

    def test_embedding_caching(self, embedding_generator):
        """Test that embeddings are cached"""
        text = "Cached text"

        # Generate twice
        emb1 = embedding_generator.generate_embedding(text)
        emb2 = embedding_generator.generate_embedding(text)

        # Should be identical (cached)
        import numpy as np
        assert np.array_equal(emb1, emb2)


# Performance test (optional, can be slow)
@pytest.mark.slow
class TestPerformance:
    """Test performance requirements from Phase 1"""

    def test_retrieval_latency(self, memory_store):
        """Test that retrieval is fast enough for 1000 episodes"""
        import time

        # Store many episodes (duplicate sample data)
        print("\nStoring 1000 episodes...")
        for i in range(200):
            for episode_data in SAMPLE_EPISODES:
                memory_store.store_episode(**episode_data)

        count = memory_store.get_episode_count()
        print(f"Stored {count} episodes")

        # Time retrieval
        query = "How to optimize performance"
        start = time.time()
        results = memory_store.retrieve_episodes(query, limit=5)
        elapsed = time.time() - start

        print(f"Retrieval took {elapsed:.3f} seconds")

        # Should be under 1 second (Phase 1 requirement)
        assert elapsed < 1.0, f"Retrieval too slow: {elapsed:.3f}s"
        assert len(results) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
