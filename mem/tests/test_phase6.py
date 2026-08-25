"""
Tests for Phase 6: Configuration, Tools, and Metrics

Tests cover:
- Configuration management (config.py)
- Agent-facing tools interface (memory_tools.py)
- Health monitoring and metrics (metrics.py)
"""

import pytest
import json
import tempfile
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from agent_memory import (
    # Config
    MemoryConfig,
    MemorySettings,
    ConsolidationSettings,
    ReflectionSettings,
    ForgettingSettings,
    AdaptationSettings,
    PerformanceSettings,
    LoggingSettings,
    load_config,
    create_default_config_file,
    get_config_template,
    # Tools
    MemoryTools,
    ToolResult,
    MemoryEntry,
    StrategyAdvice,
    # Metrics
    MemoryMetrics,
    HealthStatus,
    UsageMetrics,
    LearningMetrics,
    PerformanceMetrics,
    # Core
    MemoryStore,
    MockLLM,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for testing"""

    class MockEmbeddingGenerator:
        """Mock generator for testing without actual model"""

        def __init__(self):
            self.dimension = 384
            self._cache = {}

        def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
            """Generate deterministic 'embedding' based on text hash"""
            if use_cache and text in self._cache:
                return self._cache[text]

            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            if use_cache:
                self._cache[text] = embedding

            return embedding

        def load_model(self, *args, **kwargs):
            pass

    return MockEmbeddingGenerator()


# =============================================================================
# Configuration Tests
# =============================================================================

class TestMemorySettings:
    """Tests for MemorySettings dataclass"""

    def test_default_values(self):
        settings = MemorySettings()
        assert settings.db_path == "data/agent_memory.db"
        assert settings.vector_store_path == "data/memory_vectors"
        assert settings.embedding_model_path is None
        assert settings.working_memory_size == 20
        assert settings.short_term_ttl_seconds == 300

    def test_custom_values(self):
        settings = MemorySettings(
            db_path="/custom/path.db",
            working_memory_size=50
        )
        assert settings.db_path == "/custom/path.db"
        assert settings.working_memory_size == 50


class TestConsolidationSettings:
    """Tests for ConsolidationSettings dataclass"""

    def test_default_values(self):
        settings = ConsolidationSettings()
        assert settings.enabled is True
        assert settings.trigger_after_episodes == 20
        assert settings.min_cluster_size == 3
        assert settings.similarity_threshold == 0.75

    def test_custom_values(self):
        settings = ConsolidationSettings(
            enabled=False,
            trigger_after_episodes=50
        )
        assert settings.enabled is False
        assert settings.trigger_after_episodes == 50


class TestMemoryConfig:
    """Tests for MemoryConfig container"""

    def test_default_initialization(self):
        config = MemoryConfig()
        assert isinstance(config.memory, MemorySettings)
        assert isinstance(config.consolidation, ConsolidationSettings)
        assert isinstance(config.reflection, ReflectionSettings)
        assert isinstance(config.forgetting, ForgettingSettings)
        assert isinstance(config.adaptation, AdaptationSettings)
        assert isinstance(config.performance, PerformanceSettings)
        assert isinstance(config.logging, LoggingSettings)

    def test_to_dict(self):
        config = MemoryConfig()
        data = config.to_dict()

        assert 'memory' in data
        assert 'consolidation' in data
        assert 'reflection' in data
        assert data['memory']['db_path'] == "data/agent_memory.db"
        assert data['consolidation']['enabled'] is True

    def test_from_dict(self):
        data = {
            'memory': {'db_path': '/custom/db.sqlite'},
            'consolidation': {'enabled': False},
        }
        config = MemoryConfig.from_dict(data)

        assert config.memory.db_path == '/custom/db.sqlite'
        assert config.consolidation.enabled is False
        # Other settings should be defaults
        assert config.reflection.enabled is True

    def test_save_and_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")

            # Create and save config
            config = MemoryConfig()
            config.memory.db_path = "/test/path.db"
            config.save(config_path)

            # Verify file exists and contains JSON
            assert os.path.exists(config_path)
            with open(config_path) as f:
                data = json.load(f)
            assert data['memory']['db_path'] == "/test/path.db"

            # Load and verify
            loaded = load_config(config_path)
            assert loaded.memory.db_path == "/test/path.db"

    def test_validate_valid_config(self):
        config = MemoryConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_config(self):
        config = MemoryConfig()
        config.memory.working_memory_size = 0  # Invalid
        config.consolidation.similarity_threshold = 1.5  # Invalid

        errors = config.validate()
        assert len(errors) >= 2
        assert any("working_memory_size" in e for e in errors)
        assert any("similarity_threshold" in e for e in errors)

    def test_get_config_template(self):
        template = get_config_template()
        assert "memory:" in template
        assert "consolidation:" in template
        assert "reflection:" in template
        assert "db_path" in template


class TestLoadConfig:
    """Tests for config loading functions"""

    def test_load_nonexistent_returns_defaults(self):
        config = load_config("/nonexistent/path.yaml")
        assert config.memory.db_path == "data/agent_memory.db"

    def test_create_default_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            create_default_config_file(config_path)

            assert os.path.exists(config_path)

            # Load and verify it's valid JSON
            with open(config_path) as f:
                data = json.load(f)
            assert 'memory' in data

            # Load via config system
            config = load_config(config_path)
            assert config.memory.db_path == "data/agent_memory.db"


# =============================================================================
# Memory Tools Tests
# =============================================================================

class TestToolResult:
    """Tests for ToolResult dataclass"""

    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        result = ToolResult(success=True, data={"count": 5}, message="Found 5 items")
        data = result.to_dict()

        assert data['success'] is True
        assert data['data'] == {"count": 5}
        assert data['message'] == "Found 5 items"


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass"""

    def test_creation(self):
        entry = MemoryEntry(
            id=1,
            context="Test context",
            action="Test action",
            outcome="Test outcome",
            success_score=0.8,
            tags=["test", "unit"],
            timestamp="2024-01-01T12:00:00"
        )
        assert entry.id == 1
        assert entry.success_score == 0.8
        assert "test" in entry.tags
        assert entry.timestamp == "2024-01-01T12:00:00"

    def test_with_similarity(self):
        entry = MemoryEntry(
            id=1,
            context="Test",
            action="Action",
            outcome="Outcome",
            success_score=0.5,
            tags=[],
            timestamp="2024-01-01",
            similarity=0.95
        )
        assert entry.similarity == 0.95


class TestMemoryTools:
    """Tests for MemoryTools agent interface"""

    @pytest.fixture
    def tools(self, tmp_path, mock_embedding_generator):
        """Create MemoryTools instance with temp database"""
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"

        store = MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(vector_path),
            embedding_generator=mock_embedding_generator
        )
        return MemoryTools(store)

    def test_store_memory(self, tools):
        result = tools.store_memory(
            context="Testing Python code",
            action="Added unit test",
            outcome="Test passed",
            importance=8,
            tags=["python", "testing"]
        )

        assert result.success is True
        assert 'episode_id' in result.data
        assert result.data['episode_id'] > 0

    def test_store_memory_minimal(self, tools):
        result = tools.store_memory(
            context="Minimal test",
            action="Did something"
        )

        assert result.success is True

    def test_recall_recent(self, tools):
        # Store some memories
        tools.store_memory("Context 1", "Action 1", "Outcome 1")
        tools.store_memory("Context 2", "Action 2", "Outcome 2")

        result = tools.recall_recent(limit=5)

        assert result.success is True
        assert 'memories' in result.data
        assert len(result.data['memories']) == 2

    def test_recall_by_tags(self, tools):
        # Store with tags
        tools.store_memory("Python code", "Debug", "Fixed", tags=["python"])
        tools.store_memory("JavaScript code", "Debug", "Fixed", tags=["javascript"])

        result = tools.recall_by_tags(tags=["python"])

        assert result.success is True
        memories = result.data['memories']
        assert len(memories) == 1
        assert "Python" in memories[0]['context']

    def test_learn_from_outcome_success(self, tools):
        # First store a memory
        store_result = tools.store_memory("Test context", "Test action", "Test outcome")
        episode_id = store_result.data['episode_id']

        # Learn from it
        result = tools.learn_from_outcome(
            episode_id=episode_id,
            success=True,
            reasoning="It worked because of good approach"
        )

        assert result.success is True
        assert result.data['category'] == 'success'
        assert result.data['updated'] is True

    def test_learn_from_outcome_failure(self, tools):
        store_result = tools.store_memory("Test context", "Test action", "Test outcome")
        episode_id = store_result.data['episode_id']

        result = tools.learn_from_outcome(
            episode_id=episode_id,
            success=False,
            reasoning="Missing validation"
        )

        assert result.success is True
        assert result.data['category'] == 'failure'
        assert result.data['updated'] is True

    def test_learn_from_outcome_invalid_id(self, tools):
        result = tools.learn_from_outcome(
            episode_id=99999,
            success=True
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_get_memory_stats(self, tools):
        # Add some data
        tools.store_memory("Context", "Action", "Outcome")

        result = tools.get_memory_stats()

        assert result.success is True
        assert 'total_episodes' in result.data
        assert result.data['total_episodes'] >= 1

    def test_get_success_rate(self, tools):
        # Store memories with outcomes
        for i in range(5):
            store_result = tools.store_memory(
                f"Python context {i}", "Python action", f"Outcome {i}",
                tags=["python"]
            )
            tools.learn_from_outcome(store_result.data['episode_id'], success=(i < 3))

        result = tools.get_success_rate(tags=["python"])

        assert result.success is True
        assert 'success_rate' in result.data

    def test_get_tool_definitions(self):
        definitions = MemoryTools.get_tool_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) > 0

        # Check structure
        for tool in definitions:
            assert 'name' in tool
            assert 'description' in tool
            assert 'parameters' in tool

        # Check expected tools exist
        tool_names = [t['name'] for t in definitions]
        assert 'store_memory' in tool_names
        assert 'recall_similar' in tool_names
        assert 'learn_from_outcome' in tool_names


class TestMemoryToolsWithLLM:
    """Tests for MemoryTools that require LLM"""

    @pytest.fixture
    def tools_with_llm(self, tmp_path, mock_embedding_generator):
        """Create MemoryTools with mock LLM"""
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"

        store = MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(vector_path),
            embedding_generator=mock_embedding_generator
        )
        llm = MockLLM()
        return MemoryTools(store, llm=llm)

    def test_reflect_on_recent(self, tools_with_llm):
        # Store some memories with scores
        for i in range(3):
            result = tools_with_llm.store_memory(
                f"Context {i}", f"Action {i}", f"Outcome {i}",
                importance=3  # Low importance = low score
            )

        result = tools_with_llm.reflect_on_recent(hours=24)

        assert result.success is True
        # May or may not have reflections based on scores

    def test_get_strategy_advice(self, tools_with_llm):
        # Add some successful patterns
        for i in range(3):
            store_result = tools_with_llm.store_memory(
                "Python debugging", "Used print statements", "Found the bug"
            )
            tools_with_llm.learn_from_outcome(store_result.data['episode_id'], success=True)

        result = tools_with_llm.get_strategy_advice(
            context="I have a Python error",
            goal="Debug the code"
        )

        assert result.success is True


# =============================================================================
# Metrics Tests
# =============================================================================

class TestHealthStatus:
    """Tests for HealthStatus dataclass"""

    def test_healthy_status(self):
        status = HealthStatus(healthy=True, score=0.95)
        assert status.healthy is True
        assert status.score == 0.95
        assert status.warnings == []
        assert status.errors == []

    def test_unhealthy_status(self):
        status = HealthStatus(
            healthy=False,
            score=0.5,
            warnings=["Warning 1"],
            errors=["Error 1"]
        )
        assert status.healthy is False
        assert len(status.warnings) == 1
        assert len(status.errors) == 1

    def test_to_dict(self):
        status = HealthStatus(healthy=True, score=0.8, warnings=["Test warning"])
        data = status.to_dict()

        assert data['healthy'] is True
        assert data['score'] == 0.8
        assert "Test warning" in data['warnings']


class TestUsageMetrics:
    """Tests for UsageMetrics dataclass"""

    def test_default_values(self):
        metrics = UsageMetrics()
        assert metrics.total_episodes == 0
        assert metrics.total_patterns == 0
        assert metrics.total_reflections == 0

    def test_to_dict(self):
        metrics = UsageMetrics(total_episodes=100, total_patterns=10)
        data = metrics.to_dict()

        assert data['total_episodes'] == 100
        assert data['total_patterns'] == 10


class TestLearningMetrics:
    """Tests for LearningMetrics dataclass"""

    def test_default_values(self):
        metrics = LearningMetrics()
        assert metrics.avg_success_score is None
        assert metrics.success_rate == 0.0
        assert metrics.success_trend == "stable"

    def test_with_values(self):
        metrics = LearningMetrics(
            avg_success_score=0.75,
            success_rate=0.8,
            failure_rate=0.2,
            success_trend="improving"
        )
        assert metrics.avg_success_score == 0.75
        assert metrics.success_trend == "improving"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass"""

    def test_default_values(self):
        metrics = PerformanceMetrics()
        assert metrics.retrieval_latency_ms is None
        assert metrics.working_memory_size == 0

    def test_to_dict(self):
        metrics = PerformanceMetrics(
            retrieval_latency_ms=50.5,
            working_memory_size=10
        )
        data = metrics.to_dict()

        assert data['retrieval_latency_ms'] == 50.5
        assert data['working_memory_size'] == 10


class TestMemoryMetrics:
    """Tests for MemoryMetrics class"""

    @pytest.fixture
    def metrics(self, tmp_path, mock_embedding_generator):
        """Create MemoryMetrics with temp database"""
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"

        store = MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(vector_path),
            embedding_generator=mock_embedding_generator
        )
        return MemoryMetrics(store)

    @pytest.fixture
    def populated_metrics(self, tmp_path, mock_embedding_generator):
        """Create MemoryMetrics with some data"""
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"

        store = MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(vector_path),
            embedding_generator=mock_embedding_generator
        )

        # Add episodes
        for i in range(10):
            store.store_episode(
                context=f"Context {i}",
                action=f"Action {i}",
                outcome=f"Outcome {i}",
                success_score=0.5 + (i * 0.05),  # 0.5 to 0.95
                tags=["test"]
            )

        return MemoryMetrics(store)

    def test_get_usage_metrics_empty(self, metrics):
        usage = metrics.get_usage_metrics()

        assert isinstance(usage, UsageMetrics)
        assert usage.total_episodes == 0

    def test_get_usage_metrics_with_data(self, populated_metrics):
        usage = populated_metrics.get_usage_metrics()

        assert usage.total_episodes == 10
        assert usage.episodes_last_24h == 10
        assert usage.episodes_last_7d == 10

    def test_get_learning_metrics(self, populated_metrics):
        learning = populated_metrics.get_learning_metrics()

        assert isinstance(learning, LearningMetrics)
        assert learning.avg_success_score is not None
        assert learning.avg_success_score > 0.5

    def test_get_performance_metrics(self, populated_metrics):
        performance = populated_metrics.get_performance_metrics()

        assert isinstance(performance, PerformanceMetrics)
        assert performance.working_memory_size >= 0

    def test_get_health_report(self, populated_metrics):
        report = populated_metrics.get_health_report()

        assert 'timestamp' in report
        assert 'health' in report
        assert 'usage' in report
        assert 'learning' in report
        assert 'performance' in report

        # Health should be assessed
        assert 'healthy' in report['health']
        assert 'score' in report['health']

    def test_get_summary(self, populated_metrics):
        summary = populated_metrics.get_summary()

        assert isinstance(summary, str)
        assert "MEMORY SYSTEM HEALTH REPORT" in summary
        assert "Episodes:" in summary
        assert "Success Rate:" in summary

    def test_health_assessment_healthy(self, populated_metrics):
        report = populated_metrics.get_health_report()

        # With 10 episodes and decent scores, should be healthy
        assert report['health']['healthy'] is True
        assert report['health']['score'] > 0.7


class TestMemoryMetricsHealthAssessment:
    """Tests for health assessment edge cases"""

    @pytest.fixture
    def store(self, tmp_path, mock_embedding_generator):
        db_path = tmp_path / "test.db"
        vector_path = tmp_path / "vectors"
        return MemoryStore(
            db_path=str(db_path),
            vector_store_path=str(vector_path),
            embedding_generator=mock_embedding_generator
        )

    def test_warning_on_low_success_score(self, store):
        # Add episodes with low success scores
        for i in range(20):
            store.store_episode(
                context=f"Context {i}",
                action=f"Action {i}",
                outcome=f"Outcome {i}",
                success_score=0.3,  # Low score
                tags=["test"]
            )

        metrics = MemoryMetrics(store)
        report = metrics.get_health_report()

        # Should have warning about low success score
        assert any("success" in w.lower() for w in report['health']['warnings'])

    def test_empty_database_healthy(self, store):
        metrics = MemoryMetrics(store)
        report = metrics.get_health_report()

        # Empty database should still be healthy (no issues)
        assert report['health']['healthy'] is True
