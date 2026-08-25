"""
Agent Memory System

A hierarchical memory system for AI agents with:
- Episodic memory (store and retrieve experiences)
- Working memory (hot cache for current session)
- Short-term memory (TTL-based cache)
- Long-term memory (SQLite + ChromaDB)
- Pattern learning (cluster and extract insights)
- Reflection (analyze successes and failures)
- Adaptation & transfer learning (apply knowledge across domains)

Basic usage:
    from agent_memory import MemoryStore, EmbeddingGenerator

    # Initialize
    embedding_gen = EmbeddingGenerator("/path/to/model.gguf")
    store = MemoryStore(embedding_generator=embedding_gen)

    # Store an episode
    store.store_episode(
        context="Debugging Python TypeError",
        action="Added null check",
        outcome="Bug fixed",
        success_score=0.9,
        tags=["python", "debugging"]
    )

    # Retrieve similar experiences
    results = store.retrieve_episodes("Python error handling", limit=5)

    # Get recommendations
    advice = store.recommend_actions("I have a TypeError")

    # Transfer learning: get adapted strategies
    from agent_memory import StrategySelector
    selector = StrategySelector(store, llm)
    selection = selector.select_strategy("Docker container won't start")
"""

__version__ = "0.5.0"

# Core classes
from .memory_store import (
    MemoryStore,
    Episode,
    Reflection,
    CausalFactor,
)

# Embedding support
from .embeddings import (
    EmbeddingGenerator,
    LlamaCppEmbeddingGenerator,
    OpenAICompatEmbeddingGenerator,
    cosine_similarity,
    euclidean_distance,
    compute_text_similarity,
)

# Consolidation
from .consolidation import (
    ConsolidationEngine,
    EpisodeClusterer,
    PatternExtractor,
    LearnedPattern,
    ConsolidationReport,
)

# Reflection
from .reflector import (
    Reflector,
    ReflectionConfig,
)

# LLM interface
from .llm_interface import (
    LLMInterface,
    LlamaCppLLM,
    MockLLM,
    LLMResponse,
)

# Memory tiers
from .working_memory import WorkingMemory
from .short_term_memory import ShortTermMemory, TTLCache

# Phase 5: Adaptation & Transfer Learning
from .analogy_finder import (
    AnalogyFinder,
    AnalogousMatch,
    find_structural_analogies,
)
from .adapter import (
    StrategyAdapter,
    AdaptationResult,
    ProblemType,
)
from .strategy_selector import (
    StrategySelector,
    StrategyCandidate,
    StrategySelection,
    quick_select,
)
from .domain_learner import (
    DomainLearner,
    LearningReport,
    KeywordCandidate,
    seed_domains,
)

# Phase 6: Configuration, Tools, and Metrics
from .config import (
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
)
from .memory_tools import (
    MemoryTools,
    ToolResult,
    MemoryEntry,
    StrategyAdvice,
)
from .metrics import (
    MemoryMetrics,
    HealthStatus,
    UsageMetrics,
    LearningMetrics,
    PerformanceMetrics,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "MemoryStore",
    "Episode",
    "Reflection",
    "CausalFactor",
    # Embeddings
    "EmbeddingGenerator",
    "LlamaCppEmbeddingGenerator",
    "OpenAICompatEmbeddingGenerator",
    "cosine_similarity",
    "euclidean_distance",
    "compute_text_similarity",
    # Consolidation
    "ConsolidationEngine",
    "EpisodeClusterer",
    "PatternExtractor",
    "LearnedPattern",
    "ConsolidationReport",
    # Reflection
    "Reflector",
    "ReflectionConfig",
    # LLM
    "LLMInterface",
    "LlamaCppLLM",
    "MockLLM",
    "LLMResponse",
    # Memory tiers
    "WorkingMemory",
    "ShortTermMemory",
    "TTLCache",
    # Adaptation & Transfer Learning
    "AnalogyFinder",
    "AnalogousMatch",
    "find_structural_analogies",
    "StrategyAdapter",
    "AdaptationResult",
    "ProblemType",
    "StrategySelector",
    "StrategyCandidate",
    "StrategySelection",
    "quick_select",
    # Domain Learning
    "DomainLearner",
    "LearningReport",
    "KeywordCandidate",
    "seed_domains",
    # Configuration
    "MemoryConfig",
    "MemorySettings",
    "ConsolidationSettings",
    "ReflectionSettings",
    "ForgettingSettings",
    "AdaptationSettings",
    "PerformanceSettings",
    "LoggingSettings",
    "load_config",
    "create_default_config_file",
    "get_config_template",
    # Memory Tools (Agent API)
    "MemoryTools",
    "ToolResult",
    "MemoryEntry",
    "StrategyAdvice",
    # Metrics
    "MemoryMetrics",
    "HealthStatus",
    "UsageMetrics",
    "LearningMetrics",
    "PerformanceMetrics",
]
