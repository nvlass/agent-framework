"""
Configuration Management for Agent Memory System

Provides a centralized configuration system with:
- Default values for all settings
- Loading from YAML files
- Environment variable overrides
- Validation and type checking

Usage:
    from agent_memory import MemoryConfig, load_config

    # Load from file
    config = load_config("config.yaml")

    # Or use defaults
    config = MemoryConfig()

    # Access settings
    print(config.memory.db_path)
    print(config.consolidation.trigger_after_episodes)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


@dataclass
class MemorySettings:
    """Core memory storage settings"""
    db_path: str = "data/agent_memory.db"
    vector_store_path: str = "data/memory_vectors"
    embedding_model_path: Optional[str] = None  # Auto-detect if None
    llm_model_path: Optional[str] = None

    # Memory hierarchy
    working_memory_size: int = 20
    short_term_ttl_seconds: int = 300  # 5 minutes
    short_term_window_hours: int = 24


@dataclass
class ConsolidationSettings:
    """Settings for memory consolidation"""
    enabled: bool = True
    trigger_after_episodes: int = 20
    min_cluster_size: int = 3
    similarity_threshold: float = 0.75
    hours_back: int = 168  # 1 week
    auto_consolidate: bool = False  # Run automatically after threshold


@dataclass
class ReflectionSettings:
    """Settings for reflection and meta-learning"""
    enabled: bool = True
    auto_reflect_on_failure: bool = True
    failure_threshold: float = 0.3
    auto_reflect_on_success: bool = True
    success_threshold: float = 0.9
    max_reflections_per_day: int = 10


@dataclass
class ForgettingSettings:
    """Settings for memory forgetting/pruning"""
    enabled: bool = True
    min_access_count: int = 2
    age_threshold_days: int = 30
    redundancy_threshold: float = 0.95
    max_forget_per_run: int = 50
    keep_failures: bool = True  # Keep failures for learning
    keep_successes: bool = True  # Keep high-success episodes


@dataclass
class AdaptationSettings:
    """Settings for transfer learning and adaptation"""
    enabled: bool = True
    min_analogy_similarity: float = 0.4
    domain_distance_weight: float = 0.3
    min_confidence_for_recommendation: float = 0.5
    max_adaptations_per_query: int = 3


@dataclass
class PerformanceSettings:
    """Performance and resource limits"""
    max_episodes_in_memory: int = 10000
    max_patterns: int = 500
    max_reflections: int = 1000
    embedding_batch_size: int = 10
    retrieval_timeout_seconds: float = 2.0
    max_retrieval_results: int = 20


@dataclass
class LoggingSettings:
    """Logging configuration"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = False
    log_file_path: str = "logs/agent_memory.log"
    log_consolidation: bool = True
    log_reflections: bool = True


@dataclass
class MemoryConfig:
    """
    Main configuration container.

    Groups all settings into logical categories.
    """
    memory: MemorySettings = field(default_factory=MemorySettings)
    consolidation: ConsolidationSettings = field(default_factory=ConsolidationSettings)
    reflection: ReflectionSettings = field(default_factory=ReflectionSettings)
    forgetting: ForgettingSettings = field(default_factory=ForgettingSettings)
    adaptation: AdaptationSettings = field(default_factory=AdaptationSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'memory': self._dataclass_to_dict(self.memory),
            'consolidation': self._dataclass_to_dict(self.consolidation),
            'reflection': self._dataclass_to_dict(self.reflection),
            'forgetting': self._dataclass_to_dict(self.forgetting),
            'adaptation': self._dataclass_to_dict(self.adaptation),
            'performance': self._dataclass_to_dict(self.performance),
            'logging': self._dataclass_to_dict(self.logging),
        }

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert a dataclass to dict"""
        result = {}
        for key in obj.__dataclass_fields__:
            result[key] = getattr(obj, key)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Create config from dictionary"""
        config = cls()

        if 'memory' in data:
            config.memory = MemorySettings(**data['memory'])
        if 'consolidation' in data:
            config.consolidation = ConsolidationSettings(**data['consolidation'])
        if 'reflection' in data:
            config.reflection = ReflectionSettings(**data['reflection'])
        if 'forgetting' in data:
            config.forgetting = ForgettingSettings(**data['forgetting'])
        if 'adaptation' in data:
            config.adaptation = AdaptationSettings(**data['adaptation'])
        if 'performance' in data:
            config.performance = PerformanceSettings(**data['performance'])
        if 'logging' in data:
            config.logging = LoggingSettings(**data['logging'])

        return config

    def save(self, path: str) -> None:
        """Save config to YAML or JSON file based on extension"""
        path = Path(path)

        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                with open(path, 'w') as f:
                    yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
            except ImportError:
                raise ImportError("PyYAML required to save YAML config. Install with: pip install pyyaml")
        else:
            # JSON for all other extensions
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Memory settings
        if self.memory.working_memory_size < 1:
            errors.append("working_memory_size must be >= 1")
        if self.memory.short_term_ttl_seconds < 1:
            errors.append("short_term_ttl_seconds must be >= 1")

        # Consolidation settings
        if self.consolidation.min_cluster_size < 2:
            errors.append("min_cluster_size must be >= 2")
        if not 0.0 <= self.consolidation.similarity_threshold <= 1.0:
            errors.append("similarity_threshold must be between 0.0 and 1.0")

        # Reflection settings
        if not 0.0 <= self.reflection.failure_threshold <= 1.0:
            errors.append("failure_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.reflection.success_threshold <= 1.0:
            errors.append("success_threshold must be between 0.0 and 1.0")

        # Forgetting settings
        if self.forgetting.age_threshold_days < 1:
            errors.append("age_threshold_days must be >= 1")

        # Adaptation settings
        if not 0.0 <= self.adaptation.min_analogy_similarity <= 1.0:
            errors.append("min_analogy_similarity must be between 0.0 and 1.0")
        if not 0.0 <= self.adaptation.domain_distance_weight <= 1.0:
            errors.append("domain_distance_weight must be between 0.0 and 1.0")

        # Performance settings
        if self.performance.max_episodes_in_memory < 100:
            errors.append("max_episodes_in_memory should be >= 100")
        if self.performance.retrieval_timeout_seconds < 0.1:
            errors.append("retrieval_timeout_seconds must be >= 0.1")

        return errors


def load_config(path: Optional[str] = None) -> MemoryConfig:
    """
    Load configuration from file.

    Supports YAML and JSON formats. If no path provided,
    looks for config in standard locations.

    Args:
        path: Path to config file (optional)

    Returns:
        MemoryConfig instance
    """
    # Standard config locations to search
    search_paths = [
        path,
        os.environ.get('AGENT_MEMORY_CONFIG'),
        'agent_memory.yaml',
        'agent_memory.json',
        'config/agent_memory.yaml',
        'config/agent_memory.json',
        Path.home() / '.agent_memory' / 'config.yaml',
    ]

    for config_path in search_paths:
        if config_path and Path(config_path).exists():
            return _load_config_file(config_path)

    # Return defaults if no config found
    return MemoryConfig()


def _load_config_file(path: str) -> MemoryConfig:
    """Load config from a specific file"""
    path = Path(path)

    with open(path) as f:
        if path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required to load YAML config. Install with: pip install pyyaml")
        else:
            data = json.load(f)

    config = MemoryConfig.from_dict(data or {})

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: MemoryConfig) -> MemoryConfig:
    """Apply environment variable overrides to config"""

    # Memory settings
    if os.environ.get('AGENT_MEMORY_DB_PATH'):
        config.memory.db_path = os.environ['AGENT_MEMORY_DB_PATH']
    if os.environ.get('AGENT_MEMORY_VECTOR_PATH'):
        config.memory.vector_store_path = os.environ['AGENT_MEMORY_VECTOR_PATH']
    if os.environ.get('EMBEDDING_MODEL_PATH'):
        config.memory.embedding_model_path = os.environ['EMBEDDING_MODEL_PATH']
    if os.environ.get('LLM_MODEL_PATH'):
        config.memory.llm_model_path = os.environ['LLM_MODEL_PATH']

    # Logging
    if os.environ.get('AGENT_MEMORY_LOG_LEVEL'):
        config.logging.level = os.environ['AGENT_MEMORY_LOG_LEVEL']

    return config


def create_default_config_file(path: str = "agent_memory.yaml") -> None:
    """
    Create a default configuration file.

    Useful for bootstrapping a new setup.

    Args:
        path: Where to create the config file
    """
    config = MemoryConfig()
    config.save(path)
    print(f"Created default config at: {path}")


def get_config_template() -> str:
    """
    Get a YAML template with all config options documented.

    Returns:
        YAML string with comments
    """
    return '''# Agent Memory System Configuration
# ===================================

memory:
  # Database and storage paths
  db_path: "data/agent_memory.db"
  vector_store_path: "data/memory_vectors"

  # Model paths (null = auto-detect)
  embedding_model_path: null
  llm_model_path: null

  # Memory hierarchy settings
  working_memory_size: 20        # Hot cache size
  short_term_ttl_seconds: 300    # 5 minutes
  short_term_window_hours: 24

consolidation:
  enabled: true
  trigger_after_episodes: 20     # Consolidate every N episodes
  min_cluster_size: 3            # Minimum episodes to form pattern
  similarity_threshold: 0.75     # Embedding similarity for clustering
  hours_back: 168                # Look back 1 week
  auto_consolidate: false        # Run automatically

reflection:
  enabled: true
  auto_reflect_on_failure: true
  failure_threshold: 0.3         # Reflect if score < this
  auto_reflect_on_success: true
  success_threshold: 0.9         # Reflect if score > this
  max_reflections_per_day: 10

forgetting:
  enabled: true
  min_access_count: 2            # Keep if accessed more than this
  age_threshold_days: 30         # Consider forgetting after this age
  redundancy_threshold: 0.95     # Forget if 95% similar to another
  max_forget_per_run: 50
  keep_failures: true            # Always keep failures for learning
  keep_successes: true           # Always keep high-success episodes

adaptation:
  enabled: true
  min_analogy_similarity: 0.4    # Minimum similarity for analogies
  domain_distance_weight: 0.3    # Bonus for cross-domain analogies
  min_confidence_for_recommendation: 0.5
  max_adaptations_per_query: 3

performance:
  max_episodes_in_memory: 10000
  max_patterns: 500
  max_reflections: 1000
  embedding_batch_size: 10
  retrieval_timeout_seconds: 2.0
  max_retrieval_results: 20

logging:
  level: "INFO"                  # DEBUG, INFO, WARNING, ERROR
  log_to_file: false
  log_file_path: "logs/agent_memory.log"
  log_consolidation: true
  log_reflections: true
'''
