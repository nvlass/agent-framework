"""Configuration dataclasses for agent-core."""

import logging
from dataclasses import dataclass, field

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass
class ReactConfig:
    """Configuration for the ReAct pattern."""

    max_iterations: int = 10


@dataclass
class ReflexionConfig:
    """Configuration for the Reflexion pattern."""

    max_attempts: int = 3
    max_iterations_per_attempt: int = 10


@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    pattern: str = "react"
    max_replans: int = 3
    react: ReactConfig = field(default_factory=ReactConfig)
    reflexion: ReflexionConfig = field(default_factory=ReflexionConfig)
    soul: str = ""
    log_level: str = "WARNING"

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if self.pattern not in ("react", "plan_and_execute", "reflexion"):
            errors.append(f"Unsupported pattern: {self.pattern!r}")
        if self.react.max_iterations < 1:
            errors.append("react.max_iterations must be >= 1")
        if self.max_replans < 0:
            errors.append("max_replans must be >= 0")
        if self.reflexion.max_attempts < 1:
            errors.append("reflexion.max_attempts must be >= 1")
        if self.reflexion.max_iterations_per_attempt < 1:
            errors.append("reflexion.max_iterations_per_attempt must be >= 1")
        if self.log_level.upper() not in _VALID_LOG_LEVELS:
            errors.append(f"Invalid log_level: {self.log_level!r}")
        return errors

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "pattern": self.pattern,
            "max_replans": self.max_replans,
            "react": {"max_iterations": self.react.max_iterations},
            "reflexion": {
                "max_attempts": self.reflexion.max_attempts,
                "max_iterations_per_attempt": self.reflexion.max_iterations_per_attempt,
            },
            "soul": self.soul,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        """Create from dictionary."""
        react_d = d.get("react", {})
        ref_d = d.get("reflexion", {})
        return cls(
            pattern=d.get("pattern", "react"),
            max_replans=d.get("max_replans", 3),
            react=ReactConfig(max_iterations=react_d.get("max_iterations", 10)),
            reflexion=ReflexionConfig(
                max_attempts=ref_d.get("max_attempts", 3),
                max_iterations_per_attempt=ref_d.get("max_iterations_per_attempt", 10),
            ),
            soul=d.get("soul", ""),
            log_level=d.get("log_level", "WARNING"),
        )
