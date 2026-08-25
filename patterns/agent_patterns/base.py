"""Base Pattern interface — all execution patterns implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from agent_mind.introspection.triggers import ReflectionTrigger

from agent_patterns.context import SharedContext


@dataclass
class PatternResult:
    """Unified result from any pattern execution.

    Every pattern returns this, making patterns interchangeable.
    Pattern-specific details go in metadata.
    """
    success: bool
    summary: str = ""
    iterations: int = 0
    reflection_triggered: Optional[ReflectionTrigger] = None
    aborted: bool = False
    metadata: dict = field(default_factory=dict)


class Pattern(ABC):
    """Base interface for all execution patterns.

    Patterns are interchangeable: any code that runs a pattern just calls
    pattern.run(context) without knowing the concrete implementation.

    Dependencies (tool_executor, reasoner, plan, etc.) are injected at
    construction time. The run() method only takes SharedContext.
    """

    @abstractmethod
    def run(self, context: SharedContext) -> PatternResult:
        """Execute the pattern against the given context.

        The context provides the shared state (goal, plan, monitor, event_bus).
        Returns a PatternResult with the outcome.
        """
