"""SharedContext — mutable state shared between pattern and mind components."""

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_mind.goals.model import Goal
from agent_mind.planning.model import Plan
from agent_mind.introspection.monitor import ProgressMonitor

from agent_patterns.events.types import EventBus


@dataclass
class SharedContext:
    """Mutable state object shared between pattern and mind layers."""
    goal: Optional[Goal] = None
    plan: Optional[Plan] = None
    monitor: ProgressMonitor = field(default_factory=ProgressMonitor)
    event_bus: EventBus = field(default_factory=EventBus)
    observations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
