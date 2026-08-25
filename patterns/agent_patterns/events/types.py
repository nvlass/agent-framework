"""Event types for pattern↔mind communication and EventBus."""

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_mind.introspection.monitor import ActionResult
from agent_mind.introspection.triggers import ReflectionTrigger


# --- Events from patterns → mind (observation) ---

@dataclass
class ActionCompleted:
    """A tool action was executed."""
    action: str
    result: str
    classification: ActionResult


@dataclass
class StepFailed:
    """A plan step failed."""
    step_id: str
    reason: str


@dataclass
class Stuck:
    """Pattern detected it's stuck."""
    attempts: int
    last_actions: list[str]


@dataclass
class PatternComplete:
    """Pattern finished execution."""
    goal_id: str
    success: bool
    summary: str


# --- Events from mind → patterns (control) ---

@dataclass
class Reflect:
    """Mind requests the pattern to pause for reflection."""
    trigger: ReflectionTrigger


@dataclass
class Replan:
    """Mind requests replanning."""
    reason: str


@dataclass
class Abort:
    """Mind requests pattern to stop."""
    reason: str


# --- EventBus ---

class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def publish(self, event: Any) -> None:
        """Dispatch event to all subscribers of its type."""
        event_type = type(event)
        for handler in self._handlers.get(event_type, []):
            handler(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
