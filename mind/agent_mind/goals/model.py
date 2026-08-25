"""
Goal model — the data structures for goal management.

A Goal is a node in a tree. Goals can be decomposed into sub-goals,
forming a hierarchy from high-level objectives down to actionable tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class GoalState(Enum):
    """Lifecycle states for a goal.

    Transitions:
        PENDING → ACTIVE → COMPLETED
        ACTIVE → BLOCKED → ACTIVE (when unblocked)
        any → ABANDONED
    """
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """A single goal in the goal tree.

    Attributes:
        id: Unique identifier (auto-generated UUID)
        description: What this goal is trying to achieve
        state: Current lifecycle state
        priority: 1 (low) to 10 (urgent), determines which branch to work on
        parent_id: ID of parent goal (None for root goals)
        children_ids: IDs of sub-goals
        blocked_reason: Why this goal is blocked (only when state is BLOCKED)
        unblocked_by: What needs to happen to unblock (human-readable)
        created_at: When the goal was created
        completed_at: When the goal was completed or abandoned
        metadata: Arbitrary key-value data
    """
    description: str
    id: str = field(default_factory=lambda: str(uuid4()))
    state: GoalState = GoalState.PENDING
    priority: int = 5
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    unblocked_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """A leaf goal has no children — it's directly actionable."""
        return len(self.children_ids) == 0

    @property
    def is_terminal(self) -> bool:
        """A terminal goal is completed or abandoned — no more work needed."""
        return self.state in (GoalState.COMPLETED, GoalState.ABANDONED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "state": self.state.value,
            "priority": self.priority,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "blocked_reason": self.blocked_reason,
            "unblocked_by": self.unblocked_by,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }
