"""
Planning model — data structures for plans and steps.

A Plan is a sequence of steps to achieve a goal.
Steps can have dependencies on other steps (DAG, not just linear).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class StepStatus(Enum):
    """Lifecycle states for a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan.

    Attributes:
        id: Unique identifier
        description: What this step does (human-readable)
        tool_name: Which tool to use (None if no specific tool)
        tool_args: Arguments to pass to the tool
        depends_on: IDs of steps that must complete before this one
        status: Current lifecycle state
        result: Output from execution (set by the executor)
        error: Error message if failed
    """
    description: str
    id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        """Step is done (completed, failed, or skipped)."""
        return self.status in (
            StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Plan:
    """A plan to achieve a goal.

    Contains an ordered list of steps. Steps can have dependencies,
    forming a DAG. The executor (in agent-patterns) walks the DAG.

    Attributes:
        goal_id: The goal this plan is for
        steps: Ordered list of plan steps
        created_at: When the plan was created
        revised_at: When the plan was last revised
        revision_reason: Why the plan was revised
    """
    goal_id: str
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    revised_at: Optional[datetime] = None
    revision_reason: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """All steps are terminal."""
        return len(self.steps) > 0 and all(s.is_terminal for s in self.steps)

    @property
    def has_failures(self) -> bool:
        """Any step has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def next_steps(self) -> list[PlanStep]:
        """Return steps that are ready to execute.

        A step is ready if:
        - It is PENDING
        - All its dependencies are COMPLETED
        """
        completed = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        next_avail = [s for s in self.steps if s.status == StepStatus.PENDING
                      and all(dep in completed for dep in s.depends_on)]
        return next_avail

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "revised_at": self.revised_at.isoformat() if self.revised_at else None,
            "revision_reason": self.revision_reason,
        }
