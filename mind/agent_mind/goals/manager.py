"""
Goal manager — operations on the goal tree.

The GoalManager holds all goals and provides operations to manipulate them.
Goals form a tree (or forest — multiple root goals are allowed).
"""

from datetime import datetime
from typing import Optional

from .model import Goal, GoalState


class GoalManager:
    """Manages a tree (forest) of goals.

    Usage:
        mgr = GoalManager()
        root_id = mgr.push("Build the agent framework")
        sub_ids = mgr.decompose(root_id, ["Build tools", "Build mind"])
        mgr.activate(sub_ids[0])
        mgr.complete(sub_ids[0])
        next_goal = mgr.get_next()  # returns "Build mind"
    """

    def __init__(self):
        self._goals: dict[str, Goal] = {}

    def push(self, description: str, priority: int = 5,
             parent_id: Optional[str] = None,
             metadata: Optional[dict] = None) -> str:
        """Add a new goal. Returns its ID.

        If parent_id is given, the new goal becomes a child of that parent.
        """
        if parent_id is not None and parent_id not in self._goals:
            raise KeyError(f"Parent goal not found: {parent_id}")

        goal = Goal(
            description=description,
            priority=priority,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._goals[goal.id] = goal

        if parent_id is not None:
            self._goals[parent_id].children_ids.append(goal.id)

        return goal.id

    def get(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def activate(self, goal_id: str) -> None:
        """Mark a goal as active (currently being worked on)."""
        goal = self._require(goal_id)
        if goal.state not in (GoalState.PENDING, GoalState.BLOCKED):
            raise ValueError(
                f"Cannot activate goal in state {goal.state.value}"
            )
        goal.state = GoalState.ACTIVE
        goal.blocked_reason = None
        goal.unblocked_by = None

    def complete(self, goal_id: str) -> None:
        """Mark a goal as completed.

        If all siblings are completed, the parent is auto-completed too.
        """
        goal = self._require(goal_id)
        if goal.is_terminal:
            raise ValueError(
                f"Goal already {goal.state.value}"
            )
        goal.state = GoalState.COMPLETED
        goal.completed_at = datetime.now()

        # Propagate: if all siblings are done, complete the parent
        if goal.parent_id is not None:
            self._try_complete_parent(goal.parent_id)

    def block(self, goal_id: str, reason: str,
              unblocked_by: Optional[str] = None) -> None:
        """Block a goal with a reason."""
        goal = self._require(goal_id)
        if goal.state != GoalState.ACTIVE:
            raise ValueError(
                f"Can only block active goals, got {goal.state.value}"
            )
        goal.state = GoalState.BLOCKED
        goal.blocked_reason = reason
        goal.unblocked_by = unblocked_by

    def unblock(self, goal_id: str) -> None:
        """Unblock a goal (returns it to active)."""
        goal = self._require(goal_id)
        if goal.state != GoalState.BLOCKED:
            raise ValueError(
                f"Goal is not blocked (state: {goal.state.value})"
            )
        goal.state = GoalState.ACTIVE
        goal.blocked_reason = None
        goal.unblocked_by = None

    def abandon(self, goal_id: str) -> None:
        """Abandon a goal (no longer relevant)."""
        goal = self._require(goal_id)
        if goal.is_terminal:
            raise ValueError(
                f"Goal already {goal.state.value}"
            )
        goal.state = GoalState.ABANDONED
        goal.completed_at = datetime.now()

    def decompose(self, goal_id: str,
                  sub_descriptions: list[str]) -> list[str]:
        """Decompose a goal into sub-goals. Returns their IDs.

        The sub-goals inherit the parent's priority by default.
        """
        goal = self._require(goal_id)
        ids = []
        for desc in sub_descriptions:
            sub_id = self.push(
                description=desc,
                priority=goal.priority,
                parent_id=goal_id,
            )
            ids.append(sub_id)
        return ids

    def reprioritize(self, goal_id: str, new_priority: int) -> None:
        """Change a goal's priority."""
        if not 1 <= new_priority <= 10:
            raise ValueError(f"Priority must be 1-10, got {new_priority}")
        goal = self._require(goal_id)
        goal.priority = new_priority

    def get_next(self) -> Optional[Goal]:
        """Return the highest-priority active leaf goal.
        """
        candidates = [g for g in self._goals.values()
                      if g.state in (GoalState.ACTIVE, GoalState.PENDING)
                      and g.is_leaf]

        def comp_goals(g):
            # taking advantage that priorities are 1 to 10 for this
            state_dict = {GoalState.ACTIVE: 200, GoalState.PENDING: 100}
            return g.priority + state_dict[g.state]

        if not candidates:
            return None

        sorted_candidates = sorted(candidates, key=comp_goals, reverse=True)
        return sorted_candidates[0]

    def roots(self) -> list[Goal]:
        """Return all root goals (no parent)."""
        return [g for g in self._goals.values() if g.parent_id is None]

    def children(self, goal_id: str) -> list[Goal]:
        """Return the children of a goal."""
        goal = self._require(goal_id)
        return [self._goals[cid] for cid in goal.children_ids]

    def all_goals(self) -> list[Goal]:
        """Return all goals."""
        return list(self._goals.values())

    def _require(self, goal_id: str) -> Goal:
        """Get a goal or raise KeyError."""
        goal = self._goals.get(goal_id)
        if goal is None:
            raise KeyError(f"Goal not found: {goal_id}")
        return goal

    def _try_complete_parent(self, parent_id: str) -> None:
        """Auto-complete parent if all children are terminal."""
        parent = self._goals[parent_id]
        children = self.children(parent_id)
        if all(c.is_terminal for c in children):
            # All children done — check if any were abandoned
            if any(c.state == GoalState.ABANDONED for c in children):
                # Some children abandoned — don't auto-complete,
                # let the agent decide
                return
            parent.state = GoalState.COMPLETED
            parent.completed_at = datetime.now()
            # Recurse up
            if parent.parent_id is not None:
                self._try_complete_parent(parent.parent_id)

    def __len__(self) -> int:
        return len(self._goals)
