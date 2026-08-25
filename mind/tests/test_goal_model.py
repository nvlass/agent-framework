"""Tests for Goal model.

Run with:
    cd mind/
    python -m pytest tests/test_goal_model.py -v
"""

from agent_mind.goals.model import Goal, GoalState


class TestGoalState:

    def test_values(self):
        assert GoalState.PENDING.value == "pending"
        assert GoalState.ACTIVE.value == "active"
        assert GoalState.BLOCKED.value == "blocked"
        assert GoalState.COMPLETED.value == "completed"
        assert GoalState.ABANDONED.value == "abandoned"


class TestGoal:

    def test_defaults(self):
        g = Goal(description="test goal")
        assert g.description == "test goal"
        assert g.state == GoalState.PENDING
        assert g.priority == 5
        assert g.parent_id is None
        assert g.children_ids == []
        assert g.is_leaf is True
        assert g.is_terminal is False

    def test_id_auto_generated(self):
        g1 = Goal(description="a")
        g2 = Goal(description="b")
        assert g1.id != g2.id

    def test_is_leaf(self):
        g = Goal(description="parent", children_ids=["child-1"])
        assert g.is_leaf is False

    def test_is_terminal_completed(self):
        g = Goal(description="done", state=GoalState.COMPLETED)
        assert g.is_terminal is True

    def test_is_terminal_abandoned(self):
        g = Goal(description="nope", state=GoalState.ABANDONED)
        assert g.is_terminal is True

    def test_is_terminal_active(self):
        g = Goal(description="wip", state=GoalState.ACTIVE)
        assert g.is_terminal is False

    def test_to_dict(self):
        g = Goal(description="test")
        d = g.to_dict()
        assert d["description"] == "test"
        assert d["state"] == "pending"
        assert d["priority"] == 5
        assert d["parent_id"] is None
        assert d["children_ids"] == []

    def test_custom_priority(self):
        g = Goal(description="urgent", priority=10)
        assert g.priority == 10

    def test_metadata(self):
        g = Goal(description="tagged", metadata={"source": "user"})
        assert g.metadata["source"] == "user"
