"""Tests for Plan and PlanStep models.

Run with:
    cd mind/
    python -m pytest tests/test_plan_model.py -v
"""

import pytest

from agent_mind.planning.model import Plan, PlanStep, StepStatus


class TestStepStatus:

    def test_values(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


class TestPlanStep:

    def test_defaults(self):
        s = PlanStep(description="do something")
        assert s.description == "do something"
        assert s.status == StepStatus.PENDING
        assert s.tool_name is None
        assert s.tool_args == {}
        assert s.depends_on == []
        assert s.result is None
        assert s.error is None

    def test_id_auto_generated(self):
        s1 = PlanStep(description="a")
        s2 = PlanStep(description="b")
        assert s1.id != s2.id

    def test_with_tool(self):
        s = PlanStep(
            description="read config",
            tool_name="read_file",
            tool_args={"path": "/etc/config"},
        )
        assert s.tool_name == "read_file"
        assert s.tool_args["path"] == "/etc/config"

    def test_is_terminal_pending(self):
        s = PlanStep(description="x")
        assert s.is_terminal is False

    def test_is_terminal_in_progress(self):
        s = PlanStep(description="x", status=StepStatus.IN_PROGRESS)
        assert s.is_terminal is False

    def test_is_terminal_completed(self):
        s = PlanStep(description="x", status=StepStatus.COMPLETED)
        assert s.is_terminal is True

    def test_is_terminal_failed(self):
        s = PlanStep(description="x", status=StepStatus.FAILED)
        assert s.is_terminal is True

    def test_is_terminal_skipped(self):
        s = PlanStep(description="x", status=StepStatus.SKIPPED)
        assert s.is_terminal is True

    def test_to_dict(self):
        s = PlanStep(description="test", tool_name="read_file")
        d = s.to_dict()
        assert d["description"] == "test"
        assert d["tool_name"] == "read_file"
        assert d["status"] == "pending"


class TestPlan:

    def test_empty_plan(self):
        p = Plan(goal_id="g1")
        assert p.goal_id == "g1"
        assert p.steps == []
        assert p.is_complete is False
        assert p.has_failures is False

    def test_is_complete_all_done(self):
        p = Plan(goal_id="g1", steps=[
            PlanStep(description="a", status=StepStatus.COMPLETED),
            PlanStep(description="b", status=StepStatus.COMPLETED),
        ])
        assert p.is_complete is True

    def test_is_complete_mixed(self):
        p = Plan(goal_id="g1", steps=[
            PlanStep(description="a", status=StepStatus.COMPLETED),
            PlanStep(description="b", status=StepStatus.PENDING),
        ])
        assert p.is_complete is False

    def test_is_complete_with_skipped(self):
        p = Plan(goal_id="g1", steps=[
            PlanStep(description="a", status=StepStatus.COMPLETED),
            PlanStep(description="b", status=StepStatus.SKIPPED),
        ])
        assert p.is_complete is True

    def test_has_failures(self):
        p = Plan(goal_id="g1", steps=[
            PlanStep(description="a", status=StepStatus.COMPLETED),
            PlanStep(description="b", status=StepStatus.FAILED),
        ])
        assert p.has_failures is True

    def test_no_failures(self):
        p = Plan(goal_id="g1", steps=[
            PlanStep(description="a", status=StepStatus.COMPLETED),
        ])
        assert p.has_failures is False

    def test_get_step(self):
        s = PlanStep(description="target")
        p = Plan(goal_id="g1", steps=[
            PlanStep(description="other"),
            s,
        ])
        assert p.get_step(s.id) is s

    def test_get_step_not_found(self):
        p = Plan(goal_id="g1", steps=[PlanStep(description="a")])
        assert p.get_step("nonexistent") is None

    def test_to_dict(self):
        p = Plan(goal_id="g1", steps=[PlanStep(description="step 1")])
        d = p.to_dict()
        assert d["goal_id"] == "g1"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["description"] == "step 1"


class TestNextSteps:
    """Test Plan.next_steps() — returns steps ready to execute.

    Implement next_steps() in Plan to make these pass.
    """

    def test_no_deps_all_ready(self):
        """Steps with no dependencies are immediately ready."""
        s1 = PlanStep(description="a")
        s2 = PlanStep(description="b")
        p = Plan(goal_id="g1", steps=[s1, s2])
        ready = p.next_steps()
        assert len(ready) == 2

    def test_dep_not_met(self):
        """Step with unmet dependency is not ready."""
        s1 = PlanStep(description="first")
        s2 = PlanStep(description="second", depends_on=[s1.id])
        p = Plan(goal_id="g1", steps=[s1, s2])
        ready = p.next_steps()
        assert len(ready) == 1
        assert ready[0].id == s1.id

    def test_dep_met(self):
        """Step becomes ready when dependency is completed."""
        s1 = PlanStep(description="first", status=StepStatus.COMPLETED)
        s2 = PlanStep(description="second", depends_on=[s1.id])
        p = Plan(goal_id="g1", steps=[s1, s2])
        ready = p.next_steps()
        assert len(ready) == 1
        assert ready[0].id == s2.id

    def test_in_progress_not_ready(self):
        """Steps already in progress are not returned."""
        s1 = PlanStep(description="running", status=StepStatus.IN_PROGRESS)
        p = Plan(goal_id="g1", steps=[s1])
        ready = p.next_steps()
        assert len(ready) == 0

    def test_completed_not_ready(self):
        """Completed steps are not returned."""
        s1 = PlanStep(description="done", status=StepStatus.COMPLETED)
        p = Plan(goal_id="g1", steps=[s1])
        ready = p.next_steps()
        assert len(ready) == 0

    def test_diamond_dag(self):
        """Diamond dependency: A → B, A → C, B+C → D."""
        a = PlanStep(description="a", status=StepStatus.COMPLETED)
        b = PlanStep(description="b", depends_on=[a.id], status=StepStatus.COMPLETED)
        c = PlanStep(description="c", depends_on=[a.id])
        d = PlanStep(description="d", depends_on=[b.id, c.id])
        p = Plan(goal_id="g1", steps=[a, b, c, d])
        ready = p.next_steps()
        # Only c is ready (d waits on c)
        assert len(ready) == 1
        assert ready[0].id == c.id

    def test_empty_plan(self):
        """Empty plan has no next steps."""
        p = Plan(goal_id="g1")
        assert p.next_steps() == []
