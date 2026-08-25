"""Tests for PlanExecutor."""

import pytest
from dataclasses import dataclass
from typing import Any, Optional

from agent_mind.planning.model import Plan, PlanStep, StepStatus
from agent_mind.introspection.monitor import ActionResult, ProgressMonitor
from agent_mind.introspection.triggers import ReflectionTrigger, TriggerType

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.executor import PlanExecutor, PlanExecutionResult
from agent_patterns.context import SharedContext
from agent_patterns.events.types import Abort, Reflect, StepFailed, ActionCompleted, PatternComplete


# --- Test helpers ---

@dataclass
class FakeToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    tool_name: str = ""


class FakeToolExecutor:
    """Fake ToolExecutor that returns scripted results."""

    def __init__(self, results: dict[str, FakeToolResult] | None = None, default_success: bool = True):
        self._results = results or {}
        self._default_success = default_success
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, **kwargs) -> FakeToolResult:
        self.calls.append((tool_name, kwargs))
        if tool_name in self._results:
            return self._results[tool_name]
        if self._default_success:
            return FakeToolResult(success=True, output=f"{tool_name} done", tool_name=tool_name)
        return FakeToolResult(success=False, error=f"{tool_name} failed", tool_name=tool_name)


def make_plan(*steps: PlanStep) -> Plan:
    return Plan(goal_id="g1", steps=list(steps))


def make_step(desc: str, tool: str = "test_tool", depends: list[str] | None = None, **tool_args) -> PlanStep:
    s = PlanStep(description=desc, tool_name=tool, tool_args=tool_args)
    if depends:
        s.depends_on = depends
    return s


# --- Tests ---

class TestClassifyResult:
    def test_success_is_progress(self):
        assert PlanExecutor._classify_result(FakeToolResult(success=True)) == ActionResult.PROGRESS

    def test_failure_is_failure(self):
        assert PlanExecutor._classify_result(FakeToolResult(success=False)) == ActionResult.FAILURE


class TestExecuteStep:
    def test_step_with_tool(self):
        fake = FakeToolExecutor()
        executor = PlanExecutor(make_plan(), fake)
        step = make_step("do thing", tool="read_file", path="/tmp/x")
        result = executor._execute_step(step, SharedContext())
        assert result.success
        assert fake.calls == [("read_file", {"path": "/tmp/x"})]

    def test_step_without_tool(self):
        executor = PlanExecutor(make_plan(), FakeToolExecutor())
        step = PlanStep(description="think about it", tool_name=None)
        result = executor._execute_step(step, SharedContext())
        assert result.success
        assert result.output == "no-op step"


class TestExecutePlan:
    """Tests for the full execute_plan loop."""

    def test_empty_plan_succeeds(self):
        """A plan with no steps is trivially complete."""
        plan = make_plan()
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert result.success
        assert result.steps_completed == 0

    def test_single_step_success(self):
        step = make_step("do it")
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert result.success
        assert result.steps_completed == 1
        assert step.status == StepStatus.COMPLETED

    def test_single_step_failure(self):
        step = make_step("fail it", tool="bad_tool")
        plan = make_plan(step)
        fake = FakeToolExecutor(results={"bad_tool": FakeToolResult(success=False, error="boom")})
        executor = PlanExecutor(plan, fake)
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert not result.success
        assert result.steps_failed == 1
        assert step.status == StepStatus.FAILED

    def test_dag_ordering(self):
        """Step B depends on step A — A must run first."""
        step_a = make_step("step A", tool="tool_a")
        step_b = make_step("step B", tool="tool_b")
        step_b.depends_on = [step_a.id]
        plan = make_plan(step_a, step_b)
        fake = FakeToolExecutor()
        executor = PlanExecutor(plan, fake)
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert result.success
        assert result.steps_completed == 2
        # tool_a must have been called before tool_b
        call_names = [c[0] for c in fake.calls]
        assert call_names.index("tool_a") < call_names.index("tool_b")

    def test_blocked_step_not_executed(self):
        """If step A fails, step B (depends on A) should not run."""
        step_a = make_step("step A", tool="fail_tool")
        step_b = make_step("step B", tool="tool_b")
        step_b.depends_on = [step_a.id]
        plan = make_plan(step_a, step_b)
        fake = FakeToolExecutor(results={"fail_tool": FakeToolResult(success=False, error="nope")})
        executor = PlanExecutor(plan, fake)
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert not result.success
        assert result.steps_failed == 1
        assert result.steps_completed == 0
        assert step_b.status == StepStatus.PENDING  # never ran

    def test_monitor_integration(self):
        """Actions should be recorded in the progress monitor."""
        step = make_step("do it")
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        executor.execute_plan(ctx)
        assert len(ctx.monitor.history) == 1
        assert ctx.monitor.history[0].classification == ActionResult.PROGRESS

    def test_events_published(self):
        """ActionCompleted and PatternComplete should be published."""
        step = make_step("do it")
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        events = []
        ctx.event_bus.subscribe(ActionCompleted, lambda e: events.append(("action", e)))
        ctx.event_bus.subscribe(PatternComplete, lambda e: events.append(("complete", e)))
        executor.execute_plan(ctx)
        action_events = [e for t, e in events if t == "action"]
        complete_events = [e for t, e in events if t == "complete"]
        assert len(action_events) == 1
        assert len(complete_events) == 1
        assert complete_events[0].success

    def test_step_failed_event(self):
        """StepFailed event should be published on failure."""
        step = make_step("fail", tool="bad")
        plan = make_plan(step)
        fake = FakeToolExecutor(results={"bad": FakeToolResult(success=False, error="err")})
        executor = PlanExecutor(plan, fake)
        ctx = SharedContext()
        fails = []
        ctx.event_bus.subscribe(StepFailed, lambda e: fails.append(e))
        executor.execute_plan(ctx)
        assert len(fails) == 1
        assert fails[0].step_id == step.id

    def test_abort_stops_execution(self):
        """Publishing Abort mid-execution should stop the loop."""
        step_a = make_step("step A", tool="tool_a")
        step_b = make_step("step B", tool="tool_b")
        plan = make_plan(step_a, step_b)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        # Abort after first action
        ctx.event_bus.subscribe(ActionCompleted, lambda e: ctx.event_bus.publish(Abort(reason="stop")))
        result = executor.execute_plan(ctx)
        assert result.aborted
        assert result.steps_completed == 1  # only first step ran

    def test_reflection_stops_execution(self):
        """Reflection trigger should stop the loop."""
        steps = [make_step(f"step {i}") for i in range(5)]
        plan = make_plan(*steps)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        # Trigger reflect after first action
        trigger = ReflectionTrigger(type=TriggerType.EVENT, reason="test")
        ctx.event_bus.subscribe(
            ActionCompleted,
            lambda e: ctx.event_bus.publish(Reflect(trigger=trigger)),
        )
        result = executor.execute_plan(ctx)
        assert result.reflection_triggered is not None
        assert result.steps_completed < 5

    def test_multiple_independent_steps(self):
        """Steps without dependencies can all run."""
        steps = [make_step(f"step {i}", tool=f"tool_{i}") for i in range(3)]
        plan = make_plan(*steps)
        fake = FakeToolExecutor()
        executor = PlanExecutor(plan, fake)
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert result.success
        assert result.steps_completed == 3


class TestPatternInterface:
    """Test that PlanExecutor implements Pattern."""

    def test_is_pattern(self):
        executor = PlanExecutor(make_plan(), FakeToolExecutor())
        assert isinstance(executor, Pattern)

    def test_run_returns_pattern_result(self):
        step = make_step("do it")
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        result = executor.run(ctx)
        assert isinstance(result, PatternResult)
        assert result.success
        assert result.metadata["steps_completed"] == 1


class TestStepResultObservations:
    """Completed step results flow into context.observations."""

    def test_completed_step_appended_to_observations(self):
        step = make_step("fetch data", tool="fetch", url="http://example.com")
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        executor.execute_plan(ctx)
        assert any("fetch data" in obs for obs in ctx.observations)

    def test_failed_step_not_in_observations(self):
        step = make_step("bad step", tool="broken")
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor(default_success=False))
        ctx = SharedContext()
        executor.execute_plan(ctx)
        assert not any("bad step" in obs for obs in ctx.observations)

    def test_prior_step_result_visible_in_observations_for_next_step(self):
        step_a = make_step("step A", tool="tool_a")
        step_b = make_step("step B", tool="tool_b")
        step_b.depends_on = [step_a.id]
        plan = make_plan(step_a, step_b)
        executor = PlanExecutor(plan, FakeToolExecutor())
        ctx = SharedContext()
        executor.execute_plan(ctx)
        # After step_a, its result is in observations; step_b sees it (2 obs total)
        assert len(ctx.observations) == 2


class TestReActPerStep:
    """Steps without tool_name run a mini-ReactLoop when reasoner is provided."""

    def test_no_tool_no_reasoner_is_noop(self):
        step = PlanStep(description="think about it", tool_name=None)
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor(), reasoner=None)
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert result.success
        assert result.steps_completed == 1

    def test_no_tool_with_reasoner_runs_react(self):
        """Step without tool_name uses a mini-ReactLoop when reasoner present."""
        from agent_patterns.react import ReasonerInterface, ReasoningResult
        from agent_mind.goals.model import Goal

        class _ImmediateReasoner(ReasonerInterface):
            def reason(self, goal, observations, available_tools):
                return ReasoningResult(thought="done", answer="react result")

        step = PlanStep(description="do something smart", tool_name=None)
        plan = make_plan(step)
        executor = PlanExecutor(plan, FakeToolExecutor(), reasoner=_ImmediateReasoner())
        ctx = SharedContext()
        result = executor.execute_plan(ctx)
        assert result.success
        assert result.steps_completed == 1

    def test_react_per_step_inherits_parent_observations(self):
        """Mini-ReactLoop sees prior step results from parent context."""
        from agent_patterns.react import ReasonerInterface, ReasoningResult

        seen_observations = []

        class _ObservingReasoner(ReasonerInterface):
            def reason(self, goal, observations, available_tools):
                seen_observations.extend(observations)
                return ReasoningResult(thought="done", answer="ok")

        step_a = make_step("step A", tool="tool_a")
        step_b = PlanStep(description="use A result", tool_name=None)
        step_b.depends_on = [step_a.id]
        plan = make_plan(step_a, step_b)

        executor = PlanExecutor(plan, FakeToolExecutor(), reasoner=_ObservingReasoner())
        ctx = SharedContext()
        executor.execute_plan(ctx)
        # step_b's ReactLoop should have seen step_a's result in its observations
        assert any("step A" in obs for obs in seen_observations)
