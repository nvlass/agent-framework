"""Tests for PlanAndExecute pattern."""

import pytest
from dataclasses import dataclass
from typing import Any, Optional

from agent_mind.goals.model import Goal
from agent_mind.planning.model import Plan, PlanStep, StepStatus
from agent_mind.planning.planner import PlannerInterface
from agent_mind.introspection.triggers import ReflectionTrigger, TriggerType

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.plan_and_execute import PlanAndExecute, PlanAndExecuteResult
from agent_patterns.context import SharedContext
from agent_patterns.events.types import (
    ActionCompleted, PatternComplete, Abort, Reflect,
)


# --- Test helpers ---

@dataclass
class FakeToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    tool_name: str = ""


class FakeToolExecutor:
    """Fake ToolExecutor that returns scripted results."""

    def __init__(self, results: dict[str, FakeToolResult] | None = None,
                 default_success: bool = True):
        self._results = results or {}
        self._default_success = default_success
        self.calls: list[tuple[str, dict]] = []

    def execute(self, tool_name: str, **kwargs) -> FakeToolResult:
        self.calls.append((tool_name, kwargs))
        if tool_name in self._results:
            return self._results[tool_name]
        if self._default_success:
            return FakeToolResult(success=True, output=f"{tool_name} done",
                                 tool_name=tool_name)
        return FakeToolResult(success=False, error=f"{tool_name} failed",
                              tool_name=tool_name)


class FakePlanner(PlannerInterface):
    """Returns scripted plans for testing."""

    def __init__(self, plans: list[Plan]):
        self._plans = list(plans)
        self._index = 0
        self.revise_calls: list[tuple[Plan, str]] = []

    def create_plan(self, goal, context=None):
        return self._next()

    def revise_plan(self, plan, reason, context=None):
        self.revise_calls.append((plan, reason))
        return self._next()

    def _next(self) -> Plan:
        plan = self._plans[self._index]
        self._index = min(self._index + 1, len(self._plans) - 1)
        return plan


def make_goal(desc: str = "test goal") -> Goal:
    return Goal(description=desc)


def make_plan(*steps: PlanStep, goal_id: str = "g1") -> Plan:
    return Plan(goal_id=goal_id, steps=list(steps))


def make_step(desc: str, tool: str = "test_tool") -> PlanStep:
    return PlanStep(description=desc, tool_name=tool)


# --- Tests ---

class TestPlanAndExecute:

    def test_single_plan_succeeds(self):
        """Happy path: plan works on first attempt."""
        plan = make_plan(make_step("do it"))
        planner = FakePlanner([plan])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext(goal=make_goal())
        result = pae.run(ctx)
        assert result.success
        assert result.iterations == 1
        assert result.metadata["plan_attempts"] == 1

    def test_no_goal_fails(self):
        """Must have a goal in context."""
        planner = FakePlanner([make_plan()])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext()  # no goal
        result = pae.run(ctx)
        assert not result.success
        assert result.metadata["error"] == "missing_goal"

    def test_replan_on_failure(self):
        """First plan fails, revised plan succeeds."""
        bad_plan = make_plan(make_step("fail", tool="bad"))
        good_plan = make_plan(make_step("succeed", tool="good"))
        planner = FakePlanner([bad_plan, good_plan])
        fake = FakeToolExecutor(results={
            "bad": FakeToolResult(success=False, error="boom"),
            "good": FakeToolResult(success=True, output="ok"),
        })
        pae = PlanAndExecute(planner, fake)
        ctx = SharedContext(goal=make_goal())
        result = pae.run(ctx)
        assert result.success
        assert result.iterations == 2
        assert len(planner.revise_calls) == 1

    def test_max_replans_exhausted(self):
        """All attempts fail → overall failure."""
        bad_plan = make_plan(make_step("fail", tool="bad"))
        planner = FakePlanner([bad_plan])  # same plan every time
        fake = FakeToolExecutor(results={
            "bad": FakeToolResult(success=False, error="nope"),
        })
        pae = PlanAndExecute(planner, fake, max_replans=2)
        ctx = SharedContext(goal=make_goal())
        result = pae.run(ctx)
        assert not result.success
        assert result.iterations == 3  # 1 initial + 2 replans

    def test_abort_stops_replanning(self):
        """Abort during execution → no further attempts."""
        plan = make_plan(make_step("step1"), make_step("step2"))
        planner = FakePlanner([plan])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext(goal=make_goal())
        # Abort after first action
        ctx.event_bus.subscribe(
            ActionCompleted,
            lambda e: ctx.event_bus.publish(Abort(reason="stop")),
        )
        result = pae.run(ctx)
        assert result.aborted
        assert result.iterations == 1  # no retry after abort

    def test_reflection_stops_replanning(self):
        """Reflection trigger → stop, don't retry."""
        plan = make_plan(make_step("step1"), make_step("step2"))
        planner = FakePlanner([plan])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext(goal=make_goal())
        trigger = ReflectionTrigger(type=TriggerType.EVENT, reason="test")
        ctx.event_bus.subscribe(
            ActionCompleted,
            lambda e: ctx.event_bus.publish(Reflect(trigger=trigger)),
        )
        result = pae.run(ctx)
        assert result.reflection_triggered is not None
        assert result.iterations == 1

    def test_plan_stored_in_context(self):
        """context.plan should be set to the current plan."""
        plan = make_plan(make_step("do it"))
        planner = FakePlanner([plan])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext(goal=make_goal())
        pae.run(ctx)
        assert ctx.plan is plan

    def test_zero_max_replans(self):
        """max_replans=0 means single attempt only."""
        bad_plan = make_plan(make_step("fail", tool="bad"))
        planner = FakePlanner([bad_plan])
        fake = FakeToolExecutor(results={
            "bad": FakeToolResult(success=False, error="nope"),
        })
        pae = PlanAndExecute(planner, fake, max_replans=0)
        ctx = SharedContext(goal=make_goal())
        result = pae.run(ctx)
        assert not result.success
        assert result.iterations == 1
        assert len(planner.revise_calls) == 0

    def test_planner_receives_failure_info(self):
        """revise_plan should get a reason describing failed steps."""
        bad_plan = make_plan(make_step("read data", tool="read"))
        good_plan = make_plan(make_step("ok", tool="ok"))
        planner = FakePlanner([bad_plan, good_plan])
        fake = FakeToolExecutor(results={
            "read": FakeToolResult(success=False, error="file not found"),
            "ok": FakeToolResult(success=True, output="done"),
        })
        pae = PlanAndExecute(planner, fake)
        ctx = SharedContext(goal=make_goal())
        pae.run(ctx)
        assert len(planner.revise_calls) == 1
        reason = planner.revise_calls[0][1]
        assert "read data" in reason
        assert "file not found" in reason

    def test_is_a_pattern(self):
        planner = FakePlanner([make_plan()])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        assert isinstance(pae, Pattern)

    def test_run_returns_pattern_result(self):
        plan = make_plan(make_step("do it"))
        planner = FakePlanner([plan])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext(goal=make_goal())
        result = pae.run(ctx)
        assert isinstance(result, PatternResult)

    def test_events_flow_through(self):
        """ActionCompleted and PatternComplete from inner executor should fire."""
        plan = make_plan(make_step("do it"))
        planner = FakePlanner([plan])
        pae = PlanAndExecute(planner, FakeToolExecutor())
        ctx = SharedContext(goal=make_goal())
        events = []
        ctx.event_bus.subscribe(ActionCompleted, lambda e: events.append("action"))
        ctx.event_bus.subscribe(PatternComplete, lambda e: events.append("complete"))
        pae.run(ctx)
        assert "action" in events
        assert "complete" in events
