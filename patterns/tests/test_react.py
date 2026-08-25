"""Tests for ReactLoop."""

import pytest
from dataclasses import dataclass
from typing import Any, Optional

from agent_mind.goals.model import Goal
from agent_mind.introspection.monitor import ActionResult, ProgressMonitor
from agent_mind.introspection.triggers import ReflectionTrigger, TriggerType

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.react import (
    ReasonerInterface,
    ReasoningResult,
    ReactLoop,
    ReactResult,
    MockReasoner,
)
from agent_patterns.context import SharedContext
from agent_patterns.events.types import Abort, Reflect, ActionCompleted, PatternComplete


# --- Test helpers ---

@dataclass
class FakeToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    tool_name: str = ""


class FakeRegistry:
    def to_schemas(self):
        return [{"name": "test_tool", "description": "a test tool"}]


class FakeToolExecutor:
    """Fake ToolExecutor for testing ReactLoop."""

    def __init__(self, results: dict[str, FakeToolResult] | None = None):
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []
        self.registry = FakeRegistry()

    def execute(self, tool_name: str, **kwargs) -> FakeToolResult:
        self.calls.append((tool_name, kwargs))
        if tool_name in self._results:
            return self._results[tool_name]
        return FakeToolResult(success=True, output=f"{tool_name} result", tool_name=tool_name)


def make_goal(desc: str = "test goal") -> Goal:
    return Goal(description=desc)


def make_context(goal: Goal | None = None) -> SharedContext:
    return SharedContext(goal=goal or make_goal())


# --- Tests ---

class TestMockReasoner:
    def test_scripted_sequence(self):
        steps = [
            ReasoningResult(thought="think", action="tool1", action_args={"x": 1}),
            ReasoningResult(thought="done", answer="42"),
        ]
        r = MockReasoner(steps)
        goal = make_goal()
        res1 = r.reason(goal, [], [])
        assert res1.action == "tool1"
        res2 = r.reason(goal, [], [])
        assert res2.answer == "42"

    def test_exhausted(self):
        r = MockReasoner([])
        res = r.reason(make_goal(), [], [])
        assert res.answer == "exhausted"

    def test_beyond_steps(self):
        r = MockReasoner([ReasoningResult(thought="t", answer="a")])
        r.reason(make_goal(), [], [])
        res = r.reason(make_goal(), [], [])
        assert res.answer == "exhausted"


class TestReactLoop:
    """Tests for the ReactLoop.execute method."""

    def test_immediate_answer(self):
        """Reasoner returns answer on first call."""
        reasoner = MockReasoner([
            ReasoningResult(thought="I know this", answer="42"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        result = loop.execute(ctx)
        assert result.success
        assert result.answer == "42"
        assert result.iterations == 1

    def test_one_action_then_answer(self):
        """Reasoner does one action, then answers."""
        reasoner = MockReasoner([
            ReasoningResult(thought="need to check", action="read_file", action_args={"path": "/tmp"}),
            ReasoningResult(thought="now I know", answer="found it"),
        ])
        fake = FakeToolExecutor()
        loop = ReactLoop(fake, reasoner)
        ctx = make_context()
        result = loop.execute(ctx)
        assert result.success
        assert result.answer == "found it"
        assert result.iterations == 2
        assert len(fake.calls) == 1

    def test_observations_accumulated(self):
        """Tool results should be added to context.observations."""
        reasoner = MockReasoner([
            ReasoningResult(thought="t1", action="tool_a"),
            ReasoningResult(thought="t2", action="tool_b"),
            ReasoningResult(thought="done", answer="ok"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        loop.execute(ctx)
        assert len(ctx.observations) == 2

    def test_max_iterations(self):
        """Loop should stop at max_iterations."""
        # Reasoner that never answers
        steps = [ReasoningResult(thought=f"t{i}", action="tool") for i in range(20)]
        reasoner = MockReasoner(steps)
        loop = ReactLoop(FakeToolExecutor(), reasoner, max_iterations=3)
        ctx = make_context()
        result = loop.execute(ctx)
        assert not result.success
        assert result.iterations == 3
        assert result.answer is None

    def test_tool_failure_recorded(self):
        """Failed tool calls should still be observed."""
        reasoner = MockReasoner([
            ReasoningResult(thought="try", action="bad_tool"),
            ReasoningResult(thought="done", answer="tried"),
        ])
        fake = FakeToolExecutor(results={"bad_tool": FakeToolResult(success=False, error="nope")})
        loop = ReactLoop(fake, reasoner)
        ctx = make_context()
        result = loop.execute(ctx)
        assert result.success
        assert any("FAILED" in obs for obs in ctx.observations)

    def test_monitor_integration(self):
        """Actions should be recorded in the progress monitor."""
        reasoner = MockReasoner([
            ReasoningResult(thought="t", action="tool"),
            ReasoningResult(thought="done", answer="ok"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        loop.execute(ctx)
        assert len(ctx.monitor.history) == 1

    def test_action_completed_events(self):
        """ActionCompleted events should be published."""
        reasoner = MockReasoner([
            ReasoningResult(thought="t", action="tool"),
            ReasoningResult(thought="done", answer="ok"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        events = []
        ctx.event_bus.subscribe(ActionCompleted, lambda e: events.append(e))
        loop.execute(ctx)
        assert len(events) == 1

    def test_pattern_complete_event_on_answer(self):
        """PatternComplete should be published when answer found."""
        reasoner = MockReasoner([
            ReasoningResult(thought="done", answer="42"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        events = []
        ctx.event_bus.subscribe(PatternComplete, lambda e: events.append(e))
        loop.execute(ctx)
        assert len(events) == 1
        assert events[0].success

    def test_abort_stops_loop(self):
        """Abort event should stop the loop."""
        steps = [ReasoningResult(thought=f"t{i}", action="tool") for i in range(5)]
        reasoner = MockReasoner(steps)
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        ctx.event_bus.subscribe(ActionCompleted, lambda e: ctx.event_bus.publish(Abort(reason="stop")))
        result = loop.execute(ctx)
        assert not result.success
        assert result.iterations < 5

    def test_reflection_trigger_stops_loop(self):
        """Reflection trigger should stop the loop and be reported."""
        steps = [ReasoningResult(thought=f"t{i}", action="tool") for i in range(5)]
        reasoner = MockReasoner(steps)
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        trigger = ReflectionTrigger(type=TriggerType.EVENT, reason="test")
        ctx.event_bus.subscribe(
            ActionCompleted,
            lambda e: ctx.event_bus.publish(Reflect(trigger=trigger)),
        )
        result = loop.execute(ctx)
        assert not result.success
        assert result.reflection_triggered is not None

    def test_no_action_no_answer_counts_as_iteration(self):
        """ReasoningResult with neither action nor answer should still iterate."""
        reasoner = MockReasoner([
            ReasoningResult(thought="hmm, not sure"),  # no action, no answer
            ReasoningResult(thought="done", answer="ok"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        result = loop.execute(ctx)
        assert result.success
        assert result.iterations == 2

    def test_empty_tool_args(self):
        """Action with no args should work fine."""
        reasoner = MockReasoner([
            ReasoningResult(thought="t", action="simple_tool"),
            ReasoningResult(thought="done", answer="ok"),
        ])
        fake = FakeToolExecutor()
        loop = ReactLoop(fake, reasoner)
        ctx = make_context()
        loop.execute(ctx)
        assert fake.calls[0] == ("simple_tool", {})


class TestPatternInterface:
    """Test that ReactLoop implements Pattern."""

    def test_is_pattern(self):
        loop = ReactLoop(FakeToolExecutor(), MockReasoner([]))
        assert isinstance(loop, Pattern)

    def test_run_returns_pattern_result(self):
        reasoner = MockReasoner([
            ReasoningResult(thought="done", answer="42"),
        ])
        loop = ReactLoop(FakeToolExecutor(), reasoner)
        ctx = make_context()
        result = loop.run(ctx)
        assert isinstance(result, PatternResult)
        assert result.success
        assert result.summary == "42"
        assert result.metadata["answer"] == "42"

    def test_run_without_goal_fails(self):
        loop = ReactLoop(FakeToolExecutor(), MockReasoner([]))
        ctx = SharedContext()  # no goal
        result = loop.run(ctx)
        assert not result.success
        assert "no goal" in result.summary
