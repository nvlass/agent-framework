"""Tests for SharedContext."""

import pytest

from agent_mind.goals.model import Goal
from agent_mind.planning.model import Plan
from agent_mind.introspection.monitor import ProgressMonitor

from agent_patterns.context import SharedContext
from agent_patterns.events.types import EventBus


class TestSharedContext:
    """Test SharedContext creation and defaults."""

    def test_default_creation(self):
        ctx = SharedContext()
        assert ctx.goal is None
        assert ctx.plan is None
        assert isinstance(ctx.monitor, ProgressMonitor)
        assert isinstance(ctx.event_bus, EventBus)
        assert ctx.observations == []
        assert ctx.metadata == {}

    def test_with_goal(self):
        goal = Goal(description="test goal")
        ctx = SharedContext(goal=goal)
        assert ctx.goal.description == "test goal"

    def test_with_plan(self):
        plan = Plan(goal_id="g1")
        ctx = SharedContext(plan=plan)
        assert ctx.plan.goal_id == "g1"

    def test_observations_mutable(self):
        ctx = SharedContext()
        ctx.observations.append("saw something")
        assert len(ctx.observations) == 1

    def test_metadata_mutable(self):
        ctx = SharedContext(metadata={"key": "val"})
        assert ctx.metadata["key"] == "val"
        ctx.metadata["new"] = 42
        assert ctx.metadata["new"] == 42

    def test_independent_instances(self):
        ctx1 = SharedContext()
        ctx2 = SharedContext()
        ctx1.observations.append("only ctx1")
        assert len(ctx2.observations) == 0
