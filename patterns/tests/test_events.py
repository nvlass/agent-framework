"""Tests for the event system."""

import pytest

from agent_mind.introspection.monitor import ActionResult
from agent_mind.introspection.triggers import ReflectionTrigger, TriggerType

from agent_patterns.events.types import (
    ActionCompleted,
    StepFailed,
    Stuck,
    PatternComplete,
    Reflect,
    Replan,
    Abort,
    EventBus,
)


class TestEventDataclasses:
    """Test event creation and fields."""

    def test_action_completed(self):
        e = ActionCompleted(action="read_file", result="contents", classification=ActionResult.PROGRESS)
        assert e.action == "read_file"
        assert e.result == "contents"
        assert e.classification == ActionResult.PROGRESS

    def test_step_failed(self):
        e = StepFailed(step_id="s1", reason="timeout")
        assert e.step_id == "s1"
        assert e.reason == "timeout"

    def test_stuck(self):
        e = Stuck(attempts=5, last_actions=["a", "b"])
        assert e.attempts == 5
        assert e.last_actions == ["a", "b"]

    def test_pattern_complete(self):
        e = PatternComplete(goal_id="g1", success=True, summary="done")
        assert e.goal_id == "g1"
        assert e.success is True

    def test_reflect(self):
        trigger = ReflectionTrigger(type=TriggerType.EVENT, reason="failure")
        e = Reflect(trigger=trigger)
        assert e.trigger.type == TriggerType.EVENT

    def test_replan(self):
        e = Replan(reason="step failed")
        assert e.reason == "step failed"

    def test_abort(self):
        e = Abort(reason="user cancelled")
        assert e.reason == "user cancelled"


class TestEventBus:
    """Test EventBus pub/sub."""

    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe(Abort, lambda e: received.append(e))
        bus.publish(Abort(reason="stop"))
        assert len(received) == 1
        assert received[0].reason == "stop"

    def test_multiple_handlers_same_type(self):
        bus = EventBus()
        results = []
        bus.subscribe(Abort, lambda e: results.append("h1"))
        bus.subscribe(Abort, lambda e: results.append("h2"))
        bus.publish(Abort(reason="x"))
        assert results == ["h1", "h2"]

    def test_different_event_types_independent(self):
        bus = EventBus()
        aborts = []
        replans = []
        bus.subscribe(Abort, lambda e: aborts.append(e))
        bus.subscribe(Replan, lambda e: replans.append(e))
        bus.publish(Abort(reason="a"))
        bus.publish(Replan(reason="r"))
        assert len(aborts) == 1
        assert len(replans) == 1

    def test_publish_no_subscribers(self):
        bus = EventBus()
        bus.publish(Abort(reason="nobody listening"))  # no error

    def test_clear(self):
        bus = EventBus()
        received = []
        bus.subscribe(Abort, lambda e: received.append(e))
        bus.clear()
        bus.publish(Abort(reason="after clear"))
        assert len(received) == 0

    def test_publish_preserves_event_data(self):
        bus = EventBus()
        received = []
        bus.subscribe(ActionCompleted, lambda e: received.append(e))
        bus.publish(ActionCompleted(action="test", result="ok", classification=ActionResult.PROGRESS))
        assert received[0].action == "test"
        assert received[0].classification == ActionResult.PROGRESS
