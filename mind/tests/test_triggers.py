"""Tests for reflection trigger types.

Run with:
    cd mind/
    python -m pytest tests/test_triggers.py -v
"""

from agent_mind.introspection.triggers import TriggerType, ReflectionTrigger


class TestTriggerType:

    def test_values(self):
        assert TriggerType.EVENT.value == "event"
        assert TriggerType.IDLE.value == "idle"
        assert TriggerType.SELF_PROMPTED.value == "self_prompted"


class TestReflectionTrigger:

    def test_defaults(self):
        t = ReflectionTrigger(type=TriggerType.EVENT, reason="3 failures")
        assert t.type == TriggerType.EVENT
        assert t.reason == "3 failures"
        assert t.episode_id is None

    def test_with_episode_id(self):
        t = ReflectionTrigger(
            type=TriggerType.EVENT, reason="big failure", episode_id="ep-42"
        )
        assert t.episode_id == "ep-42"

    def test_to_dict(self):
        t = ReflectionTrigger(type=TriggerType.IDLE, reason="downtime")
        d = t.to_dict()
        assert d["type"] == "idle"
        assert d["reason"] == "downtime"
        assert d["episode_id"] is None

    def test_to_dict_with_episode_id(self):
        t = ReflectionTrigger(
            type=TriggerType.SELF_PROMPTED, reason="curious", episode_id="ep-1"
        )
        d = t.to_dict()
        assert d["episode_id"] == "ep-1"
