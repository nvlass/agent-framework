"""Tests for Pattern base interface."""

import pytest

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.context import SharedContext


class TestPatternResult:
    def test_defaults(self):
        r = PatternResult(success=True)
        assert r.success
        assert r.summary == ""
        assert r.iterations == 0
        assert r.reflection_triggered is None
        assert r.aborted is False
        assert r.metadata == {}

    def test_with_metadata(self):
        r = PatternResult(success=False, summary="failed", metadata={"reason": "timeout"})
        assert not r.success
        assert r.metadata["reason"] == "timeout"


class TestPatternABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Pattern()

    def test_subclass_must_implement_run(self):
        class Incomplete(Pattern):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_with_run(self):
        class Complete(Pattern):
            def run(self, context):
                return PatternResult(success=True, summary="done")
        p = Complete()
        result = p.run(SharedContext())
        assert result.success
