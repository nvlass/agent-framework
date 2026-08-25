"""Tests for ReflexionLoop."""

import pytest
from dataclasses import dataclass
from typing import Optional

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.context import SharedContext
from agent_patterns.reflexion import ReflexionLoop
from agent_mind.goals.model import Goal, GoalState


# --- Test helpers ---

def _make_goal(desc: str = "test goal") -> Goal:
    return Goal(description=desc, state=GoalState.ACTIVE)


def _make_context(desc: str = "test goal") -> SharedContext:
    ctx = SharedContext(goal=_make_goal(desc))
    return ctx


class _FixedPattern(Pattern):
    """Pattern that returns a pre-scripted sequence of results."""

    def __init__(self, results: list[PatternResult]) -> None:
        self._results = list(results)
        self._calls = 0

    def run(self, context: SharedContext) -> PatternResult:
        result = self._results[min(self._calls, len(self._results) - 1)]
        self._calls += 1
        return result

    @property
    def calls(self) -> int:
        return self._calls


def _succeed(summary: str = "done") -> PatternResult:
    return PatternResult(success=True, summary=summary)


def _fail(summary: str = "failed") -> PatternResult:
    return PatternResult(success=False, summary=summary)


def _reflect_fn(prompt: str) -> str:
    return f"reflection: {prompt[:30]}"


# --- Core behaviour ---

class TestReflexionLoop:
    def test_succeeds_on_first_attempt(self):
        pattern = _FixedPattern([_succeed("all good")])
        loop = ReflexionLoop(pattern, _reflect_fn, max_attempts=3)
        result = loop.run(_make_context())
        assert result.success
        assert result.summary == "all good"
        assert result.iterations == 1
        assert pattern.calls == 1

    def test_retries_on_failure(self):
        pattern = _FixedPattern([_fail(), _succeed("second try")])
        loop = ReflexionLoop(pattern, _reflect_fn, max_attempts=3)
        result = loop.run(_make_context())
        assert result.success
        assert result.iterations == 2
        assert pattern.calls == 2

    def test_fails_after_max_attempts(self):
        pattern = _FixedPattern([_fail("nope")])
        loop = ReflexionLoop(pattern, _reflect_fn, max_attempts=3)
        result = loop.run(_make_context())
        assert not result.success
        assert result.iterations == 3
        assert pattern.calls == 3

    def test_no_reflect_call_on_last_attempt(self):
        """Reflection is only generated between attempts, not after the final one."""
        reflect_calls = []
        def counting_reflect(prompt: str) -> str:
            reflect_calls.append(prompt)
            return "reflection"

        pattern = _FixedPattern([_fail(), _fail(), _fail()])
        loop = ReflexionLoop(pattern, counting_reflect, max_attempts=3)
        loop.run(_make_context())
        # 3 attempts → 2 reflections (between attempt 1→2 and 2→3)
        assert len(reflect_calls) == 2

    def test_success_on_last_attempt(self):
        pattern = _FixedPattern([_fail(), _fail(), _succeed("third time lucky")])
        loop = ReflexionLoop(pattern, _reflect_fn, max_attempts=3)
        result = loop.run(_make_context())
        assert result.success
        assert result.iterations == 3

    def test_max_attempts_one_no_retry(self):
        pattern = _FixedPattern([_fail()])
        loop = ReflexionLoop(pattern, _reflect_fn, max_attempts=1)
        result = loop.run(_make_context())
        assert not result.success
        assert result.iterations == 1
        assert pattern.calls == 1

    def test_no_goal_returns_failure(self):
        pattern = _FixedPattern([_succeed()])
        loop = ReflexionLoop(pattern, _reflect_fn)
        ctx = SharedContext()  # no goal
        result = loop.run(ctx)
        assert not result.success
        assert "goal" in result.summary.lower()


class TestReflexionContextIsolation:
    """Each attempt gets a fresh context with only prior reflections."""

    def test_fresh_context_per_attempt(self):
        """Inner pattern sees a clean observation list each attempt."""
        seen_observations: list[list[str]] = []

        class _ObservingPattern(Pattern):
            def run(self, ctx: SharedContext) -> PatternResult:
                seen_observations.append(list(ctx.observations))
                return _fail()

        loop = ReflexionLoop(_ObservingPattern(), _reflect_fn, max_attempts=2)
        loop.run(_make_context())
        # First attempt: no observations (fresh)
        assert seen_observations[0] == []
        # Second attempt: only the reflection from attempt 1
        assert len(seen_observations[1]) == 1
        assert "[Reflection from attempt 1]" in seen_observations[1][0]

    def test_reflections_accumulate_across_attempts(self):
        """Each retry sees all prior reflections, not just the last one."""
        seen_observations: list[list[str]] = []

        class _ObservingPattern(Pattern):
            def run(self, ctx: SharedContext) -> PatternResult:
                seen_observations.append(list(ctx.observations))
                return _fail()

        loop = ReflexionLoop(_ObservingPattern(), lambda p: "insight", max_attempts=3)
        loop.run(_make_context())
        # Attempt 3 should see reflections from attempt 1 and 2
        assert len(seen_observations[2]) == 2
        assert "[Reflection from attempt 1]" in seen_observations[2][0]
        assert "[Reflection from attempt 2]" in seen_observations[2][1]

    def test_original_context_observations_not_modified(self):
        """ReflexionLoop does not mutate the original context."""
        original_ctx = _make_context()
        original_ctx.observations.append("pre-existing observation")

        pattern = _FixedPattern([_fail(), _succeed()])
        loop = ReflexionLoop(pattern, _reflect_fn, max_attempts=3)
        loop.run(original_ctx)
        # Original context untouched
        assert original_ctx.observations == ["pre-existing observation"]

    def test_goal_passed_through_to_inner_pattern(self):
        """Inner pattern receives the same goal as the outer context."""
        seen_goals = []

        class _GoalCapture(Pattern):
            def run(self, ctx: SharedContext) -> PatternResult:
                seen_goals.append(ctx.goal.description)
                return _succeed()

        ctx = _make_context("specific goal description")
        loop = ReflexionLoop(_GoalCapture(), _reflect_fn)
        loop.run(ctx)
        assert seen_goals == ["specific goal description"]


class TestReflexionMetadata:
    def test_reflections_in_metadata(self):
        reflect_outputs = []
        def capture_reflect(prompt: str) -> str:
            r = f"reflection {len(reflect_outputs) + 1}"
            reflect_outputs.append(r)
            return r

        pattern = _FixedPattern([_fail(), _fail(), _succeed()])
        loop = ReflexionLoop(pattern, capture_reflect, max_attempts=3)
        result = loop.run(_make_context())
        assert result.metadata["attempts"] == 3
        assert len(result.metadata["reflections"]) == 2
        assert result.metadata["reflections"][0] == "reflection 1"
        assert result.metadata["reflections"][1] == "reflection 2"

    def test_no_reflections_on_immediate_success(self):
        pattern = _FixedPattern([_succeed()])
        loop = ReflexionLoop(pattern, _reflect_fn)
        result = loop.run(_make_context())
        assert result.metadata["reflections"] == []
        assert result.metadata["attempts"] == 1

    def test_reflect_fn_exception_does_not_crash(self):
        """If reflect_fn raises, loop continues with fallback reflection."""
        def broken_reflect(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        pattern = _FixedPattern([_fail(), _succeed()])
        loop = ReflexionLoop(pattern, broken_reflect, max_attempts=3)
        result = loop.run(_make_context())
        assert result.success
        # Fallback reflection still injected
        assert len(result.metadata["reflections"]) == 1
        assert "failed" in result.metadata["reflections"][0].lower()


class TestReflexionIntegration:
    """Integration with agent_core config and AgentInstance."""

    def test_config_validates_reflexion_pattern(self):
        from agent_core.config import AgentConfig, ReflexionConfig
        config = AgentConfig(pattern="reflexion")
        errors = config.validate()
        assert errors == []

    def test_config_validates_reflexion_fields(self):
        from agent_core.config import AgentConfig, ReflexionConfig
        config = AgentConfig(
            pattern="reflexion",
            reflexion=ReflexionConfig(max_attempts=0),
        )
        errors = config.validate()
        assert any("max_attempts" in e for e in errors)

    def test_reflexion_config_to_dict_from_dict(self):
        from agent_core.config import AgentConfig, ReflexionConfig
        config = AgentConfig(
            pattern="reflexion",
            reflexion=ReflexionConfig(max_attempts=5, max_iterations_per_attempt=8),
        )
        d = config.to_dict()
        assert d["reflexion"]["max_attempts"] == 5
        assert d["reflexion"]["max_iterations_per_attempt"] == 8
        restored = AgentConfig.from_dict(d)
        assert restored.reflexion.max_attempts == 5
        assert restored.reflexion.max_iterations_per_attempt == 8

    def test_agent_instance_builds_reflexion_pattern(self):
        from agent_core import AgentRole, AgentInstance, AgentConfig, ReflexionConfig
        from agent_core.llm import MockChatLLM

        config = AgentConfig(
            pattern="reflexion",
            reflexion=ReflexionConfig(max_attempts=2),
        )
        role = AgentRole(name="test", config=config)
        llm = MockChatLLM([
            "Thought: I know\nAnswer: done",
        ])
        agent = AgentInstance(role, llm)
        result = agent.run("do something")
        assert result.success
