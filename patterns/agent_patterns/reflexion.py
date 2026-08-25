"""ReflexionLoop — attempt → reflect on failure → retry with reflection.

Based on: Reflexion: Language Agents with Verbal Reinforcement Learning
(Shinn et al., 2023)

The agent attempts a task via an inner pattern (usually ReactLoop). On failure,
an LLM generates a verbal reflection: what went wrong, what to try differently.
The reflection is injected into the next attempt's context. Only reflections are
carried forward (not raw observation history) to prevent context bloat.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.context import SharedContext

logger = logging.getLogger(__name__)

_REFLECT_PROMPT = """\
You attempted a task but did not succeed. Reflect on what went wrong and what \
should be done differently.

Task: {goal}

What happened during the attempt:
{observations}

Outcome: {summary}

Provide a specific, actionable reflection covering:
1. What went wrong or was missed
2. What you should do differently next time
3. Any tools or approaches to try or avoid

Be concise and specific."""


@dataclass
class ReflexionResult:
    """Result from a ReflexionLoop run."""
    success: bool
    attempts: int
    final_summary: str = ""
    reflections: list[str] = field(default_factory=list)


class ReflexionLoop(Pattern):
    """Wraps any Pattern with reflect-and-retry on failure.

    Each attempt:
    1. Run the inner pattern (ReactLoop or any Pattern)
    2. If success → done
    3. If failure → call reflect_fn to generate verbal reflection
    4. Build fresh context with only prior reflections injected
    5. Retry up to max_attempts total

    Args:
        pattern: Inner pattern to attempt (typically ReactLoop).
        reflect_fn: Callable (prompt: str) -> str. Called with a structured
            failure description, returns a reflection string. Injected as an
            LLM call in agent_core.
        max_attempts: Total number of attempts (1 initial + retries).
    """

    def __init__(
        self,
        pattern: Pattern,
        reflect_fn: Callable[[str], str],
        max_attempts: int = 3,
    ) -> None:
        self._pattern = pattern
        self._reflect_fn = reflect_fn
        self._max_attempts = max_attempts

    def run(self, context: SharedContext) -> PatternResult:
        """Execute with reflect-and-retry. Implements Pattern.run()."""
        if context.goal is None:
            return PatternResult(success=False, summary="no goal set in context")

        result = self._run_reflexion(context)
        return self._to_pattern_result(result)

    def _run_reflexion(self, context: SharedContext) -> ReflexionResult:
        goal = context.goal
        reflections: list[str] = []
        last_result: Optional[PatternResult] = None

        for attempt in range(1, self._max_attempts + 1):
            logger.info("Reflexion attempt %d/%d", attempt, self._max_attempts)

            # Build fresh context for this attempt — inject prior reflections only
            attempt_context = self._build_attempt_context(context, reflections)
            result = self._pattern.run(attempt_context)
            last_result = result

            if result.success:
                logger.info("Reflexion succeeded on attempt %d", attempt)
                return ReflexionResult(
                    success=True,
                    attempts=attempt,
                    final_summary=result.summary,
                    reflections=reflections,
                )

            # Last attempt — no point reflecting
            if attempt == self._max_attempts:
                break

            # Generate reflection from the failed attempt
            reflection = self._reflect(
                goal=goal.description,
                observations=attempt_context.observations,
                summary=result.summary,
                attempt=attempt,
            )
            reflections.append(reflection)
            logger.info("Reflexion attempt %d reflection: %.120s", attempt, reflection)

        return ReflexionResult(
            success=False,
            attempts=self._max_attempts,
            final_summary=last_result.summary if last_result else "",
            reflections=reflections,
        )

    def _build_attempt_context(
        self,
        original: SharedContext,
        reflections: list[str],
    ) -> SharedContext:
        """Create a fresh SharedContext seeded with prior reflections."""
        ctx = SharedContext(goal=original.goal)
        for i, ref in enumerate(reflections, 1):
            ctx.observations.append(f"[Reflection from attempt {i}]: {ref}")
        return ctx

    def _reflect(
        self,
        goal: str,
        observations: list[str],
        summary: str,
        attempt: int,
    ) -> str:
        """Call the reflection LLM to generate a verbal reflection."""
        obs_text = "\n".join(f"- {o}" for o in observations) or "(no observations recorded)"
        prompt = _REFLECT_PROMPT.format(
            goal=goal,
            observations=obs_text,
            summary=summary or "(no summary)",
        )
        try:
            reflection = self._reflect_fn(prompt)
            return reflection.strip() or "No specific reflection generated."
        except Exception as e:
            logger.warning("Reflection LLM call failed: %s", e)
            return f"Reflection failed: {e}"

    @staticmethod
    def _to_pattern_result(r: ReflexionResult) -> PatternResult:
        return PatternResult(
            success=r.success,
            summary=r.final_summary,
            iterations=r.attempts,
            metadata={
                "attempts": r.attempts,
                "reflections": r.reflections,
            },
        )
