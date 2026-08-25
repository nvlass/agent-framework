"""ReAct loop — Reason → Act → Observe → Repeat."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

from agent_mind.goals.model import Goal
from agent_mind.introspection.monitor import ActionResult
from agent_mind.introspection.triggers import ReflectionTrigger

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.events.types import (
    ActionCompleted,
    Stuck,
    PatternComplete,
    Abort,
    Reflect,
)
from agent_patterns.context import SharedContext


@dataclass
class ReasoningResult:
    """Output from the reasoner: thought + optional action or answer."""
    thought: str
    action: Optional[str] = None       # tool name (None = done)
    action_args: dict = field(default_factory=dict)
    answer: Optional[str] = None       # final answer if done


class ReasonerInterface(ABC):
    """Abstract interface for the LLM reasoning component."""

    @abstractmethod
    def reason(
        self,
        goal: Goal,
        observations: list[str],
        available_tools: list[dict],
    ) -> ReasoningResult:
        """Given goal and observations, produce next thought+action or answer."""


@dataclass
class ReactResult:
    """Result of a ReAct loop run."""
    success: bool
    answer: Optional[str] = None
    iterations: int = 0
    reflection_triggered: Optional[ReflectionTrigger] = None


# FIXME: move the Mock* interface implementations to tests??
class MockReasoner(ReasonerInterface):
    """Scripted reasoner for testing — returns pre-defined sequence of results."""

    def __init__(self, steps: list[ReasoningResult]) -> None:
        self._steps = list(steps)
        self._index = 0

    def reason(
        self,
        goal: Goal,
        observations: list[str],
        available_tools: list[dict],
    ) -> ReasoningResult:
        if self._index >= len(self._steps):
            return ReasoningResult(thought="no more steps", answer="exhausted")
        result = self._steps[self._index]
        self._index += 1
        return result


class ReactLoop(Pattern):
    """ReAct execution loop: Reason → Act → Observe → Repeat.

    Integrates with ProgressMonitor for reflection triggers and
    EventBus for mind↔patterns communication.
    """

    def __init__(self, tool_executor, reasoner: ReasonerInterface,
                 max_iterations: int = 10) -> None:
        self._tool_executor = tool_executor
        self._reasoner = reasoner
        self._max_iterations = max_iterations

    def run(self, context: SharedContext) -> PatternResult:
        """Execute the ReAct loop. Implements Pattern.run().

        Requires context.goal to be set.
        """
        if context.goal is None:
            return PatternResult(success=False, summary="no goal set in context")
        result = self.execute(context)
        return PatternResult(
            success=result.success,
            summary=result.answer or "",
            iterations=result.iterations,
            reflection_triggered=result.reflection_triggered,
            metadata={"answer": result.answer},
        )

    def execute(self, context: SharedContext) -> ReactResult:
        """Run the ReAct loop.

        Each iteration:
        1. Call reasoner.reason(goal, observations, tool_schemas)
        2. If reasoner returns answer → done (success)
        3. If reasoner returns action → execute tool
        4. Add tool result to observations
        5. Feed result to context.monitor
        6. Check for reflection trigger
        7. Publish events

        Stops on:
        - Reasoner returns an answer
        - Max iterations reached
        - Reflection triggered
        - Abort event received

        Returns ReactResult with summary.
        """
        aborted = False
        reflect_trigger: Optional[ReflectionTrigger] = None

        def handle_abort(event: Abort) -> None:
            nonlocal aborted
            aborted = True

        def handle_reflect(event: Reflect) -> None:
            nonlocal reflect_trigger
            reflect_trigger = event.trigger

        context.event_bus.subscribe(Abort, handle_abort)
        context.event_bus.subscribe(Reflect, handle_reflect)

        goal = context.goal
        tool_executor = self._tool_executor
        reasoner = self._reasoner

        # Build tool schemas for reasoner
        available_tools = tool_executor.registry.to_schemas() if hasattr(tool_executor, 'registry') else []

        iterations = 0
        for i in range(self._max_iterations):
            if aborted:
                break

            if reflect_trigger:
                break

            iterations = i + 1

            reasoning = reasoner.reason(goal, context.observations, available_tools)
            if reasoning.answer is not None:
                logger.info("Iteration %d: answer found", iterations)
                context.event_bus.publish(PatternComplete(
                    goal_id = goal.id,
                    success = True,
                    summary = reasoning.answer,
                ))
                return ReactResult(success = True,
                                   answer = reasoning.answer,
                                   iterations = i + 1)
            if reasoning.action:
                logger.info("Iteration %d: action=%s args=%s", iterations,
                            reasoning.action, reasoning.action_args)
                logger.debug("Iteration %d: thought=%s", iterations, reasoning.thought)
                tool_result = tool_executor.execute(reasoning.action, **reasoning.action_args)
                observation = f"{reasoning.action}: {tool_result.output}" if tool_result.success \
                    else f"{reasoning.action} FAILED: {tool_result.error}"
                logger.info("Iteration %d: tool %s %s (%dms)", iterations, reasoning.action,
                            "succeeded" if tool_result.success else "FAILED",
                            getattr(tool_result, 'duration_ms', 0))
                if not tool_result.success:
                    logger.warning("Iteration %d: tool %s error: %s", iterations,
                                   reasoning.action, tool_result.error)
                logger.debug("Observation: %s", observation)
                context.observations.append(observation)
                classification = ActionResult.PROGRESS if tool_result.success \
                    else ActionResult.FAILURE
                context.monitor.record_action(reasoning.action, observation, classification)
                context.event_bus.publish(ActionCompleted(reasoning.action,
                                                          observation,
                                                          classification))

                if aborted or reflect_trigger:
                    break

                trigger = context.monitor.should_reflect()
                if trigger:
                    reflect_trigger = trigger
                    return ReactResult(success = False,
                                       iterations = i+1,
                                       reflection_triggered = trigger)

        if aborted:
            logger.info("ReAct loop aborted after %d iterations", iterations)
        elif reflect_trigger:
            logger.info("ReAct loop paused for reflection after %d iterations", iterations)
        else:
            logger.info("ReAct loop hit max iterations (%d)", iterations)
        return ReactResult(
            success=False,
            iterations=iterations,
            reflection_triggered=reflect_trigger,
        )
