"""PlanExecutor — runs Plan steps via ToolExecutor with DAG scheduling."""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

from agent_mind.planning.model import Plan, PlanStep, StepStatus
from agent_mind.introspection.monitor import ActionResult
from agent_mind.introspection.triggers import ReflectionTrigger

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.events.types import (
    ActionCompleted,
    StepFailed,
    PatternComplete,
    Abort,
    Reflect,
)
from agent_patterns.context import SharedContext


@dataclass
class PlanExecutionResult:
    """Result of executing a plan."""
    success: bool
    steps_completed: int = 0
    steps_failed: int = 0
    reflection_triggered: Optional[ReflectionTrigger] = None
    aborted: bool = False


class PlanExecutor(Pattern):
    """Executes a Plan's steps using ToolExecutor with DAG scheduling.

    Integrates with ProgressMonitor for reflection triggers and
    EventBus for mind↔patterns communication.

    When a reasoner is provided, steps without a pre-specified tool_name
    are executed via a mini-ReactLoop (ReAct-per-step) instead of as
    no-ops. This lets the planner produce high-level descriptions and
    delegate execution details to the reasoner.
    """

    def __init__(self, plan: Plan, tool_executor, reasoner=None) -> None:
        self._plan = plan
        self._tool_executor = tool_executor
        self._reasoner = reasoner
        self._aborted = False
        self._abort_reason: Optional[str] = None
        self._reflect_trigger: Optional[ReflectionTrigger] = None

    def run(self, context: SharedContext) -> PatternResult:
        """Execute the plan. Implements Pattern.run()."""
        result = self.execute_plan(context)
        return PatternResult(
            success=result.success,
            summary=f"completed={result.steps_completed}, failed={result.steps_failed}",
            reflection_triggered=result.reflection_triggered,
            aborted=result.aborted,
            metadata={"steps_completed": result.steps_completed,
                      "steps_failed": result.steps_failed},
        )

    def execute_plan(self, context: SharedContext) -> PlanExecutionResult:
        """Execute a plan step by step using DAG scheduling.

        Loop:
        1. Get next ready steps via plan.next_steps()
        2. Execute each step via tool_executor
        3. Update step status based on result
        4. Feed result to context.monitor
        5. Check for reflection trigger
        6. Publish events via context.event_bus

        Stops on:
        - Plan complete (all steps terminal)
        - All remaining steps blocked/failed (no next_steps and plan not complete)
        - Abort event received
        - Reflection triggered

        Returns PlanExecutionResult with summary.
        """
        # Subscribe to control events
        self._aborted = False
        self._abort_reason = None
        self._reflect_trigger = None
        context.event_bus.subscribe(Abort, self._handle_abort)
        context.event_bus.subscribe(Reflect, self._handle_reflect)

        steps_completed = 0
        steps_failed = 0

        while True:
            if self._aborted:
                break

            if self._reflect_trigger:
                break

            ready = self._plan.next_steps()

            if not ready:
                break

            logger.info("DAG: %d steps ready", len(ready))
            for step in ready:
                logger.info("Step %s started: %s", step.id, step.description)
                step.status = StepStatus.IN_PROGRESS
                tool_result = self._execute_step(step, context)
                classification = self._classify_result(tool_result)
                if tool_result.success:
                    step.status = StepStatus.COMPLETED
                    step.result = str(tool_result.output)
                    steps_completed += 1
                    logger.info("Step %s completed", step.id)
                    # Make result visible to subsequent steps via observations
                    context.observations.append(
                        f"Step '{step.description}' result: {step.result}"
                    )
                else:
                    step.status = StepStatus.FAILED
                    step.error = tool_result.error
                    steps_failed += 1
                    logger.warning("Step %s failed: %s", step.id, tool_result.error)
                    context.event_bus.publish(StepFailed(step.id, tool_result.error or "unknown"))

                # So, there's an intermediate "class" for decoupling -- it's OK
                context.monitor.record_action(step.description,
                                              str(tool_result.output),
                                              classification)

                context.event_bus.publish(ActionCompleted(step.description,
                                          str(tool_result.output),
                                          classification))

                if self._reflect_trigger or self._aborted:
                    break

                trigger = context.monitor.should_reflect()
                if trigger:
                    self._reflect_trigger = trigger
                    break

        plan = self._plan
        success = (not plan.steps or plan.is_complete) and not plan.has_failures
        context.event_bus.publish(PatternComplete(
            goal_id=plan.goal_id, success=success,
            summary=f"completed={steps_completed}, failed={steps_failed}",
        ))

        return PlanExecutionResult(
            success=success,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            reflection_triggered=self._reflect_trigger,
            aborted=self._aborted,
        )

    def _execute_step(self, step: PlanStep, context: SharedContext) -> "ToolResult":
        """Execute a single plan step.

        If the step has a tool_name, call it directly.
        If not, and a reasoner is available, run a mini-ReactLoop (ReAct-per-step)
        so the agent can figure out how to execute the open-ended description.
        Otherwise, treat as a no-op.
        """
        from agent_tools import ToolResult

        if step.tool_name:
            return self._tool_executor.execute(step.tool_name, **step.tool_args)

        if self._reasoner is not None:
            return self._execute_step_with_react(step, context)

        return ToolResult(success=True, output="no-op step", tool_name="")

    def _execute_step_with_react(self, step: PlanStep,
                                 parent_context: SharedContext) -> "ToolResult":
        """Run a mini-ReactLoop to execute an open-ended step description.

        The sub-context inherits parent observations so the ReactLoop sees
        results from prior steps.
        """
        from agent_tools import ToolResult
        from agent_patterns.react import ReactLoop
        from agent_mind.goals.model import Goal, GoalState
        from agent_patterns.context import SharedContext as SubContext

        sub_goal = Goal(description=step.description, state=GoalState.ACTIVE)
        sub_context = SubContext(goal=sub_goal)
        sub_context.observations = list(parent_context.observations)

        react = ReactLoop(
            tool_executor=self._tool_executor,
            reasoner=self._reasoner,
            max_iterations=10,
        )
        result = react.run(sub_context)
        logger.info("ReAct-per-step for %r: success=%s", step.description, result.success)
        return ToolResult(
            success=result.success,
            output=result.summary or "",
            tool_name="react_step",
        )

    @staticmethod
    def _classify_result(tool_result) -> ActionResult:
        """Map ToolResult.success to ActionResult."""
        if tool_result.success:
            return ActionResult.PROGRESS
        return ActionResult.FAILURE

    def _handle_abort(self, event: Abort) -> None:
        self._aborted = True
        self._abort_reason = event.reason

    def _handle_reflect(self, event: Reflect) -> None:
        self._reflect_trigger = event.trigger
