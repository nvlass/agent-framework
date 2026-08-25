"""PlanAndExecute — creates a plan then executes it, with replanning on failure."""

from dataclasses import dataclass, field
from typing import Optional

from agent_mind.planning.model import Plan, StepStatus
from agent_mind.planning.planner import PlannerInterface
from agent_mind.introspection.triggers import ReflectionTrigger

from agent_patterns.base import Pattern, PatternResult
from agent_patterns.executor import PlanExecutor
from agent_patterns.context import SharedContext


@dataclass
class PlanAndExecuteResult:
    """Result from a PlanAndExecute run."""
    success: bool
    plan_attempts: int
    final_plan: Optional[Plan] = None
    steps_completed: int = 0
    steps_failed: int = 0
    reflection_triggered: Optional[ReflectionTrigger] = None
    aborted: bool = False


class PlanAndExecute(Pattern):
    """Higher-level pattern: create a plan via PlannerInterface, execute via PlanExecutor.

    On execution failure, builds a failure reason from failed steps and asks the
    planner to revise. Retries up to max_replans times.
    """

    def __init__(self, planner: PlannerInterface, tool_executor,
                 max_replans: int = 3, reasoner=None) -> None:
        self._planner = planner
        self._tool_executor = tool_executor
        self._max_replans = max_replans
        self._reasoner = reasoner

    def run(self, context: SharedContext) -> PatternResult:
        """Execute: plan → execute → replan on failure. Implements Pattern.run()."""
        if context.goal is None:
            return PatternResult(
                success=False,
                summary="No goal provided",
                metadata={"error": "missing_goal"},
            )

        plan = None
        attempts = 0
        total_completed = 0
        total_failed = 0

        for attempt in range(self._max_replans + 1):
            attempts = attempt + 1

            if attempt == 0:
                plan = self._planner.create_plan(context.goal)
            else:
                reason = self._build_failure_reason(plan)
                plan = self._planner.revise_plan(plan, reason)

            context.plan = plan
            executor = PlanExecutor(plan, self._tool_executor, reasoner=self._reasoner)
            result = executor.execute_plan(context)

            total_completed += result.steps_completed
            total_failed += result.steps_failed

            if result.success or result.aborted or result.reflection_triggered:
                pe_result = PlanAndExecuteResult(
                    success=result.success,
                    plan_attempts=attempts,
                    final_plan=plan,
                    steps_completed=total_completed,
                    steps_failed=total_failed,
                    reflection_triggered=result.reflection_triggered,
                    aborted=result.aborted,
                )
                return self._to_pattern_result(pe_result)

        # All attempts exhausted
        pe_result = PlanAndExecuteResult(
            success=False,
            plan_attempts=attempts,
            final_plan=plan,
            steps_completed=total_completed,
            steps_failed=total_failed,
        )
        return self._to_pattern_result(pe_result)

    @staticmethod
    def _build_failure_reason(plan: Plan) -> str:
        """Summarize failed steps into a reason string for the planner."""
        failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
        parts = [f"Step '{s.description}' failed: {s.error or 'unknown'}"
                 for s in failed]
        return "; ".join(parts) if parts else "plan failed"

    @staticmethod
    def _to_pattern_result(r: PlanAndExecuteResult) -> PatternResult:
        return PatternResult(
            success=r.success,
            summary=f"attempts={r.plan_attempts}, completed={r.steps_completed}, failed={r.steps_failed}",
            iterations=r.plan_attempts,
            reflection_triggered=r.reflection_triggered,
            aborted=r.aborted,
            metadata={
                "plan_attempts": r.plan_attempts,
                "steps_completed": r.steps_completed,
                "steps_failed": r.steps_failed,
            },
        )
