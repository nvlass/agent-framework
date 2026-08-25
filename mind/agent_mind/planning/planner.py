"""
Planner — generates plans from goals.

The Planner is the cognitive capability that breaks goals into steps.
agent-patterns owns the Plan Executor that runs the plans.

This module defines:
- PlannerInterface: Abstract interface for all planners
- SimplePlanner: A basic rule-based planner (no LLM needed)
- LLMPlanner: LLM-based planner that decomposes goals into multi-step plans
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Optional

from ..goals.model import Goal
from .model import Plan, PlanStep, StepStatus

logger = logging.getLogger(__name__)


class PlannerInterface(ABC):
    """Abstract interface for planners.

    All planners implement create_plan() and revise_plan().
    """

    @abstractmethod
    def create_plan(self, goal: Goal,
                    context: Optional[dict[str, Any]] = None) -> Plan:
        """Generate a plan to achieve the goal.

        Args:
            goal: The goal to plan for
            context: Optional context (available tools, constraints, etc.)

        Returns:
            A Plan with steps to achieve the goal
        """

    @abstractmethod
    def revise_plan(self, plan: Plan, reason: str,
                    context: Optional[dict[str, Any]] = None) -> Plan:
        """Revise an existing plan based on new information.

        Args:
            plan: The current plan
            reason: Why revision is needed (e.g., "step 3 failed")
            context: Optional updated context

        Returns:
            A revised Plan
        """


class SimplePlanner(PlannerInterface):
    """A minimal planner that creates single-step plans.

    This is a bootstrap planner — it wraps the goal description
    as a single step. Useful for testing the plan execution loop
    without needing an LLM.

    For real decomposition, use an LLM-based planner.
    """

    def create_plan(self, goal: Goal,
                    context: Optional[dict[str, Any]] = None) -> Plan:
        """Create a single-step plan from the goal description."""
        step = PlanStep(description=goal.description)
        return Plan(goal_id=goal.id, steps=[step])

    def revise_plan(self, plan: Plan, reason: str,
                    context: Optional[dict[str, Any]] = None) -> Plan:
        """Revise by creating a new single-step plan with the reason as context."""
        step = PlanStep(
            description=f"{plan.steps[0].description} (revised: {reason})",
        )
        return Plan(
            goal_id=plan.goal_id,
            steps=[step],
            revised_at=datetime.now(),
            revision_reason=reason,
        )


class LLMPlanner(PlannerInterface):
    """LLM-based planner that decomposes goals into multi-step plans.

    The LLM is injected as a callable (prompt in, text out) to keep
    agent-mind independent of agent-core. Same pattern as agent-memory's
    Reflector.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        available_tools: list[dict] | None = None,
        soul: str = "",
    ) -> None:
        self._llm_fn = llm_fn
        self._available_tools = available_tools or []
        self._soul = soul

    def create_plan(self, goal: Goal,
                    context: Optional[dict[str, Any]] = None) -> Plan:
        """Generate a multi-step plan by asking the LLM to decompose the goal."""
        prompt = self._build_create_prompt(goal, context)
        response = self._llm_fn(prompt)
        return self._parse_plan(goal.id, response)

    def revise_plan(self, plan: Plan, reason: str,
                    context: Optional[dict[str, Any]] = None) -> Plan:
        """Revise a plan by asking the LLM, including previous step outcomes."""
        prompt = self._build_revise_prompt(plan, reason, context)
        response = self._llm_fn(prompt)
        revised = self._parse_plan(plan.goal_id, response)
        revised.revised_at = datetime.now()
        revised.revision_reason = reason
        return revised

    def _build_create_prompt(self, goal: Goal,
                             context: Optional[dict[str, Any]]) -> str:
        parts = []
        if self._soul:
            parts += [self._soul, ""]
        parts += [
            "Break down the following goal into a sequence of steps.",
            "",
            f"Goal: {goal.description}",
        ]
        tools_text = self._format_tools()
        if tools_text:
            parts += ["", "Available tools:", tools_text]

        if context:
            parts += ["", f"Context: {json.dumps(context)}"]

        parts += [
            "",
            "Respond with a JSON array of steps:",
            '[',
            '  {"description": "what to do", "tool_name": "tool_name", "tool_args": {"key": "value"}},',
            '  ...',
            ']',
            "",
            "If a step doesn't use a tool, omit tool_name and tool_args.",
            "Keep the plan concise — minimum steps needed.",
        ]
        return "\n".join(parts)

    def _build_revise_prompt(self, plan: Plan, reason: str,
                             context: Optional[dict[str, Any]]) -> str:
        parts = []
        if self._soul:
            parts += [self._soul, ""]
        parts += [
            "The previous plan failed. Revise it.",
            "",
            f"Goal: {plan.goal_id}",
            "",
            "Previous steps:",
        ]
        for i, step in enumerate(plan.steps, 1):
            status = step.status.value.upper()
            line = f'{i}. "{step.description}" — {status}'
            if step.error:
                line += f" (error: {step.error})"
            parts.append(line)

        parts += [
            "",
            f"Failure reason: {reason}",
        ]

        tools_text = self._format_tools()
        if tools_text:
            parts += ["", "Available tools:", tools_text]

        if context:
            parts += ["", f"Context: {json.dumps(context)}"]

        parts += [
            "",
            "Respond with a revised JSON array of steps (same format as before).",
        ]
        return "\n".join(parts)

    def _parse_plan(self, goal_id: str, response: str) -> Plan:
        """Parse LLM response into a Plan.

        Tries JSON first, falls back to line-based parsing. Never crashes.
        """
        steps = self._try_parse_json(response)
        if steps is None:
            steps = self._try_parse_lines(response)
        if not steps:
            # Ultimate fallback: single step with raw response
            steps = [PlanStep(description=response.strip() or "Execute goal")]

        # Set linear dependencies: each step depends on the previous,
        # unless it has explicit depends_on from the LLM response
        for i in range(1, len(steps)):
            if not getattr(steps[i], '_explicit_deps', False) and not steps[i].depends_on:
                steps[i].depends_on = [steps[i - 1].id]

        return Plan(goal_id=goal_id, steps=steps)

    def _try_parse_json(self, response: str) -> list[PlanStep] | None:
        """Try to extract a JSON array from the response."""
        # Strip ```json fences
        text = response.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        # Find the JSON array
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            items = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        if not isinstance(items, list):
            return None

        steps = []
        for item in items:
            if not isinstance(item, dict):
                continue
            desc = item.get("description", "")
            if not desc:
                continue
            step = PlanStep(
                description=desc,
                tool_name=item.get("tool_name"),
                tool_args=item.get("tool_args", {}),
            )
            # Optional explicit dependencies — mark if present
            if "depends_on" in item and isinstance(item["depends_on"], list):
                step.depends_on = item["depends_on"]
                step._explicit_deps = True  # type: ignore[attr-defined]
            steps.append(step)

        return steps if steps else None

    def _try_parse_lines(self, response: str) -> list[PlanStep] | None:
        """Fallback: parse numbered lines as description-only steps."""
        steps = []
        for line in response.strip().splitlines():
            line = line.strip()
            # Match "1. something" or "1) something" or "- something"
            m = re.match(r'^(?:\d+[.)]\s*|-\s*)(.*)', line)
            if m and m.group(1).strip():
                steps.append(PlanStep(description=m.group(1).strip()))
        return steps if steps else None

    def _format_tools(self) -> str:
        """Format available tools into a readable string for the prompt."""
        if not self._available_tools:
            return ""
        lines = []
        for tool in self._available_tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {}).get("properties", {})
            param_strs = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                param_strs.append(f"{pname}: {ptype}")
            sig = f"{name}({', '.join(param_strs)})"
            line = f"- {sig}"
            if desc:
                line += f" — {desc}"
            lines.append(line)
        return "\n".join(lines)
