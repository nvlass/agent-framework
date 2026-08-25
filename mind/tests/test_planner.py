"""Tests for Planner implementations.

Run with:
    cd mind/
    python -m pytest tests/test_planner.py -v
"""

import json

from agent_mind.goals.model import Goal
from agent_mind.planning.model import StepStatus
from agent_mind.planning.planner import LLMPlanner, SimplePlanner


class TestSimplePlanner:

    def test_create_plan(self):
        planner = SimplePlanner()
        goal = Goal(description="fix the bug")
        plan = planner.create_plan(goal)
        assert plan.goal_id == goal.id
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "fix the bug"

    def test_create_plan_step_is_pending(self):
        planner = SimplePlanner()
        goal = Goal(description="test")
        plan = planner.create_plan(goal)
        from agent_mind.planning.model import StepStatus
        assert plan.steps[0].status == StepStatus.PENDING

    def test_revise_plan(self):
        planner = SimplePlanner()
        goal = Goal(description="fix the bug")
        plan = planner.create_plan(goal)
        revised = planner.revise_plan(plan, reason="step failed")
        assert revised.goal_id == goal.id
        assert revised.revised_at is not None
        assert revised.revision_reason == "step failed"
        assert "revised" in revised.steps[0].description

    def test_revise_creates_new_plan(self):
        planner = SimplePlanner()
        goal = Goal(description="test")
        plan = planner.create_plan(goal)
        revised = planner.revise_plan(plan, reason="retry")
        # Original plan unchanged
        assert plan.revised_at is None
        assert revised.revised_at is not None


class TestLLMPlanner:

    def _make_planner(self, response, available_tools=None):
        """Helper: create LLMPlanner with a fixed response."""
        return LLMPlanner(
            llm_fn=lambda prompt: response,
            available_tools=available_tools,
        )

    def test_json_parsing(self):
        """Valid JSON array → plan with correct steps, tool_name, tool_args."""
        resp = json.dumps([
            {"description": "Fetch news", "tool_name": "fetch", "tool_args": {"url": "https://hn.com"}},
            {"description": "Send email", "tool_name": "send_email", "tool_args": {"to": "me"}},
        ])
        planner = self._make_planner(resp)
        goal = Goal(description="news digest")
        plan = planner.create_plan(goal)
        assert len(plan.steps) == 2
        assert plan.steps[0].tool_name == "fetch"
        assert plan.steps[0].tool_args == {"url": "https://hn.com"}
        assert plan.steps[1].tool_name == "send_email"
        assert plan.steps[1].description == "Send email"

    def test_json_with_fences(self):
        """JSON wrapped in ```json fences is parsed correctly."""
        resp = '```json\n[{"description": "step one"}, {"description": "step two"}]\n```'
        planner = self._make_planner(resp)
        plan = planner.create_plan(Goal(description="test"))
        assert len(plan.steps) == 2

    def test_linear_dependencies(self):
        """Steps get linear dependencies by default."""
        resp = json.dumps([
            {"description": "A"},
            {"description": "B"},
            {"description": "C"},
        ])
        planner = self._make_planner(resp)
        plan = planner.create_plan(Goal(description="test"))
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == [plan.steps[0].id]
        assert plan.steps[2].depends_on == [plan.steps[1].id]

    def test_fallback_line_parsing(self):
        """Numbered text lines → description-only steps."""
        resp = "1. Fetch the data\n2. Process it\n3. Send results"
        planner = self._make_planner(resp)
        plan = planner.create_plan(Goal(description="test"))
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Fetch the data"
        assert plan.steps[0].tool_name is None

    def test_fallback_dash_lines(self):
        """Dash-prefixed lines also parsed."""
        resp = "- First thing\n- Second thing"
        planner = self._make_planner(resp)
        plan = planner.create_plan(Goal(description="test"))
        assert len(plan.steps) == 2

    def test_malformed_response_fallback(self):
        """Garbage response → single fallback step."""
        planner = self._make_planner("I don't know what to do honestly")
        plan = planner.create_plan(Goal(description="test"))
        assert len(plan.steps) == 1
        assert "I don't know" in plan.steps[0].description

    def test_empty_json_array(self):
        """Empty JSON array → single fallback step."""
        planner = self._make_planner("[]")
        plan = planner.create_plan(Goal(description="test"))
        assert len(plan.steps) == 1

    def test_empty_response(self):
        """Empty string → single fallback step."""
        planner = self._make_planner("")
        plan = planner.create_plan(Goal(description="test"))
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "Execute goal"

    def test_available_tools_in_prompt(self):
        """Tool descriptions appear in the create prompt."""
        prompts = []
        tools = [{"name": "fetch", "description": "Fetch URL", "parameters": {"properties": {"url": {"type": "string"}}}}]
        planner = LLMPlanner(
            llm_fn=lambda p: (prompts.append(p), '[]')[1],
            available_tools=tools,
        )
        planner.create_plan(Goal(description="test"))
        assert "fetch(url: string)" in prompts[0]
        assert "Fetch URL" in prompts[0]

    def test_revise_includes_failure_context(self):
        """Revise prompt contains failure reason and step outcomes."""
        prompts = []
        planner = LLMPlanner(
            llm_fn=lambda p: (prompts.append(p), '[{"description": "retry"}]')[1],
        )
        # Build a plan with step outcomes
        from agent_mind.planning.model import Plan, PlanStep
        step1 = PlanStep(description="Fetch news", status=StepStatus.COMPLETED)
        step2 = PlanStep(description="Send email", status=StepStatus.FAILED, error="sendmail not found")
        plan = Plan(goal_id="g1", steps=[step1, step2])

        revised = planner.revise_plan(plan, reason="sendmail not found")
        assert "COMPLETED" in prompts[0]
        assert "FAILED" in prompts[0]
        assert "sendmail not found" in prompts[0]
        assert revised.revised_at is not None
        assert revised.revision_reason == "sendmail not found"

    def test_explicit_depends_on(self):
        """Steps with explicit depends_on override linear default."""
        resp = json.dumps([
            {"description": "A"},
            {"description": "B", "depends_on": []},
            {"description": "C"},
        ])
        planner = self._make_planner(resp)
        plan = planner.create_plan(Goal(description="test"))
        # B has explicit empty deps, so no linear dep added
        assert plan.steps[1].depends_on == []
        # C gets linear dep on B
        assert plan.steps[2].depends_on == [plan.steps[1].id]

    def test_steps_are_pending(self):
        """All parsed steps start as PENDING."""
        resp = json.dumps([{"description": "A"}, {"description": "B"}])
        planner = self._make_planner(resp)
        plan = planner.create_plan(Goal(description="test"))
        for step in plan.steps:
            assert step.status == StepStatus.PENDING

    def test_create_prompt_includes_goal(self):
        """The create prompt contains the goal description."""
        prompts = []
        planner = LLMPlanner(llm_fn=lambda p: (prompts.append(p), '[]')[1])
        planner.create_plan(Goal(description="build a rocket"))
        assert "build a rocket" in prompts[0]

    def test_soul_included_in_create_prompt(self):
        """When soul is provided, it appears at the top of the create prompt."""
        prompts = []
        planner = LLMPlanner(
            llm_fn=lambda p: (prompts.append(p), '[]')[1],
            soul="You are a cautious planner.",
        )
        planner.create_plan(Goal(description="do something"))
        assert "You are a cautious planner." in prompts[0]
        assert prompts[0].index("You are a cautious planner.") < prompts[0].index("do something")

    def test_soul_included_in_revise_prompt(self):
        """Soul also appears in the revise prompt."""
        from agent_mind.planning.model import Plan, PlanStep
        prompts = []
        planner = LLMPlanner(
            llm_fn=lambda p: (prompts.append(p), '[]')[1],
            soul="You are a cautious planner.",
        )
        plan = Plan(goal_id="g", steps=[PlanStep(description="step 1")])
        planner.revise_plan(plan, "it failed")
        assert "You are a cautious planner." in prompts[0]

    def test_no_soul_prompt_starts_with_instruction(self):
        """Without soul, prompt starts with the instruction as before."""
        prompts = []
        planner = LLMPlanner(llm_fn=lambda p: (prompts.append(p), '[]')[1])
        planner.create_plan(Goal(description="test"))
        assert prompts[0].startswith("Break down")
