"""Tests for AgentRole and AgentInstance."""

import logging
import os
import tempfile
from pathlib import Path

from agent_core.agent import AgentRole, AgentInstance, _default_permission_checker, _configure_logging
from agent_core.config import AgentConfig, ReactConfig
from agent_patterns.base import PatternResult
from agent_core.llm import MockChatLLM
from agent_tools import ToolRegistry
from agent_core.prompt import PromptAssembler
from agent_tools.core.definition import ToolDefinition, PermissionLevel
from agent_tools.core.registry import ToolRegistry


def _make_llm_answer(answer: str) -> MockChatLLM:
    """Create a MockChatLLM that immediately answers."""
    return MockChatLLM([f"Thought: done\nAnswer: {answer}"])


def _make_llm_tool_then_answer(tool: str, args: str, obs: str, answer: str) -> MockChatLLM:
    """LLM that calls a tool, then answers."""
    return MockChatLLM([
        f"Thought: need tool\nAction: {tool}\nAction Args: {args}",
        f"Thought: got it\nAnswer: {answer}",
    ])


class TestAgentRole:
    def test_basic_creation(self):
        role = AgentRole(name="test-agent")
        assert role.name == "test-agent"
        assert role.soul == ""

    def test_with_soul(self):
        role = AgentRole(name="x", soul="be helpful")
        assert role.soul == "be helpful"

    def test_from_soul_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("I am a soul")
            f.flush()
            role = AgentRole.from_soul_file("agent", f.name)
        assert role.soul == "I am a soul"
        assert role.name == "agent"

    def test_default_config(self):
        role = AgentRole(name="x")
        assert role.config.pattern == "react"


class TestPermissionChecker:
    def test_allows_safe(self):
        tool = ToolDefinition(name="t", description="t", permission=PermissionLevel.SAFE, execute=lambda: None)
        assert _default_permission_checker(tool) is True

    def test_allows_read(self):
        tool = ToolDefinition(name="t", description="t", permission=PermissionLevel.READ, execute=lambda: None)
        assert _default_permission_checker(tool) is True

    def test_allows_write(self):
        tool = ToolDefinition(name="t", description="t", permission=PermissionLevel.WRITE, execute=lambda: None)
        assert _default_permission_checker(tool) is True

    def test_blocks_dangerous(self):
        tool = ToolDefinition(name="t", description="t", permission=PermissionLevel.DANGEROUS, execute=lambda: None)
        assert _default_permission_checker(tool) is False


class TestAgentInstance:
    def test_immediate_answer(self):
        llm = _make_llm_answer("hello world")
        role = AgentRole(name="test")
        agent = AgentInstance(role, llm)
        result = agent.run("say hello")
        assert result.success
        assert "hello world" in result.summary or result.metadata.get("answer") == "hello world"

    def test_tool_call_then_answer(self):
        """Agent calls a tool, gets result, then answers."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="greet",
            description="Greet someone",
            permission=PermissionLevel.SAFE,
            execute=lambda: "hi there",
        ))
        llm = _make_llm_tool_then_answer("greet", "{}", "hi there", "done")
        role = AgentRole(name="test")
        agent = AgentInstance(role, llm, registry=registry)
        result = agent.run("greet someone")
        assert result.success
        assert len(llm.calls) == 2

    def test_max_iterations_respected(self):
        """Agent stops after max iterations even without answer."""
        config = AgentConfig(react=ReactConfig(max_iterations=2))
        role = AgentRole(name="test", config=config)
        # LLM always tries to use a tool that doesn't exist
        llm = MockChatLLM(["Thought: try\nAction: nonexistent\nAction Args: {}"])
        agent = AgentInstance(role, llm)
        result = agent.run("do something")
        assert not result.success
        assert result.iterations <= 2

    def test_invalid_config_returns_failure(self):
        config = AgentConfig(pattern="unknown")
        role = AgentRole(name="test", config=config)
        llm = MockChatLLM(["x"])
        agent = AgentInstance(role, llm)
        result = agent.run("x")
        assert not result.success
        assert "config" in result.summary.lower()

    def test_default_registry_has_tools(self):
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm)
        assert len(agent.registry) > 0

    def test_custom_registry(self):
        registry = ToolRegistry()
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm, registry=registry)
        assert len(agent.registry) == 0

    def test_custom_assembler(self):
        asm = PromptAssembler(
            system_template="CUSTOM{soul_section}{tools_section}",
            user_template="{goal}{observations_section}",
        )
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        role = AgentRole(name="test")
        # Empty registry so no tools → text parsing path → custom template used
        agent = AgentInstance(role, llm, assembler=asm, registry=ToolRegistry())
        agent.run("test")
        assert llm.calls[0][0].content.startswith("CUSTOM")

    def test_soul_from_role(self):
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        role = AgentRole(name="test", soul="be wise")
        agent = AgentInstance(role, llm)
        agent.run("test")
        system_msg = llm.calls[0][0]
        assert "be wise" in system_msg.content

    def test_two_instances_independent(self):
        llm1 = MockChatLLM(["Thought: x\nAnswer: a1"])
        llm2 = MockChatLLM(["Thought: x\nAnswer: a2"])
        role = AgentRole(name="shared-role")
        agent1 = AgentInstance(role, llm1)
        agent2 = AgentInstance(role, llm2)
        r1 = agent1.run("task1")
        r2 = agent2.run("task2")
        assert r1.success and r2.success
        assert len(llm1.calls) == 1
        assert len(llm2.calls) == 1

    def test_role_property(self):
        role = AgentRole(name="test")
        llm = MockChatLLM(["x"])
        agent = AgentInstance(role, llm)
        assert agent.role.name == "test"

    def test_custom_permission_checker_allows_dangerous(self):
        """Custom permission_checker can allow DANGEROUS tools."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="risky",
            description="A dangerous tool",
            permission=PermissionLevel.DANGEROUS,
            execute=lambda: "boom",
        ))
        llm = _make_llm_tool_then_answer("risky", "{}", "boom", "done")
        role = AgentRole(name="test")
        agent = AgentInstance(role, llm, registry=registry, permission_checker=lambda t: True)
        result = agent.run("do risky thing")
        assert result.success
        assert len(llm.calls) == 2

    def test_default_permission_checker_blocks_dangerous(self):
        """Default permission_checker blocks DANGEROUS tools."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="risky",
            description="A dangerous tool",
            permission=PermissionLevel.DANGEROUS,
            execute=lambda: "boom",
        ))
        llm = MockChatLLM([
            "Thought: try\nAction: risky\nAction Args: {}",
            "Thought: ok\nAnswer: blocked",
        ])
        role = AgentRole(name="test")
        agent = AgentInstance(role, llm, registry=registry)
        result = agent.run("do risky thing")
        # Should still complete (agent gets permission denied, then answers)
        assert result.success


class TestLogging:
    def test_default_level_is_warning(self):
        _configure_logging("WARNING")
        assert logging.getLogger("agent_core").level == logging.WARNING

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv("AGENT_LOG_LEVEL", "DEBUG")
        _configure_logging("WARNING")
        assert logging.getLogger("agent_core").level == logging.DEBUG

    def test_all_packages_configured(self):
        _configure_logging("INFO")
        for pkg in ("agent_core", "agent_patterns", "agent_tools"):
            assert logging.getLogger(pkg).level == logging.INFO


class _StubToolResult:
    """Minimal stand-in for agent_memory ToolResult."""
    def __init__(self, success=True, data=None, error=None):
        self.success = success
        self.data = data
        self.error = error


class _StubMemoryTools:
    """Stub MemoryTools for testing without a real DB."""
    def __init__(self, recall_data=None):
        self._recall_data = recall_data
        self.stored = []

    def recall_similar(self, query, limit=5):
        return _StubToolResult(success=True, data=self._recall_data)

    def store_memory(self, context, action, outcome, tags=None):
        self.stored.append({"context": context, "action": action,
                            "outcome": outcome, "tags": tags})
        return _StubToolResult(success=True, data={"id": len(self.stored)})

    def reflect_on_recent(self, hours=24, focus=None):
        return _StubToolResult(success=True, data="no insights")


class TestMemoryIntegration:
    def test_memory_tools_registered(self):
        """When memory is provided, memory tools appear in the registry."""
        mem = _StubMemoryTools()
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm, memory=mem)
        assert agent.registry.get("memory_store") is not None
        assert agent.registry.get("memory_recall") is not None
        assert agent.registry.get("memory_reflect") is not None

    def test_auto_recall_injects_observations(self):
        """Auto-recall injects past experiences into context observations."""
        mem = _StubMemoryTools(recall_data=[{"context": "past task", "outcome": "worked"}])
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: done"])
        agent = AgentInstance(role, llm, memory=mem)
        result = agent.run("new task")
        assert result.success
        # The LLM should have seen the recalled memories in the prompt
        user_msg = llm.calls[0][1]  # second message is user message
        assert "Relevant past experiences" in user_msg.content

    def test_auto_store_after_run(self):
        """After run completes, an episode is auto-stored."""
        mem = _StubMemoryTools()
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: done"])
        agent = AgentInstance(role, llm, memory=mem)
        agent.run("test task")
        assert len(mem.stored) == 1
        assert mem.stored[0]["context"] == "test task"
        assert mem.stored[0]["tags"] == ["auto"]

    def test_no_memory_backward_compat(self):
        """Without memory, agent works exactly as before."""
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm)
        result = agent.run("hello")
        assert result.success

    def test_recall_failure_does_not_crash(self):
        """If recall raises, agent continues without memories."""
        class _BrokenMemory(_StubMemoryTools):
            def recall_similar(self, query, limit=5):
                raise RuntimeError("db gone")
        mem = _BrokenMemory()
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm, memory=mem)
        result = agent.run("task")
        assert result.success
        # store should still have been attempted
        assert len(mem.stored) == 1

    def test_store_failure_does_not_crash(self):
        """If store raises, agent still returns result."""
        class _BrokenStore(_StubMemoryTools):
            def store_memory(self, **kwargs):
                raise RuntimeError("disk full")
        mem = _BrokenStore()
        role = AgentRole(name="test")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm, memory=mem)
        result = agent.run("task")
        assert result.success


class TestPlanAndExecute:
    def test_builds_and_runs(self):
        """plan_and_execute pattern builds and runs via LLMPlanner."""
        config = AgentConfig(pattern="plan_and_execute")
        role = AgentRole(name="test", config=config)
        # First response: LLMPlanner plan creation; second: step execution
        plan_json = '[{"description": "do the thing"}]'
        llm = MockChatLLM([plan_json, "Thought: done\nAnswer: planned"])
        agent = AgentInstance(role, llm)
        result = agent.run("do a complex task")
        assert isinstance(result, PatternResult)

    def test_custom_planner_used(self):
        """When a custom planner is injected, it's used instead of SimplePlanner."""
        from agent_mind.planning.planner import PlannerInterface, SimplePlanner
        from agent_mind.planning.model import Plan

        class _TrackingPlanner(PlannerInterface):
            def __init__(self):
                self.called = False
            def create_plan(self, goal_description, context=None):
                self.called = True
                return SimplePlanner().create_plan(goal_description, context)
            def revise_plan(self, plan, failure_reason, context=None):
                return SimplePlanner().revise_plan(plan, failure_reason, context)

        planner = _TrackingPlanner()
        config = AgentConfig(pattern="plan_and_execute")
        role = AgentRole(name="test", config=config)
        llm = MockChatLLM(["Thought: done\nAnswer: ok"])
        agent = AgentInstance(role, llm, planner=planner)
        agent.run("task")
        assert planner.called

    def test_default_llm_planner(self):
        """Without custom planner, LLMPlanner is used by default."""
        config = AgentConfig(pattern="plan_and_execute")
        role = AgentRole(name="test", config=config)
        plan_json = '[{"description": "do it"}]'
        llm = MockChatLLM([plan_json, "Thought: done\nAnswer: ok"])
        agent = AgentInstance(role, llm)
        result = agent.run("task")
        assert isinstance(result, PatternResult)
        # LLMPlanner consumed at least one LLM call for plan creation
        assert len(llm.calls) >= 1
        # First call is the planner prompt, not a ReAct prompt
        assert "Break down" in llm.calls[0][0].content

    def test_max_replans_from_config(self):
        """max_replans config value is passed through to PlanAndExecute."""
        config = AgentConfig(pattern="plan_and_execute", max_replans=5)
        role = AgentRole(name="test", config=config)
        plan_json = '[{"description": "do it"}]'
        llm = MockChatLLM([plan_json, "Thought: done\nAnswer: ok"])
        agent = AgentInstance(role, llm)
        pattern = agent._build_pattern(config)
        assert pattern._max_replans == 5
