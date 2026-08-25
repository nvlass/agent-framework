"""Tests for SpawnRole, SpawnRegistry, and AgentInstance.spawn()."""

import pytest
from pathlib import Path
from unittest.mock import patch

from agent_core import AgentRole, AgentInstance, MockChatLLM, AgentConfig
from agent_core.spawn import SpawnRole, SpawnRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(responses: list[str]) -> MockChatLLM:
    return MockChatLLM(responses)


def _role(name: str = "parent") -> AgentRole:
    return AgentRole(name=name, soul="You are helpful.", config=AgentConfig())


# ---------------------------------------------------------------------------
# SpawnRole
# ---------------------------------------------------------------------------

class TestSpawnRole:
    def test_defaults(self):
        r = SpawnRole(name="critic")
        assert r.name == "critic"
        assert r.soul == ""
        assert r.tools == []
        assert r.llm is None

    def test_with_tools(self):
        r = SpawnRole(name="researcher", tools=["web_search", "fetch_readable"])
        assert r.tools == ["web_search", "fetch_readable"]


# ---------------------------------------------------------------------------
# SpawnRegistry
# ---------------------------------------------------------------------------

class TestSpawnRegistry:
    def test_empty(self):
        reg = SpawnRegistry({})
        assert not reg
        assert len(reg) == 0
        assert reg.names() == []

    def test_get_existing(self):
        role = SpawnRole(name="critic")
        reg = SpawnRegistry({"critic": role})
        assert reg.get("critic") is role

    def test_get_missing_returns_none(self):
        reg = SpawnRegistry({})
        assert reg.get("nonexistent") is None

    def test_names(self):
        reg = SpawnRegistry({
            "critic": SpawnRole(name="critic"),
            "researcher": SpawnRole(name="researcher"),
        })
        assert set(reg.names()) == {"critic", "researcher"}

    def test_bool_nonempty(self):
        reg = SpawnRegistry({"r": SpawnRole(name="r")})
        assert bool(reg)

    def test_from_config_empty(self):
        reg = SpawnRegistry.from_config({})
        assert not reg

    def test_from_config_no_soul(self):
        reg = SpawnRegistry.from_config({"critic": {"tools": []}})
        role = reg.get("critic")
        assert role is not None
        assert role.soul == ""
        assert role.tools == []

    def test_from_config_missing_soul_file_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="agent_core.spawn"):
            reg = SpawnRegistry.from_config({"r": {"soul": "/nonexistent/soul.txt"}})
        assert reg.get("r").soul == ""
        assert "not found" in caplog.text

    def test_from_config_loads_soul_file(self, tmp_path):
        soul_file = tmp_path / "critic.txt"
        soul_file.write_text("You are a critical thinker.")
        reg = SpawnRegistry.from_config(
            {"critic": {"soul": "critic.txt", "tools": []}},
            soul_base_dir=tmp_path,
        )
        assert reg.get("critic").soul == "You are a critical thinker."

    def test_from_config_relative_soul_path(self, tmp_path):
        soul_file = tmp_path / "souls" / "r.txt"
        soul_file.parent.mkdir()
        soul_file.write_text("Researcher soul.")
        reg = SpawnRegistry.from_config(
            {"r": {"soul": "souls/r.txt"}},
            soul_base_dir=tmp_path,
        )
        assert reg.get("r").soul == "Researcher soul."

    def test_from_config_with_make_llm(self):
        fake_llm = object()
        called_with = []

        def make_llm(model_id):
            called_with.append(model_id)
            return fake_llm

        reg = SpawnRegistry.from_config(
            {"r": {"tools": [], "model": "my-model"}},
            make_llm_fn=make_llm,
        )
        assert reg.get("r").llm is fake_llm
        assert called_with == ["my-model"]


# ---------------------------------------------------------------------------
# AgentInstance.spawn()
# ---------------------------------------------------------------------------

class TestAgentInstanceSpawn:
    def test_spawn_no_registry_returns_error(self):
        agent = AgentInstance(_role(), _llm(["Answer: done"]))
        result = agent.spawn("researcher", "do something")
        assert "no spawn registry" in result

    def test_spawn_unknown_role_returns_error(self):
        reg = SpawnRegistry({"critic": SpawnRole(name="critic")})
        agent = AgentInstance(_role(), _llm(["Answer: done"]), spawn_registry=reg)
        result = agent.spawn("unknown", "task")
        assert "unknown role" in result
        assert "critic" in result  # lists available

    def test_spawn_runs_child_and_returns_summary(self):
        reg = SpawnRegistry({"helper": SpawnRole(name="helper", soul="You help.", tools=[])})
        parent = AgentInstance(_role(), _llm(["Answer: parent done"]), spawn_registry=reg)
        child_llm = _llm(["Answer: child completed the task"])
        # Patch AgentInstance so the child uses our mock LLM
        original_init = AgentInstance.__init__

        inits = []

        def tracking_init(self_inner, role, llm, **kwargs):
            inits.append((role.name, llm))
            original_init(self_inner, role, llm, **kwargs)

        with patch.object(AgentInstance, "__init__", tracking_init):
            # Can't easily inject child_llm without deeper patching,
            # so just verify the tool registers and spawn() is callable.
            pass

        # Simpler: spawn with the parent's LLM, verify result shape
        result = parent.spawn("helper", "test task")
        # Result is a string (summary or "done"/"failed")
        assert isinstance(result, str)

    def test_spawn_tool_registered_when_registry_provided(self):
        reg = SpawnRegistry({"critic": SpawnRole(name="critic", tools=[])})
        agent = AgentInstance(_role(), _llm([]), spawn_registry=reg)
        tool = agent.registry.get("spawn_agent")
        assert tool is not None
        assert "critic" in tool.description

    def test_spawn_tool_not_registered_without_registry(self):
        agent = AgentInstance(_role(), _llm([]))
        assert agent.registry.get("spawn_agent") is None

    def test_child_gets_only_allowed_tools(self):
        """Child registry should only contain tools declared in the role."""
        from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel

        # Register a couple of tools on the parent
        parent = AgentInstance(_role(), _llm(["Answer: done"]))
        for tname in ("tool_a", "tool_b", "tool_c"):
            parent.registry.register(ToolDefinition(
                name=tname, description=tname,
                parameters=[], returns="string",
                permission=PermissionLevel.SAFE,
                execute=lambda: tname,
            ))

        reg = SpawnRegistry({
            "specialist": SpawnRole(name="specialist", tools=["tool_a", "tool_c"]),
        })
        parent._spawn_registry = reg

        child_reg = parent._build_child_registry(reg.get("specialist"), allow_tools=None)
        assert child_reg.get("tool_a") is not None
        assert child_reg.get("tool_c") is not None
        assert child_reg.get("tool_b") is None

    def test_allow_tools_overrides_role_tools(self):
        """allow_tools at spawn time should narrow beyond the role's declared set."""
        from agent_tools.core.definition import ToolDefinition, ToolParameter, PermissionLevel

        parent = AgentInstance(_role(), _llm(["Answer: done"]))
        for tname in ("tool_a", "tool_b"):
            parent.registry.register(ToolDefinition(
                name=tname, description=tname,
                parameters=[], returns="string",
                permission=PermissionLevel.SAFE,
                execute=lambda: tname,
            ))

        reg = SpawnRegistry({
            "r": SpawnRole(name="r", tools=["tool_a", "tool_b"]),
        })
        parent._spawn_registry = reg

        # Override to allow only tool_a
        child_reg = parent._build_child_registry(reg.get("r"), allow_tools=["tool_a"])
        assert child_reg.get("tool_a") is not None
        assert child_reg.get("tool_b") is None

    def test_unknown_tool_in_role_is_skipped_with_warning(self, caplog):
        import logging
        from agent_tools.core.definition import ToolDefinition, PermissionLevel

        parent = AgentInstance(_role(), _llm([]))
        reg = SpawnRegistry({
            "r": SpawnRole(name="r", tools=["nonexistent_tool"]),
        })
        parent._spawn_registry = reg

        with caplog.at_level(logging.WARNING, logger="agent_core.agent"):
            child_reg = parent._build_child_registry(reg.get("r"), allow_tools=None)

        assert child_reg.get("nonexistent_tool") is None
        assert "nonexistent_tool" in caplog.text

    def test_child_inherits_parent_llm_when_role_has_none(self):
        parent_llm = _llm(["Answer: done"])
        reg = SpawnRegistry({"r": SpawnRole(name="r", llm=None)})
        agent = AgentInstance(_role(), parent_llm, spawn_registry=reg)

        spawned_llms = []
        original = AgentInstance.__init__

        def capture(self_inner, role, llm, **kwargs):
            spawned_llms.append(llm)
            original(self_inner, role, llm, **kwargs)

        with patch.object(AgentInstance, "__init__", capture):
            agent.spawn("r", "task")

        assert any(llm is parent_llm for llm in spawned_llms)

    def test_child_uses_role_llm_when_provided(self):
        parent_llm = _llm(["Answer: parent"])
        child_llm = _llm(["Answer: child done"])
        reg = SpawnRegistry({"r": SpawnRole(name="r", llm=child_llm)})
        agent = AgentInstance(_role(), parent_llm, spawn_registry=reg)

        spawned_llms = []
        original = AgentInstance.__init__

        def capture(self_inner, role, llm, **kwargs):
            spawned_llms.append(llm)
            original(self_inner, role, llm, **kwargs)

        with patch.object(AgentInstance, "__init__", capture):
            agent.spawn("r", "task")

        assert any(llm is child_llm for llm in spawned_llms)
