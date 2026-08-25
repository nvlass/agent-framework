"""Tests for ToolRegistry."""

import pytest

from agent_tools.core.definition import (
    ToolDefinition,
    ToolParameter,
    PermissionLevel,
)
from agent_tools.core.registry import ToolRegistry


def _make_tool(name: str = "test_tool", permission: PermissionLevel = PermissionLevel.SAFE):
    """Helper to create a minimal tool."""
    return ToolDefinition(
        name=name,
        description=f"Tool: {name}",
        permission=permission,
        execute=lambda: "ok",
    )


class TestToolRegistry:
    """Test ToolRegistry."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = _make_tool("my_tool")
        registry.register(tool)
        assert registry.get("my_tool") is tool

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(_make_tool("my_tool"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_tool("my_tool"))

    def test_register_no_execute_raises(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="broken", description="No execute")
        with pytest.raises(ValueError, match="must have execute"):
            registry.register(tool)

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(_make_tool("my_tool"))
        registry.unregister("my_tool")
        assert registry.get("my_tool") is None

    def test_unregister_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")

    def test_has(self):
        registry = ToolRegistry()
        registry.register(_make_tool("my_tool"))
        assert registry.has("my_tool") is True
        assert registry.has("other") is False

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        assert len(registry.list_tools()) == 2

    def test_list_tools_filtered(self):
        registry = ToolRegistry()
        registry.register(_make_tool("safe", PermissionLevel.SAFE))
        registry.register(_make_tool("read", PermissionLevel.READ))
        registry.register(_make_tool("dangerous", PermissionLevel.DANGEROUS))

        safe_tools = registry.list_tools(permission=PermissionLevel.SAFE)
        assert len(safe_tools) == 1
        assert safe_tools[0].name == "safe"

    def test_tool_names(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        assert sorted(registry.tool_names()) == ["a", "b"]

    def test_to_schemas(self):
        registry = ToolRegistry()
        registry.register(_make_tool("my_tool"))
        schemas = registry.to_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "my_tool"

    def test_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(_make_tool("a"))
        assert len(registry) == 1

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        assert "a" in registry
        assert "b" not in registry


class TestRegisterDefaults:
    """Test register_defaults()."""

    def test_registers_all_defaults(self):
        registry = ToolRegistry()
        registry.register_defaults()
        assert "read_file" in registry
        assert "get_env" in registry
        assert "list_directory" in registry
        assert "syntax_check" in registry

    def test_default_count(self):
        registry = ToolRegistry()
        registry.register_defaults()
        assert len(registry) == 4

    def test_no_dangerous_tools(self):
        registry = ToolRegistry()
        registry.register_defaults()
        dangerous = registry.list_tools(permission=PermissionLevel.DANGEROUS)
        assert len(dangerous) == 0

    def test_can_add_more_after_defaults(self):
        registry = ToolRegistry()
        registry.register_defaults()
        registry.register(_make_tool("custom"))
        assert len(registry) == 5
        assert "custom" in registry

    def test_duplicate_defaults_raises(self):
        """Calling register_defaults twice should fail (names already taken)."""
        registry = ToolRegistry()
        registry.register_defaults()
        with pytest.raises(ValueError, match="already registered"):
            registry.register_defaults()
