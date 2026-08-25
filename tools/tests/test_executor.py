"""Tests for ToolExecutor.

STATUS: Tests are written. Your job is to make them pass by implementing
ToolExecutor.execute() in agent_tools/core/executor.py.

Run with:
    cd tools/
    python -m pytest tests/test_executor.py -v

Start with test_execute_simple and work your way down.
The tests are ordered by complexity — each one tests a new aspect.
"""

import pytest

from agent_tools.core.definition import (
    ToolDefinition,
    ToolParameter,
    PermissionLevel,
)
from agent_tools.core.registry import ToolRegistry
from agent_tools.core.executor import ToolExecutor, ToolResult

def _make_registry_with_tool(
    name: str = "greet",
    permission: PermissionLevel = PermissionLevel.SAFE,
    execute=None,
    parameters=None,
):
    """Helper to create a registry with a single tool."""
    if execute is None:
        execute=lambda name="World": f"Hello, {name}!"

    tool = ToolDefinition(
        name=name,
        description=f"Tool: {name}",
        permission=permission,
        # So, there is a small bug here --
        # the first test has `[]` params, and `or` treats this as false
        parameters=parameters if parameters is not None else [
            ToolParameter(name="name", type="string", description="Name to greet",
                          required=False, default="World"),
        ],
        execute=execute,
    )
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _allow_all(tool: ToolDefinition) -> bool:
    return True


def _deny_dangerous(tool: ToolDefinition) -> bool:
    return tool.permission != PermissionLevel.DANGEROUS


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_success_result(self):
        result = ToolResult(success=True, output="hello", tool_name="greet")
        assert result.success is True
        assert result.output == "hello"
        assert result.error is None

    def test_failure_result(self):
        result = ToolResult(success=False, error="boom", tool_name="greet")
        assert result.success is False
        assert result.output is None
        assert result.error == "boom"


class TestToolExecutor:
    """Test ToolExecutor.

    Implement ToolExecutor.execute() to make these pass, one at a time.
    """

    # --- Start here ---

    def test_execute_simple(self):
        """Most basic test: execute a tool that takes no args."""
        registry = _make_registry_with_tool(
            execute=lambda: "hello",
            parameters=[],
        )
        executor = ToolExecutor(registry, _allow_all)
        result = executor.execute("greet")
        assert result.success is True
        assert result.output == "hello"
        assert result.tool_name == "greet"

    def test_execute_with_kwargs(self):
        """Execute a tool with keyword arguments."""
        registry = _make_registry_with_tool()
        executor = ToolExecutor(registry, _allow_all)
        result = executor.execute("greet", name="Nikos")
        assert result.success is True
        assert result.output == "Hello, Nikos!"

    def test_execute_with_default(self):
        """Missing optional param should use default value."""
        registry = _make_registry_with_tool()
        executor = ToolExecutor(registry, _allow_all)
        result = executor.execute("greet")
        assert result.success is True
        assert result.output == "Hello, World!"

    # --- Error handling ---

    def test_execute_unknown_tool(self):
        """Executing an unregistered tool should fail gracefully."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry, _allow_all)
        result = executor.execute("nonexistent")
        assert result.success is False
        assert "Unknown tool: nonexistent" in result.error

    def test_execute_permission_denied(self):
        """Permission checker returning False should block execution."""
        registry = _make_registry_with_tool(
            name="danger",
            permission=PermissionLevel.DANGEROUS,
        )
        executor = ToolExecutor(registry, _deny_dangerous)
        result = executor.execute("danger")
        assert result.success is False
        assert "Permission denied" in result.error

    def test_execute_validation_error(self):
        """Invalid arguments should fail before execution."""
        registry = _make_registry_with_tool(
            parameters=[
                ToolParameter(name="query", type="string", description="Query"),
            ],
            execute=lambda query: query,
        )
        executor = ToolExecutor(registry, _allow_all)
        # Missing required param 'query'
        result = executor.execute("greet")
        assert result.success is False
        assert "Missing required parameter: query" in result.error

    def test_execute_exception_in_tool(self):
        """Tool raising an exception should be caught and wrapped."""
        def bad_tool():
            raise RuntimeError("something broke")

        registry = _make_registry_with_tool(
            execute=bad_tool,
            parameters=[],
        )
        executor = ToolExecutor(registry, _allow_all)
        result = executor.execute("greet")
        assert result.success is False
        assert "something broke" in result.error

    # --- Timing ---

    def test_execute_records_duration(self):
        """Result should include execution duration."""
        import time

        def slow_tool():
            time.sleep(0.05)
            return "done"

        registry = _make_registry_with_tool(
            execute=slow_tool,
            parameters=[],
        )
        executor = ToolExecutor(registry, _allow_all)
        result = executor.execute("greet")
        assert result.success is True
        assert result.duration_ms >= 40  # at least ~50ms, with some tolerance
