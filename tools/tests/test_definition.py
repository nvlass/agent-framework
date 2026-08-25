"""Tests for ToolDefinition and related types."""

import pytest

from agent_tools.core.definition import (
    ToolDefinition,
    ToolParameter,
    PermissionLevel,
)


class TestPermissionLevel:
    """Test permission level enum."""

    def test_values(self):
        assert PermissionLevel.SAFE.value == "safe"
        assert PermissionLevel.READ.value == "read"
        assert PermissionLevel.WRITE.value == "write"
        assert PermissionLevel.DANGEROUS.value == "dangerous"

    def test_ordering_by_severity(self):
        """Permissions have increasing severity."""
        levels = [PermissionLevel.SAFE, PermissionLevel.READ,
                  PermissionLevel.WRITE, PermissionLevel.DANGEROUS]
        assert len(levels) == 4


class TestToolParameter:
    """Test ToolParameter dataclass."""

    def test_basic_parameter(self):
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )
        assert param.name == "query"
        assert param.type == "string"
        assert param.required is True
        assert param.default is None
        assert param.enum is None

    def test_optional_parameter_with_default(self):
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_enum_parameter(self):
        param = ToolParameter(
            name="status",
            type="string",
            description="Task status",
            enum=["pending", "running", "completed"],
        )
        assert param.enum == ["pending", "running", "completed"]

    def test_to_schema(self):
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )
        schema = param.to_schema()
        assert schema == {
            "type": "string",
            "description": "Search query",
        }

    def test_to_schema_with_enum(self):
        param = ToolParameter(
            name="status",
            type="string",
            description="Status filter",
            enum=["pending", "done"],
        )
        schema = param.to_schema()
        assert schema["enum"] == ["pending", "done"]

    def test_to_schema_with_default(self):
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        schema = param.to_schema()
        assert schema["default"] == 10


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def _make_tool(self, **overrides):
        """Helper to create a tool with defaults."""
        defaults = dict(
            name="test_tool",
            description="A test tool",
            execute=lambda: "ok",
        )
        defaults.update(overrides)
        return ToolDefinition(**defaults)

    def test_basic_tool(self):
        tool = self._make_tool()
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.permission == PermissionLevel.SAFE
        assert tool.async_capable is False
        assert tool.timeout_seconds == 30

    def test_tool_with_parameters(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="path", type="string", description="File path"),
                ToolParameter(name="encoding", type="string", description="Encoding",
                              required=False, default="utf-8"),
            ],
        )
        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "path"
        assert tool.parameters[1].required is False

    def test_to_schema(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )
        schema = tool.to_schema()
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert "query" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["query"]

    def test_to_schema_no_required(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="limit", type="integer", description="Limit",
                              required=False, default=10),
            ],
        )
        schema = tool.to_schema()
        assert "required" not in schema["parameters"]

    def test_validate_args_valid(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )
        errors = tool.validate_args(query="hello")
        assert errors == []

    def test_validate_args_missing_required(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )
        errors = tool.validate_args()
        assert len(errors) == 1
        assert "Missing required" in errors[0]

    def test_validate_args_unknown_parameter(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )
        errors = tool.validate_args(query="hello", bogus="value")
        assert len(errors) == 1
        assert "Unknown parameter" in errors[0]

    def test_validate_args_enum_violation(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="status", type="string", description="Status",
                              enum=["pending", "done"]),
            ],
        )
        errors = tool.validate_args(status="invalid")
        assert len(errors) == 1
        assert "Invalid value" in errors[0]

    def test_validate_args_enum_valid(self):
        tool = self._make_tool(
            parameters=[
                ToolParameter(name="status", type="string", description="Status",
                              enum=["pending", "done"]),
            ],
        )
        errors = tool.validate_args(status="pending")
        assert errors == []
