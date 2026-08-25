"""
Tool registry — where tools live.

The registry is the central place to register, discover, and list tools.
It enforces naming uniqueness and provides filtering by permission level.
"""

from typing import Optional

from .definition import ToolDefinition, PermissionLevel


class ToolRegistry:
    """Registry for discovering and managing tools.

    Usage:
        registry = ToolRegistry()
        registry.register(my_tool)
        tool = registry.get("my_tool")
        all_tools = registry.list_tools()
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool.

        Raises:
            ValueError: If a tool with the same name is already registered.
            ValueError: If the tool has no execute callable.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if tool.execute is None and tool.execute_async is None:
            raise ValueError(f"Tool must have execute or execute_async: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        del self._tools[name]

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(
        self,
        permission: Optional[PermissionLevel] = None,
    ) -> list[ToolDefinition]:
        """List all registered tools, optionally filtered by permission level."""
        tools = list(self._tools.values())
        if permission is not None:
            tools = [t for t in tools if t.permission == permission]
        return tools

    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def to_schemas(self) -> list[dict]:
        """Export all tools as JSON schemas for LLM function calling."""
        return [t.to_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def register_defaults(self) -> None:
        """Register all built-in safe/read tools.

        This adds: read_file, get_env, list_directory, syntax_check.
        DANGEROUS tools (shell, python_exec) are never included —
        they must be explicitly created and registered.
        """
        from ..tools.system import read_file, get_env, list_directory
        from ..tools.code import syntax_check

        for tool in (read_file, get_env, list_directory, syntax_check):
            self.register(tool)
