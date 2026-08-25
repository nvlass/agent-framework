"""
Core abstractions for agent-tools.

This module provides the foundation:
- ToolDefinition: What a tool is (name, params, permissions)
- ToolRegistry: Where tools live (register, discover, list)
- ToolExecutor: How tools run (permissions, execution, error handling)
"""

from .definition import ToolDefinition, ToolParameter, PermissionLevel
from .registry import ToolRegistry
from .executor import ToolExecutor, ToolResult

__all__ = [
    "ToolDefinition",
    "ToolParameter",
    "PermissionLevel",
    "ToolRegistry",
    "ToolExecutor",
    "ToolResult",
]
