"""
agent-tools: Unified abstraction for agent tool use.

This package provides:
- Tool definition and registration
- Execution with permissions and error handling
- Built-in tools (read_file, list_directory, get_env, syntax_check)
- Template-based command tool factory (create_command_tool)
"""

__version__ = "0.1.0"

from .core.definition import ToolDefinition, ToolParameter, PermissionLevel
from .core.registry import ToolRegistry
from .core.executor import ToolExecutor, ToolResult
from .tools.shell import create_command_tool
from .tools.system import read_file, get_env, list_directory
from .tools.code import syntax_check

__all__ = [
    # Core
    "ToolDefinition",
    "ToolParameter",
    "PermissionLevel",
    "ToolRegistry",
    "ToolExecutor",
    "ToolResult",
    # Built-in tools
    "read_file",
    "get_env",
    "list_directory",
    "syntax_check",
    # Factories
    "create_command_tool",
]
