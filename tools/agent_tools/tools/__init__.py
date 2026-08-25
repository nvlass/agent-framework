"""
Built-in tools and tool factories.

- system: Read-only system tools (read_file, list_directory, get_env)
- code: Code tools (syntax_check)
- shell: Template-based command tool factory (DANGEROUS, explicit only)
"""

from .shell import create_command_tool
from .system import read_file, get_env, list_directory
from .code import syntax_check

__all__ = [
    "create_command_tool",
    "read_file",
    "get_env",
    "list_directory",
    "syntax_check",
]
