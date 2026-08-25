"""
Built-in system tools.

Safe, read-only tools that are auto-registered in every registry.
These have no side effects on the system — they only read state.
"""

import os
from pathlib import Path
from typing import Optional

from ..core.definition import ToolDefinition, ToolParameter, PermissionLevel


MAX_FILE_SIZE = 1024 * 1024  # 1MB


def _read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a file and return its contents as a string."""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    if p.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {p.stat().st_size} bytes (max {MAX_FILE_SIZE})"
        )

    return p.read_text(encoding=encoding)


def _get_env(name: str, default: str = "") -> str:
    """Read an environment variable."""
    return os.environ.get(name, default)


read_file = ToolDefinition(
    name="read_file",
    description="Read the contents of a file. Returns the file text. "
                "Refuses files larger than 1MB.",
    parameters=[
        ToolParameter(
            name="path", type="string",
            description="Path to the file to read",
        ),
        ToolParameter(
            name="encoding", type="string",
            description="Text encoding",
            required=False, default="utf-8",
        ),
    ],
    returns="string (file contents)",
    permission=PermissionLevel.READ,
    execute=_read_file,
)

get_env = ToolDefinition(
    name="get_env",
    description="Read the value of an environment variable.",
    parameters=[
        ToolParameter(
            name="name", type="string",
            description="Environment variable name",
        ),
        ToolParameter(
            name="default", type="string",
            description="Value to return if the variable is not set",
            required=False, default="",
        ),
    ],
    returns="string (variable value or default)",
    permission=PermissionLevel.READ,
    execute=_get_env,
)



# --- TODO (Nikos): list_directory ---
#
# Create a ToolDefinition named "list_directory" that lists the contents
# of a directory. Permission level: READ.
#
# Suggested signature:
#   _list_directory(path: str) -> str
#
# Decisions to make:
#   - Return format: one entry per line? JSON? Just names or with metadata?
#   - Hidden files: include by default or add a flag?
#   - What info per entry: name only, or name + type (file/dir)?
#     Hint: pathlib.Path.iterdir() gives you Path objects,
#           and p.is_dir() / p.is_file() can distinguish types.
#
# Skeleton:
#
#   def _list_directory(path: str) -> str:
#       p = Path(path)
#       if not p.exists():
#           raise FileNotFoundError(f"Directory not found: {path}")
#       if not p.is_dir():
#           raise ValueError(f"Not a directory: {path}")
#       ...
#
#   list_directory = ToolDefinition(
#       name="list_directory",
#       description="List the contents of a directory.",
#       parameters=[
#           ToolParameter(name="path", type="string",
#                         description="Path to the directory"),
#       ],
#       returns="string (directory listing)",
#       permission=PermissionLevel.READ,
#       execute=_list_directory,
#   )


def _list_directory(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Directory not found: {path}')
    if not p.is_dir():
        raise ValueError(f'Not a directory: {path}')
    return '\n'.join([d.name for d in sorted(p.iterdir())])

list_directory = ToolDefinition(
    name='list_directory',
    description='List the contents of a directory',
    parameters=[
        ToolParameter(name='path', type='string', description='Path to the directory'),
    ],
    returns='string (directory listing)',
    permission=PermissionLevel.READ,
    execute=_list_directory,
)
