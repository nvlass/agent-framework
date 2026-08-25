"""
Template-based command tool factory.

Creates command execution tools with fixed command structures.
The agent fills in parameter slots but cannot modify the command shape.
This prevents shell injection by design — no filtering, no parsing,
no bypasses possible.

WHY NOT STRING FILTERING?
Text-based command filtering (allowlists, prefix matching) is fundamentally
broken as a security model. The attack surface is infinite:
- Variable expansion: X=r F=m bash -c '$X$F -rf /'
- Command substitution: git fetch $(rm -rf /)
- Pipe injection: git status | bash
- Encoding tricks: echo "cm0gLXJm" | base64 -d | sh
- Heredocs, backticks, eval, xargs, find -exec...

You'd need a complete bash parser to catch everything. Instead, we use
fixed command templates with subprocess(shell=False), which eliminates
the entire class of injection attacks.

Example usage:
    # Fixed command, no parameters
    git_fetch = create_command_tool(
        name="git_fetch",
        command=["git", "fetch"],
    )

    # Command with a parameter slot
    git_checkout = create_command_tool(
        name="git_checkout",
        command=["git", "checkout", "{branch}"],
        parameters=[
            ToolParameter(name="branch", type="string", description="Branch name"),
        ],
    )
    # Agent calls: tool.execute(branch="main")
    # Runs: subprocess.run(["git", "checkout", "main"], shell=False)
"""

import subprocess
from typing import Optional

from ..core.definition import ToolDefinition, ToolParameter, PermissionLevel


def extract_param(param: str, kwargs: dict) -> str:
    if param.startswith('{') and param.endswith('}') and len(param) > 2:
        name = param[1:-1]
        if name.isidentifier():
            return str(kwargs[name])
    return param

def create_command_tool(
    name: str,
    command: list[str],
    parameters: Optional[list[ToolParameter]] = None,
    description: Optional[str] = None,
    timeout_seconds: int = 30,
    working_dir: Optional[str] = None,
) -> ToolDefinition:
    """Factory that creates a template-based command execution tool.

    The command is a fixed list of strings with optional {parameter} placeholders.
    At execution time, placeholders are replaced with validated parameter values.
    subprocess.run is called with shell=False — no shell interpretation occurs.

    Args:
        name: Tool name (e.g., "git_fetch", "run_pytest")
        command: Command as a list of strings. Use {param_name} for parameter slots.
            Example: ["git", "checkout", "{branch}"]
            Example: ["pytest", "{path}", "-v"]
            Example: ["git", "fetch"]  (no parameters)
        parameters: ToolParameter definitions for each {placeholder} in command.
            Must match the placeholders exactly.
        description: Optional description (auto-generated if not provided)
        timeout_seconds: Maximum execution time
        working_dir: Optional working directory for the command

    Returns:
        ToolDefinition with DANGEROUS permission level

    """

    def _execute_command(**kwargs) -> str:
        """Execute a template command with parameter substitution.
        """

        # check the final args for dangerous chars
        DANGEROUS_CHARS = set(";|&$`(){}<>\n")
        for value in kwargs.values():
            if any(c in str(value) for c in DANGEROUS_CHARS):
                raise ValueError(f"Parameter contains forbidden characters: {value}")

        final_args = []
        for v in command:
            final_args.append(extract_param(v, kwargs))

        result = subprocess.run(
            final_args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=working_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed (exit {result.returncode}): {result.stderr.strip()}",
            )
        return result.stdout

    if parameters is None:
        parameters = []

    if description is None:
        cmd_str = " ".join(command)
        description = f"Execute: {cmd_str}"

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        returns="string (command output)",
        permission=PermissionLevel.DANGEROUS,
        timeout_seconds=timeout_seconds,
        execute=_execute_command,
    )
