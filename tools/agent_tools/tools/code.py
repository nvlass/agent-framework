"""
Code analysis tools.

Safe tools for validating and inspecting code without executing it.
"""

import ast

from ..core.definition import ToolDefinition, ToolParameter, PermissionLevel


def _syntax_check(code: str, filename: str = "<input>") -> str:
    """Check Python code for syntax errors.

    Returns "OK" if valid, or a description of the syntax error.
    Does not execute the code.
    """
    try:
        ast.parse(code, filename=filename)
        return "OK"
    except SyntaxError as e:
        parts = [f"SyntaxError: {e.msg}"]
        if e.lineno is not None:
            parts.append(f"line {e.lineno}")
        if e.offset is not None:
            parts.append(f"col {e.offset}")
        return ", ".join(parts)


syntax_check = ToolDefinition(
    name="syntax_check",
    description="Check Python code for syntax errors without executing it. "
                "Returns 'OK' if valid, or a description of the error.",
    parameters=[
        ToolParameter(
            name="code", type="string",
            description="Python source code to check",
        ),
        ToolParameter(
            name="filename", type="string",
            description="Filename for error messages",
            required=False, default="<input>",
        ),
    ],
    returns="string ('OK' or error description)",
    permission=PermissionLevel.SAFE,
    execute=_syntax_check,
)
