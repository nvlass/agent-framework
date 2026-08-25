"""
Tool definition types.

A ToolDefinition is the complete specification of what a tool does,
what parameters it takes, what permissions it needs, and how to execute it.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class PermissionLevel(Enum):
    """Permission levels for tool execution.

    Determines how the tool is treated by the security model:
    - SAFE: No side effects, always allowed (e.g., syntax_check)
    - READ: Reads system state, low risk (e.g., read_file, list_directory)
    - WRITE: Modifies state, requires approval (e.g., memory_store)
    - DANGEROUS: System-level operations, must be explicitly registered
                 with scoping — never auto-registered (e.g., shell, python_exec)
    """
    SAFE = "safe"
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass
class ToolParameter:
    """Definition of a single tool parameter.

    Maps to JSON Schema properties for LLM function calling.

    Attributes:
        name: Parameter name (used as keyword argument)
        type: JSON Schema type — "string", "integer", "boolean", "array", "object"
        description: Human-readable description for the LLM
        required: Whether the parameter must be provided
        default: Default value if not required and not provided
        enum: Optional list of allowed values
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list] = None

    def to_schema(self) -> dict:
        """Export as JSON Schema property."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete definition of a tool.

    This is the fundamental unit of agent-tools. It describes:
    - What the tool does (name, description)
    - What it accepts (parameters)
    - What it returns (returns description)
    - What permission level it needs (permission)
    - How to run it (execute / execute_async callables)

    Attributes:
        name: Unique tool name (used for registration and invocation)
        description: Human-readable description for the LLM to understand when to use this tool
        parameters: List of ToolParameter definitions
        returns: Description of what the tool returns
        permission: Required permission level
        async_capable: Whether this tool supports async execution
        timeout_seconds: Maximum execution time before timeout
        execute: Synchronous execution callable
        execute_async: Asynchronous execution callable (optional)
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: str = "string"
    permission: PermissionLevel = PermissionLevel.SAFE
    async_capable: bool = False
    timeout_seconds: int = 30
    execute: Optional[Callable] = None
    execute_async: Optional[Callable] = None

    ## TODO: add a simpler, single line description of how to execute
    ## think "EXEC: git-fetch", "EXEC: git-checkout <branch>"

    def to_schema(self) -> dict:
        """Export as JSON Schema for LLM function calling.

        Returns a dict compatible with the OpenAI function calling format,
        which most LLMs support.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }

        if required:
            schema["parameters"]["required"] = required

        return schema

    def validate_args(self, **kwargs) -> list[str]:
        """Validate arguments against parameter definitions.

        Returns a list of error messages. Empty list means valid.
        """
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                errors.append(f"Missing required parameter: {param.name}")

        # Check for unknown parameters
        known_names = {p.name for p in self.parameters}
        for key in kwargs:
            if key not in known_names:
                errors.append(f"Unknown parameter: {key}")

        # Check enum constraints
        for param in self.parameters:
            if param.enum and param.name in kwargs:
                if kwargs[param.name] not in param.enum:
                    errors.append(
                        f"Invalid value for {param.name}: {kwargs[param.name]}. "
                        f"Must be one of: {param.enum}"
                    )

        return errors
