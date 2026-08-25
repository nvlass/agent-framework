"""
Tool executor — how tools run.

The executor handles permission checking, argument validation,
execution, timing, and error handling.

STATUS: Skeleton — implementation left for Nikos to complete.
See TODOs and hints below.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from threading import Timer, Thread

from .definition import ToolDefinition, PermissionLevel
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution.

    Every tool execution returns a ToolResult, whether it succeeded or failed.

    Attributes:
        success: Whether the execution completed without errors
        output: The tool's return value (None on failure)
        error: Error message if execution failed
        duration_ms: Execution time in milliseconds
        tool_name: Name of the tool that was executed
    """
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    tool_name: str = ""


class ToolExecutor:
    """Executes tools with permission checking and error handling.

    The executor sits between the agent and the tools. It:
    1. Looks up the tool in the registry
    2. Checks permissions via the permission_checker callback
    3. Validates arguments against the tool's parameter definitions
    4. Executes the tool with timing
    5. Wraps the result (or error) in a ToolResult

    Usage:
        def my_permission_checker(tool: ToolDefinition) -> bool:
            # Your permission logic here
            return tool.permission != PermissionLevel.DANGEROUS

        executor = ToolExecutor(registry, my_permission_checker)
        result = executor.execute("read_file", path="/tmp/foo.txt")

    Args:
        registry: The ToolRegistry to look up tools from
        permission_checker: Callback that decides if a tool is allowed to run.
                           Receives the ToolDefinition, returns True if allowed.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_checker: Callable[[ToolDefinition], bool],
    ):
        self.registry = registry
        self.permission_checker = permission_checker

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool synchronously.

        TODO (Nikos): Implement this method. Here's the flow:

        1. Look up the tool in self.registry
           - If not found, return a failed ToolResult with error message

        2. Check permissions via self.permission_checker(tool)
           - If denied, return a failed ToolResult with "Permission denied"
           - Log the denial

        3. Validate arguments via tool.validate_args(**kwargs)
           - If validation errors, return a failed ToolResult with the errors

        4. Apply defaults for missing optional parameters
           Hint: for param in tool.parameters:
                     if not param.required and param.name not in kwargs and param.default is not None:
                         kwargs[param.name] = param.default

        5. Execute with timing:
           - Record start time
           - Call tool.execute(**kwargs)
           - Record duration
           - Wrap in ToolResult(success=True, output=result, ...)

        6. Handle exceptions:
           - Catch Exception, log it
           - Return ToolResult(success=False, error=str(e), ...)

        7. (Stretch) Handle timeout:
           - If execution exceeds tool.timeout_seconds, kill it
           - This is tricky for sync execution — consider signal.alarm or threading
           - Can skip this for now, will be easier with async

        Returns:
            ToolResult with success/failure, output, error, and timing
        """
        errors = []

        tool = self.registry.get(tool_name)
        if tool is None:
            logger.warning("Unknown tool: %s", tool_name)
            errors += [f'Unknown tool: {tool_name}']

        if tool and not self.permission_checker(tool):
            logger.warning("Permission denied for tool: %s", tool_name)
            errors += ['Permission denied']

        if tool:
            validation_errors = tool.validate_args(**kwargs)
            if validation_errors:
                logger.warning("Validation errors for %s: %s", tool_name, validation_errors)
            errors += validation_errors

        if errors:
            return ToolResult(success = False,
                              output = None,
                              error = "; ".join(errors),
                              tool_name = tool_name)

        for param in tool.parameters:
            if not param.required and param.name not in kwargs and \
               param.default is not None:
                kwargs[param.name] = param.default

        def timeout_handler():
            # leave as a noop for now, couldn't find a method for
            # cancelling tool execution and has to be tool specific
            #
            # something like:
            # tool.cancel()
            pass

        monitor: Timer = Timer(tool.timeout_seconds, timeout_handler)

        logger.debug("Tool %s called with args: %s", tool_name, kwargs)
        tstart = time.time()
        try:
            monitor.start()
            result = tool.execute(**kwargs)
            monitor.cancel()
            duration = (time.time() - tstart) * 1000.0
            logger.info("Tool %s executed in %dms", tool_name, round(duration))
            logger.debug("Tool %s output (%s): %.500s", tool_name, type(result).__name__, str(result))
            return ToolResult(success = True,
                              output = result,
                              error = None,
                              duration_ms = round(duration),
                              tool_name = tool_name)
        except Exception as ex:
            duration = (time.time() - tstart) * 1000.0
            logger.error("Tool %s raised exception: %s: %s", tool_name, type(ex).__name__, ex)
            return ToolResult(success = False,
                              error = str(ex),
                              duration_ms = round(duration),
                              tool_name = tool_name)

    async def execute_async(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool asynchronously.

        TODO (Nikos — later, after sync execute works):

        Same flow as execute(), but:
        - Use tool.execute_async if available, fall back to tool.execute
        - Use asyncio.wait_for() for timeout handling (much cleaner than sync)
        - Consider asyncio.to_thread() for wrapping sync tools

        Returns:
            ToolResult with success/failure, output, error, and timing
        """
        raise NotImplementedError("TODO: Implement execute_async()")
