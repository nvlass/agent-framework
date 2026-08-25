"""
Memory bridge tools.

Wraps agent-memory's MemoryTools as ToolDefinitions for use in the
agent-tools registry. Requires a MemoryTools instance to be provided
at creation time (dependency injection, no globals).

Usage:
    from agent_memory import MemoryStore, MemoryTools
    from agent_tools.tools.memory import create_memory_tools

    store = MemoryStore(db_path="agent.db")
    memory_tools = MemoryTools(store=store)
    tools = create_memory_tools(memory_tools)
    # tools is a dict: {"memory_store": ToolDefinition, ...}

    registry = ToolRegistry()
    for tool in tools.values():
        registry.register(tool)
"""

import json
from typing import Any

from ..core.definition import ToolDefinition, ToolParameter, PermissionLevel


def _format_result(data: Any) -> str:
    """Format memory ToolResult data as a string for the agent."""
    if isinstance(data, dict):
        return json.dumps(data, indent=2, default=str)
    if isinstance(data, list):
        return json.dumps(data, indent=2, default=str)
    return str(data)


def create_memory_tools(memory_tools) -> dict[str, ToolDefinition]:
    """Create memory bridge tools from a MemoryTools instance.

    Args:
        memory_tools: An agent_memory.MemoryTools instance.

    Returns:
        Dict mapping tool names to ToolDefinitions.
    """

    def _memory_store(context: str, action: str, outcome: str = "",
                      tags: str = "") -> str:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        result = memory_tools.store_memory(
            context=context, action=action, outcome=outcome, tags=tag_list,
        )
        if not result.success:
            raise RuntimeError(result.error or "Failed to store memory")
        return _format_result(result.data)

    def _memory_recall(query: str, limit: int = 5) -> str:
        result = memory_tools.recall_similar(query=query, limit=limit)
        if not result.success:
            raise RuntimeError(result.error or "Failed to recall memories")
        return _format_result(result.data)

    def _memory_reflect(hours: int = 24, focus: str = "") -> str:
        result = memory_tools.reflect_on_recent(
            hours=hours, focus=focus or None,
        )
        if not result.success:
            raise RuntimeError(result.error or "Failed to reflect")
        return _format_result(result.data)

    return {
        "memory_store": ToolDefinition(
            name="memory_store",
            description="Store an experience in memory. Records what happened "
                        "(context), what was done (action), and the result (outcome).",
            parameters=[
                ToolParameter(
                    name="context", type="string",
                    description="What was the situation or problem",
                ),
                ToolParameter(
                    name="action", type="string",
                    description="What action was taken",
                ),
                ToolParameter(
                    name="outcome", type="string",
                    description="What was the result",
                    required=False, default="",
                ),
                ToolParameter(
                    name="tags", type="string",
                    description="Comma-separated tags for categorization",
                    required=False, default="",
                ),
            ],
            returns="string (stored episode details)",
            permission=PermissionLevel.WRITE,
            execute=_memory_store,
        ),
        "memory_recall": ToolDefinition(
            name="memory_recall",
            description="Recall memories similar to a query. Returns past "
                        "experiences relevant to the current situation.",
            parameters=[
                ToolParameter(
                    name="query", type="string",
                    description="What to search for in memory",
                ),
                ToolParameter(
                    name="limit", type="integer",
                    description="Maximum number of memories to return",
                    required=False, default=5,
                ),
            ],
            returns="string (matching memories with similarity scores)",
            permission=PermissionLevel.READ,
            execute=_memory_recall,
        ),
        "memory_reflect": ToolDefinition(
            name="memory_reflect",
            description="Reflect on recent experiences to find patterns and "
                        "insights. Can focus on failures, successes, or a tag.",
            parameters=[
                ToolParameter(
                    name="hours", type="integer",
                    description="How many hours back to reflect on",
                    required=False, default=24,
                ),
                ToolParameter(
                    name="focus", type="string",
                    description="Focus area: 'failures', 'successes', or a tag name",
                    required=False, default="",
                ),
            ],
            returns="string (reflection insights and takeaways)",
            permission=PermissionLevel.WRITE,
            execute=_memory_reflect,
        ),
    }
