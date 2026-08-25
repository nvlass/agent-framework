"""Prompt assembly for LLM interactions."""

from dataclasses import dataclass
from typing import Any

from agent_core.llm import ChatMessage


DEFAULT_SYSTEM_TEMPLATE = """\
You are an autonomous agent working to achieve the user's goal.

{soul_section}\
{tools_section}\
## Response Format

You MUST respond in one of two formats:

**To use a tool:**
Thought: <your reasoning>
Action: <tool_name>
Action Args: {{"key": "value"}}

**To give a final answer:**
Thought: <your reasoning>
Answer: <your final answer>

Rules:
- Always start with a Thought line.
- Action Args MUST be a single line of valid JSON. No markdown, no code fences.
- Do NOT wrap your response in code blocks."""

NATIVE_TOOLS_SYSTEM_TEMPLATE = """\
You are an autonomous agent working to achieve the user's goal.

{soul_section}\
Think step by step. Use the provided tools when needed. \
When you have the final answer, respond with it directly."""

DEFAULT_USER_TEMPLATE = """\
## Goal
{goal}

{observations_section}"""


def _format_tools_section(tool_schemas: list[dict]) -> str:
    """Format tool schemas into a readable section."""
    if not tool_schemas:
        return ""
    lines = ["## Available Tools\n"]
    for schema in tool_schemas:
        name = schema.get("name", "unknown")
        desc = schema.get("description", "")
        lines.append(f"- **{name}**: {desc}")
        params = schema.get("parameters", {}).get("properties", {})
        required = schema.get("parameters", {}).get("required", [])
        for pname, pinfo in params.items():
            req = " (required)" if pname in required else ""
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            lines.append(f"  - `{pname}` ({ptype}{req}): {pdesc}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _format_observations_section(observations: list[str]) -> str:
    """Format observations into a readable section."""
    if not observations:
        return ""
    lines = ["## Observations so far"]
    for obs in observations:
        lines.append(obs)
    lines.append("")
    return "\n".join(lines)


@dataclass
class PromptAssembler:
    """Builds chat messages from goal, observations, tools, and soul."""

    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    user_template: str = DEFAULT_USER_TEMPLATE

    def build(
        self,
        goal: str,
        observations: list[str],
        tool_schemas: list[dict],
        soul: str = "",
        native_tools: bool = False,
    ) -> list[ChatMessage]:
        """Assemble system and user messages.

        Args:
            native_tools: When True, tools are passed via the API (not in
                the prompt). Omits tool descriptions and text-format
                instructions from the system message.

        Returns:
            List of [system_message, user_message].
        """
        soul_section = f"## Identity\n{soul}\n\n" if soul else ""

        if native_tools:
            system_content = NATIVE_TOOLS_SYSTEM_TEMPLATE.format(
                soul_section=soul_section,
            )
        else:
            tools_section = _format_tools_section(tool_schemas)
            system_content = self.system_template.format(
                soul_section=soul_section,
                tools_section=tools_section,
            )

        observations_section = _format_observations_section(observations)
        user_content = self.user_template.format(
            goal=goal,
            observations_section=observations_section,
        )

        return [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]
