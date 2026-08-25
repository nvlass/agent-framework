"""LLM-based reasoner bridging ChatLLMInterface to ReasonerInterface."""

import json
import logging
from typing import Optional

from agent_mind.goals.model import Goal
from agent_patterns.react import ReasonerInterface, ReasoningResult

from agent_core.llm import ChatLLMInterface, ChatResponse
from agent_core.prompt import PromptAssembler

logger = logging.getLogger(__name__)


class LLMReasoner(ReasonerInterface):
    """Bridges ChatLLMInterface to ReasonerInterface.

    Calls assembler.build() -> llm.chat() -> _parse() each turn.
    """

    def __init__(
        self,
        llm: ChatLLMInterface,
        assembler: Optional[PromptAssembler] = None,
        soul: str = "",
    ) -> None:
        self._llm = llm
        self._assembler = assembler or PromptAssembler()
        self._soul = soul

    def reason(
        self,
        goal: Goal,
        observations: list[str],
        available_tools: list[dict],
    ) -> ReasoningResult:
        """Generate next thought+action or answer via LLM.

        When tool schemas are available, passes them to the LLM for native
        function calling. If the response contains tool_calls, uses those
        directly (bypassing text parsing). Falls back to _parse() otherwise.
        """
        native_tools = available_tools if available_tools else None
        messages = self._assembler.build(
            goal=goal.description,
            observations=observations,
            tool_schemas=available_tools,
            soul=self._soul,
            native_tools=bool(native_tools),
        )
        msg_sizes = [len(m.content) for m in messages]
        logger.debug("Calling LLM with %d messages, sizes=%s (total %d chars)",
                      len(messages), msg_sizes, sum(msg_sizes))
        try:
            response = self._llm.chat(messages, tools=native_tools)
        except Exception as e:
            logger.error("LLM chat failed: %s: %s", type(e).__name__, e)
            return ReasoningResult(
                thought=f"LLM error: {e}",
                answer=None,
                action=None,
            )
        logger.debug("LLM raw response (%d chars): %s", len(response.content), response.content)

        # Native tool calling path
        if response.tool_calls:
            tc = response.tool_calls[0]  # Use first tool call
            thought = response.content if response.content else f"Calling {tc.name}"
            logger.info("Native tool call: %s", tc.name)
            return ReasoningResult(
                thought=thought,
                action=tc.name,
                action_args=tc.arguments,
                answer=None,
            )

        # Text parsing fallback
        logger.debug("No tool_calls in response, falling back to text parsing")
        result = self._parse(response.content)
        if result.answer is not None:
            logger.info("Reasoner produced answer: %.80s", result.answer)
        elif result.action:
            logger.info("Reasoner produced action: %s", result.action)
        return result

    @staticmethod
    def _parse(text: str) -> ReasoningResult:
        """Parse LLM output into ReasoningResult.

        Expected formats:
            Thought: ...
            Action: tool_name
            Action Args: {"key": "value"}

        Or:
            Thought: ...
            Answer: final answer

        Tolerates markdown code fences and multiline Action Args.
        """
        thought = ""
        action = None
        action_args: dict = {}
        answer = None

        # Strip code fences (```json, ```, etc.)
        cleaned = text.strip()
        lines = cleaned.splitlines()

        collecting_args = False
        args_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            # Skip code fence lines
            if stripped.startswith("```"):
                continue

            if collecting_args:
                # Check if we hit the next known prefix — stop collecting
                if stripped.startswith(("Thought:", "Action:", "Answer:")):
                    collecting_args = False
                    # Fall through to handle this line normally
                else:
                    args_lines.append(stripped)
                    continue

            if stripped.startswith("Thought:"):
                thought = stripped[len("Thought:"):].strip()
            elif stripped.startswith("Action Args:"):
                rest = stripped[len("Action Args:"):].strip()
                if rest:
                    args_lines.append(rest)
                collecting_args = True
            elif stripped.startswith("Action:"):
                action = stripped[len("Action:"):].strip()
            elif stripped.startswith("Answer:"):
                answer = stripped[len("Answer:"):].strip()

        # Parse collected Action Args
        if args_lines:
            raw = " ".join(args_lines)
            try:
                action_args = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse Action Args: %s", raw[:200])
                action_args = {}

        # If both action and answer present, action wins — the LLM wants to act.
        # (Small models often dump analysis as Answer then add the Action.)
        if action and answer is not None:
            logger.info("Both Action and Answer found; keeping action=%s", action)
            answer = None
        # If only answer, clear any stale action state
        elif answer is not None:
            action = None
            action_args = {}

        # If nothing parsed, treat whole text as thought with answer
        if not thought and action is None and answer is None:
            logger.warning("No structured output detected, treating as raw answer")
            thought = text.strip()
            answer = text.strip()

        return ReasoningResult(
            thought=thought,
            action=action,
            action_args=action_args,
            answer=answer,
        )
