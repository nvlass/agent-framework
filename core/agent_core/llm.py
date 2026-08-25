"""Chat LLM interface and mock implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ToolCall:
    """A tool call returned by the LLM via native function calling."""

    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """Response from a chat LLM."""

    content: str
    model: str
    tokens_used: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatLLMInterface(ABC):
    """Abstract interface for chat-based LLMs."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Send messages and get a response.

        Args:
            messages: Conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Optional list of tool schemas (OpenAI format) for native
                function calling. When provided, the LLM may return tool_calls
                instead of text content.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model."""


class MockChatLLM(ChatLLMInterface):
    """Mock LLM that returns pre-configured responses. For testing.

    Supports both text responses (list[str]) and ChatResponse objects
    for testing native tool calling.
    """

    def __init__(self, responses: list[str | ChatResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[list[ChatMessage]] = []
        self.tools_passed: list[Optional[list[dict]]] = []

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        self.tools_passed.append(tools)
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
        else:
            # Fallback: repeat last response
            resp = self._responses[-1] if self._responses else ""
        if isinstance(resp, ChatResponse):
            return resp
        return ChatResponse(content=resp, model="mock")

    def is_available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock"
