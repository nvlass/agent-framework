"""agent-core: Agent identity, orchestration, and binding layer."""

from agent_core.config import AgentConfig, ReactConfig, ReflexionConfig
from agent_core.llm import ChatMessage, ChatResponse, ChatLLMInterface, MockChatLLM, ToolCall
from agent_core.prompt import PromptAssembler
from agent_core.reasoner import LLMReasoner
from agent_core.agent import AgentRole, AgentInstance
from agent_core.soul import Soul, SoulProposal, SoulManager
from agent_core.spawn import SpawnRole, SpawnRegistry
from agent_core.mailbox import AgentMailbox
from agent_core.conversation import ConversationBus, ConversationError

__all__ = [
    "AgentConfig",
    "ReactConfig",
    "ReflexionConfig",
    "ChatMessage",
    "ChatResponse",
    "ChatLLMInterface",
    "MockChatLLM",
    "ToolCall",
    "PromptAssembler",
    "LLMReasoner",
    "AgentRole",
    "AgentInstance",
    "Soul",
    "SoulProposal",
    "SoulManager",
    "SpawnRole",
    "SpawnRegistry",
    "AgentMailbox",
    "ConversationBus",
    "ConversationError",
]

from agent_core.llm_cloud import AnthropicLLM, OpenAILLM, FireworksLLM

__all__ += ["AnthropicLLM", "OpenAILLM", "FireworksLLM"]
