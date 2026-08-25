"""Adapters: bridge FireworksLLM to agent_memory interfaces.

- FireworksMemoryLLM  — wraps FireworksLLM as agent_memory.LLMInterface
- FireworksEmbeddingGenerator — thin alias over OpenAICompatEmbeddingGenerator
  pre-configured for the Fireworks /v1/embeddings endpoint.
"""

import os

from agent_memory.llm_interface import LLMInterface, LLMResponse
from agent_memory.embeddings import OpenAICompatEmbeddingGenerator

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_DEFAULT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class FireworksMemoryLLM(LLMInterface):
    """Wraps a FireworksLLM instance so it can be passed to MemoryTools."""

    def __init__(self, fireworks_llm) -> None:
        self._llm = fireworks_llm

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences=None,
    ) -> LLMResponse:
        from assistant.conversation import _call_llm_raw

        response = _call_llm_raw(
            self._llm,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response["choices"][0]["message"].get("content", "") or ""
        from assistant.config import strip_channel_markup
        return LLMResponse(text=strip_channel_markup(text.strip()),
                           model=self._llm.model_name)

    def is_available(self) -> bool:
        return self._llm.is_available()

    @property
    def model_name(self) -> str:
        return self._llm.model


def FireworksEmbeddingGenerator(
    api_key: str | None = None,
    model: str = _DEFAULT_EMBED_MODEL,
) -> OpenAICompatEmbeddingGenerator:
    """Return an OpenAICompatEmbeddingGenerator pre-configured for Fireworks.

    Kept as a named factory so existing call sites don't need to change.
    """
    return OpenAICompatEmbeddingGenerator(
        base_url=_FIREWORKS_BASE_URL,
        model=model,
        api_key=api_key or os.environ.get("FIREWORKS_API_KEY", ""),
        api_key_env="FIREWORKS_API_KEY",
    )
