"""
LLM Interface for Reflection Generation

This module provides an abstraction layer for LLM backends used in
reflection generation. Supports llama.cpp (local) and can be extended
for other backends (API-based, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMResponse:
    """Response from an LLM generation"""
    text: str
    model: str
    tokens_used: int = 0
    generation_time_ms: float = 0.0


class LLMInterface(ABC):
    """
    Abstract base class for LLM backends.

    Implement this interface to add support for different LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            stop_sequences: Sequences that stop generation

        Returns:
            LLMResponse with generated text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is available and ready"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the model"""
        pass


class LlamaCppLLM(LLMInterface):
    """
    LLM interface using llama.cpp via llama-cpp-python.

    Uses the same llama.cpp infrastructure as our embedding model,
    but with a text generation model instead.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize llama.cpp LLM.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of threads (None = auto)
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self._model = None
        self._model_name = "unloaded"

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load a llama.cpp model"""
        from llama_cpp import Llama

        self.model_path = model_path
        self._model_name = Path(model_path).stem

        print(f"Loading LLM from {model_path}...")
        self._model = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=self.verbose,
        )
        print(f"LLM loaded: {self._model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Generate text using llama.cpp"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import time
        start_time = time.time()

        response = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences or [],
            echo=False,
        )

        generation_time = (time.time() - start_time) * 1000

        text = response["choices"][0]["text"]
        tokens_used = response.get("usage", {}).get("total_tokens", 0)

        return LLMResponse(
            text=text.strip(),
            model=self._model_name,
            tokens_used=tokens_used,
            generation_time_ms=generation_time,
        )

    def is_available(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name


class MockLLM(LLMInterface):
    """
    Mock LLM for testing.

    Returns predefined responses based on keywords in the prompt.
    Useful for unit testing without requiring a real model.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize mock LLM.

        Args:
            responses: Dict mapping keywords to response text
        """
        self.responses = responses or {}
        self._default_response = "This is a mock LLM response for testing."

    def set_response(self, keyword: str, response: str) -> None:
        """Set a response for a specific keyword"""
        self.responses[keyword] = response

    def set_default_response(self, response: str) -> None:
        """Set the default response when no keyword matches"""
        self._default_response = response

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Return mock response based on prompt keywords"""
        prompt_lower = prompt.lower()

        # Check for keyword matches
        for keyword, response in self.responses.items():
            if keyword.lower() in prompt_lower:
                return LLMResponse(
                    text=response,
                    model="mock",
                    tokens_used=len(response.split()),
                    generation_time_ms=1.0,
                )

        return LLMResponse(
            text=self._default_response,
            model="mock",
            tokens_used=len(self._default_response.split()),
            generation_time_ms=1.0,
        )

    def is_available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock"


# Reflection prompts
FAILURE_REFLECTION_PROMPT = """You are analyzing an agent's failed attempt. Based on the information below, provide a reflection that will help the agent learn from this failure.

CONTEXT: {context}
ACTION TAKEN: {action}
OUTCOME: {outcome}
SUCCESS SCORE: {score}

Analyze what went wrong and respond in this exact format:

INSIGHT: [One clear sentence describing the key lesson learned]

CAUSAL FACTORS:
- [Factor 1]: [positive/negative] (confidence: [0.0-1.0])
- [Factor 2]: [positive/negative] (confidence: [0.0-1.0])

ACTIONABLE TAKEAWAY: [One specific action to take in similar situations]
"""

SUCCESS_REFLECTION_PROMPT = """You are analyzing an agent's successful attempt. Based on the information below, provide a reflection that will help the agent replicate this success.

CONTEXT: {context}
ACTION TAKEN: {action}
OUTCOME: {outcome}
SUCCESS SCORE: {score}

Analyze why this worked and respond in this exact format:

INSIGHT: [One clear sentence describing why this approach succeeded]

CAUSAL FACTORS:
- [Factor 1]: [positive/negative] (confidence: [0.0-1.0])
- [Factor 2]: [positive/negative] (confidence: [0.0-1.0])

ACTIONABLE TAKEAWAY: [One specific principle to apply in similar situations]
"""

PATTERN_DISCOVERY_PROMPT = """You are analyzing a cluster of similar experiences to discover patterns. Based on the episodes below, identify what makes this pattern effective.

EPISODES:
{episodes}

COMMON TAGS: {tags}
AVERAGE SUCCESS RATE: {success_rate}

Analyze this pattern and respond in this exact format:

INSIGHT: [One clear sentence describing the pattern and why it works]

CAUSAL FACTORS:
- [Factor 1]: [positive/negative] (confidence: [0.0-1.0])
- [Factor 2]: [positive/negative] (confidence: [0.0-1.0])

ACTIONABLE TAKEAWAY: [One specific guideline derived from this pattern]
"""


def parse_reflection_response(response_text: str) -> Dict[str, Any]:
    """
    Parse an LLM reflection response into structured data.

    Args:
        response_text: Raw LLM response text

    Returns:
        Dict with 'insight', 'causal_factors', 'actionable_takeaway'
    """
    result = {
        "insight": "",
        "causal_factors": [],
        "actionable_takeaway": "",
    }

    lines = response_text.strip().split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("INSIGHT:"):
            result["insight"] = line[8:].strip()
            current_section = None
        elif line.startswith("CAUSAL FACTORS:"):
            current_section = "causal"
        elif line.startswith("ACTIONABLE TAKEAWAY:"):
            result["actionable_takeaway"] = line[20:].strip()
            current_section = None
        elif current_section == "causal" and line.startswith("- "):
            # Parse causal factor line: "- [Factor]: [contribution] (confidence: [value])"
            factor_text = line[2:]
            factor = _parse_causal_factor(factor_text)
            if factor:
                result["causal_factors"].append(factor)

    return result


def _parse_causal_factor(text: str) -> Optional[Dict[str, Any]]:
    """Parse a single causal factor line"""
    try:
        # Format: "Factor name: positive/negative (confidence: 0.8)"
        if ":" not in text:
            return None

        # Split on first colon for factor name
        parts = text.split(":", 1)
        factor_name = parts[0].strip()

        rest = parts[1].strip() if len(parts) > 1 else ""

        # Extract contribution
        contribution = "neutral"
        if "positive" in rest.lower():
            contribution = "positive"
        elif "negative" in rest.lower():
            contribution = "negative"

        # Extract confidence
        confidence = 0.5
        if "confidence:" in rest.lower():
            try:
                conf_part = rest.lower().split("confidence:")[1]
                conf_str = "".join(c for c in conf_part if c.isdigit() or c == ".")
                if conf_str:
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                pass

        return {
            "factor": factor_name,
            "contribution": contribution,
            "confidence": confidence,
        }
    except Exception:
        return None
