"""Embedding generation — protocol + concrete backends.

Three things live here:

1. ``EmbeddingGenerator`` — a structural Protocol.  Any class that implements
   ``generate_embedding`` and ``generate_embeddings_batch`` satisfies it without
   needing to inherit.

2. ``LlamaCppEmbeddingGenerator`` — local inference via a llama.cpp ``.gguf``
   model.  Original backend; requires ``llama-cpp-python``.

3. ``OpenAICompatEmbeddingGenerator`` — HTTP backend for any provider that
   exposes an OpenAI-compatible ``/v1/embeddings`` endpoint: Fireworks AI,
   OpenAI, Ollama, Together AI, xAI, etc.  Pass ``base_url`` + ``model``
   and the right ``api_key``; the rest is automatic.

Module-level helpers ``cosine_similarity``, ``euclidean_distance``, and
``compute_text_similarity`` work with any generator.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingGenerator(Protocol):
    """Structural protocol for embedding generators.

    Any class that implements these two methods satisfies the protocol — no
    inheritance required.  Use this as the type hint wherever a generator is
    accepted (``MemoryStore``, ``AnalogyFinder``, etc.).
    """

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray: ...

    def generate_embeddings_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[np.ndarray]: ...


# ---------------------------------------------------------------------------
# Backend: llama.cpp (local)
# ---------------------------------------------------------------------------

class LlamaCppEmbeddingGenerator:
    """Embedding generator backed by a local llama.cpp model (.gguf file).

    Requires ``llama-cpp-python``.  Use when you want fully offline, local
    inference with no API calls.

    Args:
        model_path: Path to a ``.gguf`` embedding model.  If ``None``, call
                    ``load_model()`` before generating embeddings.
        n_ctx:      Context window size (512 is standard for embeddings).
        embedding:  Must be ``True`` for embedding mode.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 512,
        embedding: bool = True,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._embedding_cache: dict[str, np.ndarray] = {}
        self.model = None

        if model_path:
            self.load_model(model_path, n_ctx, embedding)

    def load_model(self, model_path: str, n_ctx: int = 512, embedding: bool = True) -> None:
        from llama_cpp import Llama
        print(f"Loading embedding model from {model_path}...")
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, embedding=embedding, verbose=False)
        self.model_path = model_path
        print("Model loaded successfully!")

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if use_cache:
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self._embedding_cache:
                return self._embedding_cache[key]
        arr = np.array(self.model.embed(text), dtype=np.float32)
        if use_cache:
            self._embedding_cache[key] = arr
        return arr

    def generate_embeddings_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[np.ndarray]:
        return [self.generate_embedding(t, use_cache) for t in texts]

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def get_cache_size(self) -> int:
        return len(self._embedding_cache)

    def __repr__(self) -> str:
        return f"LlamaCppEmbeddingGenerator(model_path={self.model_path}, cache_size={self.get_cache_size()})"


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible HTTP (Fireworks, OpenAI, Ollama, xAI, …)
# ---------------------------------------------------------------------------

class OpenAICompatEmbeddingGenerator:
    """Embedding generator backed by any OpenAI-compatible /v1/embeddings endpoint.

    Works out of the box with:
    - Fireworks AI: ``base_url="https://api.fireworks.ai/inference/v1"``
    - OpenAI:       ``base_url="https://api.openai.com/v1"``
    - Ollama:       ``base_url="http://localhost:11434/v1"``
    - Together AI:  ``base_url="https://api.together.xyz/v1"``
    - xAI:          ``base_url="https://api.x.ai/v1"``  (when embeddings land)

    Results are cached in-memory by MD5(text) to avoid redundant API calls.

    Args:
        base_url:     Provider base URL (no trailing slash, no ``/embeddings``).
        model:        Embedding model name as the provider expects it.
        api_key:      API key.  If ``None``, falls back to ``api_key_env``.
        api_key_env:  Environment variable to read when ``api_key`` is absent.
        timeout:      HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 30,
    ) -> None:
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._api_key = api_key or os.environ.get(api_key_env, "")
        self._timeout = timeout
        self._cache: dict[str, np.ndarray] = {}

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        if use_cache:
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self._cache:
                return self._cache[key]

        import requests
        resp = requests.post(
            self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        arr = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)

        if use_cache:
            self._cache[key] = arr
        return arr

    def generate_embeddings_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[np.ndarray]:
        return [self.generate_embedding(t, use_cache) for t in texts]

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"OpenAICompatEmbeddingGenerator(url={self._url}, model={self._model})"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors (-1 to 1)."""
    n1, n2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (n1 * n2))


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Euclidean distance between two embedding vectors (0 = identical)."""
    return float(np.linalg.norm(embedding1 - embedding2))


def compute_text_similarity(
    text1: str, text2: str, generator: EmbeddingGenerator
) -> float:
    """Convenience: embed two texts and return their cosine similarity."""
    return cosine_similarity(generator.generate_embedding(text1), generator.generate_embedding(text2))
