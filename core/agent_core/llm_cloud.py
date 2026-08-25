"""Cloud LLM backends — Anthropic and OpenAI-compatible APIs.

Both use direct HTTP (no SDK dependencies required).

Usage:
    from agent_core.llm_cloud import AnthropicLLM, OpenAILLM, FireworksLLM

    # Anthropic — reads ANTHROPIC_API_KEY from env by default
    llm = AnthropicLLM(model="claude-opus-4-6")

    # OpenAI — reads OPENAI_API_KEY from env by default
    llm = OpenAILLM(model="gpt-4o")

    # OpenAI-compatible (Groq, Together, Mistral, etc.)
    llm = OpenAILLM(model="llama-3.1-8b-instant",
                    base_url="https://api.groq.com/openai",
                    api_key="gsk_...")

    # Fireworks AI — reads FIREWORKS_API_KEY from env by default
    llm = FireworksLLM(model="accounts/fireworks/models/deepseek-v4-pro")
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from agent_core.llm import ChatLLMInterface, ChatMessage, ChatResponse, ToolCall


def _parse_retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
    """Extract retry-after seconds from a 429 response, or None."""
    val = exc.headers.get("retry-after") or exc.headers.get("Retry-After")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return None


def _parse_sse_stream(resp) -> dict:
    """Reassemble a streaming SSE response into a single response-like dict.

    Handles both text content and tool calls, which arrive as fragments:

        data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x",
                "function":{"name":"send_email","arguments":""}}]}}]}
        data: {"choices":[{"delta":{"tool_calls":[{"index":0,
                "function":{"arguments":"{\"subject\":"}}]}}]}
        ...
        data: [DONE]

    Returns a dict with the same shape as a non-streaming chat completion
    response, so callers can parse it identically.
    """
    content_parts: list[str] = []
    # tool_calls_map: index → {id, name, arguments fragments}
    tool_calls_acc: dict[int, dict] = {}
    usage: dict = {}
    model: str = ""

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if not model:
            model = chunk.get("model", "")
        # Fireworks returns usage in the final chunk
        if chunk.get("usage"):
            usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}

        # Text content
        if delta.get("content"):
            content_parts.append(delta["content"])

        # Tool call fragments
        for tc_delta in delta.get("tool_calls") or []:
            idx = tc_delta.get("index", 0)
            if idx not in tool_calls_acc:
                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
            entry = tool_calls_acc[idx]
            if tc_delta.get("id"):
                entry["id"] = tc_delta["id"]
            fn = tc_delta.get("function") or {}
            if fn.get("name"):
                entry["name"] = fn["name"]
            if fn.get("arguments"):
                entry["arguments"] += fn["arguments"]

    # Rebuild into OpenAI-shaped message dict (preserve id for tool result matching)
    tool_calls_out = [
        {"id": v["id"], "type": "function", "function": {"name": v["name"], "arguments": v["arguments"]}}
        for _, v in sorted(tool_calls_acc.items())
    ]
    return {
        "model": model,
        "choices": [{
            "message": {
                "content": "".join(content_parts) or None,
                "tool_calls": tool_calls_out or None,
            }
        }],
        "usage": usage,
    }


def _post_with_retry(
    req: urllib.request.Request,
    timeout: int,
    max_retries: int,
    provider: str,
) -> dict:
    """POST req, retrying on 429 rate-limit responses.

    Each retry waits for the ``retry-after`` header value (seconds), falling
    back to exponential backoff (5 s, 10 s, 20 s, …) when the header is absent.

    The per-request socket timeout is unaffected — it still applies to each
    individual HTTP call. The sleep happens *between* calls.

    Args:
        req: Prepared urllib Request (reusable).
        timeout: Per-request socket timeout in seconds.
        max_retries: Maximum number of retries on 429 (0 = no retry).
        provider: Name used in error messages (e.g. "Anthropic", "OpenAI").

    Raises:
        PermissionError: On 401 / 403.
        RuntimeError: On other HTTP errors, or 429 after all retries exhausted.
        ConnectionError: On network failure.
    """
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                wait = _parse_retry_after(exc) or (5.0 * (2 ** attempt))
                time.sleep(wait)
                continue
            body_text = ""
            try:
                body_text = (exc.fp.read() if exc.fp else b"").decode()
            except Exception:
                pass
            if exc.code in (401, 403):
                raise PermissionError(
                    f"{provider} auth error {exc.code}: {body_text}"
                ) from exc
            if exc.code == 429:
                raise RuntimeError(
                    f"{provider} rate limit exceeded after {max_retries} retries: {body_text}"
                ) from exc
            raise RuntimeError(
                f"{provider} API error {exc.code}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach {provider} API: {exc.reason}"
            ) from exc
    raise RuntimeError(f"{provider}: unexpected retry loop exit")  # unreachable


def _translate_messages_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert OpenAI-format raw message dicts to Anthropic API format.

    Returns (system_text, anthropic_messages).

    Handles:
    - role="system"  → extracted as top-level system string
    - role="user"    → passed through
    - role="assistant" with tool_calls → content blocks (text + tool_use)
    - role="tool"    → merged into a single user message with tool_result blocks
                       (Anthropic requires all tool results for one round in one message)
    """
    system = ""
    anthropic_messages: list[dict] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "system":
            system = msg.get("content", "")
            i += 1
            continue

        if role == "user":
            anthropic_messages.append({"role": "user", "content": msg.get("content", "")})
            i += 1
            continue

        if role == "assistant":
            content_blocks: list[dict] = []
            text = msg.get("content") or ""
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if content_blocks:
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            i += 1
            continue

        if role == "tool":
            # Merge consecutive tool-result messages into a single user message
            tool_blocks: list[dict] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                t = messages[i]
                tool_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": t.get("tool_call_id", ""),
                    "content": t.get("content", ""),
                })
                i += 1
            anthropic_messages.append({"role": "user", "content": tool_blocks})
            continue

        i += 1  # skip unknown roles

    return system, anthropic_messages


def _normalize_anthropic_response(response: dict, model: str) -> dict:
    """Convert an Anthropic /v1/messages response to OpenAI-shaped completion dict.

    Returns a dict with the same shape as _parse_sse_stream so that callers
    can parse it identically regardless of provider.
    """
    text_parts: list[str] = []
    tool_calls_out: list[dict] = []

    for block in response.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls_out.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    return {
        "model": response.get("model", model),
        "choices": [{
            "message": {
                "content": "".join(text_parts) or None,
                "tool_calls": tool_calls_out or None,
            }
        }],
        "usage": {
            "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
        },
    }


class AnthropicLLM(ChatLLMInterface):
    """Anthropic Messages API.

    Handles the Anthropic-specific message format:
    - System message is a top-level field (not inside messages array)
    - Tool schemas use ``input_schema`` instead of ``parameters``
    - Response content is a list of typed blocks (text, tool_use)

    Args:
        model: Model ID, e.g. ``"claude-opus-4-6"`` or ``"claude-haiku-4-5-20251001"``.
        api_key: API key. Defaults to ``ANTHROPIC_API_KEY`` env var.
        timeout_seconds: Per-request HTTP socket timeout. Unaffected by retries.
        max_tokens: Default max output tokens (override per call with chat(max_tokens=N)).
        max_retries: How many times to retry on 429 rate-limit responses.
            Each retry sleeps for the ``retry-after`` header value, or exponential
            backoff (5 s, 10 s, 20 s, …) if the header is absent.
        base_url: Override for self-hosted or proxy deployments.
    """

    _API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        timeout_seconds: int = 120,
        max_tokens: int = 512,
        max_retries: int = 3,
        base_url: str = "https://api.anthropic.com",
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Call /v1/messages.

        Anthropic requires:
        - ``system`` as a top-level string (not a message)
        - ``messages`` alternating user/assistant (no system role in array)
        - ``max_tokens`` is mandatory

        Args:
            tools: OpenAI-format tool schemas. Automatically converted to
                Anthropic format (``input_schema`` instead of ``parameters``).

        Raises:
            PermissionError: If API key is missing or invalid (401/403).
            RuntimeError: If the API returns an error response.
            ConnectionError: On network failure.
        """
        # Anthropic: system is top-level, not a message
        system = ""
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "model": self._model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system:
            body["system"] = system
        if tools:
            # Convert OpenAI schema format → Anthropic format
            body["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._API_VERSION,
        }
        req = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        response = _post_with_retry(req, self._timeout, self._max_retries, "Anthropic")

        # Parse content blocks
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        tokens_used = response.get("usage", {}).get("output_tokens", 0)
        return ChatResponse(
            content=" ".join(text_parts).strip(),
            model=response.get("model", self._model),
            tokens_used=tokens_used,
            tool_calls=tool_calls,
        )

    def call_raw(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Call /v1/messages with raw OpenAI-format message dicts.

        Translates the OpenAI message format to Anthropic's format transparently,
        including tool calls and tool results. Returns a normalized response dict
        with the same shape as _parse_sse_stream so callers are provider-agnostic.

        Args:
            messages: Raw OpenAI-format message dicts (role/content/tool_calls/tool_call_id).
            tools: Tool schemas in OpenAI format — auto-converted to Anthropic format.
        """
        system, chat_messages = _translate_messages_to_anthropic(messages)
        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "model": self._model,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._API_VERSION,
        }
        req = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        response = _post_with_retry(req, self._timeout, self._max_retries, "Anthropic")
        return _normalize_anthropic_response(response, self._model)

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model


class OpenAILLM(ChatLLMInterface):
    """OpenAI Chat Completions API (and any compatible endpoint).

    Works with:
    - OpenAI (default)
    - Groq: ``base_url="https://api.groq.com/openai"``
    - Together: ``base_url="https://api.together.xyz"``
    - Mistral: ``base_url="https://api.mistral.ai"``
    - Local proxies (LiteLLM, Ollama OpenAI mode, etc.)

    The request/response format is identical to ``LlamaCppServerLLM`` —
    the only differences are the Authorization header and configurable URL.

    Args:
        model: Model ID, e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``.
        api_key: API key. Defaults to ``OPENAI_API_KEY`` env var.
        base_url: Base URL for the API (no trailing slash).
        timeout_seconds: Per-request HTTP socket timeout. Unaffected by retries.
        max_tokens: Default max output tokens (override per call with chat(max_tokens=N)).
        max_retries: How many times to retry on 429 rate-limit responses.
        extra_headers: Additional HTTP headers merged into every request.
            Useful for providers that require a User-Agent or custom auth headers
            (e.g. Fireworks AI behind Cloudflare).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com",
        timeout_seconds: int = 120,
        max_tokens: int = 512,
        max_retries: int = 3,
        extra_headers: Optional[dict] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._extra_headers = extra_headers or {}

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Call /v1/chat/completions.

        Identical to the llama-server format, with an Authorization header.

        Args:
            tools: Tool schemas in OpenAI format (passed through as-is).

        Raises:
            PermissionError: If API key is missing or invalid (401/403).
            RuntimeError: If the API returns an error response.
            ConnectionError: On network failure.
        """
        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        response = _post_with_retry(req, self._timeout, self._max_retries, "OpenAI")

        msg = response["choices"][0]["message"]
        content = (msg.get("content") or "").strip()

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args))

        tokens_used = response.get("usage", {}).get("completion_tokens", 0)
        return ChatResponse(
            content=content,
            model=response.get("model", self._model),
            tokens_used=tokens_used,
            tool_calls=tool_calls,
        )

    def call_raw(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Call Fireworks with raw OpenAI-format message dicts using streaming.

        Overrides OpenAILLM.call_raw() to use SSE streaming (avoiding the 4096
        non-streaming cap) and Fireworks-specific rate limiting.
        Returns a normalized dict with the same shape as _parse_sse_stream.
        """
        if self._min_request_interval > 0:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_call_time = time.monotonic()

        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "stream": True,
            "context_length_exceeded_behavior": self._context_length_exceeded_behavior,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return _parse_sse_stream(resp)
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = (exc.fp.read() if exc.fp else b"").decode()
            except Exception:
                pass
            if exc.code in (401, 403):
                raise PermissionError(f"Fireworks auth error {exc.code}: {body_text}") from exc
            if exc.code == 429:
                limit = exc.headers.get("x-ratelimit-limit", "?")
                remaining = exc.headers.get("x-ratelimit-remaining", "?")
                raise RuntimeError(
                    f"Fireworks rate limit (limit={limit}, remaining={remaining}): {body_text}"
                ) from exc
            raise RuntimeError(f"Fireworks API error {exc.code}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach Fireworks API: {exc.reason}") from exc

    def call_raw(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Call /v1/chat/completions with raw OpenAI-format message dicts.

        Unlike chat(), accepts raw dicts so callers can include role="tool"
        messages with tool_call_id. Returns the raw completion response dict.
        """
        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        return _post_with_retry(req, self._timeout, self._max_retries, "OpenAI")

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

class FireworksLLM(OpenAILLM):
    """Fireworks AI inference API.

    Thin subclass of OpenAILLM that pre-configures the Fireworks endpoint,
    reads ``FIREWORKS_API_KEY`` from the environment, and sets sensible
    defaults for Fireworks-specific behaviour.

    Fireworks differences from standard OpenAI:
    - Always uses ``stream=True`` to avoid the 4096 non-streaming output cap.
      The SSE stream is reassembled transparently — callers see a normal
      ``ChatResponse``.
    - Requires a ``User-Agent`` header to pass Cloudflare — included by default.
    - ``context_length_exceeded_behavior``: ``"truncate"`` (default) silently
      reduces ``max_tokens`` to fit the context window; ``"error"`` mirrors
      OpenAI's behaviour.
    - Model IDs use the ``accounts/fireworks/models/<name>`` prefix.

    Args:
        model: Fireworks model ID.
        api_key: API key. Defaults to ``FIREWORKS_API_KEY`` env var.
        timeout_seconds: Per-request HTTP socket timeout.
        max_tokens: Default max output tokens (no hard cap when streaming).
        max_retries: Retries on 429 rate-limit responses.
        min_request_interval: Minimum seconds between requests. Enforces a
            proactive rate limit to stay within Fireworks' RPS quota.
            Default 1.0 (= 1 req/s). Set to 0 to disable.
        context_length_exceeded_behavior: ``"truncate"`` (default) or ``"error"``.
        extra_headers: Additional HTTP headers merged on top of the default
            User-Agent. Use to pass any other provider-required headers.
    """

    _BASE_URL = "https://api.fireworks.ai/inference"
    _DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
    _USER_AGENT = "agent-core/1.0"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout_seconds: int = 120,
        max_tokens: int = 4096,
        max_retries: int = 3,
        min_request_interval: float = 1.0,
        context_length_exceeded_behavior: str = "truncate",
        extra_headers: Optional[dict] = None,
    ) -> None:
        headers = {"User-Agent": self._USER_AGENT}
        if extra_headers:
            headers.update(extra_headers)
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY", ""),
            base_url=self._BASE_URL,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            max_retries=max_retries,
            extra_headers=headers,
        )
        self._context_length_exceeded_behavior = context_length_exceeded_behavior
        self._min_request_interval = min_request_interval
        self._last_call_time: float = 0.0

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Call Fireworks /v1/chat/completions using streaming.

        Always sends ``stream=True`` to avoid Fireworks' 4096-token
        non-streaming cap. The SSE response is reassembled into a single
        ``ChatResponse`` via ``_parse_sse_stream()``.
        """
        # Proactive rate limiting: enforce minimum interval between requests.
        if self._min_request_interval > 0:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
        self._last_call_time = time.monotonic()

        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "stream": True,
            "context_length_exceeded_behavior": self._context_length_exceeded_behavior,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self._extra_headers,
        }
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response = _parse_sse_stream(resp)
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = (exc.fp.read() if exc.fp else b"").decode()
            except Exception:
                pass
            if exc.code in (401, 403):
                raise PermissionError(f"Fireworks auth error {exc.code}: {body_text}") from exc
            if exc.code == 429:
                limit = exc.headers.get("x-ratelimit-limit", "?")
                remaining = exc.headers.get("x-ratelimit-remaining", "?")
                raise RuntimeError(
                    f"Fireworks rate limit (limit={limit}, remaining={remaining}): {body_text}"
                ) from exc
            raise RuntimeError(f"Fireworks API error {exc.code}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach Fireworks API: {exc.reason}") from exc

        msg = response["choices"][0]["message"]
        content = (msg.get("content") or "").strip()

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args))

        tokens_used = response.get("usage", {}).get("completion_tokens", 0)
        return ChatResponse(
            content=content,
            model=response.get("model", self._model),
            tokens_used=tokens_used,
            tool_calls=tool_calls,
        )
