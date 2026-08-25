"""llama.cpp LLM backends — server (HTTP) and CLI (subprocess)."""

import json
import re
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from agent_core.llm import ChatLLMInterface, ChatMessage, ChatResponse, ToolCall


def _strip_thinking_blocks(text: str) -> str:
    """Remove thinking blocks from model output.

    Handles multiple formats:
    - <think>...</think>  — Qwen3, DeepSeek-R1, Gemma (standard)
    - <|channel>thought\\n...<channel|>  — Gemma-4-E2B
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
    return text.strip()


def _format_chatml(messages: list[ChatMessage]) -> str:
    """Format messages as ChatML prompt.

    ChatML is understood by most fine-tuned models. If your model uses
    a different template (e.g. Llama 3's <|start_header_id|>), swap
    this function or add a chat_template parameter.

    Returns:
        Single prompt string ending with assistant turn open.
    """
    msgs = [f'<|im_start|>{x.role}\n{x.content}<|im_end|>' for x in messages] \
        + ['<|im_start|>assistant\n']

    return '\n'.join(msgs)


class LlamaCppServerLLM(ChatLLMInterface):
    """Talks to a running llama-server via /v1/chat/completions.

    Start the server first:
        llama-server -m model.gguf --port 7788

    The server handles chat templates automatically based on the model's
    metadata, so no manual formatting needed.
    """

    # Sampling defaults mirror llama-cli's, which the OpenAI /v1 endpoint does
    # NOT apply on its own. Without repeat_penalty especially, small models loop
    # ("repeats the same thing every answer"). These are sent on every request;
    # override any of them via the `sampling` arg (e.g. from YAML) per model.
    _DEFAULT_SAMPLING = {
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
    }

    def __init__(self, port: int = 7788, host: str = "localhost",
                 base_url: Optional[str] = None, timeout_seconds: int = 120,
                 max_tokens: int = 512, model: str = "local",
                 sampling: Optional[dict] = None) -> None:
        """Initialize server backend.

        Args:
            port: Port where llama-server is listening.
            host: Host where llama-server is listening (default localhost;
                set to reach a server on another machine).
            base_url: Full base URL override (e.g. "http://pi.local:7788").
                Takes precedence over host/port.
            timeout_seconds: HTTP request timeout. Set high enough for
                slow generation on large prompts.
            max_tokens: Default max output tokens per request.
            model: Label reported as the model name (llama-server serves one
                model and echoes this back).
            sampling: Optional sampling overrides merged over _DEFAULT_SAMPLING
                (top_p, top_k, min_p, repeat_penalty, repeat_last_n). Small
                models especially need repeat_penalty to avoid degenerate loops.
        """
        self._base_url = (base_url.rstrip("/") if base_url
                          else f"http://{host}:{port}")
        self._sampling = {**self._DEFAULT_SAMPLING, **(sampling or {})}
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens
        self._model = model

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Send chat completion request to llama-server.

        POST {base_url}/v1/chat/completions with:
            {"messages": [...], "max_tokens": ..., "temperature": ...}

        When tools are provided, they are sent as OpenAI-compatible function
        schemas. The server may respond with tool_calls instead of text.

        Args:
            max_tokens: Override for this call. 0 means use instance default.
            tools: Optional tool schemas for native function calling.

        Raises:
            ConnectionError: If server is not reachable.
            RuntimeError: If server returns an error response.
        """
        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens

        req_map = {'messages': [{'content': msg.content, 'role': msg.role} for msg in messages],
                   'max_tokens': effective_max_tokens,
                   'temperature': temperature,
                   'cache_prompt': False,
                   **self._sampling}
        if tools:
            req_map['tools'] = [
                {'type': 'function', 'function': schema}
                for schema in tools
            ]
            req_map['tool_choice'] = 'auto'
        req_body = json.dumps(req_map).encode()
        try:
            req = urllib.request.Request(f'{self._base_url}/v1/chat/completions',
                                         headers = {'Content-Type': 'application/json'},
                                         data = req_body)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.getcode() >= 400:
                    raise RuntimeError()
                raw_body = resp.read().decode('utf-8')
                response = json.loads(raw_body)
                msg = response["choices"][0]["message"]
                content = _strip_thinking_blocks(msg.get("content") or "")
                tool_calls = []
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        args = json.loads(args)
                    tool_calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
                return ChatResponse(
                    content=content,
                    model=response["model"],
                    tokens_used=response["usage"]["completion_tokens"],
                    tool_calls=tool_calls,
                )
        except urllib.error.URLError as ex:
            raise ConnectionError(ex)

    def call_raw(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """Call /v1/chat/completions with raw OpenAI-format message dicts.

        This is what the assistant layer uses (unlike chat()): it accepts raw
        dicts so callers can include role="tool" messages with tool_call_id and
        assistant messages carrying tool_calls. llama-server's endpoint is
        OpenAI-compatible, so the raw messages pass through unchanged and the
        response is already OpenAI-shaped. Thinking blocks (<think>…</think>,
        emitted by small/reasoning models) are stripped from the content.

        Returns the raw completion dict:
            {"choices": [{"message": {...}}], "usage": {...}}
        """
        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        body: dict = {
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "cache_prompt": False,
            **self._sampling,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": s} for s in tools]
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(body).encode(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = (exc.fp.read() if exc.fp else b"").decode()
            except Exception:
                pass
            raise RuntimeError(f"llama-server error {exc.code}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach llama-server at {self._base_url}: {exc.reason}"
            ) from exc

        # Strip thinking blocks from the assistant content, if any.
        try:
            msg = response["choices"][0]["message"]
            if msg.get("content"):
                msg["content"] = _strip_thinking_blocks(msg["content"])
        except (KeyError, IndexError):
            pass
        return response

    def is_available(self) -> bool:
        """Check if llama-server is running and ready.

        GET {base_url}/health — returns 200 with {"status": "ok"}
        when the model is loaded and ready.
        """
        try:
            with urllib.request.urlopen(f'{self._base_url}/health', timeout = 5) as req:
                if req.getcode() == 200:
                    return True
                return False
        except:
            return False

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_name(self) -> str:
        """Return model name from server, or fallback."""
        return f"llama-server@{self._base_url}"


class LlamaCppCliLLM(ChatLLMInterface):
    """Runs llama-cli as a subprocess for each chat() call.

    No server needed — just the model file and llama-cli binary.
    Slower (loads model each time) but simpler for quick experiments.

    Usage:
        llm = LlamaCppCliLLM(model_path="/path/to/model.gguf")
    """

    def __init__(
        self,
        model_path: str | Path,
        llama_cli: str = "llama-cli",
        timeout_seconds: int = 120,
        max_tokens: int = 512,
    ) -> None:
        """Initialize CLI backend.

        Args:
            model_path: Path to the .gguf model file.
            llama_cli: Name or path of the llama-cli binary.
                Defaults to "llama-cli" (assumes it's on PATH).
            timeout_seconds: Subprocess timeout.
            max_tokens: Default max output tokens per request.
        """
        self._model_path = str(model_path)
        self._llama_cli = llama_cli
        self._timeout = timeout_seconds
        self._max_tokens = max_tokens

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 0,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Run llama-cli and capture output.

        Steps:
            1. Format messages into a single prompt via _format_chatml()
            2. Run: llama-cli -m {model} -p {prompt} -n {max_tokens}
                    --temp {temperature} --no-display-prompt
            3. Capture stdout → ChatResponse

        Note: tools parameter is accepted for interface compatibility but
        ignored — CLI mode does not support native tool calling.

        Args:
            max_tokens: Override for this call. 0 means use instance default.

        Raises:
            FileNotFoundError: If llama-cli or model not found.
            RuntimeError: If subprocess fails.
            TimeoutError: If generation exceeds timeout.
        """
        effective_max_tokens = max_tokens if max_tokens > 0 else self._max_tokens
        prompt = _format_chatml(messages)
        cmd = [self._llama_cli, '-m', self._model_path, '-p', prompt,
               '-n', str(effective_max_tokens), '--temp', str(temperature),
               '--no-display-prompt']

        try:
            completed = subprocess.run(cmd,
                                       capture_output = True,
                                       text = True,
                                       timeout = self._timeout)
            if completed.returncode > 0:
                raise RuntimeError(completed.stderr)
            return ChatResponse(content = _strip_thinking_blocks(completed.stdout.strip()),
                                model = self.model_name)
        except subprocess.TimeoutExpired as ex:
            raise TimeoutError(ex)

    def is_available(self) -> bool:
        """Check that model file exists and llama-cli is findable."""
        return Path(self._model_path).is_file() and \
            shutil.which(self._llama_cli) is not None

    @property
    def model_name(self) -> str:
        """Model filename without directory."""
        return Path(self._model_path).name
