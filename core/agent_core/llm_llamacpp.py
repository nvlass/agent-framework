"""llama.cpp LLM backends — server (HTTP) and CLI (subprocess)."""

import json
import re
import shutil
import subprocess
import urllib.request
import urllib.error
import uuid
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


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_JSON_DECODER = json.JSONDecoder()


def _loads_lenient(s: str) -> "dict | None":
    """Parse a JSON object from ``s``, recovering the breakages small models make.

    Tries, in order: strict parse; raw_decode (ignores trailing text after the
    object); then appending 1–3 closing braces (a dropped trailing ``}`` is the
    single most common small-model JSON error). Returns a dict, or None if none
    of those yield a valid object. Appending only ``}`` never fabricates content —
    it just closes an object the model forgot to close.
    """
    s = s.strip()
    if not s.startswith("{"):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:  # trailing text after a complete object (e.g. "{...} </tool_call> junk")
        obj, _ = _JSON_DECODER.raw_decode(s, 0)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for extra in range(1, 4):  # missing trailing brace(s)
        try:
            return json.loads(s + "}" * extra)
        except json.JSONDecodeError:
            continue
    return None


def _to_tool_call(obj: dict) -> "dict | None":
    """Build an OpenAI tool_call dict from a parsed ``{name, arguments}`` object."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not name:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if not isinstance(args, str):
        args = json.dumps(args)
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _extract_tool_calls(text: str) -> "tuple[list[dict], str]":
    """Lift ``<tool_call>{json}</tool_call>`` blocks out of content into OpenAI shape.

    Some models (SmolLM3, Hermes, Qwen) emit tool calls as ``<tool_call>`` XML in
    the text, and some llama.cpp builds do not parse that back into the response's
    ``tool_calls`` field — leaving the call stranded in ``content`` where an
    OpenAI client never sees it. This recovers them client-side.

    Robust to the mess small models produce: a missing/truncated ``</tool_call>``
    close tag, and slightly-malformed JSON in the body (a dropped closing brace).
    The tag between markers is parsed leniently (:func:`_loads_lenient`).

    Returns (tool_calls, remaining_text). Unparseable blocks are left in place.
    """
    calls: list[dict] = []
    out: list[str] = []
    i = 0
    while True:
        j = text.find(_TOOL_CALL_OPEN, i)
        if j == -1:
            out.append(text[i:])
            break
        out.append(text[i:j])  # keep text before the marker
        k = j + len(_TOOL_CALL_OPEN)
        close = text.find(_TOOL_CALL_CLOSE, k)
        body = text[k:close] if close != -1 else text[k:]
        call = _to_tool_call(_loads_lenient(body))
        if call:
            calls.append(call)
            i = close + len(_TOOL_CALL_CLOSE) if close != -1 else len(text)
        else:
            out.append(text[j:k])  # couldn't parse — keep marker, advance past it
            i = k
    remaining = "".join(out).strip()
    return calls, remaining


def _extract_bare_tool_call(text: str, tool_names: set) -> "dict | None":
    """Recognise a tool call emitted as bare JSON (no ``<tool_call>`` wrapper).

    Small models (e.g. SmolLM3) sometimes dump ``{"name": ..., "arguments": ...}``
    straight into content instead of wrapping it — and then neither the server's
    parser nor _extract_tool_calls (both keyed on ``<tool_call>``) catches it.

    Only converts when the parsed ``name`` matches a KNOWN tool, so ordinary prose
    or JSON-looking content is never misread as a call. Parses leniently (recovers
    a dropped brace). Returns an OpenAI tool_call dict, or None.
    """
    if not tool_names:
        return None
    obj = _loads_lenient(text)
    if not obj or obj.get("name") not in tool_names:
        return None
    return _to_tool_call(obj)


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

        # Post-process the assistant message: strip thinking blocks, and — when
        # the server left a tool call as text in content (some builds/models
        # don't structure it) — lift it into a proper tool_calls array so
        # OpenAI-format callers see a real tool call. Two shapes occur, sometimes
        # from the same model on different turns:
        #   - wrapped:  <tool_call>{json}</tool_call>   (_extract_tool_calls)
        #   - bare:     {json}  with no wrapper          (_extract_bare_tool_call)
        try:
            msg = response["choices"][0]["message"]
            if msg.get("content") and not msg.get("tool_calls"):
                content = _strip_thinking_blocks(msg["content"])
                calls, remaining = _extract_tool_calls(content)
                if not calls:
                    tool_names = {t.get("name") for t in (tools or [])
                                  if isinstance(t, dict)}
                    bare = _extract_bare_tool_call(content, tool_names)
                    if bare:
                        calls, remaining = [bare], ""
                if calls:
                    msg["tool_calls"] = calls
                    msg["content"] = remaining or None
                    response["choices"][0]["finish_reason"] = "tool_calls"
                else:
                    msg["content"] = content
            elif msg.get("content"):
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
