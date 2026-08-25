"""Tests for AnthropicLLM and OpenAILLM."""

import json
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, call, patch

import pytest

from agent_core.llm import ChatMessage, ChatResponse, ToolCall
from agent_core.llm_cloud import AnthropicLLM, OpenAILLM


def _make_429(retry_after: float = None):
    """Build a urllib HTTPError simulating a 429 rate-limit response."""
    from http.client import HTTPMessage
    hdrs = HTTPMessage()
    if retry_after is not None:
        hdrs["retry-after"] = str(retry_after)
    return urllib.error.HTTPError(
        url="https://example.com",
        code=429,
        msg="Too Many Requests",
        hdrs=hdrs,
        fp=BytesIO(b'{"error": "rate_limit_exceeded"}'),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_http_response(body: dict, status: int = 200):
    """Return a mock that behaves like urllib urlopen context manager."""
    raw = json.dumps(body).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = raw
    mock_resp.getcode.return_value = status
    return mock_resp


def _anthropic_text_response(text: str, model: str = "claude-opus-4-6") -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _anthropic_tool_response(tool_name: str, tool_input: dict) -> dict:
    return {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll use a tool."},
            {"type": "tool_use", "id": "toolu_01", "name": tool_name, "input": tool_input},
        ],
        "model": "claude-opus-4-6",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 20, "output_tokens": 15},
    }


def _openai_text_response(text: str, model: str = "gpt-4o") -> dict:
    return {
        "id": "chatcmpl-01",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _openai_tool_response(tool_name: str, tool_args: dict) -> dict:
    return {
        "id": "chatcmpl-02",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_01",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------

class TestAnthropicLLMBasic:
    def test_model_name(self):
        llm = AnthropicLLM(model="claude-haiku-4-5-20251001", api_key="sk-test")
        assert llm.model_name == "claude-haiku-4-5-20251001"

    def test_is_available_with_key(self):
        assert AnthropicLLM(api_key="sk-test").is_available()

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert not AnthropicLLM(api_key="").is_available()

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        llm = AnthropicLLM()
        assert llm.is_available()


class TestAnthropicLLMChat:
    def _llm(self) -> AnthropicLLM:
        return AnthropicLLM(model="claude-opus-4-6", api_key="sk-test")

    def test_text_response(self):
        llm = self._llm()
        fake_resp = _fake_http_response(_anthropic_text_response("Hello!"))
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = llm.chat([ChatMessage(role="user", content="Hi")])
        assert result.content == "Hello!"
        assert result.model == "claude-opus-4-6"
        assert result.tokens_used == 5
        assert result.tool_calls == []

    def test_system_message_extracted(self):
        """System message must be sent as top-level 'system' field, not in messages."""
        llm = self._llm()
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ])
        assert captured_body["system"] == "You are helpful."
        # System should NOT appear in messages array
        for m in captured_body["messages"]:
            assert m["role"] != "system"

    def test_no_system_message(self):
        """When no system message, 'system' key should not be in body."""
        llm = self._llm()
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="Hello")])
        assert "system" not in captured_body

    def test_tool_response(self):
        llm = self._llm()
        fake_resp = _fake_http_response(
            _anthropic_tool_response("get_weather", {"location": "London"})
        )
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = llm.chat([ChatMessage(role="user", content="Weather?")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "London"}
        assert result.content == "I'll use a tool."

    def test_tools_converted_to_anthropic_format(self):
        """OpenAI-format tool schemas must be converted: parameters → input_schema."""
        llm = self._llm()
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        openai_tools = [{
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }]
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="search")], tools=openai_tools)
        anthropic_tools = captured_body["tools"]
        assert len(anthropic_tools) == 1
        assert "input_schema" in anthropic_tools[0]
        assert "parameters" not in anthropic_tools[0]
        assert anthropic_tools[0]["input_schema"]["type"] == "object"

    def test_auth_header_set(self):
        llm = self._llm()
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="Hi")])
        # urllib capitalizes first char of header names
        assert "X-api-key" in captured_headers or "x-api-key" in captured_headers or \
               any("api-key" in k.lower() for k in captured_headers)

    def test_http_401_raises_permission_error(self):
        import urllib.error
        llm = self._llm()
        http_err = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=BytesIO(b'{"error": "invalid api key"}'),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(PermissionError):
                llm.chat([ChatMessage(role="user", content="hi")])

    def test_http_500_raises_runtime_error(self):
        import urllib.error
        llm = self._llm()
        http_err = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=BytesIO(b'{"error": "internal"}'),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(RuntimeError):
                llm.chat([ChatMessage(role="user", content="hi")])

    def test_connection_error(self):
        import urllib.error
        llm = self._llm()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            with pytest.raises(ConnectionError):
                llm.chat([ChatMessage(role="user", content="hi")])

    def test_anthropic_version_header(self):
        llm = self._llm()
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="Hi")])
        header_keys_lower = {k.lower() for k in captured_headers}
        assert "anthropic-version" in header_keys_lower

    def test_constructor_max_tokens_used_as_default(self):
        """max_tokens set in constructor is sent when chat() uses default (0)."""
        llm = AnthropicLLM(model="claude-opus-4-6", api_key="sk-test", max_tokens=2048)
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["max_tokens"] == 2048

    def test_per_call_max_tokens_overrides_constructor(self):
        """Explicit max_tokens in chat() overrides the constructor default."""
        llm = AnthropicLLM(model="claude-opus-4-6", api_key="sk-test", max_tokens=2048)
        fake_resp = _fake_http_response(_anthropic_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")], max_tokens=100)
        assert captured_body["max_tokens"] == 100


# ---------------------------------------------------------------------------
# OpenAILLM
# ---------------------------------------------------------------------------

class TestOpenAILLMBasic:
    def test_model_name(self):
        llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")
        assert llm.model_name == "gpt-4o-mini"

    def test_is_available_with_key(self):
        assert OpenAILLM(api_key="sk-test").is_available()

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert not OpenAILLM(api_key="").is_available()

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        llm = OpenAILLM()
        assert llm.is_available()


class TestOpenAILLMChat:
    def _llm(self, **kwargs) -> OpenAILLM:
        return OpenAILLM(model="gpt-4o", api_key="sk-test", **kwargs)

    def test_text_response(self):
        llm = self._llm()
        fake_resp = _fake_http_response(_openai_text_response("Hello!"))
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = llm.chat([ChatMessage(role="user", content="Hi")])
        assert result.content == "Hello!"
        assert result.model == "gpt-4o"
        assert result.tokens_used == 5
        assert result.tool_calls == []

    def test_tool_response(self):
        llm = self._llm()
        fake_resp = _fake_http_response(
            _openai_tool_response("get_weather", {"location": "Paris"})
        )
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = llm.chat([ChatMessage(role="user", content="Weather?")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "Paris"}

    def test_tool_args_already_dict(self):
        """Handle cases where tool arguments are returned as a dict, not a JSON string."""
        llm = self._llm()
        response = _openai_tool_response("foo", {"x": 1})
        # Patch: make arguments already a dict (some APIs do this)
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {"x": 1}
        fake_resp = _fake_http_response(response)
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = llm.chat([ChatMessage(role="user", content="go")])
        assert result.tool_calls[0].arguments == {"x": 1}

    def test_tools_sent_in_openai_format(self):
        llm = self._llm()
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        tools = [{"name": "search", "description": "search", "parameters": {}}]
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")], tools=tools)
        assert captured_body["tools"][0]["type"] == "function"
        assert captured_body["tool_choice"] == "auto"

    def test_auth_header_bearer(self):
        llm = self._llm()
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        auth = captured_headers.get("Authorization") or captured_headers.get("authorization", "")
        assert auth.startswith("Bearer ")

    def test_custom_base_url(self):
        """OpenAILLM hits the custom base URL, enabling Groq/Together/etc."""
        llm = self._llm(base_url="https://api.groq.com/openai")
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_urls = []
        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert "groq.com" in captured_urls[0]

    def test_system_message_in_messages_array(self):
        """OpenAI accepts system role inside the messages array (unlike Anthropic)."""
        llm = self._llm()
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hello"),
            ])
        roles = [m["role"] for m in captured_body["messages"]]
        assert "system" in roles

    def test_http_401_raises_permission_error(self):
        import urllib.error
        llm = self._llm()
        http_err = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=BytesIO(b'{"error": {"message": "invalid api key"}}'),
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(PermissionError):
                llm.chat([ChatMessage(role="user", content="hi")])

    def test_connection_error(self):
        import urllib.error
        llm = self._llm()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(ConnectionError):
                llm.chat([ChatMessage(role="user", content="hi")])

    def test_null_content_with_tool_calls(self):
        """content may be null when tool_calls are present — should return empty string."""
        llm = self._llm()
        fake_resp = _fake_http_response(_openai_tool_response("foo", {}))
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = llm.chat([ChatMessage(role="user", content="go")])
        assert result.content == ""
        assert len(result.tool_calls) == 1

    def test_constructor_max_tokens_used_as_default(self):
        """max_tokens set in constructor is sent when chat() uses default (0)."""
        llm = OpenAILLM(model="gpt-4o", api_key="sk-test", max_tokens=4096)
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["max_tokens"] == 4096

    def test_per_call_max_tokens_overrides_constructor(self):
        """Explicit max_tokens in chat() overrides the constructor default."""
        llm = OpenAILLM(model="gpt-4o", api_key="sk-test", max_tokens=4096)
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")], max_tokens=200)
        assert captured_body["max_tokens"] == 200


# ---------------------------------------------------------------------------
# Rate-limit retry behaviour (shared via _post_with_retry)
# ---------------------------------------------------------------------------

class TestRateLimitRetry:
    """429 responses trigger retry with sleep; other errors raise immediately."""

    def _anthropic(self, max_retries=2) -> AnthropicLLM:
        return AnthropicLLM(model="claude-opus-4-6", api_key="sk-test", max_retries=max_retries)

    def _openai(self, max_retries=2) -> OpenAILLM:
        return OpenAILLM(model="gpt-4o", api_key="sk-test", max_retries=max_retries)

    def test_anthropic_retries_on_429_then_succeeds(self):
        """One 429 followed by success → result returned, sleep called once."""
        llm = self._anthropic()
        ok_resp = _fake_http_response(_anthropic_text_response("ok"))
        side_effects = [_make_429(retry_after=1.0), ok_resp]
        with patch("urllib.request.urlopen", side_effect=side_effects), \
             patch("time.sleep") as mock_sleep:
            result = llm.chat([ChatMessage(role="user", content="hi")])
        assert result.content == "ok"
        mock_sleep.assert_called_once_with(1.0)

    def test_openai_retries_on_429_then_succeeds(self):
        llm = self._openai()
        ok_resp = _fake_http_response(_openai_text_response("ok"))
        side_effects = [_make_429(retry_after=2.0), ok_resp]
        with patch("urllib.request.urlopen", side_effect=side_effects), \
             patch("time.sleep") as mock_sleep:
            result = llm.chat([ChatMessage(role="user", content="hi")])
        assert result.content == "ok"
        mock_sleep.assert_called_once_with(2.0)

    def test_raises_after_max_retries_exhausted(self):
        """After max_retries 429s, RuntimeError is raised."""
        llm = self._anthropic(max_retries=2)
        side_effects = [_make_429(1.0), _make_429(1.0), _make_429(1.0)]
        with patch("urllib.request.urlopen", side_effect=side_effects), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="rate limit"):
                llm.chat([ChatMessage(role="user", content="hi")])

    def test_exponential_backoff_when_no_retry_after_header(self):
        """Without retry-after header, sleep uses 5s * 2^attempt backoff."""
        llm = self._anthropic(max_retries=2)
        ok_resp = _fake_http_response(_anthropic_text_response("ok"))
        side_effects = [_make_429(), _make_429(), ok_resp]
        with patch("urllib.request.urlopen", side_effect=side_effects), \
             patch("time.sleep") as mock_sleep:
            llm.chat([ChatMessage(role="user", content="hi")])
        assert mock_sleep.call_args_list == [call(5.0), call(10.0)]

    def test_max_retries_zero_raises_immediately_on_429(self):
        """max_retries=0 means no retry — 429 raises RuntimeError immediately."""
        llm = self._anthropic(max_retries=0)
        with patch("urllib.request.urlopen", side_effect=[_make_429(1.0)]), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                llm.chat([ChatMessage(role="user", content="hi")])
        mock_sleep.assert_not_called()

    def test_non_429_errors_not_retried(self):
        """500 errors raise immediately without retry."""
        llm = self._anthropic(max_retries=3)
        http_500 = urllib.error.HTTPError(
            url="https://x", code=500, msg="err", hdrs=None, fp=BytesIO(b"internal")
        )
        with patch("urllib.request.urlopen", side_effect=[http_500]), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                llm.chat([ChatMessage(role="user", content="hi")])
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAILLM extra_headers
# ---------------------------------------------------------------------------

class TestOpenAIExtraHeaders:
    def test_extra_headers_sent(self):
        llm = OpenAILLM(model="gpt-4o", api_key="sk-test",
                        extra_headers={"User-Agent": "my-agent/1.0", "X-Custom": "yes"})
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        keys_lower = {k.lower() for k in captured_headers}
        assert "user-agent" in keys_lower
        assert "x-custom" in keys_lower

    def test_no_extra_headers_by_default(self):
        """Default OpenAILLM only sends Content-Type and Authorization."""
        llm = OpenAILLM(model="gpt-4o", api_key="sk-test")
        fake_resp = _fake_http_response(_openai_text_response("ok"))
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return fake_resp
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        keys_lower = {k.lower() for k in captured_headers}
        assert "content-type" in keys_lower
        assert "authorization" in keys_lower


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------

def _sse_lines(*chunks: dict, done: bool = True) -> list[bytes]:
    """Build a list of SSE byte lines from chunk dicts."""
    lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks]
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


def _sse_resp(lines: list[bytes]):
    """Mock response that iterates SSE lines."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.__iter__ = lambda s: iter(lines)
    return mock_resp


def _sse_text(text: str, model: str = "accounts/fireworks/models/deepseek-v4-pro") -> list[bytes]:
    """SSE stream for a simple text response, split across two content chunks."""
    half = len(text) // 2
    return _sse_lines(
        {"model": model, "choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"content": text[:half]}}]},
        {"choices": [{"delta": {"content": text[half:]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )


def _sse_tool(tool_name: str, tool_args: dict,
              model: str = "accounts/fireworks/models/deepseek-v4-pro") -> list[bytes]:
    """SSE stream for a tool call response, arguments split across two chunks."""
    args_str = json.dumps(tool_args)
    half = len(args_str) // 2
    return _sse_lines(
        {"model": model, "choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_01", "type": "function",
             "function": {"name": tool_name, "arguments": ""}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": args_str[:half]}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": args_str[half:]}}
        ]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 15}},
    )


# ---------------------------------------------------------------------------
# FireworksLLM
# ---------------------------------------------------------------------------

class TestFireworksLLM:
    def _llm(self, **kwargs):
        from agent_core.llm_cloud import FireworksLLM
        return FireworksLLM(api_key="fw-test", **kwargs)

    def test_default_model(self):
        from agent_core.llm_cloud import FireworksLLM
        llm = FireworksLLM(api_key="fw-test")
        assert "fireworks" in llm.model_name

    def test_api_key_from_env(self, monkeypatch):
        from agent_core.llm_cloud import FireworksLLM
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-env-key")
        llm = FireworksLLM()
        assert llm.is_available()

    def test_text_response(self):
        llm = self._llm()
        with patch("urllib.request.urlopen", return_value=_sse_resp(_sse_text("Hello!"))):
            result = llm.chat([ChatMessage(role="user", content="hi")])
        assert result.content == "Hello!"
        assert result.tool_calls == []

    def test_tool_call_response(self):
        llm = self._llm()
        with patch("urllib.request.urlopen",
                   return_value=_sse_resp(_sse_tool("send_email", {"subject": "s", "body": "b"}))):
            result = llm.chat([ChatMessage(role="user", content="send")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "send_email"
        assert result.tool_calls[0].arguments == {"subject": "s", "body": "b"}

    def test_tool_args_reassembled_from_fragments(self):
        """Tool call arguments arriving as fragments are correctly joined."""
        llm = self._llm()
        # Use a longer args string so it genuinely splits across chunks
        long_body = "x" * 200
        with patch("urllib.request.urlopen",
                   return_value=_sse_resp(_sse_tool("send_email", {"body": long_body}))):
            result = llm.chat([ChatMessage(role="user", content="send")])
        assert result.tool_calls[0].arguments["body"] == long_body

    def test_stream_true_in_request_body(self):
        """Fireworks always sends stream=True."""
        llm = self._llm()
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["stream"] is True

    def test_base_url_is_fireworks(self):
        llm = self._llm()
        captured_urls = []
        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert "fireworks.ai" in captured_urls[0]

    def test_user_agent_header_sent(self):
        """Fireworks requires User-Agent to pass Cloudflare."""
        llm = self._llm()
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        keys_lower = {k.lower() for k in captured_headers}
        assert "user-agent" in keys_lower

    def test_context_length_exceeded_behavior_in_body(self):
        llm = self._llm(context_length_exceeded_behavior="error")
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["context_length_exceeded_behavior"] == "error"

    def test_default_context_length_exceeded_behavior_is_truncate(self):
        llm = self._llm()
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["context_length_exceeded_behavior"] == "truncate"

    def test_default_max_tokens_is_4096(self):
        llm = self._llm()
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["max_tokens"] == 4096

    def test_extra_headers_merged_with_user_agent(self):
        llm = self._llm(extra_headers={"X-Custom": "value"})
        captured_headers = {}
        def fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        keys_lower = {k.lower() for k in captured_headers}
        assert "user-agent" in keys_lower
        assert "x-custom" in keys_lower

    def test_tokens_used_from_final_chunk(self):
        llm = self._llm()
        with patch("urllib.request.urlopen", return_value=_sse_resp(_sse_text("hi"))):
            result = llm.chat([ChatMessage(role="user", content="hi")])
        assert result.tokens_used == 5  # from usage in final chunk

    def test_high_max_tokens_allowed(self):
        """Streaming removes the 4096 cap — large max_tokens accepted."""
        llm = self._llm(max_tokens=16384)
        captured_body = {}
        def fake_urlopen(req, timeout=None):
            captured_body.update(json.loads(req.data.decode()))
            return _sse_resp(_sse_text("ok"))
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            llm.chat([ChatMessage(role="user", content="hi")])
        assert captured_body["max_tokens"] == 16384

    def test_rate_limit_interval_respected(self):
        """Second call sleeps to maintain min_request_interval."""
        import time
        llm = self._llm(min_request_interval=0.5)
        with patch("urllib.request.urlopen", return_value=_sse_resp(_sse_text("ok"))):
            llm.chat([ChatMessage(role="user", content="hi")])
        # Simulate a very recent first call so the second must wait
        llm._last_call_time = time.monotonic()
        slept = []
        with patch("time.sleep", side_effect=slept.append):
            with patch("urllib.request.urlopen", return_value=_sse_resp(_sse_text("ok"))):
                llm.chat([ChatMessage(role="user", content="hi")])
        assert slept and slept[0] > 0

    def test_rate_limit_disabled(self):
        """min_request_interval=0 disables throttling."""
        import time
        llm = self._llm(min_request_interval=0)
        llm._last_call_time = time.monotonic()
        slept = []
        with patch("time.sleep", side_effect=slept.append):
            with patch("urllib.request.urlopen", return_value=_sse_resp(_sse_text("ok"))):
                llm.chat([ChatMessage(role="user", content="hi")])
        assert not slept
