"""Tests for ChatLLMInterface, MockChatLLM, and message dataclasses."""

from agent_core.llm import ChatMessage, ChatResponse, MockChatLLM, ToolCall


class TestChatMessage:
    def test_fields(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_system_role(self):
        msg = ChatMessage(role="system", content="you are helpful")
        assert msg.role == "system"


class TestToolCall:
    def test_fields(self):
        tc = ToolCall(name="read_file", arguments={"path": "/tmp"})
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp"}

    def test_empty_args(self):
        tc = ToolCall(name="list_dir", arguments={})
        assert tc.arguments == {}


class TestChatResponse:
    def test_fields(self):
        r = ChatResponse(content="hi", model="gpt-4")
        assert r.content == "hi"
        assert r.model == "gpt-4"
        assert r.tokens_used == 0
        assert r.tool_calls == []

    def test_tokens_used(self):
        r = ChatResponse(content="x", model="m", tokens_used=42)
        assert r.tokens_used == 42

    def test_with_tool_calls(self):
        tc = ToolCall(name="foo", arguments={"a": 1})
        r = ChatResponse(content="", model="m", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "foo"


class TestMockChatLLM:
    def test_returns_responses_in_order(self):
        llm = MockChatLLM(["first", "second"])
        msgs = [ChatMessage(role="user", content="hi")]
        assert llm.chat(msgs).content == "first"
        assert llm.chat(msgs).content == "second"

    def test_fallback_repeats_last(self):
        llm = MockChatLLM(["only"])
        msgs = [ChatMessage(role="user", content="hi")]
        llm.chat(msgs)
        assert llm.chat(msgs).content == "only"

    def test_records_calls(self):
        llm = MockChatLLM(["r"])
        msgs = [ChatMessage(role="user", content="a")]
        llm.chat(msgs)
        assert len(llm.calls) == 1
        assert llm.calls[0][0].content == "a"

    def test_is_available(self):
        assert MockChatLLM(["x"]).is_available()

    def test_model_name(self):
        assert MockChatLLM(["x"]).model_name == "mock"

    def test_response_model_is_mock(self):
        llm = MockChatLLM(["hi"])
        r = llm.chat([ChatMessage(role="user", content="x")])
        assert r.model == "mock"

    def test_accepts_chat_response_objects(self):
        tc = ToolCall(name="read_file", arguments={"path": "/tmp"})
        resp = ChatResponse(content="", model="mock", tool_calls=[tc])
        llm = MockChatLLM([resp])
        r = llm.chat([ChatMessage(role="user", content="x")])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "read_file"

    def test_records_tools_passed(self):
        llm = MockChatLLM(["r"])
        tools = [{"name": "foo", "description": "bar"}]
        llm.chat([ChatMessage(role="user", content="x")], tools=tools)
        assert llm.tools_passed[0] == tools

    def test_tools_default_none(self):
        llm = MockChatLLM(["r"])
        llm.chat([ChatMessage(role="user", content="x")])
        assert llm.tools_passed[0] is None
