"""Tests for LLMReasoner."""

from agent_mind.goals.model import Goal
from agent_core.llm import ChatResponse, MockChatLLM, ToolCall
from agent_core.reasoner import LLMReasoner
from agent_core.prompt import PromptAssembler


class TestParse:
    def test_action_with_args(self):
        text = 'Thought: need to read\nAction: read_file\nAction Args: {"path": "/tmp/x"}'
        r = LLMReasoner._parse(text)
        assert r.thought == "need to read"
        assert r.action == "read_file"
        assert r.action_args == {"path": "/tmp/x"}
        assert r.answer is None

    def test_answer(self):
        text = "Thought: I know the answer\nAnswer: 42"
        r = LLMReasoner._parse(text)
        assert r.thought == "I know the answer"
        assert r.answer == "42"
        assert r.action is None

    def test_action_wins_over_answer(self):
        """When both Action and Answer are present, action wins."""
        text = "Thought: hmm\nAnswer: done\nAction: foo\nAction Args: {}"
        r = LLMReasoner._parse(text)
        assert r.action == "foo"
        assert r.answer is None

    def test_answer_alone_still_works(self):
        """Answer without Action is still treated as final answer."""
        text = "Thought: hmm\nAnswer: done"
        r = LLMReasoner._parse(text)
        assert r.answer == "done"
        assert r.action is None

    def test_malformed_falls_back(self):
        text = "just some text"
        r = LLMReasoner._parse(text)
        assert r.thought == "just some text"
        assert r.answer == "just some text"

    def test_bad_json_args(self):
        text = "Thought: x\nAction: foo\nAction Args: not json"
        r = LLMReasoner._parse(text)
        assert r.action == "foo"
        assert r.action_args == {}

    def test_action_no_args(self):
        text = "Thought: try it\nAction: list_dir"
        r = LLMReasoner._parse(text)
        assert r.action == "list_dir"
        assert r.action_args == {}

    def test_empty_string(self):
        r = LLMReasoner._parse("")
        assert r.answer == ""

    def test_code_fenced_args(self):
        text = (
            "Thought: sending email\n"
            "Action: send_email\n"
            "Action Args:\n"
            "```\n"
            '{"subject": "Hello", "body": "World"}\n'
            "```"
        )
        r = LLMReasoner._parse(text)
        assert r.action == "send_email"
        assert r.action_args == {"subject": "Hello", "body": "World"}

    def test_code_fenced_json_tag(self):
        text = (
            "Thought: x\n"
            "Action: foo\n"
            "Action Args:\n"
            "```json\n"
            '{"a": 1}\n'
            "```"
        )
        r = LLMReasoner._parse(text)
        assert r.action_args == {"a": 1}

    def test_multiline_args(self):
        text = (
            "Thought: composing\n"
            "Action: send_email\n"
            "Action Args:\n"
            '{"subject": "News",\n'
            ' "body": "stuff"}'
        )
        r = LLMReasoner._parse(text)
        assert r.action == "send_email"
        assert r.action_args == {"subject": "News", "body": "stuff"}

    def test_args_on_same_line_still_works(self):
        text = 'Thought: x\nAction: foo\nAction Args: {"k": "v"}'
        r = LLMReasoner._parse(text)
        assert r.action_args == {"k": "v"}


class TestReason:
    def test_calls_llm_and_parses(self):
        llm = MockChatLLM(["Thought: thinking\nAnswer: yes"])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test goal")
        result = reasoner.reason(goal, [], [])
        assert result.answer == "yes"
        assert len(llm.calls) == 1

    def test_soul_forwarded_to_assembler(self):
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        reasoner = LLMReasoner(llm, soul="be kind")
        goal = Goal(description="test")
        reasoner.reason(goal, [], [])
        # Soul should appear in the system message sent to LLM
        system_msg = llm.calls[0][0]
        assert "be kind" in system_msg.content

    def test_observations_forwarded(self):
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        reasoner.reason(goal, ["obs1"], [])
        user_msg = llm.calls[0][1]
        assert "obs1" in user_msg.content

    def test_tools_forwarded(self):
        """Tools are passed via the API (native tool calling), not in prompt."""
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        tools = [{"name": "mytool", "description": "does stuff"}]
        reasoner.reason(goal, [], tools)
        # Tools should be passed via tools param, not in prompt text
        assert llm.tools_passed[0] == tools

    def test_custom_assembler(self):
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        asm = PromptAssembler(
            system_template="CUSTOM{soul_section}{tools_section}",
            user_template="{goal}{observations_section}",
        )
        reasoner = LLMReasoner(llm, assembler=asm)
        goal = Goal(description="test")
        reasoner.reason(goal, [], [])
        assert llm.calls[0][0].content.startswith("CUSTOM")


class TestNativeToolCalling:
    """Tests for native tool calling path in LLMReasoner."""

    def test_tool_call_response_used_directly(self):
        """When LLM returns tool_calls, reasoner uses them (no text parsing)."""
        tc = ToolCall(name="read_file", arguments={"path": "/tmp/x"})
        resp = ChatResponse(content="Let me read that file", model="mock", tool_calls=[tc])
        llm = MockChatLLM([resp])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        tools = [{"name": "read_file", "description": "read a file"}]
        result = reasoner.reason(goal, [], tools)
        assert result.action == "read_file"
        assert result.action_args == {"path": "/tmp/x"}
        assert result.answer is None
        assert result.thought == "Let me read that file"

    def test_tool_call_synthetic_thought(self):
        """When content is empty, a synthetic thought is generated."""
        tc = ToolCall(name="list_dir", arguments={})
        resp = ChatResponse(content="", model="mock", tool_calls=[tc])
        llm = MockChatLLM([resp])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        result = reasoner.reason(goal, [], [{"name": "list_dir", "description": "list"}])
        assert result.action == "list_dir"
        assert "list_dir" in result.thought

    def test_no_tool_calls_falls_back_to_parsing(self):
        """When no tool_calls, falls back to text parsing."""
        llm = MockChatLLM(["Thought: thinking\nAction: foo\nAction Args: {}"])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        result = reasoner.reason(goal, [], [{"name": "foo", "description": "x"}])
        assert result.action == "foo"

    def test_tools_passed_to_llm(self):
        """Tool schemas are forwarded to llm.chat() as tools param."""
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        tools = [{"name": "mytool", "description": "stuff"}]
        reasoner.reason(goal, [], tools)
        assert llm.tools_passed[0] == tools

    def test_no_tools_passes_none(self):
        """When no tools available, tools=None is passed to LLM."""
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        reasoner.reason(goal, [], [])
        assert llm.tools_passed[0] is None

    def test_first_tool_call_used_when_multiple(self):
        """When LLM returns multiple tool_calls, only the first is used."""
        tc1 = ToolCall(name="first", arguments={"a": 1})
        tc2 = ToolCall(name="second", arguments={"b": 2})
        resp = ChatResponse(content="", model="mock", tool_calls=[tc1, tc2])
        llm = MockChatLLM([resp])
        reasoner = LLMReasoner(llm)
        goal = Goal(description="test")
        result = reasoner.reason(goal, [], [{"name": "first", "description": "x"}])
        assert result.action == "first"
        assert result.action_args == {"a": 1}
