"""Tests for PromptAssembler."""

from agent_core.prompt import PromptAssembler


class TestPromptAssembler:
    def test_build_returns_two_messages(self):
        pa = PromptAssembler()
        msgs = pa.build("do X", [], [])
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_goal_in_user_message(self):
        pa = PromptAssembler()
        msgs = pa.build("find the answer", [], [])
        assert "find the answer" in msgs[1].content

    def test_soul_in_system_message(self):
        pa = PromptAssembler()
        msgs = pa.build("x", [], [], soul="I am helpful")
        assert "I am helpful" in msgs[0].content

    def test_no_soul_no_identity_section(self):
        pa = PromptAssembler()
        msgs = pa.build("x", [], [])
        assert "Identity" not in msgs[0].content

    def test_tools_in_system_message(self):
        schemas = [{"name": "read_file", "description": "Read a file"}]
        pa = PromptAssembler()
        msgs = pa.build("x", [], schemas)
        assert "read_file" in msgs[0].content

    def test_observations_in_user_message(self):
        pa = PromptAssembler()
        msgs = pa.build("x", ["saw a cat", "found key"], [])
        assert "saw a cat" in msgs[1].content
        assert "found key" in msgs[1].content

    def test_no_observations_no_section(self):
        pa = PromptAssembler()
        msgs = pa.build("x", [], [])
        assert "Observations" not in msgs[1].content

    def test_custom_templates(self):
        pa = PromptAssembler(
            system_template="SYS:{soul_section}{tools_section}",
            user_template="USR:{goal}{observations_section}",
        )
        msgs = pa.build("mygoal", [], [])
        assert msgs[0].content == "SYS:"
        assert "mygoal" in msgs[1].content

    def test_tool_parameters_formatted(self):
        schemas = [{
            "name": "search",
            "description": "Search things",
            "parameters": {
                "properties": {"query": {"type": "string", "description": "what"}},
                "required": ["query"],
            },
        }]
        pa = PromptAssembler()
        msgs = pa.build("x", [], schemas)
        assert "query" in msgs[0].content
        assert "(required)" in msgs[0].content

    def test_response_format_in_system(self):
        pa = PromptAssembler()
        msgs = pa.build("x", [], [])
        assert "Action:" in msgs[0].content
        assert "Answer:" in msgs[0].content


class TestNativeToolsPrompt:
    """Tests for native_tools=True mode."""

    def test_no_action_format_instructions(self):
        pa = PromptAssembler()
        msgs = pa.build("x", [], [{"name": "foo"}], native_tools=True)
        assert "Action:" not in msgs[0].content
        assert "Action Args:" not in msgs[0].content

    def test_no_tool_descriptions_in_prompt(self):
        pa = PromptAssembler()
        schemas = [{"name": "read_file", "description": "Read a file"}]
        msgs = pa.build("x", [], schemas, native_tools=True)
        assert "read_file" not in msgs[0].content

    def test_soul_still_included(self):
        pa = PromptAssembler()
        msgs = pa.build("x", [], [], soul="be kind", native_tools=True)
        assert "be kind" in msgs[0].content

    def test_goal_still_in_user_message(self):
        pa = PromptAssembler()
        msgs = pa.build("find answer", [], [], native_tools=True)
        assert "find answer" in msgs[1].content
