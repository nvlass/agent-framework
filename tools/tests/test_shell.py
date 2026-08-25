"""Tests for template-based command tool factory.

Run with:
    cd tools/
    python -m pytest tests/test_shell.py -v
"""

import pytest

from agent_tools.tools.shell import extract_param, create_command_tool
from agent_tools.core.definition import ToolParameter, PermissionLevel


class TestExtractParam:
    """Test the extract_param helper."""

    # --- Plain strings (not parameters) ---

    def test_plain_string(self):
        """A regular string is returned as-is."""
        assert extract_param("git", {}) == "git"

    def test_plain_string_with_kwargs(self):
        """Plain string unaffected even when kwargs exist."""
        assert extract_param("fetch", {"branch": "main"}) == "fetch"

    def test_empty_string(self):
        """Empty string returned as-is."""
        assert extract_param("", {}) == ""

    def test_string_with_dashes(self):
        """Dashes are not valid in Python identifiers — not a param."""
        assert extract_param("{not-a-param}", {}) == "{not-a-param}"

    def test_string_with_spaces(self):
        """Spaces inside braces — not a valid param."""
        assert extract_param("{not a param}", {}) == "{not a param}"

    def test_partial_braces(self):
        """Only opening brace — not a param."""
        assert extract_param("{orphan", {}) == "{orphan"

    def test_braces_in_middle(self):
        """Braces embedded in text — not a param placeholder."""
        assert extract_param("pre{x}post", {}) == "pre{x}post"

    # --- Parameter substitution ---

    def test_simple_param(self):
        """A {name} placeholder is replaced with the kwarg value."""
        assert extract_param("{branch}", {"branch": "main"}) == "main"

    def test_param_value_is_stringified(self):
        """Non-string kwarg values are converted to str."""
        assert extract_param("{count}", {"count": 42}) == "42"

    def test_param_missing_from_kwargs(self):
        """A valid placeholder with no matching kwarg should raise KeyError."""
        with pytest.raises(KeyError):
            extract_param("{branch}", {})

    def test_param_with_underscores(self):
        """Python identifiers can have underscores."""
        assert extract_param("{file_path}", {"file_path": "/tmp/x"}) == "/tmp/x"

    def test_param_with_digits(self):
        """Python identifiers can contain digits (not leading)."""
        assert extract_param("{arg2}", {"arg2": "value"}) == "value"

    def test_param_leading_digit_not_valid(self):
        """{2arg} is not a valid Python identifier — returned as-is."""
        assert extract_param("{2arg}", {}) == "{2arg}"

    def test_param_single_char(self):
        """Single character param names work."""
        assert extract_param("{x}", {"x": "y"}) == "y"


class TestCreateCommandTool:
    """Integration tests for create_command_tool."""

    # --- Factory ---

    def test_returns_tool_definition(self):
        """Factory returns a ToolDefinition."""
        tool = create_command_tool(name="test", command=["echo"])
        assert tool.name == "test"
        assert tool.permission == PermissionLevel.DANGEROUS

    def test_auto_description(self):
        """Description is auto-generated from command if not provided."""
        tool = create_command_tool(name="test", command=["git", "fetch"])
        assert tool.description == "Execute: git fetch"

    def test_custom_description(self):
        """Explicit description overrides auto-generation."""
        tool = create_command_tool(
            name="test", command=["git", "fetch"], description="Fetch remote"
        )
        assert tool.description == "Fetch remote"

    # --- Execution: basic commands ---

    def test_simple_command(self):
        """Run a command with no parameters."""
        tool = create_command_tool(name="echo_hi", command=["echo", "hello"])
        result = tool.execute()
        assert result.strip() == "hello"

    def test_command_with_param(self):
        """Run a command with a substituted parameter."""
        tool = create_command_tool(
            name="echo_msg",
            command=["echo", "{msg}"],
            parameters=[
                ToolParameter(name="msg", type="string", description="Message"),
            ],
        )
        result = tool.execute(msg="world")
        assert result.strip() == "world"

    def test_command_multiple_params(self):
        """Multiple parameters are substituted correctly."""
        tool = create_command_tool(
            name="printf_test",
            command=["printf", "%s %s", "{a}", "{b}"],
            parameters=[
                ToolParameter(name="a", type="string", description="First"),
                ToolParameter(name="b", type="string", description="Second"),
            ],
        )
        result = tool.execute(a="hello", b="world")
        assert result == "hello world"

    def test_working_dir(self):
        """working_dir is passed to subprocess."""
        tool = create_command_tool(
            name="pwd_test", command=["pwd"], working_dir="/tmp"
        )
        result = tool.execute()
        # macOS /tmp is a symlink to /private/tmp
        assert result.strip() in ("/tmp", "/private/tmp")

    # --- Error handling ---

    def test_nonzero_exit_raises(self):
        """Non-zero exit code raises RuntimeError."""
        tool = create_command_tool(
            name="fail", command=["false"]
        )
        with pytest.raises(RuntimeError, match="command failed"):
            tool.execute()

    def test_timeout_raises(self):
        """Command exceeding timeout raises TimeoutExpired."""
        import subprocess
        tool = create_command_tool(
            name="slow", command=["sleep", "10"], timeout_seconds=1
        )
        with pytest.raises(subprocess.TimeoutExpired):
            tool.execute()

    # --- Dangerous character rejection ---

    def test_rejects_semicolon(self):
        """Semicolons in parameter values are rejected."""
        tool = create_command_tool(
            name="test",
            command=["echo", "{msg}"],
            parameters=[
                ToolParameter(name="msg", type="string", description="Message"),
            ],
        )
        with pytest.raises(ValueError, match="forbidden characters"):
            tool.execute(msg="hello; rm -rf /")

    def test_rejects_pipe(self):
        """Pipes in parameter values are rejected."""
        tool = create_command_tool(
            name="test",
            command=["echo", "{msg}"],
            parameters=[
                ToolParameter(name="msg", type="string", description="Message"),
            ],
        )
        with pytest.raises(ValueError, match="forbidden characters"):
            tool.execute(msg="hello | bash")

    def test_rejects_newline(self):
        """Newlines in parameter values are rejected."""
        tool = create_command_tool(
            name="test",
            command=["echo", "{msg}"],
            parameters=[
                ToolParameter(name="msg", type="string", description="Message"),
            ],
        )
        with pytest.raises(ValueError, match="forbidden characters"):
            tool.execute(msg="hello\nworld")

    def test_rejects_dollar(self):
        """Dollar signs in parameter values are rejected."""
        tool = create_command_tool(
            name="test",
            command=["echo", "{msg}"],
            parameters=[
                ToolParameter(name="msg", type="string", description="Message"),
            ],
        )
        with pytest.raises(ValueError, match="forbidden characters"):
            tool.execute(msg="$(whoami)")

    def test_allows_clean_values(self):
        """Normal values without metacharacters pass through fine."""
        tool = create_command_tool(
            name="test",
            command=["echo", "{msg}"],
            parameters=[
                ToolParameter(name="msg", type="string", description="Message"),
            ],
        )
        result = tool.execute(msg="feature/my-branch")
        assert result.strip() == "feature/my-branch"
