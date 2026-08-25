"""Tests for code analysis tools.

Run with:
    cd tools/
    python -m pytest tests/test_code.py -v
"""

from agent_tools.tools.code import syntax_check


class TestSyntaxCheck:
    """Test syntax_check tool."""

    def test_valid_code(self):
        result = syntax_check.execute(code="x = 1 + 2")
        assert result == "OK"

    def test_valid_function(self):
        code = "def greet(name):\n    return f'Hello, {name}!'"
        result = syntax_check.execute(code=code)
        assert result == "OK"

    def test_valid_class(self):
        code = "class Foo:\n    def bar(self):\n        pass"
        result = syntax_check.execute(code=code)
        assert result == "OK"

    def test_empty_string(self):
        result = syntax_check.execute(code="")
        assert result == "OK"

    def test_syntax_error(self):
        result = syntax_check.execute(code="def f(")
        assert "SyntaxError" in result

    def test_syntax_error_includes_line(self):
        result = syntax_check.execute(code="x = 1\ny = \n")
        assert "line" in result

    def test_syntax_error_includes_col(self):
        result = syntax_check.execute(code="x = )")
        assert "col" in result

    def test_custom_filename(self):
        result = syntax_check.execute(code="x = )", filename="test.py")
        assert "SyntaxError" in result

    def test_indentation_error(self):
        code = "def f():\nx = 1"
        result = syntax_check.execute(code=code)
        assert "SyntaxError" in result or "indent" in result.lower()

    def test_tool_metadata(self):
        assert syntax_check.name == "syntax_check"
        from agent_tools.core.definition import PermissionLevel
        assert syntax_check.permission == PermissionLevel.SAFE
