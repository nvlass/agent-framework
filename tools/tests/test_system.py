"""Tests for built-in system tools.

Run with:
    cd tools/
    python -m pytest tests/test_system.py -v
"""

import os
import pytest
from pathlib import Path

from agent_tools.tools.system import read_file, get_env


@pytest.fixture
def tmp_file(tmp_path):
    """Create a temporary file with known contents."""
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    return f


@pytest.fixture
def tmp_dir_with_entries(tmp_path):
    """Create a temporary directory with a mix of files and subdirs."""
    (tmp_path / "file_a.txt").write_text("a")
    (tmp_path / "file_b.py").write_text("b")
    (tmp_path / "subdir").mkdir()
    (tmp_path / ".hidden").write_text("secret")
    return tmp_path


class TestReadFile:
    """Test read_file tool."""

    def test_read_simple_file(self, tmp_file):
        result = read_file.execute(path=str(tmp_file))
        assert result == "hello world"

    def test_read_with_encoding(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes("café".encode("latin-1"))
        result = read_file.execute(path=str(f), encoding="latin-1")
        assert result == "café"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            read_file.execute(path="/nonexistent/path/file.txt")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(ValueError, match="Not a file"):
            read_file.execute(path=str(tmp_path))

    def test_file_too_large(self, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        with pytest.raises(ValueError, match="File too large"):
            read_file.execute(path=str(big))

    def test_tool_metadata(self):
        assert read_file.name == "read_file"
        from agent_tools.core.definition import PermissionLevel
        assert read_file.permission == PermissionLevel.READ

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = read_file.execute(path=str(f))
        assert result == ""


class TestGetEnv:
    """Test get_env tool."""

    def test_existing_variable(self):
        result = get_env.execute(name="PATH")
        assert result == os.environ["PATH"]

    def test_missing_variable_default(self):
        result = get_env.execute(name="AGENT_TOOLS_TEST_NONEXISTENT_VAR_XYZ")
        assert result == ""

    def test_custom_default(self):
        result = get_env.execute(
            name="AGENT_TOOLS_TEST_NONEXISTENT_VAR_XYZ", default="fallback"
        )
        assert result == "fallback"

    def test_tool_metadata(self):
        assert get_env.name == "get_env"
        from agent_tools.core.definition import PermissionLevel
        assert get_env.permission == PermissionLevel.READ


# --- Tests for list_directory (Nikos) ---
# Implement list_directory in system.py to make these pass.
# The fixture tmp_dir_with_entries creates:
#   file_a.txt, file_b.py, subdir/, .hidden

class TestListDirectory:
    """Test list_directory tool."""

    @pytest.fixture(autouse=True)
    def _import_list_directory(self):
        from agent_tools.tools.system import list_directory
        self.list_directory = list_directory

    def test_lists_entries(self, tmp_dir_with_entries):
        """Should include all visible entries."""
        result = self.list_directory.execute(path=str(tmp_dir_with_entries))
        assert "file_a.txt" in result
        assert "file_b.py" in result
        assert "subdir" in result

    def test_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.list_directory.execute(path="/nonexistent/dir")

    def test_not_a_directory(self, tmp_file):
        with pytest.raises(ValueError, match="Not a directory"):
            self.list_directory.execute(path=str(tmp_file))

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = self.list_directory.execute(path=str(empty))
        assert result.strip() == ""

    def test_tool_metadata(self):
        assert self.list_directory.name == "list_directory"
        from agent_tools.core.definition import PermissionLevel
        assert self.list_directory.permission == PermissionLevel.READ
