"""Tests for memory bridge tools.

Uses a mock MemoryTools to keep agent-tools independent of agent-memory.

Run with:
    cd tools/
    python -m pytest tests/test_memory.py -v
"""

import json
from dataclasses import dataclass
from typing import Any, Optional, List

import pytest

from agent_tools.tools.memory import create_memory_tools
from agent_tools.core.definition import PermissionLevel


@dataclass
class MockToolResult:
    """Mimics agent_memory.MemoryTools.ToolResult."""
    success: bool
    data: Any = None
    message: str = ""
    error: Optional[str] = None


class MockMemoryTools:
    """Minimal mock of agent_memory.MemoryTools."""

    def __init__(self):
        self.calls = []

    def store_memory(self, context: str, action: str, outcome: str = "",
                     tags: Optional[List[str]] = None, **kwargs) -> MockToolResult:
        self.calls.append(("store_memory", {
            "context": context, "action": action,
            "outcome": outcome, "tags": tags,
        }))
        return MockToolResult(
            success=True,
            data={"episode_id": 42, "stored": True},
        )

    def recall_similar(self, query: str, limit: int = 5,
                       **kwargs) -> MockToolResult:
        self.calls.append(("recall_similar", {"query": query, "limit": limit}))
        return MockToolResult(
            success=True,
            data={
                "memories": [
                    {"context": "similar situation", "action": "did X",
                     "similarity": 0.85},
                ],
                "count": 1,
            },
        )

    def reflect_on_recent(self, hours: int = 24,
                          focus: Optional[str] = None,
                          **kwargs) -> MockToolResult:
        self.calls.append(("reflect_on_recent", {"hours": hours, "focus": focus}))
        return MockToolResult(
            success=True,
            data={
                "reflection_id": 7,
                "insight": "Pattern found: X leads to Y",
                "actionable_takeaway": "Try Y next time",
            },
        )


class MockFailingMemoryTools:
    """Mock that returns failures."""

    def store_memory(self, **kwargs):
        return MockToolResult(success=False, error="DB connection failed")

    def recall_similar(self, **kwargs):
        return MockToolResult(success=False, error="Vector store unavailable")

    def reflect_on_recent(self, **kwargs):
        return MockToolResult(success=False, error="No LLM configured")


@pytest.fixture
def mock_memory():
    return MockMemoryTools()


@pytest.fixture
def tools(mock_memory):
    return create_memory_tools(mock_memory)


class TestCreateMemoryTools:
    """Test that create_memory_tools returns the right structure."""

    def test_returns_three_tools(self, tools):
        assert len(tools) == 3
        assert "memory_store" in tools
        assert "memory_recall" in tools
        assert "memory_reflect" in tools

    def test_store_is_write(self, tools):
        assert tools["memory_store"].permission == PermissionLevel.WRITE

    def test_recall_is_read(self, tools):
        assert tools["memory_recall"].permission == PermissionLevel.READ

    def test_reflect_is_write(self, tools):
        assert tools["memory_reflect"].permission == PermissionLevel.WRITE


class TestMemoryStore:
    """Test memory_store bridge tool."""

    def test_basic_store(self, tools, mock_memory):
        result = tools["memory_store"].execute(
            context="debugging auth", action="checked logs",
        )
        parsed = json.loads(result)
        assert parsed["episode_id"] == 42
        assert parsed["stored"] is True

    def test_store_with_tags(self, tools, mock_memory):
        tools["memory_store"].execute(
            context="test", action="test", tags="python, debugging",
        )
        call = mock_memory.calls[-1]
        assert call[1]["tags"] == ["python", "debugging"]

    def test_store_empty_tags(self, tools, mock_memory):
        tools["memory_store"].execute(
            context="test", action="test", tags="",
        )
        call = mock_memory.calls[-1]
        assert call[1]["tags"] is None

    def test_store_failure_raises(self):
        tools = create_memory_tools(MockFailingMemoryTools())
        with pytest.raises(RuntimeError, match="DB connection failed"):
            tools["memory_store"].execute(
                context="test", action="test",
            )


class TestMemoryRecall:
    """Test memory_recall bridge tool."""

    def test_basic_recall(self, tools):
        result = tools["memory_recall"].execute(query="auth problems")
        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert parsed["memories"][0]["similarity"] == 0.85

    def test_recall_with_limit(self, tools, mock_memory):
        tools["memory_recall"].execute(query="test", limit=3)
        call = mock_memory.calls[-1]
        assert call[1]["limit"] == 3

    def test_recall_failure_raises(self):
        tools = create_memory_tools(MockFailingMemoryTools())
        with pytest.raises(RuntimeError, match="Vector store unavailable"):
            tools["memory_recall"].execute(query="test")


class TestMemoryReflect:
    """Test memory_reflect bridge tool."""

    def test_basic_reflect(self, tools):
        result = tools["memory_reflect"].execute()
        parsed = json.loads(result)
        assert parsed["reflection_id"] == 7
        assert "Pattern found" in parsed["insight"]

    def test_reflect_with_focus(self, tools, mock_memory):
        tools["memory_reflect"].execute(focus="failures")
        call = mock_memory.calls[-1]
        assert call[1]["focus"] == "failures"

    def test_reflect_empty_focus_becomes_none(self, tools, mock_memory):
        tools["memory_reflect"].execute(focus="")
        call = mock_memory.calls[-1]
        assert call[1]["focus"] is None

    def test_reflect_with_hours(self, tools, mock_memory):
        tools["memory_reflect"].execute(hours=48)
        call = mock_memory.calls[-1]
        assert call[1]["hours"] == 48

    def test_reflect_failure_raises(self):
        tools = create_memory_tools(MockFailingMemoryTools())
        with pytest.raises(RuntimeError, match="No LLM configured"):
            tools["memory_reflect"].execute()
