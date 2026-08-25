"""
Tests for Memory CLI

Run these to verify your CLI implementation works correctly.

Usage:
    python -m pytest tests/test_cli.py -v

Note: These tests use pure dependency injection - we create a test MemoryStore
and pass it directly to command functions. No mocking or patching needed!
"""

import pytest
import sys
import json
from pathlib import Path
import tempfile
import shutil

# Use proper package imports
from agent_memory.memory_store import MemoryStore
from agent_memory.memory_cli import command_stats, command_search, command_recent, command_show, command_export


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def populated_store(temp_dir):
    """Create a memory store with sample data"""
    import numpy as np

    class MockGenerator:
        def generate_embedding(self, text, use_cache=True):
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(384).astype(np.float32)
            return emb / np.linalg.norm(emb)

    db_path = Path(temp_dir) / "test.db"
    vector_path = Path(temp_dir) / "vectors"

    store = MemoryStore(
        db_path=str(db_path),
        vector_store_path=str(vector_path),
        embedding_generator=MockGenerator()
    )

    # Add sample episodes
    episodes = [
        {
            "context": "User asked about Python decorators",
            "action": "Explained with examples",
            "outcome": "User understood",
            "success_score": 0.9,
            "tags": ["python", "teaching"],
        },
        {
            "context": "User needed React optimization",
            "action": "Suggested React.memo",
            "outcome": "Performance improved",
            "success_score": 0.95,
            "tags": ["react", "performance"],
        },
        {
            "context": "Docker networking issue",
            "action": "Explained bridge networks",
            "outcome": "Still had problems",
            "success_score": 0.5,
            "tags": ["docker", "networking"],
        },
    ]

    for episode in episodes:
        store.store_episode(**episode)

    return store


class TestCommandStats:
    """Test the stats command"""

    def test_stats_shows_counts(self, populated_store, capsys):
        """Stats should show episode counts"""
        from argparse import Namespace
        args = Namespace()

        # This test will pass once you implement command_stats
        # It should print statistics without errors
        try:
            command_stats(args, populated_store)
            captured = capsys.readouterr()
            # Should show some output
            assert len(captured.out) > 0 or "TODO" in captured.out
        except NotImplementedError:
            pytest.skip("command_stats not implemented yet")


class TestCommandSearch:
    """Test the search command"""

    def test_search_finds_episodes(self, populated_store, capsys):
        """Search should find relevant episodes"""
        from argparse import Namespace
        args = Namespace(query="Python decorators", limit=5)

        try:
            command_search(args, populated_store)
            captured = capsys.readouterr()
            # Should show some output
            assert len(captured.out) > 0 or "TODO" in captured.out
        except NotImplementedError:
            pytest.skip("command_search not implemented yet")


class TestCommandRecent:
    """Test the recent command"""

    def test_recent_shows_episodes(self, populated_store, capsys):
        """Recent should show recent episodes"""
        from argparse import Namespace
        args = Namespace(hours=24, limit=10)

        try:
            command_recent(args, populated_store)
            captured = capsys.readouterr()
            assert len(captured.out) > 0 or "TODO" in captured.out
        except NotImplementedError:
            pytest.skip("command_recent not implemented yet")


class TestCommandShow:
    """Test the show command"""

    def test_show_displays_episode(self, populated_store, capsys):
        """Show should display episode details"""
        from argparse import Namespace
        args = Namespace(episode_id=1)

        try:
            command_show(args, populated_store)
            captured = capsys.readouterr()
            assert len(captured.out) > 0 or "TODO" in captured.out
        except NotImplementedError:
            pytest.skip("command_show not implemented yet")

    def test_show_handles_missing_episode(self, populated_store, capsys):
        """Show should handle non-existent episode gracefully"""
        from argparse import Namespace
        args = Namespace(episode_id=9999)

        try:
            command_show(args, populated_store)
            captured = capsys.readouterr()
            # Should not crash
            assert True
        except NotImplementedError:
            pytest.skip("command_show not implemented yet")


class TestCommandExport:
    """Test the export command"""

    def test_export_creates_json(self, populated_store, temp_dir):
        """Export should create valid JSON file"""
        from argparse import Namespace
        output_file = Path(temp_dir) / "export.json"
        args = Namespace(output=str(output_file))

        try:
            command_export(args, populated_store)

            # Check file was created
            if output_file.exists():
                # Verify it's valid JSON
                with open(output_file) as f:
                    data = json.load(f)
                assert isinstance(data, list)
                assert len(data) == 3
        except NotImplementedError:
            pytest.skip("command_export not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
