"""
Tests for WorkingMemory class - hot cache for recent episodes
"""
import pytest
from datetime import datetime
from agent_memory.working_memory import WorkingMemory
from agent_memory.memory_store import Episode


def create_test_episode(id_num: int, context: str = None) -> Episode:
    """Helper to create test episodes"""
    return Episode(
        id=id_num,
        timestamp=datetime.now(),
        context=context or f"context_{id_num}",
        action=f"action_{id_num}",
        outcome=f"outcome_{id_num}",
        success_score=0.8,
        tags=["test"]
    )


def test_working_memory_initialization():
    """Test WorkingMemory initializes correctly"""
    wm = WorkingMemory(max_size=20)

    assert len(wm.buffer) == 0
    assert wm.session_id is None
    assert wm.session_started is None


def test_add_episode():
    """Test adding episodes to working memory"""
    wm = WorkingMemory(max_size=5)

    ep1 = create_test_episode(1)
    ep2 = create_test_episode(2)

    wm.add(ep1)
    assert len(wm.buffer) == 1

    wm.add(ep2)
    assert len(wm.buffer) == 2


def test_get_all():
    """Test getting all episodes from working memory"""
    wm = WorkingMemory(max_size=5)

    episodes = [create_test_episode(i) for i in range(1, 4)]  # 1, 2, 3
    for ep in episodes:
        wm.add(ep)

    all_eps = wm.get_all()
    assert len(all_eps) == 3
    assert all_eps[0].id == 1
    assert all_eps[1].id == 2
    assert all_eps[2].id == 3


def test_auto_eviction_when_full():
    """Test that oldest episodes are auto-evicted when buffer is full"""
    wm = WorkingMemory(max_size=3)

    # Add 5 episodes to a buffer with max_size=3
    for i in range(1, 6):
        wm.add(create_test_episode(i))

    all_eps = wm.get_all()

    # Should only have 3 episodes (the last 3 added)
    assert len(all_eps) == 3

    # Should be episodes 3, 4, 5 (oldest 1, 2 were evicted)
    assert all_eps[0].id == 3
    assert all_eps[1].id == 4
    assert all_eps[2].id == 5


def test_get_recent():
    """Test getting last N episodes"""
    wm = WorkingMemory(max_size=10)

    # Add 5 episodes
    for i in range(1, 6):
        wm.add(create_test_episode(i))

    # Get last 3
    recent = wm.get_recent(n=3)
    assert len(recent) == 3
    assert recent[0].id == 3
    assert recent[1].id == 4
    assert recent[2].id == 5


def test_get_recent_more_than_available():
    """Test get_recent when n > buffer size"""
    wm = WorkingMemory(max_size=10)

    # Add only 3 episodes
    for i in range(1, 4):
        wm.add(create_test_episode(i))

    # Request 10 (more than available)
    recent = wm.get_recent(n=10)

    # Should return all 3 available
    assert len(recent) == 3
    assert recent[0].id == 1
    assert recent[1].id == 2
    assert recent[2].id == 3


def test_get_recent_with_zero():
    """Test get_recent with n=0"""
    wm = WorkingMemory(max_size=10)

    for i in range(1, 4):
        wm.add(create_test_episode(i))

    recent = wm.get_recent(n=0)
    assert len(recent) == 0


def test_clear():
    """Test clearing working memory"""
    wm = WorkingMemory(max_size=5)

    # Add some episodes
    for i in range(3):
        wm.add(create_test_episode(i))

    assert len(wm.buffer) == 3

    # Clear
    wm.clear()

    assert len(wm.buffer) == 0


def test_start_session():
    """Test starting a new session"""
    wm = WorkingMemory(max_size=5)

    # Add some episodes
    for i in range(3):
        wm.add(create_test_episode(i))

    # Start new session
    session_id = "session_123"
    wm.start_session(session_id)

    # Buffer should be cleared
    assert len(wm.buffer) == 0

    # Session info should be set
    assert wm.session_id == session_id
    assert wm.session_started is not None
    assert isinstance(wm.session_started, datetime)


def test_start_session_clears_previous():
    """Test that starting new session clears previous session data"""
    wm = WorkingMemory(max_size=5)

    # First session
    wm.start_session("session_1")
    wm.add(create_test_episode(1))
    wm.add(create_test_episode(2))

    assert len(wm.buffer) == 2
    assert wm.session_id == "session_1"

    # Second session
    wm.start_session("session_2")

    # Buffer cleared, session ID updated
    assert len(wm.buffer) == 0
    assert wm.session_id == "session_2"


def test_working_memory_is_fifo():
    """Test that working memory follows FIFO eviction"""
    wm = WorkingMemory(max_size=3)

    # Add episodes 1, 2, 3
    for i in range(1, 4):
        wm.add(create_test_episode(i))

    all_eps = wm.get_all()
    assert [ep.id for ep in all_eps] == [1, 2, 3]

    # Add episode 4 (should evict 1)
    wm.add(create_test_episode(4))
    all_eps = wm.get_all()
    assert [ep.id for ep in all_eps] == [2, 3, 4]

    # Add episode 5 (should evict 2)
    wm.add(create_test_episode(5))
    all_eps = wm.get_all()
    assert [ep.id for ep in all_eps] == [3, 4, 5]


def test_empty_buffer_operations():
    """Test operations on empty buffer"""
    wm = WorkingMemory(max_size=5)

    # Get all from empty buffer
    assert wm.get_all() == []

    # Get recent from empty buffer
    assert wm.get_recent(n=5) == []

    # Clear empty buffer (should not error)
    wm.clear()
    assert len(wm.buffer) == 0


def test_custom_max_size():
    """Test different max_size configurations"""
    # Very small buffer
    wm_small = WorkingMemory(max_size=1)
    wm_small.add(create_test_episode(1))
    wm_small.add(create_test_episode(2))
    assert len(wm_small.buffer) == 1
    assert wm_small.get_all()[0].id == 2

    # Large buffer
    wm_large = WorkingMemory(max_size=100)
    for i in range(50):
        wm_large.add(create_test_episode(i))
    assert len(wm_large.buffer) == 50
