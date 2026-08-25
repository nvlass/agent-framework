"""
Tests for ShortTermMemory - TTL cache for recent episodes
"""
import pytest
import time
from datetime import datetime
from agent_memory.short_term_memory import TTLCache, ShortTermMemory
from agent_memory.memory_store import Episode


def create_test_episode(id_num: int) -> Episode:
    """Helper to create test episodes"""
    return Episode(
        id=id_num,
        timestamp=datetime.now(),
        context=f"context_{id_num}",
        action=f"action_{id_num}",
        outcome=f"outcome_{id_num}",
        success_score=0.8,
        tags=["test"]
    )


class TestTTLCache:
    """Test TTL cache implementation"""

    def test_cache_initialization(self):
        """Test cache initializes correctly"""
        cache = TTLCache(ttl_seconds=60)
        assert cache.ttl_seconds == 60
        assert cache.size() == 0

    def test_cache_miss_loads_value(self):
        """Test cache miss calls loader function"""
        cache = TTLCache(ttl_seconds=60)

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return "loaded_value"

        result = cache.get("key1", loader)

        assert result == "loaded_value"
        assert call_count == 1
        assert cache.size() == 1

    def test_cache_hit_returns_cached_value(self):
        """Test cache hit returns cached value without calling loader"""
        cache = TTLCache(ttl_seconds=60)

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return f"loaded_{call_count}"

        # First call - miss
        result1 = cache.get("key1", loader)
        assert result1 == "loaded_1"
        assert call_count == 1

        # Second call - hit
        result2 = cache.get("key1", loader)
        assert result2 == "loaded_1"  # Same cached value
        assert call_count == 1  # Loader not called again

    def test_cache_expiry(self):
        """Test cache entries expire after TTL"""
        cache = TTLCache(ttl_seconds=1)  # 1 second TTL

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return f"loaded_{call_count}"

        # Load initial value
        result1 = cache.get("key1", loader)
        assert result1 == "loaded_1"
        assert call_count == 1

        # Wait for expiry
        time.sleep(1.1)

        # Should reload after expiry
        result2 = cache.get("key1", loader)
        assert result2 == "loaded_2"  # New value
        assert call_count == 2  # Loader called again

    def test_cache_invalidate(self):
        """Test manual cache invalidation"""
        cache = TTLCache(ttl_seconds=60)

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return f"loaded_{call_count}"

        # Load and cache
        cache.get("key1", loader)
        assert call_count == 1

        # Invalidate
        cache.invalidate("key1")

        # Next get should reload
        cache.get("key1", loader)
        assert call_count == 2

    def test_cache_clear(self):
        """Test clearing entire cache"""
        cache = TTLCache(ttl_seconds=60)

        # Load multiple keys
        cache.get("key1", lambda: "value1")
        cache.get("key2", lambda: "value2")
        cache.get("key3", lambda: "value3")

        assert cache.size() == 3

        # Clear
        cache.clear()

        assert cache.size() == 0

    def test_multiple_keys(self):
        """Test cache handles multiple keys correctly"""
        cache = TTLCache(ttl_seconds=60)

        # Different keys should have independent cached values
        val1 = cache.get("key1", lambda: "value1")
        val2 = cache.get("key2", lambda: "value2")
        val3 = cache.get("key3", lambda: "value3")

        assert val1 == "value1"
        assert val2 == "value2"
        assert val3 == "value3"
        assert cache.size() == 3

    def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        cache = TTLCache(ttl_seconds=1)

        # Add some entries
        cache.get("key1", lambda: "value1")
        cache.get("key2", lambda: "value2")

        assert cache.size() == 2

        # Wait for expiry
        time.sleep(1.1)

        # Add one more (won't be expired)
        cache.get("key3", lambda: "value3")

        # Cleanup should remove expired ones
        removed = cache.cleanup_expired()

        assert removed == 2  # key1 and key2 expired
        assert cache.size() == 1  # Only key3 remains


class TestShortTermMemory:
    """Test short-term memory implementation"""

    def test_initialization(self):
        """Test short-term memory initializes correctly"""
        stm = ShortTermMemory(ttl_seconds=300, default_time_window_hours=24)

        assert stm.cache.ttl_seconds == 300
        assert stm.default_time_window_hours == 24

    def test_get_recent_episodes_caches_result(self):
        """Test that get_recent_episodes caches the result"""
        stm = ShortTermMemory(ttl_seconds=60)

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return [create_test_episode(i) for i in range(3)]

        # First call - cache miss
        episodes1 = stm.get_recent_episodes(hours=24, loader=loader)
        assert len(episodes1) == 3
        assert call_count == 1

        # Second call - cache hit
        episodes2 = stm.get_recent_episodes(hours=24, loader=loader)
        assert len(episodes2) == 3
        assert call_count == 1  # Loader not called again

    def test_different_time_windows_cached_separately(self):
        """Test different time windows have separate cache entries"""
        stm = ShortTermMemory(ttl_seconds=60)

        def loader_24h():
            return [create_test_episode(i) for i in range(24)]

        def loader_12h():
            return [create_test_episode(i) for i in range(12)]

        # Load 24-hour window
        episodes_24 = stm.get_recent_episodes(hours=24, loader=loader_24h)
        assert len(episodes_24) == 24

        # Load 12-hour window (different cache key)
        episodes_12 = stm.get_recent_episodes(hours=12, loader=loader_12h)
        assert len(episodes_12) == 12

        # Both should be cached
        assert stm.cache.size() == 2

    def test_invalidate_on_new_episode(self):
        """Test cache invalidation when new episode is added"""
        stm = ShortTermMemory(ttl_seconds=60)

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return [create_test_episode(i) for i in range(3)]

        # Load and cache
        stm.get_recent_episodes(hours=24, loader=loader)
        assert call_count == 1

        # Invalidate on new episode
        stm.invalidate_on_new_episode()

        # Next call should reload
        stm.get_recent_episodes(hours=24, loader=loader)
        assert call_count == 2

    def test_invalidate_specific_window(self):
        """Test invalidating specific time window"""
        stm = ShortTermMemory(ttl_seconds=60)

        # Load two different windows
        stm.get_recent_episodes(hours=24, loader=lambda: [create_test_episode(1)])
        stm.get_recent_episodes(hours=12, loader=lambda: [create_test_episode(2)])

        assert stm.cache.size() == 2

        # Invalidate only 24-hour window
        stm.invalidate_window(hours=24)

        assert stm.cache.size() == 1

    def test_clear(self):
        """Test clearing all cached episodes"""
        stm = ShortTermMemory(ttl_seconds=60)

        # Load multiple windows
        stm.get_recent_episodes(hours=24, loader=lambda: [create_test_episode(1)])
        stm.get_recent_episodes(hours=12, loader=lambda: [create_test_episode(2)])
        stm.get_recent_episodes(hours=6, loader=lambda: [create_test_episode(3)])

        assert stm.cache.size() == 3

        # Clear
        stm.clear()

        assert stm.cache.size() == 0

    def test_cache_stats(self):
        """Test cache statistics"""
        stm = ShortTermMemory(ttl_seconds=60)

        # Initially empty
        stats = stm.cache_stats()
        assert stats["cached_queries"] == 0

        # Load some data
        stm.get_recent_episodes(hours=24, loader=lambda: [create_test_episode(1)])
        stm.get_recent_episodes(hours=12, loader=lambda: [create_test_episode(2)])

        stats = stm.cache_stats()
        assert stats["cached_queries"] == 2

    def test_default_time_window(self):
        """Test using default time window"""
        stm = ShortTermMemory(ttl_seconds=60, default_time_window_hours=48)

        def loader():
            return [create_test_episode(1)]

        # Pass None to use default
        episodes = stm.get_recent_episodes(hours=None, loader=loader)
        assert len(episodes) == 1

        # Should be cached with default window key
        assert stm.cache.size() == 1

    def test_ttl_expiry_integration(self):
        """Test that cached episodes expire after TTL"""
        stm = ShortTermMemory(ttl_seconds=1)  # 1 second TTL

        call_count = 0
        def loader():
            nonlocal call_count
            call_count += 1
            return [create_test_episode(call_count)]

        # Load and cache
        episodes1 = stm.get_recent_episodes(hours=24, loader=loader)
        assert episodes1[0].id == 1
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should reload with new data
        episodes2 = stm.get_recent_episodes(hours=24, loader=loader)
        assert episodes2[0].id == 2
        assert call_count == 2

    def test_cleanup_expired_episodes(self):
        """Test cleanup of expired episode caches"""
        stm = ShortTermMemory(ttl_seconds=1)

        # Load some data
        stm.get_recent_episodes(hours=24, loader=lambda: [create_test_episode(1)])
        stm.get_recent_episodes(hours=12, loader=lambda: [create_test_episode(2)])

        assert stm.cache.size() == 2

        # Wait for expiry
        time.sleep(1.1)

        # Load one more (won't be expired)
        stm.get_recent_episodes(hours=6, loader=lambda: [create_test_episode(3)])

        # Cleanup
        removed = stm.cleanup_expired()

        assert removed == 2  # 24h and 12h expired
        assert stm.cache.size() == 1  # Only 6h remains
