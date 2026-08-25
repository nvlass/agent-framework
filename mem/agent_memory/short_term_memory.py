"""
Short-Term Memory with TTL Cache

Implements a time-based cache for recent episodes, bridging the gap between
working memory (hot cache, ~20 items) and long-term storage (everything).

Pattern: Clojure core.cache style hit/miss/evict
- Hit: Return cached value if not expired
- Miss: Call loader function, cache result
- Evict: Automatic on TTL expiry
"""

from typing import List, Optional, Callable, Any, Dict, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from .memory_store import Episode


class TTLCache:
    """
    Time-To-Live cache with Clojure core.cache semantics

    Stores values with timestamps and automatically evicts expired entries.
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize TTL cache

        Args:
            ttl_seconds: Time-to-live in seconds (default: 5 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def get(self, key: str, loader: Callable[[], Any]) -> Any:
        """
        Get value from cache or load it

        Args:
            key: Cache key
            loader: Function to call on cache miss (no arguments)

        Returns:
            Cached or freshly loaded value

        Example:
            >>> cache = TTLCache(ttl_seconds=60)
            >>> value = cache.get("key", lambda: expensive_operation())
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            age = (datetime.now() - timestamp).total_seconds()

            if age < self.ttl_seconds:
                # Cache hit - return cached value
                return value

        # Cache miss or expired - load fresh value
        value = loader()
        self._cache[key] = (value, datetime.now())
        return value

    def invalidate(self, key: str) -> None:
        """
        Manually invalidate a cache entry

        Args:
            key: Cache key to invalidate
        """
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache"""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if (now - timestamp).total_seconds() >= self.ttl_seconds
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def size(self) -> int:
        """Return number of cached entries"""
        return len(self._cache)


class ShortTermMemory:
    """
    Short-term memory cache for recent episodes

    Sits between working memory (20 items) and long-term storage (everything).
    Uses TTL caching to avoid repeated database queries for recent episodes.

    Design:
    - Default: Cache episodes from last 24 hours
    - TTL: 5 minutes (configurable)
    - Invalidation: Automatic on TTL expiry + manual on new episodes
    """

    def __init__(self, ttl_seconds: int = 300, default_time_window_hours: int = 24):
        """
        Initialize short-term memory

        Args:
            ttl_seconds: Cache TTL in seconds (default: 5 minutes)
            default_time_window_hours: Default time window for "recent" (default: 24 hours)
        """
        self.cache = TTLCache(ttl_seconds)
        self.default_time_window_hours = default_time_window_hours

    def get_recent_episodes(
        self,
        hours: Optional[int],
        loader: Callable[[], List["Episode"]]
    ) -> List["Episode"]:
        """
        Get recent episodes with TTL caching

        Uses hit/miss pattern:
        - Hit: Returns cached episodes (if < TTL)
        - Miss: Calls loader to fetch from DB and caches result

        Args:
            hours: Time window in hours (None = use default)
            loader: Function to fetch episodes from database

        Returns:
            List of recent episodes

        Example:
            >>> episodes = short_term.get_recent_episodes(
            ...     hours=24,
            ...     loader=lambda: store._fetch_recent_from_db(24)
            ... )
        """
        time_window = hours if hours is not None else self.default_time_window_hours
        key = f"recent_{time_window}h"

        return self.cache.get(key, loader)

    def invalidate_on_new_episode(self) -> None:
        """
        Invalidate cache when new episode is stored

        Call this after storing a new episode to ensure subsequent
        queries include the new episode.
        """
        self.cache.clear()

    def invalidate_window(self, hours: int) -> None:
        """
        Invalidate specific time window

        Args:
            hours: Time window to invalidate
        """
        key = f"recent_{hours}h"
        self.cache.invalidate(key)

    def clear(self) -> None:
        """Clear all cached episodes"""
        self.cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries

        Returns:
            Number of entries removed
        """
        return self.cache.cleanup_expired()

    def cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache size and other stats
        """
        return {
            "cached_queries": self.cache.size(),
        }
