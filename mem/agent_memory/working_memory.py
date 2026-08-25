from collections import deque
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .memory_store import Episode

class WorkingMemory:
    """
    In-memory buffer for recent episodes (hot cache)
    Automatically evicts oldest when capacity reached
    """
    def __init__(self, max_size: int = 20):
        self.buffer = deque(maxlen=max_size)
        self.session_id: Optional[str] = None
        self.session_started: Optional[datetime] = None

    def add(self, episode: "Episode") -> None:
        """Add episode to working memory (auto-evicts oldest if full)"""
        self.buffer.append(episode)

    def get_all(self) -> List["Episode"]:
        """Get all episodes in working memory (most recent last)"""
        return list(self.buffer)

    def get_recent(self, n: int = 10) -> List["Episode"]:
        """Get last N episodes"""
        if n == 0:
            return []
        return list(self.buffer)[-n:]

    def clear(self) -> None:
        """Clear working memory (e.g., new session)"""
        self.buffer.clear()

    def start_session(self, session_id: str) -> None:
        """Start a new session, clearing previous working memory"""
        self.clear()
        self.session_id = session_id
        self.session_started = datetime.now()
