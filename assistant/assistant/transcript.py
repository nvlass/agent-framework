"""Append-only conversation transcript.

Writes a human-readable markdown log of every conversation turn to a file.
The transcript is independent of memory — it is never summarised or compacted,
and survives across sessions as a permanent record.

Format:
    ### 2026-05-15 14:23:01

    **You:** What is the capital of France?

    **Assistant:** Paris.

    ---

System events (compaction, reflection, curiosity findings) are logged as
blockquotes so they're visually distinct but still part of the record.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path


class Transcript:
    """Append-only conversation transcript writer."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._session_started = False

    def _write(self, text: str) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(text)

    def _start_session(self) -> None:
        """Write a session header on the first write of each run."""
        if self._session_started:
            return
        self._session_started = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n\n---\n\n## Session {ts}\n\n")

    def user(self, text: str) -> None:
        """Log a user message."""
        self._start_session()
        ts = datetime.now().strftime("%H:%M:%S")
        self._write(f"### {ts}\n\n**You:** {text}\n\n")

    def assistant(self, text: str) -> None:
        """Log an assistant response."""
        self._start_session()
        ts = datetime.now().strftime("%H:%M:%S")
        self._write(f"**Assistant:** {text}\n\n")

    def tool(self, name: str, result_summary: str) -> None:
        """Log a tool call (compact — just name and truncated result)."""
        self._start_session()
        summary = result_summary[:120].replace("\n", " ")
        self._write(f"> ⚙ `{name}` → {summary}\n\n")

    def system(self, event: str) -> None:
        """Log a system event (compaction, reflection, curiosity finding)."""
        self._start_session()
        ts = datetime.now().strftime("%H:%M:%S")
        self._write(f"> [{ts}] {event}\n\n")
