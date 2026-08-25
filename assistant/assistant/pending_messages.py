"""Pending messages — proactive thoughts the assistant wants to share.

When the assistant notices something significant during autonomous research or
reflection, it queues a message here. Messages are shown at the next session
start and optionally emailed if urgency is high.

This is append-only from the agent's side. Only the human dismisses messages.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class PendingMessages:
    """Persistent queue of assistant-initiated messages."""

    def __init__(self, db_path: str | Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_messages (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                topic    TEXT,
                message  TEXT NOT NULL,
                source   TEXT DEFAULT 'inner_voice',
                urgency  TEXT DEFAULT 'low',
                read     INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

    def add(
        self,
        topic: str,
        message: str,
        source: str = "inner_voice",
        urgency: str = "low",
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO pending_messages (topic, message, source, urgency) "
                "VALUES (?, ?, ?, ?)",
                (topic, message, source, urgency),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_unread(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, created_at, topic, message, source, urgency "
            "FROM pending_messages WHERE read = 0 ORDER BY created_at"
        ).fetchall()
        return [
            {
                "id": r[0], "created_at": r[1], "topic": r[2],
                "message": r[3], "source": r[4], "urgency": r[5],
            }
            for r in rows
        ]

    def mark_read(self, msg_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE pending_messages SET read = 1 WHERE id = ?", (msg_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count_unread(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pending_messages WHERE read = 0"
        ).fetchone()
        return row[0] if row else 0

    def get_all(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return all messages (read and unread), newest first."""
        rows = self._conn.execute(
            "SELECT id, created_at, topic, message, source, urgency, read "
            "FROM pending_messages ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [
            {
                "id": r[0], "created_at": r[1], "topic": r[2],
                "message": r[3], "source": r[4], "urgency": r[5], "read": bool(r[6]),
            }
            for r in rows
        ]

    def count_all(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM pending_messages").fetchone()
        return row[0] if row else 0

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """Search messages by text or topic, newest first."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            "SELECT id, created_at, topic, message, source, urgency, read "
            "FROM pending_messages "
            "WHERE message LIKE ? OR topic LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [
            {
                "id": r[0], "created_at": r[1], "topic": r[2],
                "message": r[3], "source": r[4], "urgency": r[5], "read": bool(r[6]),
            }
            for r in rows
        ]

    def format_unread(self) -> str:
        msgs = self.get_unread()
        if not msgs:
            return ""
        lines = [f"--- {len(msgs)} message(s) from your assistant ---"]
        for m in msgs:
            ts = (m["created_at"] or "")[:16]
            tag = f"[{m['urgency']}]" if m["urgency"] != "low" else ""
            lines.append(f"[{m['id']}] {ts} {tag} {m['message']}")
        return "\n".join(lines)
