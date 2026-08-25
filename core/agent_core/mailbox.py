"""AgentMailbox — persistent, cross-process inter-agent message bus.

Agents running as separate OS processes communicate by reading and writing
to a shared SQLite file.  WAL mode makes concurrent access safe: multiple
readers are always allowed, and writes are serialised by SQLite itself.

Usage (sender)::

    mailbox = AgentMailbox("/shared/agents.db", agent_name="ada")
    mailbox.send(to="smith", message="Did you finish the analysis?", topic="research")

Usage (recipient, in the turn loop)::

    mailbox = AgentMailbox("/shared/agents.db", agent_name="smith")
    unread = mailbox.inbox(unread_only=True)
    for msg in unread:
        buffer.add_background_note(
            f"[Mailbox from {msg['from_agent']}] {msg['message']}"
        )
        mailbox.mark_read(msg["id"])

Agents know each other by name.  Name is set in the agent's YAML config::

    name: agent-smith
    mailbox_db: /shared/agent_mailbox.db

Both agents must point to the same physical file for cross-process messaging.
Within a single process, multiple AgentMailbox instances on the same file
work too (e.g. spawned child agents).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mailbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    topic       TEXT DEFAULT '',
    message     TEXT NOT NULL,
    reply_to    INTEGER REFERENCES mailbox(id),
    read        INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mailbox_inbox
    ON mailbox(to_agent, read, created_at);
CREATE INDEX IF NOT EXISTS idx_mailbox_thread
    ON mailbox(reply_to);
"""


class AgentMailbox:
    """Persistent inter-agent message queue backed by SQLite.

    Thread-safe: uses a lock around writes; reads are safe without it under
    WAL mode.

    Args:
        db_path:    Path to the shared SQLite file.  Created if absent.
        agent_name: This agent's identity — used as ``from_agent`` on sends
                    and ``to_agent`` filter on reads.
    """

    def __init__(self, db_path: str | Path, agent_name: str) -> None:
        self._name = agent_name
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """This agent's name, used for routing."""
        return self._name

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def send(
        self,
        to: str,
        message: str,
        topic: str = "",
        reply_to: int | None = None,
    ) -> int:
        """Send a message to another agent.

        Args:
            to:       Recipient agent name.
            message:  Message body.
            topic:    Optional short label (e.g. ``"soul-check"``, ``"research"``).
            reply_to: ID of the message being replied to (for threading).

        Returns:
            The new message's integer ID.
        """
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO mailbox (from_agent, to_agent, topic, message, reply_to) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._name, to, topic, message, reply_to),
            )
            self._conn.commit()
            return cur.lastrowid

    def mark_read(self, msg_id: int) -> bool:
        """Mark a single message as read.  Returns True if the message existed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mailbox SET read = 1 WHERE id = ? AND to_agent = ?",
                (msg_id, self._name),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_all_read(self) -> int:
        """Mark all unread inbox messages as read.  Returns the count updated."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mailbox SET read = 1 WHERE to_agent = ? AND read = 0",
                (self._name,),
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def inbox(
        self,
        unread_only: bool = True,
        from_agent: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return messages addressed to this agent.

        Args:
            unread_only: If True, return only unread messages.
            from_agent:  Optional filter by sender.
            limit:       Maximum number of messages to return.

        Returns:
            List of message dicts, oldest first.
        """
        where = ["to_agent = ?"]
        params: list = [self._name]
        if unread_only:
            where.append("read = 0")
        if from_agent:
            where.append("from_agent = ?")
            params.append(from_agent)
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT id, created_at, from_agent, to_agent, topic, message, reply_to, read "
            f"FROM mailbox WHERE {' AND '.join(where)} "
            f"ORDER BY created_at ASC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_unread(self) -> int:
        """Number of unread messages in this agent's inbox."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM mailbox WHERE to_agent = ? AND read = 0",
            (self._name,),
        ).fetchone()
        return row[0] if row else 0

    def get_message(self, msg_id: int) -> dict | None:
        """Fetch a single message by ID (regardless of recipient)."""
        row = self._conn.execute(
            "SELECT id, created_at, from_agent, to_agent, topic, message, reply_to, read "
            "FROM mailbox WHERE id = ?",
            (msg_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_thread(self, msg_id: int) -> list[dict]:
        """Return a message and all direct replies, oldest first."""
        root = self.get_message(msg_id)
        if not root:
            return []
        rows = self._conn.execute(
            "SELECT id, created_at, from_agent, to_agent, topic, message, reply_to, read "
            "FROM mailbox WHERE id = ? OR reply_to = ? ORDER BY created_at ASC",
            (msg_id, msg_id),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def sent(self, limit: int = 50) -> list[dict]:
        """Return messages sent by this agent, newest first."""
        rows = self._conn.execute(
            "SELECT id, created_at, from_agent, to_agent, topic, message, reply_to, read "
            "FROM mailbox WHERE from_agent = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (self._name, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_inbox(self, limit: int = 5) -> str:
        """Format unread messages as a compact string for system prompt injection.

        Returns an empty string when there are no unread messages.
        """
        msgs = self.inbox(unread_only=True, limit=limit)
        if not msgs:
            return ""
        lines = [f"--- {len(msgs)} message(s) in your mailbox ---"]
        for m in msgs:
            ts = (m["created_at"] or "")[:16]
            topic_str = f" [{m['topic']}]" if m.get("topic") else ""
            reply_str = f" (reply to #{m['reply_to']})" if m.get("reply_to") else ""
            lines.append(
                f"[#{m['id']}] {ts} From {m['from_agent']}{topic_str}{reply_str}: {m['message']}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AgentMailbox(name={self._name!r}, unread={self.count_unread()})"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _row_to_dict(row: tuple) -> dict:
    return {
        "id": row[0],
        "created_at": row[1],
        "from_agent": row[2],
        "to_agent": row[3],
        "topic": row[4],
        "message": row[5],
        "reply_to": row[6],
        "read": bool(row[7]),
    }
