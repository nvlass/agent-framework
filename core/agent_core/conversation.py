"""ConversationBus — bounded, turn-taking dialogue between two agents.

Where the mailbox (see ``mailbox.py``) is fire-and-forget async messaging,
this is a *structured conversation*: two agents exchange turns on a topic with
a lifecycle (active → closed), explicit turn-taking, and a hard guarantee that
the exchange terminates — either because a participant signals ``done`` or
because a ``max_turns`` cap is hit. The cap is the backstop against two
autonomous agents ping-ponging forever.

It shares the same physical file as the mailbox — "the shared channel" — and
touches nothing else: each agent's own memory / journal / soul stay private.
Sharing is explicit (you take a turn); it is never ambient (no agent reads
another's private store).

Fits interval-ticking daemons: no simultaneous liveness is required. Each tick
an agent calls ``needs_attention()``; if it is its turn in some conversation it
replies, otherwise it moves on. The other agent picks the thread up on its own
next tick.

Turn model (2-party)::

    ada.open(peer="smith", topic="scag", message="What's your read on X?")
        # turn 1 by ada; state=active; next_turn=smith
    smith.reply(cid, "I think Y because...")
        # turn 2 by smith; next_turn=ada
    ada.reply(cid, "Good point — settling on Z.", done=True)
        # turn 3 by ada; state=closed (reason=done)

The turn claim in ``reply`` is atomic (BEGIN IMMEDIATE): if it is not your turn,
or the conversation is already closed, the reply is refused — so two daemons
acting on the same tick cannot both take the same turn.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    topic         TEXT NOT NULL DEFAULT '',
    initiator     TEXT NOT NULL,
    peer          TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'closed'
    next_turn     TEXT,                             -- whose turn (NULL when closed)
    turn_count    INTEGER NOT NULL DEFAULT 0,
    max_turns     INTEGER NOT NULL DEFAULT 6,
    closed_reason TEXT NOT NULL DEFAULT '',         -- 'done' | 'turn_limit' | 'abandoned'
    initiator_ack INTEGER NOT NULL DEFAULT 0,       -- has initiator seen the latest state
    peer_ack      INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS conversation_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    from_agent      TEXT NOT NULL,
    turn_no         INTEGER NOT NULL,
    message         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_turn ON conversations(next_turn, state);
CREATE INDEX IF NOT EXISTS idx_convturns ON conversation_turns(conversation_id, turn_no);
"""


class ConversationError(RuntimeError):
    """Raised when a conversation operation is invalid (not your turn, closed…)."""


class ConversationBus:
    """Bounded turn-taking conversations for one agent, over a shared SQLite file.

    Args:
        db_path:    The shared channel file (same one the mailbox uses).
        agent_name: This agent's identity — used for turn ownership and routing.
    """

    def __init__(self, db_path: str | Path, agent_name: str) -> None:
        self._name = agent_name
        self._lock = threading.Lock()
        # isolation_level=None → autocommit; we manage BEGIN IMMEDIATE ourselves
        # so the read-modify-write in reply() is atomic across processes.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False,
                                     isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Transaction helper
    # ------------------------------------------------------------------

    @contextmanager
    def _immediate(self):
        """A cross-process-safe write transaction (BEGIN IMMEDIATE)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, peer: str, message: str, topic: str = "",
             max_turns: int = 6) -> dict:
        """Open a conversation with *peer* and take the first turn.

        Returns the conversation dict. Raises ConversationError on bad input.
        """
        if peer == self._name:
            raise ConversationError("Cannot open a conversation with yourself.")
        if not (message or "").strip():
            raise ConversationError("Opening message is empty.")
        if max_turns < 2:
            raise ConversationError("max_turns must be at least 2 (an opener + a reply).")

        with self._immediate() as conn:
            cur = conn.execute(
                "INSERT INTO conversations "
                "(topic, initiator, peer, state, next_turn, turn_count, max_turns, "
                " initiator_ack, peer_ack) "
                "VALUES (?, ?, ?, 'active', ?, 1, ?, 1, 0)",
                (topic, self._name, peer, peer, max_turns),
            )
            cid = cur.lastrowid
            conn.execute(
                "INSERT INTO conversation_turns "
                "(conversation_id, from_agent, turn_no, message) VALUES (?, ?, 1, ?)",
                (cid, self._name, message),
            )
        return self.get(cid)

    def reply(self, conversation_id: int, message: str, done: bool = False) -> dict:
        """Take your turn in a conversation. Atomic turn-claim + termination.

        Refuses (ConversationError) if the conversation is closed or it is not
        your turn. Closes the conversation if *done* is set or the turn cap is
        reached. Returns the updated conversation dict.
        """
        if not (message or "").strip():
            raise ConversationError("Reply message is empty.")

        with self._immediate() as conn:
            row = conn.execute(
                "SELECT state, next_turn, initiator, peer, turn_count, max_turns "
                "FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise ConversationError(f"No conversation #{conversation_id}.")
            state, next_turn, initiator, peer, turn_count, max_turns = row
            if state != "active":
                raise ConversationError(
                    f"Conversation #{conversation_id} is already closed.")
            if next_turn != self._name:
                raise ConversationError(
                    f"Not your turn in #{conversation_id} (waiting on {next_turn}).")

            other = initiator if self._name == peer else peer
            new_turn_no = turn_count + 1
            conn.execute(
                "INSERT INTO conversation_turns "
                "(conversation_id, from_agent, turn_no, message) VALUES (?, ?, ?, ?)",
                (conversation_id, self._name, new_turn_no, message),
            )

            if done:
                new_state, new_next, reason = "closed", None, "done"
            elif new_turn_no >= max_turns:
                new_state, new_next, reason = "closed", None, "turn_limit"
            else:
                new_state, new_next, reason = "active", other, ""

            # The replier has, by definition, seen the state up to here; the
            # other party has new activity to see → reset their ack.
            my_ack_col = "initiator_ack" if self._name == initiator else "peer_ack"
            other_ack_col = "peer_ack" if self._name == initiator else "initiator_ack"
            conn.execute(
                f"UPDATE conversations SET state=?, next_turn=?, turn_count=?, "
                f"closed_reason=?, updated_at=CURRENT_TIMESTAMP, "
                f"{my_ack_col}=1, {other_ack_col}=0 WHERE id=?",
                (new_state, new_next, new_turn_no, reason, conversation_id),
            )
        return self.get(conversation_id)

    def abandon(self, conversation_id: int, reason: str = "abandoned") -> dict:
        """Close a conversation you participate in without a further reply."""
        with self._immediate() as conn:
            row = conn.execute(
                "SELECT initiator, peer, state FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise ConversationError(f"No conversation #{conversation_id}.")
            initiator, peer, state = row
            if self._name not in (initiator, peer):
                raise ConversationError(
                    f"You are not a participant in #{conversation_id}.")
            if state == "closed":
                return self.get(conversation_id)
            conn.execute(
                "UPDATE conversations SET state='closed', next_turn=NULL, "
                "closed_reason=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (reason or "abandoned", conversation_id),
            )
        return self.get(conversation_id)

    def acknowledge(self, conversation_id: int) -> None:
        """Mark that you have seen the conversation's current state.

        Removes it from ``needs_attention`` until there is new activity. Call
        after reading a reply or a closed outcome you don't need to answer.
        """
        col = self._ack_column(conversation_id)
        if col is None:
            return
        with self._immediate() as conn:
            conn.execute(
                f"UPDATE conversations SET {col}=1 WHERE id=?", (conversation_id,))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def needs_attention(self, limit: int = 20) -> list[dict]:
        """Conversations that want this agent's attention this tick.

        Two kinds, both surfaced so the agent can act each tick:
          - active and it is *your* turn → you should reply.
          - closed (or a fresh reply) that you have not acknowledged → read it.
        Each dict carries ``attention`` = 'your_turn' or 'unread'.
        """
        rows = self._conn.execute(
            "SELECT id, initiator, peer, state, next_turn, initiator_ack, peer_ack "
            "FROM conversations "
            "WHERE (initiator = ? OR peer = ?) "
            "ORDER BY updated_at ASC LIMIT ?",
            (self._name, self._name, limit),
        ).fetchall()
        out = []
        for cid, initiator, peer, state, next_turn, i_ack, p_ack in rows:
            my_ack = i_ack if self._name == initiator else p_ack
            if state == "active" and next_turn == self._name:
                d = self.get(cid)
                d["attention"] = "your_turn"
                out.append(d)
            elif not my_ack:
                d = self.get(cid)
                d["attention"] = "unread"
                out.append(d)
        return out

    def get(self, conversation_id: int) -> dict:
        """Return the conversation's metadata plus its latest turn."""
        row = self._conn.execute(
            "SELECT id, created_at, updated_at, topic, initiator, peer, state, "
            "next_turn, turn_count, max_turns, closed_reason "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ConversationError(f"No conversation #{conversation_id}.")
        last = self._conn.execute(
            "SELECT from_agent, message FROM conversation_turns "
            "WHERE conversation_id = ? ORDER BY turn_no DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return {
            "id": row[0], "created_at": row[1], "updated_at": row[2],
            "topic": row[3], "initiator": row[4], "peer": row[5],
            "state": row[6], "next_turn": row[7], "turn_count": row[8],
            "max_turns": row[9], "closed_reason": row[10],
            "last_from": last[0] if last else None,
            "last_message": last[1] if last else None,
        }

    def history(self, conversation_id: int) -> list[dict]:
        """All turns of a conversation, oldest first."""
        rows = self._conn.execute(
            "SELECT id, created_at, from_agent, turn_no, message "
            "FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_no ASC",
            (conversation_id,),
        ).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "from_agent": r[2],
             "turn_no": r[3], "message": r[4]}
            for r in rows
        ]

    def format_attention(self, limit: int = 3) -> str:
        """Compact summary of conversations needing attention, for prompt injection.

        Empty string when there is nothing to attend to.
        """
        items = self.needs_attention(limit=limit)
        if not items:
            return ""
        lines = [f"--- {len(items)} conversation(s) awaiting you ---"]
        for c in items:
            topic = f" [{c['topic']}]" if c["topic"] else ""
            if c["attention"] == "your_turn":
                lines.append(
                    f"[#{c['id']}]{topic} your turn — {c['last_from']} said: "
                    f"{_clip(c['last_message'])} "
                    f"(turn {c['turn_count']}/{c['max_turns']})"
                )
            else:
                state = ("closed: " + c["closed_reason"]) if c["state"] == "closed" else "new reply"
                lines.append(
                    f"[#{c['id']}]{topic} {state} — {c['last_from']} said: "
                    f"{_clip(c['last_message'])}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ack_column(self, conversation_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT initiator, peer FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        initiator, peer = row
        if self._name == initiator:
            return "initiator_ack"
        if self._name == peer:
            return "peer_ack"
        return None

    def __repr__(self) -> str:
        return f"ConversationBus(name={self._name!r})"


def _clip(text: str | None, n: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"
