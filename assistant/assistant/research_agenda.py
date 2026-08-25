"""Research agenda — SQLite-persisted focused research goals.

When the user (or agent) asks to study a topic in depth it is added here.
CuriosityEngine checks this table first and prioritises active items over the
soul's general interest list, cycling the least-researched topic first.

The agent can close a topic (mark_status to 'mature' or 'cancelled') when
it has accumulated enough understanding or the goal is no longer relevant.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_agenda (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT    NOT NULL,
    goal        TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'active',
    cycles      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    deadline    TEXT
);
"""


class ResearchAgenda:
    """Persisted list of focused research topics for the curiosity engine."""

    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API (called from tools and CuriosityEngine)
    # ------------------------------------------------------------------

    def add_topic(self, topic: str, goal: str = "", deadline: str = "") -> str:
        """Add a focused research topic. Returns a confirmation string."""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM research_agenda WHERE topic=? AND status='active'",
                (topic,),
            ).fetchone()
            if existing:
                return f"Already actively researching '{topic}' (#{existing['id']})."
            cur = conn.execute(
                "INSERT INTO research_agenda (topic, goal, deadline) VALUES (?, ?, ?)",
                (topic, goal, deadline or None),
            )
        return f"Added '{topic}' to research agenda (#{cur.lastrowid}). The curiosity engine will prioritise it."

    def get_next_topic(self) -> dict | None:
        """Return the active agenda item with fewest cycles (least explored)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, topic, goal FROM research_agenda "
                "WHERE status='active' ORDER BY cycles ASC, id ASC LIMIT 1",
            ).fetchone()
        return dict(row) if row else None

    def increment_cycles(self, topic_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE research_agenda SET cycles=cycles+1 WHERE id=?", (topic_id,)
            )

    def mark_status(self, topic_id: int, status: str) -> str:
        with self._conn() as conn:
            conn.execute(
                "UPDATE research_agenda SET status=? WHERE id=?", (status, topic_id)
            )
            if conn.total_changes == 0:
                return f"Agenda item #{topic_id} not found."
        return f"Agenda item #{topic_id} marked as '{status}'."

    def summarize(self) -> str:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, topic, goal, status, cycles, deadline "
                "FROM research_agenda ORDER BY status='active' DESC, cycles ASC, id ASC",
            ).fetchall()
        if not rows:
            return "Research agenda is empty."
        lines = []
        for r in rows:
            deadline_str = f", deadline: {r['deadline']}" if r["deadline"] else ""
            goal_str = f" — {r['goal']}" if r["goal"] else ""
            lines.append(
                f"#{r['id']} [{r['status']}] {r['topic']}{goal_str}"
                f" ({r['cycles']} cycles{deadline_str})"
            )
        return "\n".join(lines)
