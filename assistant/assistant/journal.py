"""Daily journal — timestamped entries by user or agent.

SQLite-backed. Entries are indexed by date so reading a day's log is fast.
Simple text search covers most retrieval needs; can be extended with FTS5
later without changing the tool API.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_entries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    date       TEXT    NOT NULL,
    time       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    tags       TEXT    NOT NULL DEFAULT '[]',
    author     TEXT    NOT NULL DEFAULT 'user',
    created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_journal_date ON journal_entries(date);
"""


class Journal:
    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db)
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

    def _resolve_date(self, date_str: str) -> str:
        if not date_str or date_str == "today":
            return date.today().isoformat()
        if date_str == "yesterday":
            return (date.today() - timedelta(days=1)).isoformat()
        return date_str  # expect YYYY-MM-DD

    def add_entry(self, content: str, tags: str = "", author: str = "user") -> str:
        now = datetime.now()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO journal_entries (date, time, content, tags, author) "
                "VALUES (?, ?, ?, ?, ?)",
                (now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), content,
                 json.dumps(tag_list), author),
            )
        return f"Journal entry added at {now.strftime('%H:%M')}."

    def read_day(self, date_str: str = "today") -> str:
        target = self._resolve_date(date_str)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM journal_entries WHERE date=? ORDER BY time",
                (target,),
            ).fetchall()
        if not rows:
            return f"No journal entries for {target}."
        lines = [f"Journal — {target}"]
        for row in rows:
            author_tag = f" [{row['author']}]" if row['author'] != "user" else ""
            lines.append(f"\n{row['time']}{author_tag}\n{row['content']}")
        return "\n".join(lines)

    def search_entries(self, query: str, limit: int = 10) -> list[dict]:
        """Return raw journal rows matching a LIKE query, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, time, content, tags, author FROM journal_entries "
                "WHERE content LIKE ? ORDER BY date DESC, time DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str) -> str:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT date, time, content, author FROM journal_entries "
                "WHERE content LIKE ? ORDER BY date DESC, time DESC LIMIT 10",
                (f"%{query}%",),
            ).fetchall()
        if not rows:
            return f"No journal entries matching '{query}'."
        lines = []
        for row in rows:
            author_tag = f" [{row['author']}]" if row['author'] != "user" else ""
            snippet = row['content'][:120] + ("..." if len(row['content']) > 120 else "")
            lines.append(f"{row['date']} {row['time']}{author_tag}: {snippet}")
        return "\n".join(lines)
