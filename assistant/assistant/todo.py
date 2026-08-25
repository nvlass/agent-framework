"""TODO management with agent annotation support.

SQLite-backed. Designed to grow: priorities, due dates, tags, and per-todo
notes that either the user or the agent can add (e.g. "I looked into this —
the blocker is X").

Schema lives in two tables:
  todos       — the items themselves
  todo_notes  — timestamped annotations attached to a todo
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'pending',
    priority    TEXT    NOT NULL DEFAULT 'normal',
    due_date    TEXT,
    tags        TEXT    NOT NULL DEFAULT '[]',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS todo_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    todo_id     INTEGER NOT NULL REFERENCES todos(id) ON DELETE CASCADE,
    note        TEXT    NOT NULL,
    author      TEXT    NOT NULL DEFAULT 'agent',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

_PRIORITY_SORT = "CASE priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2 WHEN 'normal' THEN 3 ELSE 4 END"
_VALID_STATUS   = {"pending", "in_progress", "done", "cancelled"}
_VALID_PRIORITY = {"urgent", "high", "normal", "low"}


class TodoDB:
    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add(self, title: str, description: str = "", priority: str = "normal",
            due_date: str = "", tags: str = "") -> str:
        if priority not in _VALID_PRIORITY:
            priority = "normal"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO todos (title, description, priority, due_date, tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (title, description, priority, due_date or None, json.dumps(tag_list)),
            )
        return f"TODO #{cur.lastrowid} added: {title}"

    def list_todos(self, status: str = "pending") -> str:
        with self._conn() as conn:
            if status == "all":
                rows = conn.execute(
                    f"SELECT * FROM todos ORDER BY {_PRIORITY_SORT}, created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM todos WHERE status=? ORDER BY {_PRIORITY_SORT}, created_at",
                    (status,),
                ).fetchall()
            if not rows:
                return f"No {status} TODOs."
            lines = []
            for row in rows:
                due = f"  due: {row['due_date']}" if row['due_date'] else ""
                lines.append(
                    f"#{row['id']} [{row['priority']}] [{row['status']}] {row['title']}{due}"
                )
                if row['description']:
                    lines.append(f"    {row['description']}")
                notes = conn.execute(
                    "SELECT * FROM todo_notes WHERE todo_id=? ORDER BY created_at",
                    (row['id'],),
                ).fetchall()
                for n in notes:
                    lines.append(f"    [{n['author']} @ {n['created_at'][:16]}] {n['note']}")
        return "\n".join(lines)

    def set_status(self, todo_id: int, status: str) -> str:
        if status not in _VALID_STATUS:
            return f"Invalid status '{status}'. Valid: {', '.join(sorted(_VALID_STATUS))}"
        with self._conn() as conn:
            conn.execute(
                "UPDATE todos SET status=?, updated_at=datetime('now','localtime') WHERE id=?",
                (status, todo_id),
            )
            if conn.total_changes == 0:
                return f"TODO #{todo_id} not found."
        return f"TODO #{todo_id} marked {status}."

    def add_note(self, todo_id: int, note: str, author: str = "agent") -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT title FROM todos WHERE id=?", (todo_id,)).fetchone()
            if not row:
                return f"TODO #{todo_id} not found."
            conn.execute(
                "INSERT INTO todo_notes (todo_id, note, author) VALUES (?, ?, ?)",
                (todo_id, note, author),
            )
        return f"Note added to TODO #{todo_id} ({row['title']})."
