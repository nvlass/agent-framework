"""Task scheduler — SQLite-persisted scheduled prompts.

Supports two schedule types:
  once  — fires at a specific ISO datetime, then deactivates
  cron  — fires on a cron expression (requires pip install croniter)

Due tasks are placed into a queue.Queue so the main loop can inject
them as autonomous agent turns without blocking input().

Usage:
    q = queue.Queue()
    sched = TaskScheduler(db_path, q)
    sched.start()
    # main loop calls q.get_nowait() before each input()
"""

import logging
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt      TEXT    NOT NULL,
    type        TEXT    NOT NULL DEFAULT 'once',
    cron_expr   TEXT,
    next_run    TEXT    NOT NULL,
    last_run    TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);
"""

_CHECK_INTERVAL = 30  # seconds between due-task checks
_DT_FMT = "%Y-%m-%d %H:%M"


def _now_str() -> str:
    return datetime.now().strftime(_DT_FMT)


def _cron_next(expr: str, after: datetime) -> str | None:
    """Return next run time for a cron expression, or None if croniter unavailable."""
    try:
        from croniter import croniter
        return croniter(expr, after).get_next(datetime).strftime(_DT_FMT)
    except ImportError:
        return None


class TaskScheduler:
    """Background scheduler that injects due prompts into a queue."""

    def __init__(self, db_path, prompt_queue: queue.Queue) -> None:
        self._db = str(db_path)
        self._queue = prompt_queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API (called from tools)
    # ------------------------------------------------------------------

    def schedule_once(self, prompt: str, when_iso: str) -> str:
        """Schedule a one-shot task. when_iso: 'YYYY-MM-DD HH:MM'."""
        try:
            dt = datetime.strptime(when_iso.strip(), _DT_FMT)
        except ValueError:
            return f"Invalid datetime format. Use YYYY-MM-DD HH:MM (got: {when_iso!r})"
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO scheduled_tasks (prompt, type, next_run) VALUES (?, 'once', ?)",
                (prompt, dt.strftime(_DT_FMT)),
            )
        return f"Scheduled task #{cur.lastrowid} for {dt.strftime(_DT_FMT)}."

    def schedule_cron(self, prompt: str, cron_expr: str) -> str:
        """Schedule a recurring task via cron expression. Requires croniter."""
        next_run = _cron_next(cron_expr, datetime.now())
        if next_run is None:
            return (
                "Cron scheduling requires: pip install croniter\n"
                "For one-shot tasks use when= with a datetime string."
            )
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO scheduled_tasks (prompt, type, cron_expr, next_run) "
                "VALUES (?, 'cron', ?, ?)",
                (prompt, cron_expr, next_run),
            )
        return f"Recurring task #{cur.lastrowid} scheduled (next: {next_run})."

    def list_tasks(self) -> str:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, type, prompt, cron_expr, next_run, last_run, active "
                "FROM scheduled_tasks ORDER BY active DESC, next_run"
            ).fetchall()
        if not rows:
            return "No scheduled tasks."
        lines = []
        for r in rows:
            status = "active" if r["active"] else "done"
            sched = r["cron_expr"] if r["cron_expr"] else r["next_run"]
            prompt_preview = r["prompt"][:60] + ("..." if len(r["prompt"]) > 60 else "")
            lines.append(f"#{r['id']} [{status}] [{r['type']}] {sched} — {prompt_preview}")
        return "\n".join(lines)

    def cancel_task(self, task_id: int) -> str:
        with self._conn() as conn:
            conn.execute(
                "UPDATE scheduled_tasks SET active=0 WHERE id=?", (task_id,)
            )
            if conn.total_changes == 0:
                return f"Task #{task_id} not found."
        return f"Task #{task_id} cancelled."

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="task-scheduler")
        self._thread.start()
        log.info("TaskScheduler started")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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

    def _run(self) -> None:
        while not self._stop.wait(_CHECK_INTERVAL):
            self._check_due()

    def _check_due(self) -> None:
        now = _now_str()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, prompt, type, cron_expr FROM scheduled_tasks "
                "WHERE active=1 AND next_run <= ?",
                (now,),
            ).fetchall()

            for row in rows:
                self._queue.put(row["prompt"])
                log.info("TaskScheduler: queued task #%d", row["id"])

                if row["type"] == "cron" and row["cron_expr"]:
                    next_run = _cron_next(row["cron_expr"], datetime.now())
                    if next_run:
                        conn.execute(
                            "UPDATE scheduled_tasks SET last_run=?, next_run=? WHERE id=?",
                            (now, next_run, row["id"]),
                        )
                    else:
                        conn.execute(
                            "UPDATE scheduled_tasks SET active=0, last_run=? WHERE id=?",
                            (now, row["id"]),
                        )
                else:
                    conn.execute(
                        "UPDATE scheduled_tasks SET active=0, last_run=? WHERE id=?",
                        (now, row["id"]),
                    )
