"""Two-layer soul management.

The soul has two layers:
  IMMUTABLE  — soul.txt (or --soul arg). Written by the human. Never touched
               by the agent.
  LEARNED    — soul_learned.txt alongside soul.txt. Appended to when a soul
               proposal is approved. The agent proposes; the human decides.

At runtime both layers are concatenated into a single soul string. The agent
sees one coherent identity, not two separate files.

Soul proposals live in the assistant_data.db database. They are surfaced at
session start if any are pending, and the human approves or rejects them
conversationally (via the decide_soul_proposal tool).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

# Paths are resolved in this priority order when no explicit path is given.
_SOUL_SEARCH_ORDER = [
    Path.cwd() / "soul.txt",
]

_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # assistant/ root
_SOUL_SEARCH_ORDER.append(_SCRIPT_DIR / "soul.txt")

_DEFAULT_SOUL = "You are a helpful personal assistant."


class SoulManager:
    """Manages the immutable + learned two-layer soul."""

    def __init__(
        self,
        soul_arg: str | None,
        data_db: str | Path,
    ) -> None:
        """
        Args:
            soul_arg:  From --soul CLI flag (path string, or None).
            data_db:   Path to assistant_data.db for proposal storage.
        """
        self._soul_path = self._resolve_soul_path(soul_arg)
        self._learned_path = (
            self._soul_path.with_stem(self._soul_path.stem + "_learned")
            if self._soul_path
            else Path.cwd() / "soul_learned.txt"
        )
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(data_db), check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # Soul loading
    # ------------------------------------------------------------------

    def load(self) -> tuple[str, str]:
        """Return (assembled soul text, human-readable source description)."""
        if self._soul_path and self._soul_path.exists():
            base = self._soul_path.read_text(encoding="utf-8").strip()
            source = str(self._soul_path)
        else:
            base = _DEFAULT_SOUL
            source = "<default>"

        if self._learned_path.exists():
            learned = self._learned_path.read_text(encoding="utf-8").strip()
            if learned:
                base = base + "\n\n## Learned preferences\n" + learned
                source += f" + {self._learned_path.name}"

        return base, source

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS soul_proposals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                section      TEXT,
                current_text TEXT DEFAULT '',
                proposed_text TEXT NOT NULL,
                reasoning    TEXT NOT NULL,
                status       TEXT DEFAULT 'pending',
                decided_at   DATETIME,
                notes        TEXT DEFAULT ''
            )
        """)
        self._conn.commit()

    def add_proposal(
        self,
        proposed_text: str,
        reasoning: str,
        section: str = "preferences",
        current_text: str = "",
    ) -> str:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO soul_proposals "
                "(section, current_text, proposed_text, reasoning) "
                "VALUES (?, ?, ?, ?)",
                (section, current_text, proposed_text, reasoning),
            )
            self._conn.commit()
        return f"Soul proposal #{cur.lastrowid} submitted."

    def list_proposals(self, status: str = "pending") -> str:
        rows = self._conn.execute(
            "SELECT id, created_at, section, proposed_text, reasoning "
            "FROM soul_proposals WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
        if not rows:
            return f"No {status} soul proposals."
        lines = [f"Soul proposals ({status}):"]
        for r in rows:
            lines.append(f"\n[#{r[0]}] {r[2]}  ({(r[1] or '')[:16]})")
            lines.append(f"  Add:    {r[3]}")
            lines.append(f"  Reason: {r[4]}")
        return "\n".join(lines)

    def decide(self, proposal_id: int, decision: str, notes: str = "") -> str:
        """Approve or reject a proposal. decision must be 'approve' or 'reject'."""
        if decision not in ("approve", "reject"):
            return "decision must be 'approve' or 'reject'."

        row = self._conn.execute(
            "SELECT proposed_text, section, status FROM soul_proposals WHERE id = ?",
            (proposal_id,),
        ).fetchone()
        if not row:
            return f"Proposal #{proposal_id} not found."
        if row[2] != "pending":
            return f"Proposal #{proposal_id} is already {row[2]}."

        status = "approved" if decision == "approve" else "rejected"
        with self._lock:
            self._conn.execute(
                "UPDATE soul_proposals "
                "SET status = ?, decided_at = ?, notes = ? WHERE id = ?",
                (status, datetime.now().isoformat(), notes, proposal_id),
            )
            self._conn.commit()

        if decision == "approve":
            self._append_to_learned(row[0])
            return (
                f"Proposal #{proposal_id} approved and added to "
                f"{self._learned_path.name}. It will take effect next session."
            )
        return f"Proposal #{proposal_id} rejected."

    def count_pending(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM soul_proposals WHERE status = 'pending'"
        ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_to_learned(self, text: str) -> None:
        with open(self._learned_path, "a", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

    @staticmethod
    def _resolve_soul_path(soul_arg: str | None) -> Path | None:
        if soul_arg:
            return Path(soul_arg)
        for candidate in _SOUL_SEARCH_ORDER:
            if candidate.exists():
                return candidate
        return None
