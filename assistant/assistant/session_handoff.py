"""File-backed session handoff — carries conversational *texture* across restarts.

Distinct from the two persistence layers around it:
  - episodic memory persists *facts* (queried, deduped) — SQLite/vectors
  - the soul persists *identity* (human-authored) — soul.txt
This is the third layer: *narrative continuity* — where the last session left
off, the tone, the open threads — so a restart resumes warm instead of cold.

Stored as a human-legible markdown append-log rather than a DB row, on purpose:
the note's whole value is being readable, hand-editable, and git-diffable, and
it sits in the soul-adjacent (identity) layer, not the queryable data layer.
Writes are once-per-session and single-writer, so SQLite buys nothing here. The
accumulating log doubles as a dated diary of the agent's session-to-session
evolution (feeds the "agent summary" report idea).

Format — markdown H2 entries, newest last::

    ## 2026-08-19T21:04

    I was mid-way through helping Nikos untangle the compaction bug and we'd
    just switched Ada to DeepSeek. The mood was collaborative and a bit playful.
    Open thread: whether the job-search task survives without deferring.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^## (.+)$", re.MULTILINE)

DEFAULT_HANDOFF_PROMPT = """\
You are about to end this session. Write a short note to your future self — the
version of you that wakes up next session with no memory of this conversation's
texture (facts you already keep in long-term memory; feel, you do not).

In first person and in your own voice, capture:
- where we left off and what we were in the middle of
- the tone and feel of the exchange
- any open threads you meant to come back to

3-5 sentences. Texture, not a transcript — just enough that you pick the thread
back up warmly instead of cold. Do not re-list facts; capture the feel."""


class SessionHandoff:
    """Append-log of first-person continuity notes at ``path``.

    Args:
        path: Markdown file to append to / read from.
        prompt: System prompt for note synthesis (overridable per agent).
        max_entries: Soft cap; the log is trimmed to the last N on write.
    """

    def __init__(self, path, prompt: str = DEFAULT_HANDOFF_PROMPT,
                 max_entries: int = 20) -> None:
        self._path = Path(path)
        self.prompt = prompt
        self._max_entries = max_entries

    # --- read/write ---------------------------------------------------------

    def read_latest(self) -> Optional[str]:
        """Return the body of the most recent entry, or None if none/missing."""
        if not self._path.exists():
            return None
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read handoff file %s: %s", self._path, exc)
            return None
        entries = self._parse(text)
        if not entries:
            return None
        return entries[-1][1].strip() or None

    def write(self, note: str) -> None:
        """Append a dated entry atomically; soft-trim to the last max_entries."""
        note = (note or "").strip()
        if not note:
            return
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
        entries = []
        if self._path.exists():
            try:
                entries = self._parse(self._path.read_text(encoding="utf-8"))
            except OSError:
                entries = []
        entries.append((ts, note))
        entries = entries[-self._max_entries:]
        out = "".join(f"## {t}\n\n{body}\n\n" for t, body in entries)
        self._atomic_write(out)

    def make_note(self, llm, conversation_text: str) -> Optional[str]:
        """Synthesize a texture note from the session via one LLM call, persist it.

        Returns the note (also written to the log), or None if there was nothing
        to summarize or the call failed. Never raises — a failed handoff must not
        break shutdown.
        """
        if not (conversation_text or "").strip():
            return None
        # Local imports avoid a module-load cycle (conversation imports config,
        # config builds SessionHandoff).
        from assistant.conversation import _call_llm_raw
        from assistant.config import strip_channel_markup
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user",
             "content": f"Here is the session you're handing off:\n\n{conversation_text}"},
        ]
        try:
            resp = _call_llm_raw(llm, messages, max_tokens=512, temperature=0.6)
            note = strip_channel_markup(
                (resp["choices"][0]["message"].get("content") or "").strip())
        except Exception as exc:
            logger.warning("Handoff note generation failed: %s", exc)
            return None
        if note:
            self.write(note)
        return note or None

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _parse(text: str) -> list[tuple[str, str]]:
        """Split '## <ts>\\n\\n<body>' blocks into [(ts, body), ...]."""
        matches = list(_HEADER_RE.finditer(text))
        out = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out.append((m.group(1).strip(), text[m.end():end].strip()))
        return out

    def _atomic_write(self, content: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, self._path)
