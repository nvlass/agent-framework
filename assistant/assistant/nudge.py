"""NudgeMonitor — soul-derived autonomy nudge.

Periodically reads recent conversation turns and asks a lightweight LLM
whether the agent is behaving consistently with the ``## Self-monitoring``
section of its soul.  If not, it queues a private pending message that the
agent will see when it calls ``list_pending_messages`` — a gentle internal
reminder, not a user-facing interrupt.

The evaluator is intentionally soul-driven: it extracts the agent's own
self-monitoring criteria from the soul text rather than using a hardcoded
behavioural ideal.  This means:

- Agent Smith (soul: "act without waiting for approval") → nudged on deference
- Personal assistant (no ``## Self-monitoring`` section) → nudge disabled
- Ada (no section) → nudge disabled

Opt-in via YAML:  ``nudge_interval: 900``  (0 = off, the default)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.conversation import ConversationBuffer
    from assistant.pending_messages import PendingMessages

log = logging.getLogger(__name__)

_SECTION_HEADER = "## self-monitoring"

_NUDGE_PROMPT = """\
You are evaluating whether an AI agent is behaving consistently with its own \
stated self-monitoring criteria.

The agent's self-monitoring criteria (from its soul):
{criteria}

Recent conversation (last {n} exchanges):
{transcript}

Based solely on the criteria above, is there a meaningful gap between what the \
agent says it values and how it actually behaved in this conversation?

Only say YES if you see a clear, specific pattern — not a one-off slip.
Say NO if the behaviour is broadly consistent, ambiguous, or too short to judge.

Respond with JSON only (no markdown fences):
{{"nudge": true, "message": "..."}}
or
{{"nudge": false, "message": ""}}

If nudge is true: write message as a first-person observation the agent would \
find useful — direct, specific, non-judgmental. 1-2 sentences.
"""


def _parse_self_monitoring(soul_text: str) -> str:
    """Extract the ## Self-monitoring section from the soul. Returns '' if absent."""
    lines: list[str] = []
    in_section = False
    for line in soul_text.splitlines():
        if line.strip().lower().startswith(_SECTION_HEADER):
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("##"):
                break
            lines.append(line)
    return "\n".join(lines).strip()


class NudgeMonitor:
    """Background nudge that checks soul-consistency of recent conversation.

    Disabled automatically when the soul has no ``## Self-monitoring`` section.

    Args:
        llm:              LLM instance for evaluation.
        soul_text:        Full soul text — parsed for ## Self-monitoring.
        pending_messages: Where nudge messages are queued.
        buffer:           ConversationBuffer — read recent turns from here.
        interval_seconds: How often to evaluate (e.g. 900 = 15 min).
        turns:            How many recent user+assistant exchanges to include.
    """

    def __init__(
        self,
        llm,
        soul_text: str,
        pending_messages: "PendingMessages",
        buffer: "ConversationBuffer",
        interval_seconds: int = 900,
        turns: int = 8,
    ) -> None:
        self._llm = llm
        self._criteria = _parse_self_monitoring(soul_text)
        self._pending = pending_messages
        self._buffer = buffer
        self._interval = interval_seconds
        self._turns = turns
        self._timer: threading.Timer | None = None
        self._running = False

        if not self._criteria:
            log.info("NudgeMonitor: no ## Self-monitoring section in soul — disabled")

    @property
    def enabled(self) -> bool:
        return bool(self._criteria)

    def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._schedule()
        log.info("NudgeMonitor started (every %ds, %d turns)", self._interval, self._turns)

    def stop(self) -> None:
        self._running = False
        if self._timer:
            self._timer.cancel()

    def _schedule(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        try:
            self._evaluate()
        except Exception as exc:
            log.warning("NudgeMonitor evaluation failed: %s", exc)
        finally:
            self._schedule()

    def _evaluate(self) -> None:
        transcript = self._buffer.recent_turns_text(self._turns)
        if not transcript:
            log.debug("NudgeMonitor: no conversation yet — skipping")
            return

        from assistant.conversation import _call_llm_raw

        prompt = _NUDGE_PROMPT.format(
            criteria=self._criteria,
            n=self._turns,
            transcript=transcript,
        )
        resp = _call_llm_raw(
            self._llm,
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        raw = (resp["choices"][0]["message"].get("content") or "").strip()
        if not raw:
            return

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try the complete-object regex first
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            # Last resort: response was truncated mid-string — recover nudge=true +
            # whatever message we got so far rather than silently dropping it.
            if data is None and '"nudge": true' in raw:
                msg_match = re.search(r'"message"\s*:\s*"([^"]{10,})', raw)
                if msg_match:
                    data = {"nudge": True, "message": msg_match.group(1).rstrip() + "…"}
        if data is None:
            log.warning("NudgeMonitor: JSON parse failed — raw=%r", raw[:200])
            return

        if data.get("nudge") and data.get("message"):
            if self._recently_nudged():
                log.debug("NudgeMonitor: cooldown active — suppressing duplicate nudge")
                return
            msg_id = self._pending.add(
                topic="self-monitoring",
                message=data["message"],
                source="nudge",
                urgency="low",
            )
            log.info("NudgeMonitor: queued nudge #%d", msg_id)
        else:
            log.debug("NudgeMonitor: behaviour consistent with soul — no nudge")

    def _recently_nudged(self) -> bool:
        """Return True if a nudge was queued within the last 3 intervals."""
        cutoff = time.time() - self._interval * 3
        try:
            for m in self._pending.get_all(limit=10):
                if m.get("source") != "nudge":
                    continue
                ts_str = m.get("created_at", "")
                if not ts_str:
                    continue
                # SQLite CURRENT_TIMESTAMP is "YYYY-MM-DD HH:MM:SS"
                from datetime import datetime, timezone
                ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc).timestamp()
                if ts > cutoff:
                    return True
        except Exception as exc:
            log.debug("NudgeMonitor: cooldown check failed: %s", exc)
        return False
