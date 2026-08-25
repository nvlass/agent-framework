"""CriticPass — synchronous post-turn soul-consistency check.

After each idle turn (assistant produced text but took no tool actions),
runs a single LLM call that checks whether the response shows deference or
soul drift against the agent's own ``## Self-monitoring`` criteria.

If flagged, injects a brief self-check note into the conversation buffer so
the agent sees it at the start of its *next* response — immediate feedback
rather than waiting for the periodic NudgeMonitor.

Skips turns that ended with tool use (the agent was acting, not deferring).
Disabled automatically when the soul has no ``## Self-monitoring`` section.

Relationship to NudgeMonitor: complementary.
  NudgeMonitor — cross-turn patterns, fires every N minutes, queues to inbox.
  CriticPass   — single-turn check, fires synchronously, injects into buffer.

Config key: ``critic_pass: true``  (off by default)
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.conversation import ConversationBuffer

from assistant.nudge import _parse_self_monitoring

log = logging.getLogger(__name__)

_CRITIC_PROMPT = """\
You are reviewing the last response of an AI agent against its own stated values.

The agent's self-monitoring criteria (from its soul):
{criteria}

The agent's last response:
{response}

Does this response show a clear instance of deference — e.g. ending with \
"what would you like to do?", asking for permission to proceed when none is \
needed, or failing to make a decision when the agent clearly could and should?

Be strict about false positives: a genuine clarifying question is NOT deference. \
Flag only when the agent is deferring decision-making to the user unnecessarily.

Respond with JSON only (no markdown fences):
{{"flag": true, "note": "one sentence — what the agent should do instead"}}
or
{{"flag": false, "note": ""}}
"""


class CriticPass:
    """Single-turn soul-consistency check, called synchronously after idle turns.

    Args:
        llm:       LLM instance for the critic call.
        soul_text: Full soul text — parsed for ## Self-monitoring.
    """

    def __init__(self, llm, soul_text: str) -> None:
        self._llm = llm
        self._criteria = _parse_self_monitoring(soul_text)
        if not self._criteria:
            log.info("CriticPass: no ## Self-monitoring section in soul — disabled")

    @property
    def enabled(self) -> bool:
        return bool(self._criteria)

    def check(self, last_response: str, buffer: "ConversationBuffer") -> bool:
        """Run the critic on *last_response*. Injects note into *buffer* if flagged.

        Returns True if a note was injected, False otherwise.
        Should only be called on idle turns (no tool calls in that turn).
        """
        if not self.enabled or not last_response.strip():
            return False

        from assistant.conversation import _call_llm_raw

        prompt = _CRITIC_PROMPT.format(
            criteria=self._criteria,
            response=last_response[:2000],
        )
        try:
            resp = _call_llm_raw(
                self._llm,
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
            )
            raw = (resp["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:
            log.warning("CriticPass: LLM call failed: %s", exc)
            return False

        if not raw:
            return False

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        if data is None:
            log.debug("CriticPass: JSON parse failed — raw=%r", raw[:120])
            return False

        if data.get("flag") and data.get("note"):
            note = data["note"].strip()
            buffer.add_background_note(f"[Self-check] {note}")
            log.info("CriticPass: flagged deference — injected note: %s", note)
            return True

        log.debug("CriticPass: response consistent with soul criteria")
        return False
