"""Inner voice — significance filter for proactive messages.

When the curiosity engine or reflection scheduler produces a thought, it passes
through here. The inner voice evaluates whether it's genuinely worth surfacing
to the user — not as a research update, but as something the user would actually
want to hear.

The filter is intentionally conservative. Most thoughts don't pass. Only things
that are surprising, that connect to something the user cares about, or that
open a real question get queued.

Rate-limited: at most one evaluation per min_interval seconds (default 10 min),
so a burst of curiosity findings doesn't trigger multiple LLM calls.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

_SIGNIFICANCE_PROMPT = """\
You are a personal assistant. While researching or reflecting, you had this thought:

Topic: {topic}
Thought: {thought}

Your soul (abbreviated):
{soul_excerpt}

Decide: is this worth queuing as a proactive message for your next conversation with the person?

Say YES if the thought:
- Is genuinely interesting, surprising, or challenges something previously believed
- Connects to something in your soul's interests or values (even loosely)
- Opens a real question you'd want to explore together
- Sparked something you'd naturally want to mention if you were talking right now

Say NO only if it is:
- Generic or completely unsurprising ("X exists and is a thing")
- Entirely off-topic with no conceivable connection to the soul or the person's world
- A near-duplicate of something already queued

When in doubt, lean toward YES — it is better to share something mildly interesting \
than to stay silent. The person can always dismiss it.

Respond with JSON only (no markdown fences):
{{"share": true, "message": "...", "urgency": "low|medium|high"}}
or
{{"share": false, "message": "", "urgency": "low"}}

If share is true: write message as a natural first-person note (1-3 sentences) to start a conversation.
urgency: high = very time-sensitive or important; medium = highlight at session start; low = queue quietly.
"""


class InnerVoice:
    """
    Significance filter: decides whether a thought is worth sharing,
    and routes it to pending_messages if so.

    Thread-safe. Rate-limited to avoid excessive inference.
    """

    def __init__(
        self,
        llm,
        soul_text: str,
        pending_messages,
        min_interval: float = 600.0,
        on_proactive: Callable[[str, str, str], None] | None = None,
    ) -> None:
        """
        Args:
            llm:              FireworksLLM instance.
            soul_text:        Current assembled soul text (for context).
            pending_messages: PendingMessages instance.
            min_interval:     Minimum seconds between evaluations (default 10 min).
            on_proactive:     Optional callback(topic, message, urgency) for urgent
                              messages — use this to trigger email if desired.
        """
        self._llm = llm
        self._soul_excerpt = soul_text[:600]
        self._pending = pending_messages
        self._min_interval = min_interval
        self._on_proactive = on_proactive
        self._last_eval: float = 0.0
        self._lock = threading.Lock()

    def evaluate(self, thought: str, topic: str, source: str = "unknown") -> bool:
        """
        Evaluate a thought for significance. If significant, queues a pending
        message and (for high urgency) calls on_proactive.

        Returns True if the thought was queued as a pending message.
        """
        with self._lock:
            now = time.time()
            if now - self._last_eval < self._min_interval:
                return False
            self._last_eval = now

        try:
            result = self._evaluate(thought, topic)
        except Exception as exc:
            log.debug("InnerVoice evaluation error: %s", exc)
            return False

        if not result:
            return False

        share, message, urgency = result
        if not share or not message:
            return False

        msg_id = self._pending.add(topic, message, source=source, urgency=urgency)
        log.info("InnerVoice queued #%d [%s] %s", msg_id, urgency, topic)

        if urgency == "high" and self._on_proactive:
            try:
                self._on_proactive(topic, message, urgency)
            except Exception as exc:
                log.warning("InnerVoice on_proactive failed: %s", exc)

        return True

    def _evaluate(self, thought: str, topic: str) -> tuple[bool, str, str] | None:
        """Returns (share, message, urgency) or None on failure."""
        from assistant.conversation import _call_llm_raw

        prompt = _SIGNIFICANCE_PROMPT.format(
            topic=topic,
            thought=thought[:800],
            soul_excerpt=self._soul_excerpt,
        )
        response = _call_llm_raw(
            self._llm,
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        raw = (response["choices"][0]["message"].get("content") or "").strip()

        # Strip markdown fences if the model wraps in ```json
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        data = json.loads(raw)
        return (
            bool(data.get("share")),
            str(data.get("message", "")),
            str(data.get("urgency", "low")),
        )
