"""WorkspaceState — a small, per-agent global workspace (Meta-Mind floor).

An agent's contexts (interactive chat loop, work-cycle daemon, curiosity,
reflection, the bus) run in parallel and share persistent state but not working
awareness — a "split brain." This is the shared bridge between them: contexts
*publish* salient, TAGGED items; readers pull a curated, by-source slice into
their next prompt so the agent knows what its other processes are doing.

Two design commitments (see docs/meta-mind-design.md):

- **Share the boundary, not just the state** (Lilith's constraint). Every entry
  carries a ``source`` provenance tag, and ``render`` groups by source — so
  "whose thought is whose" is preserved. A workspace that merges everything into
  one buffer is a blender, not a corpus callosum.
- **Passive integrator, never a controller.** It only holds and renders what
  contexts choose to publish. It decides nothing; the self is the *contents*.

Extensible by construction: a new producer/slot is one ``publish``/``note`` call
— no registration, no class edits. Thread-safe (many daemon threads write).
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _Entry:
    source: str        # which context produced it: "chat" | "work-cycle" | "bus" | …
    name: str          # slot name; "" for a free event
    value: Any
    seq: int


def _short(value: Any, limit: int = 120) -> str:
    s = str(value).replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


class WorkspaceState:
    """Thread-safe workspace of source-tagged slots + a bounded event stream.

    - **slots**: named single-value fields (``activity``, ``last_user``, …);
      the latest ``publish`` to a name wins.
    - **events**: a bounded, insertion-ordered stream of salient moments
      (cycle outcomes, replies sent, notes saved).

    Both carry a ``source`` tag. ``render`` returns a compact, by-source block
    for prompt injection (empty string when there is nothing to say).
    """

    def __init__(self, max_events: int = 12) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, _Entry] = {}
        self._events: deque[_Entry] = deque(maxlen=max_events)
        self._seq = 0

    # --- producers ----------------------------------------------------------

    def publish(self, slot: str, value: Any, source: str) -> None:
        """Set a named slot (latest wins), tagged with its source context."""
        with self._lock:
            self._seq += 1
            self._slots[slot] = _Entry(source, slot, value, self._seq)

    def note(self, text: str, source: str) -> None:
        """Append a salient event to the bounded stream, tagged with its source."""
        with self._lock:
            self._seq += 1
            self._events.append(_Entry(source, "", text, self._seq))

    def clear_slot(self, slot: str) -> None:
        """Remove a slot (e.g. an activity that has ended)."""
        with self._lock:
            self._slots.pop(slot, None)

    # --- readers ------------------------------------------------------------

    def render(self, max_events: Optional[int] = None) -> str:
        """A compact, source-grouped view of the workspace for prompt injection.

        Slots and recent events are grouped under their source so provenance is
        legible. Returns "" when empty. ``max_events`` caps how many recent
        events are shown (default: all currently held).
        """
        with self._lock:
            slots = list(self._slots.values())
            events = list(self._events)
        if not slots and not events:
            return ""
        if max_events is not None:
            events = events[-max_events:]

        by_source: dict[str, dict[str, list]] = defaultdict(
            lambda: {"slots": [], "events": []})
        for e in slots:
            by_source[e.source]["slots"].append(e)
        for e in events:
            by_source[e.source]["events"].append(e)

        lines = ["[Your working state — what you and your other processes are "
                 "doing right now; each line tagged by which of you produced it]"]
        for source in sorted(by_source):
            slot_parts = [f"{s.name}={_short(s.value)}"
                          for s in sorted(by_source[source]["slots"],
                                          key=lambda x: x.name)]
            event_parts = [_short(e.value) for e in by_source[source]["events"]]
            seg = f"{source}: "
            if slot_parts:
                seg += "; ".join(slot_parts)
            if event_parts:
                seg += (" | recent: " if slot_parts else "") + " · ".join(event_parts)
            lines.append("  " + seg)
        return "\n".join(lines)

    # --- introspection ------------------------------------------------------

    def snapshot(self) -> dict:
        """Structured view (for debugging / a future TUI tab)."""
        with self._lock:
            return {
                "slots": {n: {"value": e.value, "source": e.source}
                          for n, e in self._slots.items()},
                "events": [{"source": e.source, "text": e.value}
                           for e in self._events],
            }
