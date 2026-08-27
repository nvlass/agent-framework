"""Two-tier work queue for the autonomous loop.

Collapses the work cycle's ad-hoc work paths (reactive prompt_queue, proactive
rotation, conversation special-case) into one stream — see
`docs/work-queue-design.md`. This module is the standalone primitive (Step 1);
it does not touch the loop yet.

Two tiers, deliberately different disciplines:

- **Urgent** — strict priority, FIFO within a priority. Something/someone is
  blocked (a user prompt, a peer awaiting a turn, a scheduled task). No
  randomness: a lottery that occasionally ignores a waiting peer is a bug.

- **Background** — weighted lottery (lottery scheduling). Self-directed work with
  no deadline (todos, research, interests, dream). When no urgent item is
  pending, pick one by weighted random draw: no source starves, significance
  biases selection, and the "arbitrary long wait" downside is harmless because
  nothing is blocked on background work. This is the same weighted-random pattern
  the curiosity engine already uses for strategy_weights.

**Idempotency:** an item's ``key`` is held from ``put_*`` until ``done(key)`` — so
a polling producer (e.g. "is it my turn in a conversation?") cannot enqueue a
duplicate while one is already queued *or* in flight. In practice each background
source contributes at most one item at a time (its goal generator returns one
goal or None), so the per-item background lottery coincides with a per-source one.

The tier *assignment* (which kinds are urgent vs background, and their weights)
is policy that lives with the producers/config, not here — this primitive only
provides the mechanism.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WorkItem:
    """A unit of work for the queue.

    Args:
        kind:    Dispatch key — "user" | "conversation" | "mailbox" |
                 "scheduled" | "todos" | "research" | "interests" | "dream" | …
        payload: Whatever the handler for this kind needs (goal text, a
                 conversation id, a prompt, …).
        key:     Idempotency key. Defaults to ``kind`` (one pending item per
                 kind). Set finer for independent items, e.g. "conversation:2".
    """

    kind: str
    payload: Any = None
    key: str = ""

    def dedup_key(self) -> str:
        return self.key or self.kind


class WorkQueue:
    """Strict-priority urgent tier + weighted-lottery background tier."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._urgent: list[tuple[int, int, WorkItem]] = []   # heap: (priority, seq, item)
        self._background: dict[str, tuple[WorkItem, float]] = {}  # key -> (item, weight)
        self._queued: set[str] = set()     # keys currently in a tier
        self._inflight: set[str] = set()   # keys dequeued but not yet done()
        self._seq = 0
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def _held(self, key: str) -> bool:
        return key in self._queued or key in self._inflight

    def contains(self, key: str) -> bool:
        """True if an item with this key is queued or in flight (for producers)."""
        return self._held(key)

    def put_urgent(self, item: WorkItem, priority: int = 100) -> bool:
        """Enqueue an urgent item. Lower priority = served first. FIFO ties.

        Returns False (no-op) if an item with the same key is already held.
        """
        key = item.dedup_key()
        if self._held(key):
            return False
        heapq.heappush(self._urgent, (priority, self._seq, item))
        self._seq += 1
        self._queued.add(key)
        return True

    def put_background(self, item: WorkItem, weight: float = 1.0) -> bool:
        """Enqueue a background item with a lottery weight (higher = likelier).

        Returns False if the key is already held or weight <= 0.
        """
        key = item.dedup_key()
        if weight <= 0 or self._held(key):
            return False
        self._background[key] = (item, weight)
        self._queued.add(key)
        return True

    # ------------------------------------------------------------------
    # Dequeue
    # ------------------------------------------------------------------

    def get(self) -> Optional[WorkItem]:
        """Pop the next item: all urgent (strict priority) before any background.

        A returned item's key stays held (in flight) until ``done(key)`` — so
        producers won't re-enqueue it mid-processing. Returns None if empty.
        """
        if self._urgent:
            _, _, item = heapq.heappop(self._urgent)
            key = item.dedup_key()
            self._queued.discard(key)
            self._inflight.add(key)
            return item
        if self._background:
            keys = list(self._background)
            weights = [self._background[k][1] for k in keys]
            chosen = self._rng.choices(keys, weights=weights, k=1)[0]
            item, _ = self._background.pop(chosen)
            self._queued.discard(chosen)
            self._inflight.add(chosen)
            return item
        return None

    def done(self, key: str) -> None:
        """Release an item's key after processing, so it can be enqueued again."""
        self._inflight.discard(key)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._urgent) + len(self._background)

    def pending(self) -> dict[str, int]:
        """Counts by tier — for the startup banner / debugging."""
        return {"urgent": len(self._urgent), "background": len(self._background),
                "in_flight": len(self._inflight)}
