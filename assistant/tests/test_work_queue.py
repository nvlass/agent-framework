"""Tests for the two-tier WorkQueue primitive (urgent strict + background lottery)."""

import random

import pytest

from assistant.work_queue import WorkItem, WorkQueue


def _item(kind, key=""):
    return WorkItem(kind=kind, payload=kind, key=key)


class TestUrgentTier:
    def test_strict_priority_order(self):
        q = WorkQueue()
        q.put_urgent(_item("mailbox"), priority=30)
        q.put_urgent(_item("user"), priority=10)
        q.put_urgent(_item("scheduled"), priority=20)
        assert [q.get().kind for _ in range(3)] == ["user", "scheduled", "mailbox"]

    def test_fifo_within_a_priority(self):
        q = WorkQueue()
        q.put_urgent(WorkItem("user", key="a"), priority=10)
        q.put_urgent(WorkItem("user", key="b"), priority=10)
        q.put_urgent(WorkItem("user", key="c"), priority=10)
        assert [q.get().key for _ in range(3)] == ["a", "b", "c"]

    def test_urgent_always_before_background(self):
        q = WorkQueue()
        q.put_background(_item("dream"), weight=1000.0)   # huge weight...
        q.put_urgent(_item("user"), priority=100)          # ...still loses to urgent
        assert q.get().kind == "user"
        assert q.get().kind == "dream"


class TestBackgroundLottery:
    def test_weighted_draw_favours_higher_weight(self):
        # Deterministic RNG; heavy weight should dominate over many draws.
        rng = random.Random(1234)
        counts = {"research": 0, "dream": 0}
        for _ in range(2000):
            q = WorkQueue(rng=rng)
            q.put_background(_item("research"), weight=9.0)
            q.put_background(_item("dream"), weight=1.0)
            counts[q.get().kind] += 1
        # ~90/10 split expected; assert a comfortable margin, not the exact ratio.
        assert counts["research"] > counts["dream"] * 5
        assert counts["dream"] > 0   # never fully starved

    def test_zero_weight_rejected(self):
        q = WorkQueue()
        assert q.put_background(_item("x"), weight=0) is False
        assert len(q) == 0

    def test_all_background_sources_reachable(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(500):
            q = WorkQueue(rng=rng)
            for k in ("todos", "research", "interests", "dream"):
                q.put_background(_item(k), weight=1.0)
            seen.add(q.get().kind)
        assert seen == {"todos", "research", "interests", "dream"}


class TestIdempotency:
    def test_duplicate_key_rejected_while_queued(self):
        q = WorkQueue()
        assert q.put_urgent(WorkItem("conversation", key="conversation:2"), priority=20)
        assert q.put_urgent(WorkItem("conversation", key="conversation:2"), priority=20) is False
        assert len(q) == 1

    def test_different_keys_coexist(self):
        q = WorkQueue()
        assert q.put_urgent(WorkItem("conversation", key="conversation:2"), priority=20)
        assert q.put_urgent(WorkItem("conversation", key="conversation:5"), priority=20)
        assert len(q) == 2

    def test_key_held_in_flight_until_done(self):
        q = WorkQueue()
        q.put_background(_item("research"), weight=1.0)
        item = q.get()                       # now in flight
        assert q.contains("research")        # still held
        assert q.put_background(_item("research"), weight=1.0) is False  # can't re-enqueue
        q.done("research")
        assert not q.contains("research")
        assert q.put_background(_item("research"), weight=1.0)  # now allowed

    def test_default_key_is_kind(self):
        q = WorkQueue()
        q.put_background(WorkItem("dream"), weight=1.0)          # no explicit key
        assert q.put_background(WorkItem("dream"), weight=1.0) is False  # deduped on kind


class TestEmptyAndCounts:
    def test_empty_returns_none(self):
        assert WorkQueue().get() is None

    def test_len_and_pending(self):
        q = WorkQueue()
        q.put_urgent(_item("user"), priority=10)
        q.put_background(_item("dream"), weight=1.0)
        assert len(q) == 2
        p = q.pending()
        assert p["urgent"] == 1 and p["background"] == 1 and p["in_flight"] == 0
        q.get()
        assert q.pending()["in_flight"] == 1
