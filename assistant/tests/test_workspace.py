"""Tests for WorkspaceState — the Meta-Mind floor (source-tagged shared workspace)."""

import threading

from assistant.workspace import WorkspaceState


class TestSlotsAndEvents:
    def test_publish_slot_shows_with_source(self):
        w = WorkspaceState()
        w.publish("activity", "researching consciousness", source="work-cycle")
        r = w.render()
        assert "work-cycle:" in r
        assert "activity=researching consciousness" in r

    def test_note_event_shows_with_source(self):
        w = WorkspaceState()
        w.note("saved a note about SCAG", source="work-cycle")
        r = w.render()
        assert "work-cycle:" in r and "saved a note about SCAG" in r

    def test_latest_publish_wins(self):
        w = WorkspaceState()
        w.publish("last_user", "hello", source="chat")
        w.publish("last_user", "what's on your mind?", source="chat")
        r = w.render()
        assert "what's on your mind?" in r and "hello" not in r

    def test_events_bounded(self):
        w = WorkspaceState(max_events=3)
        for i in range(6):
            w.note(f"event {i}", source="work-cycle")
        r = w.render()
        assert "event 5" in r and "event 4" in r and "event 3" in r
        assert "event 0" not in r and "event 2" not in r

    def test_clear_slot(self):
        w = WorkspaceState()
        w.publish("activity", "x", source="work-cycle")
        w.clear_slot("activity")
        assert "activity" not in w.render()


class TestProvenanceGrouping:
    def test_grouped_by_source_whose_thought_is_whose(self):
        w = WorkspaceState()
        w.publish("last_user", "how are you?", source="chat")
        w.publish("activity", "researching X", source="work-cycle")
        w.publish("active_conversations", "#2 with smith (their turn)", source="bus")
        w.note("found paper Y", source="work-cycle")
        r = w.render()
        # each source appears as its own labelled group
        assert "chat: " in r and "work-cycle: " in r and "bus: " in r
        # the work-cycle line carries both its slot and its event
        wc_line = [ln for ln in r.splitlines() if ln.strip().startswith("work-cycle:")][0]
        assert "researching X" in wc_line and "found paper Y" in wc_line
        # provenance is preserved: chat's content is not attributed to work-cycle
        assert "how are you?" not in wc_line


class TestExtensibility:
    def test_new_slot_appears_without_registration(self):
        w = WorkspaceState()
        # a brand-new producer just publishes a new slot name + source — no class edit
        w.publish("mood", "playful and a little smug", source="soul-monitor")
        r = w.render()
        assert "soul-monitor:" in r and "mood=playful" in r


class TestEmptyAndSnapshot:
    def test_empty_renders_blank(self):
        assert WorkspaceState().render() == ""

    def test_max_events_cap_on_render(self):
        w = WorkspaceState(max_events=10)
        for i in range(5):
            w.note(f"e{i}", source="s")
        assert "e0" not in w.render(max_events=2)
        assert "e4" in w.render(max_events=2) and "e3" in w.render(max_events=2)

    def test_snapshot_structure(self):
        w = WorkspaceState()
        w.publish("activity", "x", source="work-cycle")
        w.note("did a thing", source="chat")
        snap = w.snapshot()
        assert snap["slots"]["activity"] == {"value": "x", "source": "work-cycle"}
        assert snap["events"] == [{"source": "chat", "text": "did a thing"}]


class TestThreadSafety:
    def test_concurrent_writes_do_not_corrupt(self):
        w = WorkspaceState(max_events=1000)

        def worker(src):
            for i in range(200):
                w.publish(f"slot_{src}", i, source=src)
                w.note(f"{src}:{i}", source=src)

        threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = w.snapshot()
        # 6 slots (one per thread), all events accounted for, no crash/corruption
        assert len([k for k in snap["slots"] if k.startswith("slot_")]) == 6
        assert len(snap["events"]) == 1000  # deque maxlen, no torn writes
