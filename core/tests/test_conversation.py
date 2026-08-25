"""Tests for ConversationBus — bounded turn-taking dialogue between agents."""

import pytest

from agent_core.conversation import ConversationBus, ConversationError


@pytest.fixture
def channel(tmp_path):
    """A shared channel file with two agents' buses on it (like two processes)."""
    db = tmp_path / "channel.db"
    ada = ConversationBus(db, "ada")
    smith = ConversationBus(db, "smith")
    yield ada, smith


class TestLifecycle:
    def test_open_sets_first_turn_and_hands_off(self, channel):
        ada, smith = channel
        c = ada.open(peer="smith", message="What's your read on X?", topic="scag")
        assert c["state"] == "active"
        assert c["next_turn"] == "smith"      # handed to peer
        assert c["turn_count"] == 1
        assert c["initiator"] == "ada" and c["peer"] == "smith"
        assert c["last_from"] == "ada"

    def test_full_exchange_until_done(self, channel):
        ada, smith = channel
        c = ada.open("smith", "Question?", max_turns=6)
        cid = c["id"]
        c = smith.reply(cid, "Answer.")
        assert c["next_turn"] == "ada" and c["turn_count"] == 2
        c = ada.reply(cid, "Thanks, settled.", done=True)
        assert c["state"] == "closed"
        assert c["closed_reason"] == "done"
        assert c["next_turn"] is None
        assert [t["from_agent"] for t in ada.history(cid)] == ["ada", "smith", "ada"]

    def test_abandon_closes(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "hi")["id"]
        c = smith.abandon(cid, reason="busy")
        assert c["state"] == "closed" and c["closed_reason"] == "busy"

    def test_non_participant_cannot_abandon(self, tmp_path):
        db = tmp_path / "c.db"
        ada = ConversationBus(db, "ada")
        cid = ada.open("smith", "hi")["id"]
        lilith = ConversationBus(db, "lilith")
        with pytest.raises(ConversationError):
            lilith.abandon(cid)


class TestTurnTaking:
    def test_cannot_reply_out_of_turn(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "Q?")["id"]
        # it's smith's turn — ada replying again must be refused
        with pytest.raises(ConversationError, match="Not your turn"):
            ada.reply(cid, "me again")

    def test_cannot_reply_to_closed(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "Q?")["id"]
        smith.reply(cid, "A.", done=True)
        with pytest.raises(ConversationError, match="closed"):
            ada.reply(cid, "more")

    def test_turn_alternates(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "0", max_turns=10)["id"]
        assert smith.reply(cid, "1")["next_turn"] == "ada"
        assert ada.reply(cid, "2")["next_turn"] == "smith"
        assert smith.reply(cid, "3")["next_turn"] == "ada"

    def test_double_reply_same_turn_second_refused(self, channel):
        # Simulates two daemons acting on the same tick: the atomic claim means
        # only the first reply for a given turn succeeds.
        ada, smith = channel
        cid = ada.open("smith", "Q?")["id"]
        smith.reply(cid, "first")            # smith's turn -> now ada's
        # smith tries again immediately; it's ada's turn now -> refused
        with pytest.raises(ConversationError):
            smith.reply(cid, "second")


class TestTermination:
    def test_turn_limit_closes_even_without_done(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "t1", max_turns=3)["id"]  # turn 1
        smith.reply(cid, "t2")                             # turn 2
        c = ada.reply(cid, "t3")                           # turn 3 == cap
        assert c["state"] == "closed"
        assert c["closed_reason"] == "turn_limit"
        assert c["next_turn"] is None

    def test_max_turns_must_be_at_least_two(self, channel):
        ada, _ = channel
        with pytest.raises(ConversationError):
            ada.open("smith", "hi", max_turns=1)

    def test_cannot_open_with_self(self, channel):
        ada, _ = channel
        with pytest.raises(ConversationError):
            ada.open("ada", "hi")

    def test_empty_message_rejected(self, channel):
        ada, smith = channel
        with pytest.raises(ConversationError):
            ada.open("smith", "   ")
        cid = ada.open("smith", "real")["id"]
        with pytest.raises(ConversationError):
            smith.reply(cid, "")


class TestAttention:
    def test_peer_sees_your_turn(self, channel):
        ada, smith = channel
        ada.open("smith", "ping", topic="t")
        att = smith.needs_attention()
        assert len(att) == 1
        assert att[0]["attention"] == "your_turn"
        # ada is NOT waiting on herself
        assert ada.needs_attention() == []

    def test_initiator_sees_closing_reply_as_unread(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "Q?", max_turns=2)["id"]
        smith.reply(cid, "final answer")   # turn 2 == cap -> closed
        # ada never gets a turn, but must still see the closing message
        att = ada.needs_attention()
        assert len(att) == 1
        assert att[0]["attention"] == "unread"
        assert att[0]["state"] == "closed"
        assert att[0]["last_message"] == "final answer"

    def test_acknowledge_clears_attention(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "Q?", max_turns=2)["id"]
        smith.reply(cid, "final")
        assert len(ada.needs_attention()) == 1
        ada.acknowledge(cid)
        assert ada.needs_attention() == []

    def test_new_activity_reopens_attention_after_ack(self, channel):
        ada, smith = channel
        cid = ada.open("smith", "0", max_turns=10)["id"]
        smith.reply(cid, "1")              # ada's turn now
        ada.acknowledge(cid)               # ada acks but it's still her turn...
        # replying is what advances; her turn should still surface
        att = ada.needs_attention()
        assert any(a["id"] == cid and a["attention"] == "your_turn" for a in att)

    def test_format_attention_string(self, channel):
        ada, smith = channel
        ada.open("smith", "What's the plan?", topic="planning")
        s = smith.format_attention()
        assert "your turn" in s
        assert "planning" in s
        assert smith.format_attention() != ""
        assert ada.format_attention() == ""   # nothing awaiting ada


class TestPersistence:
    def test_reopen_file_preserves_state(self, tmp_path):
        db = tmp_path / "c.db"
        cid = ConversationBus(db, "ada").open("smith", "Q?", max_turns=4)["id"]
        ConversationBus(db, "smith").reply(cid, "A.")
        # fresh bus on same file sees the ongoing conversation
        ada2 = ConversationBus(db, "ada")
        c = ada2.get(cid)
        assert c["state"] == "active" and c["next_turn"] == "ada"
        assert c["turn_count"] == 2
        assert len(ada2.history(cid)) == 2
