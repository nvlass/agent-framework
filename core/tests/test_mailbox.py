"""Tests for AgentMailbox."""

import pytest
from pathlib import Path

from agent_core.mailbox import AgentMailbox


@pytest.fixture
def db(tmp_path):
    return tmp_path / "mailbox.db"


@pytest.fixture
def smith(db):
    return AgentMailbox(db, "smith")


@pytest.fixture
def ada(db):
    return AgentMailbox(db, "ada")


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_db(self, tmp_path):
        path = tmp_path / "new.db"
        assert not path.exists()
        AgentMailbox(path, "agent")
        assert path.exists()

    def test_name(self, smith):
        assert smith.name == "smith"

    def test_idempotent_init(self, db):
        # Second open of same file should not error
        a = AgentMailbox(db, "a")
        b = AgentMailbox(db, "b")
        assert a.name == "a"
        assert b.name == "b"


# ---------------------------------------------------------------------------
# Send / inbox
# ---------------------------------------------------------------------------

class TestSendReceive:
    def test_send_returns_id(self, smith, ada):
        msg_id = smith.send(to="ada", message="Hello Ada")
        assert isinstance(msg_id, int)
        assert msg_id >= 1

    def test_recipient_sees_message(self, smith, ada):
        smith.send(to="ada", message="Hello Ada")
        msgs = ada.inbox()
        assert len(msgs) == 1
        assert msgs[0]["message"] == "Hello Ada"
        assert msgs[0]["from_agent"] == "smith"
        assert msgs[0]["to_agent"] == "ada"

    def test_sender_does_not_see_own_message_in_inbox(self, smith, ada):
        smith.send(to="ada", message="Hello Ada")
        assert smith.inbox() == []

    def test_unread_only_default(self, smith, ada):
        smith.send(to="ada", message="msg1")
        smith.send(to="ada", message="msg2")
        ada.mark_all_read()
        assert ada.inbox(unread_only=True) == []
        assert len(ada.inbox(unread_only=False)) == 2

    def test_from_agent_filter(self, smith, ada, db):
        critic = AgentMailbox(db, "critic")
        smith.send(to="ada", message="from smith")
        critic.send(to="ada", message="from critic")
        msgs = ada.inbox(from_agent="smith")
        assert len(msgs) == 1
        assert msgs[0]["from_agent"] == "smith"

    def test_topic_stored(self, smith, ada):
        smith.send(to="ada", message="check this", topic="research")
        msg = ada.inbox()[0]
        assert msg["topic"] == "research"

    def test_ordering_oldest_first(self, smith, ada):
        smith.send(to="ada", message="first")
        smith.send(to="ada", message="second")
        msgs = ada.inbox()
        assert msgs[0]["message"] == "first"
        assert msgs[1]["message"] == "second"

    def test_limit_respected(self, smith, ada):
        for i in range(10):
            smith.send(to="ada", message=f"msg {i}")
        assert len(ada.inbox(limit=3)) == 3


# ---------------------------------------------------------------------------
# count_unread / mark_read / mark_all_read
# ---------------------------------------------------------------------------

class TestReadStatus:
    def test_count_unread_empty(self, ada):
        assert ada.count_unread() == 0

    def test_count_unread_increments(self, smith, ada):
        smith.send(to="ada", message="a")
        smith.send(to="ada", message="b")
        assert ada.count_unread() == 2

    def test_mark_read_returns_true(self, smith, ada):
        msg_id = smith.send(to="ada", message="hello")
        assert ada.mark_read(msg_id) is True

    def test_mark_read_decrements_unread(self, smith, ada):
        msg_id = smith.send(to="ada", message="hello")
        ada.mark_read(msg_id)
        assert ada.count_unread() == 0

    def test_mark_read_wrong_recipient_returns_false(self, smith, ada):
        msg_id = smith.send(to="ada", message="hello")
        # Smith tries to mark a message addressed to Ada
        assert smith.mark_read(msg_id) is False

    def test_mark_read_nonexistent_returns_false(self, ada):
        assert ada.mark_read(9999) is False

    def test_mark_all_read_returns_count(self, smith, ada):
        smith.send(to="ada", message="a")
        smith.send(to="ada", message="b")
        assert ada.mark_all_read() == 2

    def test_mark_all_read_idempotent(self, smith, ada):
        smith.send(to="ada", message="a")
        ada.mark_all_read()
        assert ada.mark_all_read() == 0


# ---------------------------------------------------------------------------
# Threading (reply_to)
# ---------------------------------------------------------------------------

class TestThreading:
    def test_reply_to_stored(self, smith, ada):
        root_id = smith.send(to="ada", message="question?")
        reply_id = ada.send(to="smith", message="answer!", reply_to=root_id)
        reply = ada.get_message(reply_id)
        assert reply["reply_to"] == root_id

    def test_get_thread_returns_root_and_replies(self, smith, ada):
        root_id = smith.send(to="ada", message="root")
        ada.send(to="smith", message="reply1", reply_to=root_id)
        ada.send(to="smith", message="reply2", reply_to=root_id)
        thread = smith.get_thread(root_id)
        assert len(thread) == 3
        assert thread[0]["id"] == root_id

    def test_get_thread_nonexistent_returns_empty(self, smith):
        assert smith.get_thread(9999) == []


# ---------------------------------------------------------------------------
# get_message / sent
# ---------------------------------------------------------------------------

class TestGetMessage:
    def test_get_message_exists(self, smith, ada):
        msg_id = smith.send(to="ada", message="hi")
        msg = smith.get_message(msg_id)
        assert msg is not None
        assert msg["id"] == msg_id

    def test_get_message_nonexistent(self, smith):
        assert smith.get_message(9999) is None

    def test_sent_returns_own_messages(self, smith, ada):
        smith.send(to="ada", message="a")
        smith.send(to="ada", message="b")
        sent = smith.sent()
        assert len(sent) == 2
        assert all(m["from_agent"] == "smith" for m in sent)

    def test_sent_newest_first(self, smith, ada):
        smith.send(to="ada", message="first")
        smith.send(to="ada", message="second")
        sent = smith.sent()
        assert sent[0]["message"] == "second"


# ---------------------------------------------------------------------------
# format_inbox
# ---------------------------------------------------------------------------

class TestFormatInbox:
    def test_empty_returns_empty_string(self, ada):
        assert ada.format_inbox() == ""

    def test_format_contains_sender_and_message(self, smith, ada):
        smith.send(to="ada", message="Hello Ada", topic="greeting")
        formatted = ada.format_inbox()
        assert "smith" in formatted
        assert "Hello Ada" in formatted
        assert "greeting" in formatted

    def test_format_reply_to_shown(self, smith, ada):
        root_id = smith.send(to="ada", message="root")
        smith.send(to="ada", message="follow-up", reply_to=root_id)
        formatted = ada.format_inbox()
        assert f"reply to #{root_id}" in formatted

    def test_format_limit(self, smith, ada):
        for i in range(10):
            smith.send(to="ada", message=f"msg {i}")
        formatted = ada.format_inbox(limit=2)
        assert formatted.count("From smith") == 2


# ---------------------------------------------------------------------------
# Cross-process simulation (two connections to same file)
# ---------------------------------------------------------------------------

class TestCrossProcess:
    def test_two_connections_same_file(self, db):
        """Simulate two processes sharing the same mailbox file."""
        sender = AgentMailbox(db, "alice")
        receiver = AgentMailbox(db, "bob")

        sender.send(to="bob", message="inter-process hello")
        msgs = receiver.inbox()
        assert len(msgs) == 1
        assert msgs[0]["message"] == "inter-process hello"

    def test_read_status_visible_across_connections(self, db):
        a = AgentMailbox(db, "alice")
        b1 = AgentMailbox(db, "bob")
        b2 = AgentMailbox(db, "bob")  # second connection, same agent

        msg_id = a.send(to="bob", message="hello")
        assert b1.count_unread() == 1
        b2.mark_read(msg_id)
        assert b1.count_unread() == 0
