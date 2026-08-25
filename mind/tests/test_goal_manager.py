"""Tests for GoalManager.

Run with:
    cd mind/
    python -m pytest tests/test_goal_manager.py -v
"""

import pytest

from agent_mind.goals.model import Goal, GoalState
from agent_mind.goals.manager import GoalManager


class TestPush:

    def test_push_root(self):
        mgr = GoalManager()
        gid = mgr.push("build framework")
        assert mgr.get(gid) is not None
        assert mgr.get(gid).description == "build framework"
        assert len(mgr) == 1

    def test_push_child(self):
        mgr = GoalManager()
        root = mgr.push("parent")
        child = mgr.push("child", parent_id=root)
        assert mgr.get(child).parent_id == root
        assert child in mgr.get(root).children_ids

    def test_push_bad_parent_raises(self):
        mgr = GoalManager()
        with pytest.raises(KeyError, match="Parent goal not found"):
            mgr.push("orphan", parent_id="nonexistent")

    def test_push_with_priority(self):
        mgr = GoalManager()
        gid = mgr.push("urgent", priority=9)
        assert mgr.get(gid).priority == 9

    def test_push_with_metadata(self):
        mgr = GoalManager()
        gid = mgr.push("tagged", metadata={"source": "user"})
        assert mgr.get(gid).metadata["source"] == "user"


class TestActivate:

    def test_activate_pending(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.activate(gid)
        assert mgr.get(gid).state == GoalState.ACTIVE

    def test_activate_blocked(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.activate(gid)
        mgr.block(gid, "waiting")
        mgr.unblock(gid)
        # now active, block again to test activate from blocked
        mgr.block(gid, "waiting again")
        mgr.activate(gid)
        assert mgr.get(gid).state == GoalState.ACTIVE
        assert mgr.get(gid).blocked_reason is None

    def test_activate_completed_raises(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.complete(gid)
        with pytest.raises(ValueError, match="Cannot activate"):
            mgr.activate(gid)


class TestComplete:

    def test_complete_goal(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.complete(gid)
        assert mgr.get(gid).state == GoalState.COMPLETED
        assert mgr.get(gid).completed_at is not None

    def test_complete_already_completed_raises(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.complete(gid)
        with pytest.raises(ValueError, match="already completed"):
            mgr.complete(gid)

    def test_complete_propagates_to_parent(self):
        mgr = GoalManager()
        root = mgr.push("parent")
        children = mgr.decompose(root, ["a", "b"])
        mgr.complete(children[0])
        assert mgr.get(root).state != GoalState.COMPLETED  # one still pending
        mgr.complete(children[1])
        assert mgr.get(root).state == GoalState.COMPLETED  # all done

    def test_complete_does_not_propagate_if_abandoned_sibling(self):
        mgr = GoalManager()
        root = mgr.push("parent")
        children = mgr.decompose(root, ["a", "b"])
        mgr.complete(children[0])
        mgr.abandon(children[1])
        # Parent should NOT auto-complete — let agent decide
        assert mgr.get(root).state != GoalState.COMPLETED


class TestBlock:

    def test_block_active_goal(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.activate(gid)
        mgr.block(gid, "waiting for API", unblocked_by="API key arrives")
        assert mgr.get(gid).state == GoalState.BLOCKED
        assert mgr.get(gid).blocked_reason == "waiting for API"
        assert mgr.get(gid).unblocked_by == "API key arrives"

    def test_block_pending_raises(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        with pytest.raises(ValueError, match="Can only block active"):
            mgr.block(gid, "reason")


class TestUnblock:

    def test_unblock(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.activate(gid)
        mgr.block(gid, "waiting")
        mgr.unblock(gid)
        assert mgr.get(gid).state == GoalState.ACTIVE
        assert mgr.get(gid).blocked_reason is None

    def test_unblock_not_blocked_raises(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        with pytest.raises(ValueError, match="not blocked"):
            mgr.unblock(gid)


class TestAbandon:

    def test_abandon(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.abandon(gid)
        assert mgr.get(gid).state == GoalState.ABANDONED
        assert mgr.get(gid).completed_at is not None

    def test_abandon_completed_raises(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        mgr.complete(gid)
        with pytest.raises(ValueError, match="already completed"):
            mgr.abandon(gid)


class TestDecompose:

    def test_decompose(self):
        mgr = GoalManager()
        root = mgr.push("parent", priority=8)
        children = mgr.decompose(root, ["step 1", "step 2", "step 3"])
        assert len(children) == 3
        assert len(mgr) == 4  # root + 3 children
        for cid in children:
            child = mgr.get(cid)
            assert child.parent_id == root
            assert child.priority == 8  # inherited

    def test_decompose_makes_parent_non_leaf(self):
        mgr = GoalManager()
        root = mgr.push("parent")
        assert mgr.get(root).is_leaf is True
        mgr.decompose(root, ["child"])
        assert mgr.get(root).is_leaf is False


class TestReprioritize:

    def test_reprioritize(self):
        mgr = GoalManager()
        gid = mgr.push("test", priority=5)
        mgr.reprioritize(gid, 9)
        assert mgr.get(gid).priority == 9

    def test_reprioritize_out_of_range(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        with pytest.raises(ValueError, match="Priority must be 1-10"):
            mgr.reprioritize(gid, 11)

    def test_reprioritize_zero(self):
        mgr = GoalManager()
        gid = mgr.push("test")
        with pytest.raises(ValueError, match="Priority must be 1-10"):
            mgr.reprioritize(gid, 0)


class TestGetNext:
    """Test get_next() — returns highest-priority active/pending leaf.

    Implement get_next() in GoalManager to make these pass.
    """

    def test_single_pending_goal(self):
        mgr = GoalManager()
        gid = mgr.push("only goal")
        result = mgr.get_next()
        assert result is not None
        assert result.id == gid

    def test_highest_priority_wins(self):
        mgr = GoalManager()
        mgr.push("low", priority=3)
        high = mgr.push("high", priority=9)
        mgr.push("medium", priority=5)
        result = mgr.get_next()
        assert result.id == high

    def test_skips_completed(self):
        mgr = GoalManager()
        gid1 = mgr.push("done", priority=10)
        gid2 = mgr.push("todo", priority=5)
        mgr.complete(gid1)
        result = mgr.get_next()
        assert result.id == gid2

    def test_skips_blocked(self):
        mgr = GoalManager()
        gid1 = mgr.push("blocked", priority=10)
        gid2 = mgr.push("available", priority=5)
        mgr.activate(gid1)
        mgr.block(gid1, "waiting")
        result = mgr.get_next()
        assert result.id == gid2

    def test_skips_non_leaf(self):
        mgr = GoalManager()
        root = mgr.push("parent", priority=10)
        child = mgr.decompose(root, ["child"])[0]
        result = mgr.get_next()
        assert result.id == child  # leaf, not parent

    def test_empty_returns_none(self):
        mgr = GoalManager()
        assert mgr.get_next() is None

    def test_all_completed_returns_none(self):
        mgr = GoalManager()
        gid = mgr.push("done")
        mgr.complete(gid)
        assert mgr.get_next() is None

    def test_prefers_active_same_priority(self):
        """Active goals should come before pending at same priority."""
        mgr = GoalManager()
        pending = mgr.push("pending", priority=5)
        active = mgr.push("active", priority=5)
        mgr.activate(active)
        result = mgr.get_next()
        assert result.id == active


class TestTreeNavigation:

    def test_roots(self):
        mgr = GoalManager()
        r1 = mgr.push("root 1")
        r2 = mgr.push("root 2")
        mgr.push("child", parent_id=r1)
        roots = mgr.roots()
        assert len(roots) == 2

    def test_children(self):
        mgr = GoalManager()
        root = mgr.push("parent")
        mgr.decompose(root, ["a", "b"])
        children = mgr.children(root)
        assert len(children) == 2

    def test_children_not_found_raises(self):
        mgr = GoalManager()
        with pytest.raises(KeyError):
            mgr.children("nonexistent")
