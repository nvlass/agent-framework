"""Tests for the two-layer soul model."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_core.soul import Soul, SoulProposal, SoulManager, IMMUTABLE_FILE, LEARNABLE_FILE
from agent_core.agent import AgentRole, AgentInstance
from agent_core.llm import MockChatLLM


# --- Soul dataclass ---

class TestSoul:
    def test_merged_both(self):
        soul = Soul(immutable="I am always honest.", learnable="Be concise.")
        merged = soul.merged
        assert "Core Identity" in merged
        assert "I am always honest." in merged
        assert "Operational Guidelines" in merged
        assert "Be concise." in merged

    def test_merged_immutable_only(self):
        soul = Soul(immutable="core values", learnable="")
        assert "core values" in soul.merged
        assert "Operational Guidelines" not in soul.merged

    def test_merged_learnable_only(self):
        soul = Soul(immutable="", learnable="be brief")
        assert "be brief" in soul.merged
        assert "Core Identity" not in soul.merged

    def test_merged_empty(self):
        soul = Soul()
        assert soul.merged == ""

    def test_str_is_merged(self):
        soul = Soul(immutable="x", learnable="y")
        assert str(soul) == soul.merged


# --- SoulManager ---

@pytest.fixture
def soul_dir(tmp_path):
    (tmp_path / IMMUTABLE_FILE).write_text("I never lie.")
    (tmp_path / LEARNABLE_FILE).write_text("Be concise.")
    return tmp_path


class TestSoulManager:
    def test_load_both_files(self, soul_dir):
        manager = SoulManager(soul_dir)
        soul = manager.load()
        assert soul.immutable == "I never lie."
        assert soul.learnable == "Be concise."

    def test_load_missing_learnable(self, tmp_path):
        (tmp_path / IMMUTABLE_FILE).write_text("core")
        manager = SoulManager(tmp_path)
        soul = manager.load()
        assert soul.immutable == "core"
        assert soul.learnable == ""

    def test_load_missing_immutable(self, tmp_path):
        (tmp_path / LEARNABLE_FILE).write_text("style")
        manager = SoulManager(tmp_path)
        soul = manager.load()
        assert soul.immutable == ""
        assert soul.learnable == "style"

    def test_load_empty_dir(self, tmp_path):
        manager = SoulManager(tmp_path)
        soul = manager.load()
        assert soul.immutable == ""
        assert soul.learnable == ""

    def test_propose_change_creates_proposal(self, soul_dir):
        manager = SoulManager(soul_dir)
        soul = manager.load()
        p = manager.propose_change(
            section="verbosity",
            current=soul.learnable,
            proposed="Be very concise.",
            reasoning="Shorter is better.",
        )
        assert p.status == "pending"
        assert p.section == "verbosity"
        assert p.proposed == "Be very concise."
        assert p.reasoning == "Shorter is better."
        assert len(p.id) > 0

    def test_propose_persists_to_disk(self, soul_dir):
        manager = SoulManager(soul_dir)
        manager.propose_change("x", "old", "new", "because")
        proposals_file = soul_dir / ".soul_proposals.json"
        assert proposals_file.exists()
        data = json.loads(proposals_file.read_text())
        assert len(data) == 1
        assert data[0]["status"] == "pending"

    def test_list_pending_proposals(self, soul_dir):
        manager = SoulManager(soul_dir)
        manager.propose_change("a", "old", "new1", "r1")
        manager.propose_change("b", "old", "new2", "r2")
        pending = manager.list_proposals(status="pending")
        assert len(pending) == 2

    def test_list_all_proposals(self, soul_dir):
        manager = SoulManager(soul_dir)
        p = manager.propose_change("a", "old", "new", "r")
        manager.approve(p.id)
        manager.propose_change("b", "old2", "new2", "r2")
        all_proposals = manager.list_proposals()
        assert len(all_proposals) == 2

    def test_approve_updates_learnable_file(self, soul_dir):
        manager = SoulManager(soul_dir)
        p = manager.propose_change("style", "Be concise.", "Be extremely concise.", "better")
        updated_soul = manager.approve(p.id)
        assert (soul_dir / LEARNABLE_FILE).read_text() == "Be extremely concise."
        assert updated_soul.learnable == "Be extremely concise."

    def test_approve_marks_proposal_approved(self, soul_dir):
        manager = SoulManager(soul_dir)
        p = manager.propose_change("x", "old", "new", "r")
        manager.approve(p.id)
        proposals = manager.list_proposals(status="approved")
        assert len(proposals) == 1
        assert proposals[0].id == p.id

    def test_reject_marks_proposal_rejected(self, soul_dir):
        manager = SoulManager(soul_dir)
        p = manager.propose_change("x", "old", "new", "r")
        manager.reject(p.id)
        proposals = manager.list_proposals(status="rejected")
        assert len(proposals) == 1

    def test_reject_does_not_modify_file(self, soul_dir):
        manager = SoulManager(soul_dir)
        original = (soul_dir / LEARNABLE_FILE).read_text()
        p = manager.propose_change("x", original, "something else", "r")
        manager.reject(p.id)
        assert (soul_dir / LEARNABLE_FILE).read_text() == original

    def test_approve_already_approved_raises(self, soul_dir):
        manager = SoulManager(soul_dir)
        p = manager.propose_change("x", "old", "new", "r")
        manager.approve(p.id)
        with pytest.raises(ValueError, match="already"):
            manager.approve(p.id)

    def test_reject_already_rejected_raises(self, soul_dir):
        manager = SoulManager(soul_dir)
        p = manager.propose_change("x", "old", "new", "r")
        manager.reject(p.id)
        with pytest.raises(ValueError, match="already"):
            manager.reject(p.id)

    def test_unknown_id_raises(self, soul_dir):
        manager = SoulManager(soul_dir)
        with pytest.raises(KeyError):
            manager.approve("nonexistent")

    def test_immutable_never_modified(self, soul_dir):
        """Approving a learnable change never touches SOUL_IMMUTABLE.md."""
        manager = SoulManager(soul_dir)
        original_immutable = (soul_dir / IMMUTABLE_FILE).read_text()
        p = manager.propose_change("learnable", "old", "new learnable", "r")
        manager.approve(p.id)
        assert (soul_dir / IMMUTABLE_FILE).read_text() == original_immutable


# --- AgentRole.from_soul_dir ---

class TestAgentRoleFromSoulDir:
    def test_loads_merged_soul(self, soul_dir):
        role = AgentRole.from_soul_dir("agent", soul_dir)
        assert "I never lie." in role.soul
        assert "Be concise." in role.soul

    def test_soul_manager_attached(self, soul_dir):
        role = AgentRole.from_soul_dir("agent", soul_dir)
        assert role.soul_manager is not None
        assert isinstance(role.soul_manager, SoulManager)

    def test_from_soul_file_no_manager(self, tmp_path):
        """from_soul_file sets no soul_manager (backwards compat)."""
        f = tmp_path / "soul.txt"
        f.write_text("be helpful")
        role = AgentRole.from_soul_file("agent", f)
        assert role.soul_manager is None
        assert role.soul == "be helpful"


# --- propose_soul_change tool integration ---

class TestSoulChangeToolRegistered:
    def test_tool_registered_when_soul_manager_present(self, soul_dir):
        role = AgentRole.from_soul_dir("agent", soul_dir)
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm)
        assert agent.registry.get("propose_soul_change") is not None

    def test_tool_not_registered_without_soul_manager(self):
        role = AgentRole(name="agent", soul="plain soul")
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm)
        assert agent.registry.get("propose_soul_change") is None

    def test_tool_call_creates_proposal(self, soul_dir):
        role = AgentRole.from_soul_dir("agent", soul_dir)
        llm = MockChatLLM([
            'Thought: propose\nAction: propose_soul_change\n'
            'Action Args: {"section": "tone", "proposed": "Be friendlier.", "reasoning": "users prefer it"}',
            "Thought: done\nAnswer: proposed",
        ])
        agent = AgentInstance(role, llm)
        result = agent.run("improve yourself")
        assert result.success
        proposals = role.soul_manager.list_proposals(status="pending")
        assert len(proposals) == 1
        assert proposals[0].section == "tone"

    def test_reload_soul_picks_up_approved_change(self, soul_dir):
        role = AgentRole.from_soul_dir("agent", soul_dir)
        llm = MockChatLLM(["Thought: x\nAnswer: y"])
        agent = AgentInstance(role, llm)
        # Simulate human approving a change
        p = role.soul_manager.propose_change("x", "old", "New learnable content.", "better")
        role.soul_manager.approve(p.id)
        # reload_soul picks up the change
        agent.reload_soul()
        assert "New learnable content." in role.soul
