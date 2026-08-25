"""Two-layer soul model: immutable core + learnable preferences.

The soul is split into two files in a directory:
  SOUL_IMMUTABLE.md   — core identity, ethics, never touched by the agent
  SOUL_LEARNABLE.md   — operational guidelines the agent can propose changes to

Change flow:
  1. Agent calls propose_change(section, proposed, reasoning)
  2. Proposal is saved to .soul_proposals.json (status: "pending")
  3. Human reviews via: python -m agent_core.soul_cli list/approve/reject
  4. On approve, SOUL_LEARNABLE.md is updated in-place
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


IMMUTABLE_FILE = "SOUL_IMMUTABLE.md"
LEARNABLE_FILE = "SOUL_LEARNABLE.md"
PROPOSALS_FILE = ".soul_proposals.json"


@dataclass
class Soul:
    """Merged soul from immutable + learnable layers."""

    immutable: str = ""
    learnable: str = ""

    @property
    def merged(self) -> str:
        """Combined soul text for prompt assembly."""
        parts = []
        if self.immutable:
            parts.append(f"## Core Identity\n{self.immutable}")
        if self.learnable:
            parts.append(f"## Operational Guidelines\n{self.learnable}")
        return "\n\n".join(parts)

    def __str__(self) -> str:
        return self.merged


@dataclass
class SoulProposal:
    """A proposed change to SOUL_LEARNABLE, pending human approval."""

    id: str
    section: str       # human-readable label for what's being changed
    current: str       # current text (for context in review)
    proposed: str      # the full new SOUL_LEARNABLE.md content
    reasoning: str     # why the agent wants this change
    created_at: str
    status: str = "pending"   # "pending" | "approved" | "rejected"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "section": self.section,
            "current": self.current,
            "proposed": self.proposed,
            "reasoning": self.reasoning,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoulProposal":
        return cls(**d)


class SoulManager:
    """Loads, persists, and manages proposals for a two-file soul.

    Usage:
        manager = SoulManager(Path("./soul"))
        soul = manager.load()            # Soul with .immutable / .learnable / .merged
        proposal = manager.propose_change(
            section="verbosity",
            current=soul.learnable,
            proposed="... updated text ...",
            reasoning="I found that shorter responses work better",
        )
        # Human reviews: python -m agent_core.soul_cli list/approve/reject
    """

    def __init__(self, soul_dir: Path) -> None:
        self._dir = Path(soul_dir)
        self._proposals_file = self._dir / PROPOSALS_FILE

    def load(self) -> Soul:
        """Load both soul files. Missing files produce empty strings."""
        immutable = self._read(IMMUTABLE_FILE)
        learnable = self._read(LEARNABLE_FILE)
        return Soul(immutable=immutable, learnable=learnable)

    def propose_change(
        self,
        section: str,
        current: str,
        proposed: str,
        reasoning: str,
    ) -> SoulProposal:
        """Record a proposal. Does NOT modify SOUL_LEARNABLE.md."""
        proposal = SoulProposal(
            id=str(uuid.uuid4())[:8],
            section=section,
            current=current,
            proposed=proposed,
            reasoning=reasoning,
            created_at=datetime.now().isoformat(),
            status="pending",
        )
        proposals = self._load_proposals()
        proposals.append(proposal)
        self._save_proposals(proposals)
        return proposal

    def list_proposals(self, status: Optional[str] = None) -> list[SoulProposal]:
        """Return proposals, optionally filtered by status."""
        proposals = self._load_proposals()
        if status:
            proposals = [p for p in proposals if p.status == status]
        return proposals

    def approve(self, proposal_id: str) -> Soul:
        """Apply a pending proposal to SOUL_LEARNABLE.md and mark approved."""
        proposals = self._load_proposals()
        proposal = self._find(proposals, proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"Proposal {proposal_id!r} is already {proposal.status!r}")
        # Write the proposed text to SOUL_LEARNABLE.md
        (self._dir / LEARNABLE_FILE).write_text(proposal.proposed)
        proposal.status = "approved"
        self._save_proposals(proposals)
        return self.load()

    def reject(self, proposal_id: str) -> None:
        """Mark a proposal as rejected (no file changes)."""
        proposals = self._load_proposals()
        proposal = self._find(proposals, proposal_id)
        if proposal.status != "pending":
            raise ValueError(f"Proposal {proposal_id!r} is already {proposal.status!r}")
        proposal.status = "rejected"
        self._save_proposals(proposals)

    # --- internal helpers ---

    def _read(self, filename: str) -> str:
        path = self._dir / filename
        return path.read_text().strip() if path.exists() else ""

    def _load_proposals(self) -> list[SoulProposal]:
        if not self._proposals_file.exists():
            return []
        data = json.loads(self._proposals_file.read_text())
        return [SoulProposal.from_dict(d) for d in data]

    def _save_proposals(self, proposals: list[SoulProposal]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._proposals_file.write_text(
            json.dumps([p.to_dict() for p in proposals], indent=2)
        )

    def _find(self, proposals: list[SoulProposal], proposal_id: str) -> SoulProposal:
        for p in proposals:
            if p.id == proposal_id:
                return p
        raise KeyError(f"Proposal {proposal_id!r} not found")
