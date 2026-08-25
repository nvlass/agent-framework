"""CLI for reviewing and managing soul proposals.

Usage:
    python -m agent_core.soul_cli --soul-dir ./soul list
    python -m agent_core.soul_cli --soul-dir ./soul approve <id>
    python -m agent_core.soul_cli --soul-dir ./soul reject <id>
    python -m agent_core.soul_cli --soul-dir ./soul show <id>
"""

import argparse
import sys
from pathlib import Path

from agent_core.soul import SoulManager


def cmd_list(manager: SoulManager, status: str) -> None:
    proposals = manager.list_proposals(status=status if status != "all" else None)
    if not proposals:
        label = f"({status})" if status != "all" else ""
        print(f"No proposals {label}.")
        return
    for p in proposals:
        print(f"[{p.status.upper():8s}] {p.id}  {p.created_at[:16]}  {p.section!r}")
        print(f"           Reason: {p.reasoning[:80]}")
        print()


def cmd_show(manager: SoulManager, proposal_id: str) -> None:
    proposals = manager.list_proposals()
    found = [p for p in proposals if p.id == proposal_id]
    if not found:
        print(f"Proposal {proposal_id!r} not found.", file=sys.stderr)
        sys.exit(1)
    p = found[0]
    print(f"ID:        {p.id}")
    print(f"Status:    {p.status}")
    print(f"Created:   {p.created_at}")
    print(f"Section:   {p.section}")
    print(f"Reasoning: {p.reasoning}")
    print()
    print("--- CURRENT ---")
    print(p.current or "(empty)")
    print()
    print("--- PROPOSED ---")
    print(p.proposed)


def cmd_approve(manager: SoulManager, proposal_id: str) -> None:
    try:
        soul = manager.approve(proposal_id)
        print(f"Approved {proposal_id!r}. SOUL_LEARNABLE.md updated.")
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_reject(manager: SoulManager, proposal_id: str) -> None:
    try:
        manager.reject(proposal_id)
        print(f"Rejected {proposal_id!r}.")
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage agent soul proposals")
    parser.add_argument("--soul-dir", default="./soul",
                        help="Directory containing SOUL_IMMUTABLE.md and SOUL_LEARNABLE.md")
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List proposals")
    list_p.add_argument("--status", default="pending",
                        choices=["pending", "approved", "rejected", "all"],
                        help="Filter by status (default: pending)")

    show_p = sub.add_parser("show", help="Show proposal details and diff")
    show_p.add_argument("id", help="Proposal ID")

    approve_p = sub.add_parser("approve", help="Approve and apply a proposal")
    approve_p.add_argument("id", help="Proposal ID")

    reject_p = sub.add_parser("reject", help="Reject a proposal")
    reject_p.add_argument("id", help="Proposal ID")

    args = parser.parse_args()
    manager = SoulManager(Path(args.soul_dir))

    if args.command == "list":
        cmd_list(manager, args.status)
    elif args.command == "show":
        cmd_show(manager, args.id)
    elif args.command == "approve":
        cmd_approve(manager, args.id)
    elif args.command == "reject":
        cmd_reject(manager, args.id)


if __name__ == "__main__":
    main()
