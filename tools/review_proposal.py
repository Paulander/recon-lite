#!/usr/bin/env python3
"""
M5.2: Human-in-the-loop review tool for script proposals.

Usage:
    # View a proposal
    uv run python tools/review_proposal.py --proposal proposals/tactical_fork_v1.yaml --view
    
    # Accept a proposal
    uv run python tools/review_proposal.py --proposal proposals/tactical_fork_v1.yaml \
        --action accept --reviewer "alice" --notes "Looks good, integrate with tactics subgraph"
    
    # Reject a proposal
    uv run python tools/review_proposal.py --proposal proposals/tactical_fork_v1.yaml \
        --action reject --reviewer "bob" --notes "Too few samples, needs more data"
    
    # List all proposals
    uv run python tools/review_proposal.py --list proposals/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_proposal(path: Path) -> dict:
    """Load a proposal from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_proposal(proposal: dict, path: Path) -> None:
    """Save a proposal to YAML file."""
    with open(path, "w") as f:
        yaml.dump(proposal, f, default_flow_style=False, sort_keys=False)


def print_proposal(proposal: dict) -> None:
    """Print a proposal in human-readable format."""
    print(f"\n{'=' * 60}")
    print(f"Proposal: {proposal.get('proposal_id', 'unknown')}")
    print(f"{'=' * 60}")
    
    print(f"\nCluster: {proposal.get('cluster_id', 'unknown')}")
    print(f"Status: {proposal.get('status', 'unknown')}")
    print(f"Created: {proposal.get('created_at', 'unknown')}")
    
    stats = proposal.get("statistics", {})
    print(f"\n--- Statistics ---")
    print(f"  Sample count: {stats.get('sample_count', 0)}")
    print(f"  Avg outcome: {stats.get('avg_outcome', 0):.4f}")
    print(f"  Positive ratio: {stats.get('positive_ratio', 0):.1%}")
    print(f"  Confidence: {stats.get('confidence', 0):.1%}")
    
    print(f"\n--- Preconditions ---")
    for pc in proposal.get("preconditions", []):
        req = "required" if pc.get("required") else "optional"
        print(f"  • {pc.get('sensor', 'unknown')} ({req})")
    
    print(f"\n--- Suggested Actuator ---")
    actuator = proposal.get("suggested_actuator", {})
    print(f"  Type: {actuator.get('type', 'unknown')}")
    print(f"  Priority: {actuator.get('priority', 'unknown')}")
    
    print(f"\n--- Suggested Nodes ---")
    for node in proposal.get("suggested_nodes", []):
        print(f"  • {node.get('id')}: {node.get('type')} - {node.get('description', '')}")
    
    print(f"\n--- Edges ---")
    for edge in proposal.get("edges", []):
        print(f"  • {edge.get('src')} --[{edge.get('ltype')}]--> {edge.get('dst')}")
    
    fens = proposal.get("sample_fens", [])
    if fens:
        print(f"\n--- Sample Positions ({len(fens)}) ---")
        for fen in fens[:3]:
            print(f"  • {fen}")
    
    active = proposal.get("trace_active_nodes", [])
    if active:
        print(f"\n--- Trace Active Nodes ---")
        print(f"  {', '.join(active[:10])}")
    
    review = proposal.get("review", {})
    if review.get("reviewer"):
        print(f"\n--- Review ---")
        print(f"  Reviewer: {review.get('reviewer')}")
        print(f"  Decision: {review.get('decision')}")
        print(f"  Date: {review.get('reviewed_at')}")
        if review.get("notes"):
            print(f"  Notes: {review.get('notes')}")
    
    print()


def apply_decision(
    proposal: dict,
    action: str,
    reviewer: str,
    notes: str,
) -> dict:
    """Apply a review decision to a proposal."""
    proposal["status"] = action
    proposal["review"] = {
        "reviewer": reviewer,
        "reviewed_at": datetime.utcnow().isoformat(),
        "decision": action,
        "notes": notes,
    }
    return proposal


def move_proposal(src_path: Path, action: str, proposals_dir: Path) -> Path:
    """Move proposal to accepted/rejected directory."""
    if action == "accept":
        dest_dir = proposals_dir / "accepted"
    elif action == "reject":
        dest_dir = proposals_dir / "rejected"
    else:
        return src_path
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    shutil.move(str(src_path), str(dest_path))
    return dest_path


def list_proposals(proposals_dir: Path) -> None:
    """List all proposals in a directory."""
    print(f"\n{'=' * 70}")
    print(f"Proposals in {proposals_dir}")
    print(f"{'=' * 70}")
    
    # Pending proposals
    pending = list(proposals_dir.glob("*.yaml"))
    if pending:
        print(f"\n--- Pending ({len(pending)}) ---")
        for p in sorted(pending):
            proposal = load_proposal(p)
            stats = proposal.get("statistics", {})
            print(f"  {p.name}: confidence={stats.get('confidence', 0):.1%}, samples={stats.get('sample_count', 0)}")
    
    # Accepted proposals
    accepted_dir = proposals_dir / "accepted"
    if accepted_dir.exists():
        accepted = list(accepted_dir.glob("*.yaml"))
        if accepted:
            print(f"\n--- Accepted ({len(accepted)}) ---")
            for p in sorted(accepted):
                proposal = load_proposal(p)
                review = proposal.get("review", {})
                print(f"  {p.name}: reviewer={review.get('reviewer', '?')}, date={review.get('reviewed_at', '?')[:10]}")
    
    # Rejected proposals
    rejected_dir = proposals_dir / "rejected"
    if rejected_dir.exists():
        rejected = list(rejected_dir.glob("*.yaml"))
        if rejected:
            print(f"\n--- Rejected ({len(rejected)}) ---")
            for p in sorted(rejected):
                proposal = load_proposal(p)
                review = proposal.get("review", {})
                print(f"  {p.name}: reviewer={review.get('reviewer', '?')}, reason={review.get('notes', '?')[:30]}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Review and manage script proposals."
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        help="Path to proposal YAML file",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="View proposal details",
    )
    parser.add_argument(
        "--action",
        choices=["accept", "reject", "defer"],
        help="Review action to take",
    )
    parser.add_argument(
        "--reviewer",
        type=str,
        default="anonymous",
        help="Reviewer name",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Review notes",
    )
    parser.add_argument(
        "--list",
        type=Path,
        dest="list_dir",
        help="List all proposals in directory",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Don't move proposal after accept/reject",
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list_dir:
        if not args.list_dir.exists():
            print(f"Error: Directory not found: {args.list_dir}")
            sys.exit(1)
        list_proposals(args.list_dir)
        return
    
    # Single proposal operations
    if not args.proposal:
        parser.print_help()
        sys.exit(1)
    
    if not args.proposal.exists():
        print(f"Error: Proposal not found: {args.proposal}")
        sys.exit(1)
    
    proposal = load_proposal(args.proposal)
    
    # View mode
    if args.view or not args.action:
        print_proposal(proposal)
        return
    
    # Apply action
    if args.action:
        proposal = apply_decision(proposal, args.action, args.reviewer, args.notes)
        save_proposal(proposal, args.proposal)
        print(f"Applied '{args.action}' to {args.proposal.name}")
        
        # Move to appropriate directory
        if not args.no_move and args.action in ("accept", "reject"):
            proposals_dir = args.proposal.parent
            new_path = move_proposal(args.proposal, args.action, proposals_dir)
            print(f"Moved to: {new_path}")
        
        print_proposal(proposal)


if __name__ == "__main__":
    main()

