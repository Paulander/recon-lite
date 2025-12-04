#!/usr/bin/env python3
"""
Bandit Refresh Tool (M4)

Load traces, aggregate bandit statistics across episodes, apply decay for
aging data, and export updated priors for future games.

Usage:
    uv run python tools/bandit_refresh.py reports/krk_trace.jsonl --output weights/bandit_priors.json
    uv run python tools/bandit_refresh.py reports/*.jsonl --existing weights/bandit_priors.json --decay 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recon_lite.trace_db import EpisodeRecord, TraceDB
from src.recon_lite.plasticity.bandit import (
    BanditPriors,
    merge_priors,
    save_priors,
    load_priors,
)


def aggregate_priors_from_episodes(episodes: List[EpisodeRecord]) -> BanditPriors:
    """
    Aggregate bandit statistics from episode summaries.

    Args:
        episodes: List of EpisodeRecord objects

    Returns:
        BanditPriors with aggregated statistics
    """
    priors = BanditPriors()

    for episode in episodes:
        if not episode.summary:
            continue

        for parent_id, arms in episode.summary.bandit_stats.items():
            if parent_id not in priors.arm_stats:
                priors.arm_stats[parent_id] = {}

            for arm_id, arm_summary in arms.items():
                if arm_id not in priors.arm_stats[parent_id]:
                    priors.arm_stats[parent_id][arm_id] = {
                        "pulls": 0,
                        "sum_reward": 0.0,
                        "mean_reward": 0.0,
                    }

                stats = priors.arm_stats[parent_id][arm_id]
                stats["pulls"] += arm_summary.pulls
                stats["sum_reward"] += arm_summary.sum_reward

        priors.total_episodes += 1

    # Compute mean rewards
    for parent_id, arms in priors.arm_stats.items():
        for arm_id, stats in arms.items():
            if stats["pulls"] > 0:
                stats["mean_reward"] = round(stats["sum_reward"] / stats["pulls"], 4)

    return priors


def load_traces(paths: List[Path]) -> List[EpisodeRecord]:
    """Load episodes from one or more JSONL trace files."""
    episodes = []
    for path in paths:
        if not path.exists():
            print(f"Warning: Trace file not found: {path}", file=sys.stderr)
            continue
        try:
            episodes.extend(TraceDB.load_episodes(path))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate bandit statistics and export priors for M4 consolidation"
    )
    parser.add_argument(
        "traces",
        type=Path,
        nargs="+",
        help="JSONL trace file(s) to process",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("weights/bandit_priors.json"),
        help="Output file path (default: weights/bandit_priors.json)",
    )
    parser.add_argument(
        "--existing", "-e",
        type=Path,
        default=None,
        help="Existing priors file to merge with (optional)",
    )
    parser.add_argument(
        "--decay", "-d",
        type=float,
        default=0.9,
        help="Decay factor for existing priors (default: 0.9)",
    )
    args = parser.parse_args()

    # Load traces
    print(f"Loading traces from {len(args.traces)} file(s)...")
    episodes = load_traces(args.traces)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Aggregate new priors
    print("Aggregating bandit statistics...")
    new_priors = aggregate_priors_from_episodes(episodes)
    print(f"  Episodes with summaries: {new_priors.total_episodes}")
    print(f"  Parents tracked: {len(new_priors.arm_stats)}")

    # Merge with existing if provided
    if args.existing and args.existing.exists():
        print(f"Loading existing priors from {args.existing}...")
        old_priors = load_priors(args.existing)
        print(f"  Existing episodes: {old_priors.total_episodes}")

        print(f"Merging with decay factor {args.decay}...")
        final_priors = merge_priors(old_priors, new_priors, decay=args.decay)
        print(f"  Total episodes after merge: {final_priors.total_episodes}")
    else:
        final_priors = new_priors

    # Save
    save_priors(final_priors, args.output)
    print(f"\nWrote bandit priors to {args.output}")

    # Print summary
    print("\nSummary by parent:")
    for parent_id, arms in sorted(final_priors.arm_stats.items()):
        total_pulls = sum(a.get("pulls", 0) for a in arms.values())
        print(f"  {parent_id}: {len(arms)} arms, {total_pulls:.0f} total pulls")
        for arm_id, stats in sorted(arms.items()):
            print(f"    - {arm_id}: pulls={stats['pulls']:.0f}, mean={stats['mean_reward']:.4f}")


if __name__ == "__main__":
    main()

