#!/usr/bin/env python3
"""
Batch Consolidation Tool (M4)

Load trace JSONL files, run the consolidation engine on all episodes,
and export updated weight packs and bandit priors.

Usage:
    uv run python tools/consolidate_batch.py reports/*.jsonl --output weights/krk_consolidated.json
    uv run python tools/consolidate_batch.py reports/*.jsonl --existing weights/krk_consolidated.json --min-episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recon_lite.trace_db import EpisodeRecord, EpisodeSummary, TraceDB
from src.recon_lite.plasticity.consolidate import (
    ConsolidationConfig,
    ConsolidationEngine,
)
from src.recon_lite.plasticity.bandit import (
    BanditPriors,
    merge_priors,
    save_priors,
    load_priors,
)


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


def aggregate_bandit_priors(episodes: List[EpisodeRecord]) -> BanditPriors:
    """Aggregate bandit statistics from episode summaries."""
    from src.recon_lite.trace_db import BanditArmSummary

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch consolidation on trace files"
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
        default=Path("weights/krk_consolidated.json"),
        help="Output file path for consolidation state (default: weights/krk_consolidated.json)",
    )
    parser.add_argument(
        "--existing", "-e",
        type=Path,
        default=None,
        help="Existing consolidation state to load and update (optional)",
    )
    parser.add_argument(
        "--bandit-output",
        type=Path,
        default=None,
        help="Output file path for bandit priors (optional)",
    )
    parser.add_argument(
        "--swp-output",
        type=Path,
        default=None,
        help="Output file path for SWP weight pack (optional)",
    )
    parser.add_argument(
        "--eta", "-r",
        type=float,
        default=0.01,
        help="Consolidation learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--min-episodes", "-n",
        type=int,
        default=1,
        help="Minimum episodes before applying (default: 1 for batch mode)",
    )
    parser.add_argument(
        "--outcome-weight",
        type=float,
        default=0.5,
        help="Weight for outcome vs tick rewards (default: 0.5)",
    )
    parser.add_argument(
        "--max-base-delta",
        type=float,
        default=0.5,
        help="Maximum w_base change per consolidation (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    args = parser.parse_args()

    # Load traces
    print(f"Loading traces from {len(args.traces)} file(s)...")
    episodes = load_traces(args.traces)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Count episodes with summaries
    episodes_with_summaries = [ep for ep in episodes if ep.summary]
    print(f"Episodes with summaries: {len(episodes_with_summaries)}")

    if not episodes_with_summaries:
        print("No episodes have summaries. Run with --plasticity or --bandit to generate summaries.", file=sys.stderr)
        sys.exit(1)

    # Initialize consolidation engine
    config = ConsolidationConfig(
        eta_consolidate=args.eta,
        min_episodes=args.min_episodes,
        outcome_weight=args.outcome_weight,
        max_base_delta=args.max_base_delta,
        enabled=True,
    )
    engine = ConsolidationEngine(config)

    # Load existing state if provided
    if args.existing and args.existing.exists():
        print(f"Loading existing consolidation state from {args.existing}...")
        engine.load_state(args.existing)
        print(f"  Loaded state with {engine.total_episodes} prior episodes")

    # Accumulate all episodes
    print("Accumulating episode summaries...")
    for ep in episodes_with_summaries:
        engine.accumulate_episode(ep.summary)

    print(f"  Total episodes accumulated: {engine.total_episodes}")
    print(f"  Edges tracked: {len(engine.edge_states)}")

    # Check if we should apply
    if engine.episodes_since_apply < args.min_episodes:
        print(f"Not enough episodes ({engine.episodes_since_apply} < {args.min_episodes}). Skipping consolidation.")
    else:
        # Apply consolidation (we need a dummy graph for this)
        # In batch mode, we just update the w_base values in the engine state
        print("Computing consolidation updates...")

        # Preview changes
        preview = {}
        for edge_key, state in engine.edge_states.items():
            if state.episode_count > 0:
                mean_delta = state.mean_weighted_delta()
                delta_w_base = config.eta_consolidate * mean_delta
                delta_w_base = max(-config.max_base_delta, min(config.max_base_delta, delta_w_base))
                new_w_base = max(config.w_min, min(config.w_max, state.w_base + delta_w_base))
                actual_delta = new_w_base - state.w_base
                if abs(actual_delta) > 1e-6:
                    preview[edge_key] = {
                        "old_w_base": round(state.w_base, 4),
                        "new_w_base": round(new_w_base, 4),
                        "delta": round(actual_delta, 4),
                    }

        if preview:
            print(f"\nProposed w_base changes ({len(preview)} edges):")
            for edge_key, changes in sorted(preview.items(), key=lambda x: abs(x[1]["delta"]), reverse=True)[:10]:
                print(f"  {edge_key}: {changes['old_w_base']:.4f} -> {changes['new_w_base']:.4f} (Δ={changes['delta']:+.4f})")
            if len(preview) > 10:
                print(f"  ... and {len(preview) - 10} more")

        if not args.dry_run:
            # Apply changes to engine state
            for edge_key, state in engine.edge_states.items():
                if state.episode_count > 0:
                    mean_delta = state.mean_weighted_delta()
                    delta_w_base = config.eta_consolidate * mean_delta
                    delta_w_base = max(-config.max_base_delta, min(config.max_base_delta, delta_w_base))
                    new_w_base = max(config.w_min, min(config.w_max, state.w_base + delta_w_base))
                    state.w_base = new_w_base
                    state.accumulated_weighted_delta = 0.0
                    state.episode_count = 0

            engine.episodes_since_apply = 0

    # Save outputs
    if not args.dry_run:
        # Save consolidation state
        args.output.parent.mkdir(parents=True, exist_ok=True)
        engine.save_state(args.output)
        print(f"\nWrote consolidation state to {args.output}")

        # Save SWP if requested
        if args.swp_output:
            engine.export_w_base_pack(args.swp_output)
            print(f"Wrote SWP weight pack to {args.swp_output}")

        # Save bandit priors if requested
        if args.bandit_output:
            bandit_priors = aggregate_bandit_priors(episodes_with_summaries)
            save_priors(bandit_priors, args.bandit_output)
            print(f"Wrote bandit priors to {args.bandit_output}")

    # Print summary
    print("\n=== Consolidation Summary ===")
    print(f"Total episodes processed: {len(episodes_with_summaries)}")

    # Compute outcome stats
    wins = sum(1 for ep in episodes_with_summaries if ep.result == "1-0")
    losses = sum(1 for ep in episodes_with_summaries if ep.result == "0-1")
    draws = sum(1 for ep in episodes_with_summaries if ep.result in ("1/2-1/2", "draw"))
    print(f"Outcomes: {wins} wins, {losses} losses, {draws} draws")

    if episodes_with_summaries:
        avg_reward = sum(ep.summary.avg_reward_tick for ep in episodes_with_summaries) / len(episodes_with_summaries)
        print(f"Average reward per tick: {avg_reward:.4f}")

    print(f"Edges in consolidation state: {len(engine.edge_states)}")


if __name__ == "__main__":
    main()

