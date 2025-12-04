#!/usr/bin/env python3
"""
Trace Summarize Tool (M4)

Load JSONL traces, aggregate episode summaries, and export metrics for
analysis and consolidation planning.

Usage:
    uv run python tools/trace_summarize.py reports/krk_trace.jsonl --output reports/summary.json
    uv run python tools/trace_summarize.py reports/krk_trace.jsonl --format csv --output reports/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recon_lite.trace_db import EpisodeRecord, EpisodeSummary, TraceDB


@dataclass
class AggregatedStats:
    """Aggregated statistics across multiple episodes."""

    total_episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    unknown: int = 0

    total_ticks: int = 0
    total_reward: float = 0.0
    reward_count: int = 0

    # Per-edge stats
    edge_delta_sums: Dict[str, float] = field(default_factory=dict)
    edge_episode_counts: Dict[str, int] = field(default_factory=dict)

    # Per-phase stats
    phase_usage: Dict[str, int] = field(default_factory=dict)

    # Per-bandit-arm stats
    bandit_pulls: Dict[str, Dict[str, int]] = field(default_factory=dict)
    bandit_rewards: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add_episode(self, episode: EpisodeRecord) -> None:
        """Add an episode's data to the aggregated stats."""
        self.total_episodes += 1

        # Count outcome
        result = episode.result
        if result == "1-0":
            self.wins += 1
        elif result == "0-1":
            self.losses += 1
        elif result in ("1/2-1/2", "draw"):
            self.draws += 1
        else:
            self.unknown += 1

        # Count ticks
        self.total_ticks += len(episode.ticks)

        # Aggregate from summary if available
        if episode.summary:
            self._add_summary(episode.summary)
        else:
            # Fall back to tick-level aggregation
            self._add_from_ticks(episode)

    def _add_summary(self, summary: EpisodeSummary) -> None:
        """Add data from an EpisodeSummary."""
        # Reward stats
        self.total_reward += summary.total_reward_tick
        self.reward_count += summary.reward_tick_count

        # Edge deltas
        for edge_key, delta in summary.edge_delta_sums.items():
            self.edge_delta_sums[edge_key] = self.edge_delta_sums.get(edge_key, 0.0) + delta
            self.edge_episode_counts[edge_key] = self.edge_episode_counts.get(edge_key, 0) + 1

        # Phase usage
        for phase, count in summary.phase_usage.items():
            self.phase_usage[phase] = self.phase_usage.get(phase, 0) + count

        # Bandit stats
        for parent_id, arms in summary.bandit_stats.items():
            if parent_id not in self.bandit_pulls:
                self.bandit_pulls[parent_id] = {}
                self.bandit_rewards[parent_id] = {}
            for arm_id, arm_summary in arms.items():
                self.bandit_pulls[parent_id][arm_id] = (
                    self.bandit_pulls[parent_id].get(arm_id, 0) + arm_summary.pulls
                )
                self.bandit_rewards[parent_id][arm_id] = (
                    self.bandit_rewards[parent_id].get(arm_id, 0.0) + arm_summary.sum_reward
                )

    def _add_from_ticks(self, episode: EpisodeRecord) -> None:
        """Fall back to aggregating from tick records."""
        for tick in episode.ticks:
            if tick.reward_tick is not None:
                self.total_reward += tick.reward_tick
                self.reward_count += 1
            if tick.phase_estimate:
                self.phase_usage[tick.phase_estimate] = (
                    self.phase_usage.get(tick.phase_estimate, 0) + 1
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dict."""
        avg_reward = self.total_reward / self.reward_count if self.reward_count > 0 else 0.0
        win_rate = self.wins / self.total_episodes if self.total_episodes > 0 else 0.0

        # Compute per-edge averages
        edge_avg_deltas = {}
        for edge_key, total_delta in self.edge_delta_sums.items():
            count = self.edge_episode_counts.get(edge_key, 1)
            edge_avg_deltas[edge_key] = round(total_delta / count, 4)

        # Compute bandit arm means
        bandit_means = {}
        for parent_id, arms in self.bandit_pulls.items():
            bandit_means[parent_id] = {}
            for arm_id, pulls in arms.items():
                if pulls > 0:
                    total_reward = self.bandit_rewards[parent_id].get(arm_id, 0.0)
                    bandit_means[parent_id][arm_id] = {
                        "pulls": pulls,
                        "mean_reward": round(total_reward / pulls, 4),
                    }

        return {
            "total_episodes": self.total_episodes,
            "outcomes": {
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "unknown": self.unknown,
                "win_rate": round(win_rate, 4),
            },
            "ticks": {
                "total": self.total_ticks,
                "avg_per_episode": round(self.total_ticks / max(1, self.total_episodes), 2),
            },
            "rewards": {
                "total": round(self.total_reward, 4),
                "count": self.reward_count,
                "avg": round(avg_reward, 4),
            },
            "phase_usage": dict(self.phase_usage),
            "edge_avg_deltas": edge_avg_deltas,
            "bandit_stats": bandit_means,
        }


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


def export_json(stats: AggregatedStats, output: Path) -> None:
    """Export stats to JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(stats.to_dict(), fh, indent=2)
    print(f"Wrote JSON summary to {output}")


def export_csv(stats: AggregatedStats, output: Path) -> None:
    """Export stats to CSV (one row per edge)."""
    output.parent.mkdir(parents=True, exist_ok=True)

    data = stats.to_dict()

    with output.open("w", newline="", encoding="utf-8") as fh:
        # Write overview section
        writer = csv.writer(fh)
        writer.writerow(["# Overview"])
        writer.writerow(["total_episodes", data["total_episodes"]])
        writer.writerow(["wins", data["outcomes"]["wins"]])
        writer.writerow(["losses", data["outcomes"]["losses"]])
        writer.writerow(["draws", data["outcomes"]["draws"]])
        writer.writerow(["win_rate", data["outcomes"]["win_rate"]])
        writer.writerow(["avg_reward", data["rewards"]["avg"]])
        writer.writerow([])

        # Write edge deltas
        writer.writerow(["# Edge Deltas"])
        writer.writerow(["edge_key", "avg_delta", "episode_count"])
        for edge_key, avg_delta in sorted(data["edge_avg_deltas"].items()):
            count = stats.edge_episode_counts.get(edge_key, 0)
            writer.writerow([edge_key, avg_delta, count])
        writer.writerow([])

        # Write phase usage
        writer.writerow(["# Phase Usage"])
        writer.writerow(["phase", "tick_count"])
        for phase, count in sorted(data["phase_usage"].items()):
            writer.writerow([phase, count])
        writer.writerow([])

        # Write bandit stats
        writer.writerow(["# Bandit Stats"])
        writer.writerow(["parent", "arm", "pulls", "mean_reward"])
        for parent_id, arms in sorted(data["bandit_stats"].items()):
            for arm_id, arm_data in sorted(arms.items()):
                writer.writerow([
                    parent_id,
                    arm_id,
                    arm_data["pulls"],
                    arm_data["mean_reward"],
                ])

    print(f"Wrote CSV summary to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate trace metrics for M4 consolidation planning"
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
        default=Path("reports/trace_summary.json"),
        help="Output file path (default: reports/trace_summary.json)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    # Load traces
    print(f"Loading traces from {len(args.traces)} file(s)...")
    episodes = load_traces(args.traces)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Aggregate stats
    stats = AggregatedStats()
    for ep in episodes:
        stats.add_episode(ep)

    # Export
    if args.format == "json":
        export_json(stats, args.output)
    else:
        export_csv(stats, args.output)

    # Print summary
    data = stats.to_dict()
    print(f"\nSummary:")
    print(f"  Episodes: {data['total_episodes']}")
    print(f"  Win rate: {data['outcomes']['win_rate']:.1%}")
    print(f"  Avg reward: {data['rewards']['avg']:.4f}")
    print(f"  Edges tracked: {len(data['edge_avg_deltas'])}")


if __name__ == "__main__":
    main()

