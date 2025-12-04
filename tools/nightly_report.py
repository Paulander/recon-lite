#!/usr/bin/env python3
"""
Nightly Report Generator

Generates markdown reports from training runs and maintains a history
of results over time for tracking learning progress.

Usage:
    uv run python tools/nightly_report.py \
        --consol-state weights/krk_consol.json \
        --trace-dir reports/nightly \
        --history reports/nightly_history.json \
        --output reports/nightly_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_consolidation_state(path: Path) -> Optional[Dict[str, Any]]:
    """Load consolidation state JSON."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_trace_summaries(trace_dir: Path) -> List[Dict[str, Any]]:
    """Load all trace/summary JSONs from a directory."""
    summaries = []
    if not trace_dir.exists():
        return summaries
    
    for path in sorted(trace_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["_source_file"] = path.name
            summaries.append(data)
        except Exception:
            continue
    
    return summaries


def compute_session_stats(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stats from trace summaries."""
    if not summaries:
        return {}
    
    total_episodes = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_reward = 0.0
    total_ticks = 0
    
    for s in summaries:
        episodes = s.get("total_episodes", 0)
        total_episodes += episodes
        total_wins += s.get("wins", 0)
        total_losses += s.get("losses", 0)
        total_draws += s.get("draws", 0)
        
        if "avg_reward_tick" in s and "total_ticks" in s:
            total_reward += s["avg_reward_tick"] * s["total_ticks"]
            total_ticks += s["total_ticks"]
    
    win_rate = total_wins / total_episodes * 100 if total_episodes > 0 else 0.0
    avg_reward = total_reward / total_ticks if total_ticks > 0 else 0.0
    
    return {
        "total_episodes": total_episodes,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "win_rate": win_rate,
        "avg_reward_tick": avg_reward,
    }


def load_history(path: Path) -> List[Dict[str, Any]]:
    """Load history from JSON file."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def save_history(path: Path, history: List[Dict[str, Any]]) -> None:
    """Save history to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n")


def compute_weight_stats(consol_state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics from consolidation state."""
    w_base = consol_state.get("w_base", {})
    w_init = consol_state.get("w_init", {})
    
    if not w_base:
        return {}
    
    weights = list(w_base.values())
    deltas = [w_base[k] - w_init.get(k, w_base[k]) for k in w_base]
    
    # Top movers
    changes = [
        {"edge": k, "delta": w_base[k] - w_init.get(k, w_base[k])}
        for k in w_base
    ]
    changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    top_movers = changes[:5]
    
    return {
        "edges_tracked": len(w_base),
        "w_base_mean": sum(weights) / len(weights) if weights else 0,
        "w_base_min": min(weights) if weights else 0,
        "w_base_max": max(weights) if weights else 0,
        "delta_mean": sum(deltas) / len(deltas) if deltas else 0,
        "delta_abs_sum": sum(abs(d) for d in deltas),
        "top_movers": top_movers,
    }


def generate_histogram_ascii(values: List[float], bins: int = 8) -> str:
    """Generate ASCII histogram."""
    if not values:
        return "(no data)"
    
    min_v = min(values)
    max_v = max(values)
    
    if max_v == min_v:
        return f"All values = {min_v:.4f}"
    
    bin_width = (max_v - min_v) / bins
    counts = [0] * bins
    
    for v in values:
        idx = min(int((v - min_v) / bin_width), bins - 1)
        counts[idx] += 1
    
    max_count = max(counts)
    result = []
    for i in range(bins):
        bar_len = int(counts[i] / max_count * 20) if max_count > 0 else 0
        bar = "█" * bar_len
        low = min_v + i * bin_width
        high = min_v + (i + 1) * bin_width
        result.append(f"[{low:7.3f}, {high:7.3f}) | {bar} {counts[i]}")
    
    return "\n".join(result)


def generate_report(
    consol_state: Optional[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    run_name: str,
) -> str:
    """Generate markdown report."""
    now = datetime.now()
    report = []
    
    report.append(f"# Nightly Training Report")
    report.append(f"\n**Generated**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Run**: {run_name}")
    report.append("")
    
    # Session stats
    report.append("## Session Statistics")
    report.append("")
    
    session_stats = compute_session_stats(summaries)
    if session_stats:
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Total Episodes | {session_stats.get('total_episodes', 0)} |")
        report.append(f"| Wins | {session_stats.get('wins', 0)} |")
        report.append(f"| Losses | {session_stats.get('losses', 0)} |")
        report.append(f"| Draws | {session_stats.get('draws', 0)} |")
        report.append(f"| Win Rate | {session_stats.get('win_rate', 0):.1f}% |")
        report.append(f"| Avg Reward/Tick | {session_stats.get('avg_reward_tick', 0):.5f} |")
    else:
        report.append("*No trace summaries found.*")
    report.append("")
    
    # Consolidation state
    report.append("## Consolidation State")
    report.append("")
    
    if consol_state:
        meta = consol_state.get("consolidation_meta", {})
        config = consol_state.get("config", {})
        
        report.append(f"| Parameter | Value |")
        report.append(f"|-----------|-------|")
        report.append(f"| Total Episodes | {meta.get('total_episodes', 0)} |")
        report.append(f"| Episodes Since Apply | {meta.get('episodes_since_apply', 0)} |")
        report.append(f"| Learning Rate (η) | {config.get('eta_consolidate', 'N/A')} |")
        report.append(f"| Min Episodes | {config.get('min_episodes', 'N/A')} |")
        report.append(f"| Last Apply | {meta.get('last_apply_time', 'N/A')} |")
        report.append("")
        
        # Weight stats
        weight_stats = compute_weight_stats(consol_state)
        if weight_stats:
            report.append("### Weight Statistics")
            report.append("")
            report.append(f"- Edges tracked: {weight_stats['edges_tracked']}")
            report.append(f"- w_base mean: {weight_stats['w_base_mean']:.4f}")
            report.append(f"- w_base range: [{weight_stats['w_base_min']:.4f}, {weight_stats['w_base_max']:.4f}]")
            report.append(f"- Delta abs sum: {weight_stats['delta_abs_sum']:.4f}")
            report.append("")
            
            # Top movers
            if weight_stats.get("top_movers"):
                report.append("### Top Weight Changes")
                report.append("")
                report.append("| Edge | Delta |")
                report.append("|------|-------|")
                for m in weight_stats["top_movers"]:
                    sign = "+" if m["delta"] > 0 else ""
                    report.append(f"| {m['edge']} | {sign}{m['delta']:.4f} |")
            report.append("")
            
            # Histogram
            w_base = consol_state.get("w_base", {})
            if w_base:
                report.append("### Weight Distribution")
                report.append("")
                report.append("```")
                report.append(generate_histogram_ascii(list(w_base.values())))
                report.append("```")
                report.append("")
    else:
        report.append("*No consolidation state loaded.*")
        report.append("")
    
    # History trend
    report.append("## Historical Trend")
    report.append("")
    
    if len(history) >= 2:
        report.append("| Date | Episodes | Win Rate | Delta Sum |")
        report.append("|------|----------|----------|-----------|")
        for h in history[-10:]:  # Last 10 entries
            report.append(
                f"| {h.get('date', 'N/A')} | "
                f"{h.get('total_episodes', 0)} | "
                f"{h.get('win_rate', 0):.1f}% | "
                f"{h.get('delta_abs_sum', 0):.4f} |"
            )
        report.append("")
        
        # Trend indicators
        latest = history[-1]
        prev = history[-2]
        
        wr_diff = latest.get("win_rate", 0) - prev.get("win_rate", 0)
        wr_arrow = "📈" if wr_diff > 0 else "📉" if wr_diff < 0 else "➡️"
        
        report.append(f"**Win Rate Trend**: {wr_arrow} {wr_diff:+.1f}% from previous run")
        report.append("")
    else:
        report.append("*Not enough history for trend analysis (need ≥2 runs).*")
        report.append("")
    
    # Next steps
    report.append("## Recommended Actions")
    report.append("")
    
    if session_stats.get("win_rate", 0) < 50:
        report.append("- ⚠️ Win rate below 50% - consider reviewing weight updates")
    
    if consol_state:
        meta = consol_state.get("consolidation_meta", {})
        if meta.get("episodes_since_apply", 0) > 50:
            report.append("- 💡 Many episodes since last apply - consider running consolidation")
        
        weight_stats = compute_weight_stats(consol_state)
        if weight_stats.get("delta_abs_sum", 0) < 0.01:
            report.append("- ℹ️ Minimal weight drift - learning may have converged or stalled")
    
    report.append("")
    report.append("---")
    report.append(f"*Report generated by tools/nightly_report.py*")
    
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nightly training report with historical tracking."
    )
    parser.add_argument(
        "--consol-state",
        type=Path,
        help="Path to consolidation state JSON",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Directory containing trace/summary JSONs",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("reports/nightly_history.json"),
        help="Path to history JSON file (default: reports/nightly_history.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("reports/nightly_report.md"),
        help="Output markdown file (default: reports/nightly_report.md)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Name for this training run",
    )
    parser.add_argument(
        "--no-update-history",
        action="store_true",
        help="Generate report without updating history file",
    )
    
    args = parser.parse_args()
    
    # Load data
    consol_state = None
    if args.consol_state:
        consol_state = load_consolidation_state(args.consol_state)
        if consol_state:
            print(f"Loaded consolidation state: {args.consol_state}")
        else:
            print(f"Warning: Could not load consolidation state: {args.consol_state}")
    
    summaries = []
    if args.trace_dir:
        summaries = load_trace_summaries(args.trace_dir)
        print(f"Loaded {len(summaries)} trace summaries from {args.trace_dir}")
    
    # Load history
    history = load_history(args.history)
    print(f"Loaded {len(history)} historical entries")
    
    # Compute current stats for history
    session_stats = compute_session_stats(summaries)
    weight_stats = compute_weight_stats(consol_state) if consol_state else {}
    
    # Add to history
    if not args.no_update_history:
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "run_name": args.run_name,
            "total_episodes": session_stats.get("total_episodes", 0),
            "wins": session_stats.get("wins", 0),
            "losses": session_stats.get("losses", 0),
            "win_rate": session_stats.get("win_rate", 0),
            "avg_reward_tick": session_stats.get("avg_reward_tick", 0),
            "delta_abs_sum": weight_stats.get("delta_abs_sum", 0),
            "edges_tracked": weight_stats.get("edges_tracked", 0),
        }
        
        # Check if this is a new entry or update to today
        if history and history[-1].get("date") == entry["date"]:
            history[-1] = entry  # Update today's entry
        else:
            history.append(entry)
        
        save_history(args.history, history)
        print(f"Updated history: {args.history}")
    
    # Generate report
    report = generate_report(consol_state, summaries, history, args.run_name)
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
