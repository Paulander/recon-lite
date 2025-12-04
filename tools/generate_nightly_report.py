#!/usr/bin/env python3
"""
M8.4: Standardized Nightly Report Generator

Generates a comprehensive markdown report from training run data.
Combines consolidation state, trace summaries, and evaluation results.

Usage:
    uv run python tools/generate_nightly_report.py \
        --consol weights/nightly/krk_consol.json \
        --traces reports/krk_trace.jsonl \
        --output reports/nightly_report.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file, return empty list if not found."""
    if not path.exists():
        return []
    episodes = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return episodes


def compute_stats(episodes: List[Dict]) -> Dict[str, Any]:
    """Compute statistics from episode records."""
    if not episodes:
        return {}
    
    wins = sum(1 for e in episodes if e.get("result") == "1-0")
    losses = sum(1 for e in episodes if e.get("result") == "0-1")
    draws = sum(1 for e in episodes if e.get("result") in ("1/2-1/2", "*"))
    
    plies = [e.get("notes", {}).get("plies", 0) for e in episodes]
    avg_plies = sum(plies) / len(plies) if plies else 0
    
    return {
        "total_games": len(episodes),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(episodes) * 100 if episodes else 0,
        "checkmate_rate": wins / len(episodes) * 100 if episodes else 0,
        "avg_plies": avg_plies,
    }


def get_top_weight_changes(consol: Dict, n: int = 10) -> List[Dict]:
    """Get top N weight changes from consolidation state."""
    edges = consol.get("edges", {})
    changes = []
    for edge_key, state in edges.items():
        w_init = state.get("w_init", 1.0)
        w_base = state.get("w_base", w_init)
        delta = w_base - w_init
        if abs(delta) > 0.001:
            changes.append({
                "edge": edge_key,
                "w_init": round(w_init, 3),
                "w_base": round(w_base, 3),
                "delta": round(delta, 3),
                "delta_pct": round(delta / w_init * 100, 1) if w_init != 0 else 0,
            })
    
    changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return changes[:n]


def generate_report(
    consol_path: Optional[Path] = None,
    traces_path: Optional[Path] = None,
    first_consol_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    title: str = "Nightly Training Report",
    timestamp: Optional[str] = None,
) -> str:
    """Generate markdown report."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        f"# {title}",
        "",
        f"**Generated:** {timestamp}",
        "",
    ]
    
    # Load data
    consol = load_json(consol_path) if consol_path else None
    first_consol = load_json(first_consol_path) if first_consol_path else None
    episodes = load_jsonl(traces_path) if traces_path else []
    
    # Episode Statistics
    if episodes:
        stats = compute_stats(episodes)
        lines.extend([
            "## Training Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Games** | {stats['total_games']} |",
            f"| **Wins (Checkmates)** | {stats['wins']} |",
            f"| **Losses** | {stats['losses']} |",
            f"| **Draws/Incomplete** | {stats['draws']} |",
            f"| **Win Rate** | {stats['win_rate']:.1f}% |",
            f"| **Avg Plies to Mate** | {stats['avg_plies']:.1f} |",
            "",
        ])
    
    # Consolidation State
    if consol:
        total_episodes = consol.get("total_episodes", 0)
        lines.extend([
            "## Consolidation State",
            "",
            f"- **Total Episodes Accumulated:** {total_episodes}",
            f"- **Tracked Edges:** {len(consol.get('edges', {}))}",
            "",
        ])
        
        # Weight Changes
        changes = get_top_weight_changes(consol)
        if changes:
            lines.extend([
                "### Top Weight Changes",
                "",
                "| Edge | Initial | Current | Change | % Change |",
                "|------|---------|---------|--------|----------|",
            ])
            for c in changes:
                lines.append(
                    f"| `{c['edge']}` | {c['w_init']} | {c['w_base']} | {c['delta']:+.3f} | {c['delta_pct']:+.1f}% |"
                )
            lines.append("")
    
    # Comparison with initial state
    if first_consol and consol:
        first_edges = first_consol.get("edges", {})
        current_edges = consol.get("edges", {})
        
        improvements = []
        regressions = []
        
        for edge_key in current_edges:
            if edge_key in first_edges:
                first_w = first_edges[edge_key].get("w_base", 1.0)
                current_w = current_edges[edge_key].get("w_base", 1.0)
                delta = current_w - first_w
                if delta > 0.01:
                    improvements.append((edge_key, delta))
                elif delta < -0.01:
                    regressions.append((edge_key, delta))
        
        if improvements or regressions:
            lines.extend([
                "### Changes from Initial State",
                "",
            ])
            
            if improvements:
                lines.append(f"**Improved edges:** {len(improvements)}")
            if regressions:
                lines.append(f"**Regressed edges:** {len(regressions)}")
            lines.append("")
    
    # File References
    lines.extend([
        "## File References",
        "",
    ])
    if consol_path:
        lines.append(f"- Consolidation: `{consol_path}`")
    if traces_path:
        lines.append(f"- Traces: `{traces_path}`")
    if first_consol_path:
        lines.append(f"- Initial State: `{first_consol_path}`")
    lines.append("")
    
    # Commands for further analysis
    lines.extend([
        "## Analysis Commands",
        "",
        "```bash",
        "# View weight changes in detail",
    ])
    if consol_path and first_consol_path:
        lines.append(f"uv run python tools/pack_diff.py {first_consol_path} {consol_path}")
    if consol_path:
        lines.append(f"uv run python tools/report_consolidation.py {consol_path}")
    lines.extend([
        "",
        "# Visualize consolidation history",
        "# Open demos/visualization/consolidation_dashboard.html and load the JSON",
        "```",
        "",
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate nightly training report")
    parser.add_argument("--consol", type=Path, help="Path to consolidation state JSON")
    parser.add_argument("--traces", type=Path, help="Path to trace JSONL file")
    parser.add_argument("--first-consol", type=Path, help="Path to initial consolidation state (for comparison)")
    parser.add_argument("-o", "--output", type=Path, help="Output markdown file")
    parser.add_argument("--title", type=str, default="Nightly Training Report")
    args = parser.parse_args()
    
    report = generate_report(
        consol_path=args.consol,
        traces_path=args.traces,
        first_consol_path=args.first_consol,
        output_path=args.output,
        title=args.title,
    )
    
    if args.output:
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

