#!/usr/bin/env python3
"""M5.1 Sweep Analysis Tool.

Generates markdown comparison reports from HyperSweep results.

Usage:
    python scripts/analyze_sweep.py snapshots/sweeps/
    python scripts/analyze_sweep.py snapshots/sweeps/ --output reports/sweep_analysis.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_sweep_results(sweep_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all trial results from a sweep directory.
    
    Looks for:
    - sweep_summary.json (combined results)
    - */result.json (individual trial results)
    """
    results = []
    
    # Try combined summary first
    summary_path = sweep_dir / "sweep_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
            if "results" in data:
                return data["results"]
    
    # Fall back to individual result files
    for trial_dir in sweep_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        
        result_path = trial_dir / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
    
    return results


def generate_summary_table(results: List[Dict[str, Any]]) -> str:
    """Generate main comparison table."""
    lines = [
        "## Summary Table",
        "",
        "| Trial | Win Rate | Cycles to 80% | SOLID | Hoisted | POR | Max Depth | Branching |",
        "|-------|----------|---------------|-------|---------|-----|-----------|-----------|",
    ]
    
    for r in results:
        trial_name = r.get("trial_name", "unknown")
        win_rate = r.get("final_win_rate", 0)
        cycles_80 = r.get("cycles_to_80_percent")
        solid = r.get("solid_nodes", 0)
        hoisted = r.get("hoisted_clusters", 0)
        por = r.get("por_edges", 0)
        max_depth = r.get("max_depth", 0)
        branching = r.get("branching_factor", 0)
        
        cycles_str = str(cycles_80) if cycles_80 else "N/A"
        
        lines.append(
            f"| {trial_name} | {win_rate:.1%} | {cycles_str} | "
            f"{solid} | {hoisted} | {por} | {max_depth} | {branching:.2f} |"
        )
    
    return "\n".join(lines)


def generate_config_table(results: List[Dict[str, Any]]) -> str:
    """Generate configuration comparison table."""
    lines = [
        "## Configuration Comparison",
        "",
        "| Trial | Consistency | Hoist | Success Bypass | Speculative | Stall Recovery | Scent Shaping |",
        "|-------|-------------|-------|----------------|-------------|----------------|---------------|",
    ]
    
    for r in results:
        trial_name = r.get("trial_name", "unknown")
        config = r.get("config", {})
        
        consistency = config.get("consistency_threshold", "?")
        hoist = config.get("hoist_threshold", "?")
        bypass = "Yes" if config.get("enable_success_bypass") else "No"
        speculative = "Yes" if config.get("enable_speculative_hoisting") else "No"
        stall = "Yes" if config.get("enable_stall_recovery") else "No"
        scent = "Yes" if config.get("enable_scent_shaping") else "No"
        
        lines.append(
            f"| {trial_name} | {consistency} | {hoist} | "
            f"{bypass} | {speculative} | {stall} | {scent} |"
        )
    
    return "\n".join(lines)


def generate_progression_section(results: List[Dict[str, Any]]) -> str:
    """Generate win rate progression section."""
    lines = [
        "## Win Rate Progression",
        "",
    ]
    
    for r in results:
        trial_name = r.get("trial_name", "unknown")
        history = r.get("win_rate_history", [])
        
        if history:
            # Show first 10 cycles
            history_str = " -> ".join(f"{wr:.0%}" for wr in history[:10])
            if len(history) > 10:
                history_str += f" ... ({len(history)} cycles)"
        else:
            history_str = "No data"
        
        lines.append(f"**{trial_name}:** {history_str}")
        lines.append("")
    
    return "\n".join(lines)


def generate_learning_speed_analysis(results: List[Dict[str, Any]]) -> str:
    """Generate learning speed comparison."""
    lines = [
        "## Learning Speed Analysis",
        "",
    ]
    
    # Sort by cycles to 80%
    sorted_results = sorted(
        results,
        key=lambda r: r.get("cycles_to_80_percent") or float("inf")
    )
    
    fastest = sorted_results[0] if sorted_results else None
    slowest = sorted_results[-1] if sorted_results else None
    
    if fastest and fastest.get("cycles_to_80_percent"):
        lines.append(f"**Fastest to 80%:** {fastest['trial_name']} ({fastest['cycles_to_80_percent']} cycles)")
    else:
        lines.append("**Fastest to 80%:** None reached threshold")
    
    if slowest and slowest.get("cycles_to_80_percent"):
        lines.append(f"**Slowest to 80%:** {slowest['trial_name']} ({slowest['cycles_to_80_percent']} cycles)")
    
    lines.append("")
    
    # Compare final win rates
    lines.append("### Final Win Rates")
    lines.append("")
    
    sorted_by_win = sorted(results, key=lambda r: -r.get("final_win_rate", 0))
    for i, r in enumerate(sorted_by_win, 1):
        medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
        lines.append(f"{medal} {r['trial_name']}: {r.get('final_win_rate', 0):.1%}")
    
    return "\n".join(lines)


def generate_structural_analysis(results: List[Dict[str, Any]]) -> str:
    """Generate structural maturity analysis."""
    lines = [
        "## Structural Maturity Analysis",
        "",
    ]
    
    # Find best depth
    best_depth = max(results, key=lambda r: r.get("max_depth", 0))
    lines.append(f"**Deepest Hierarchy:** {best_depth['trial_name']} (depth {best_depth.get('max_depth', 0)})")
    
    # Find most POR edges
    best_por = max(results, key=lambda r: r.get("por_edges", 0))
    lines.append(f"**Most POR Edges:** {best_por['trial_name']} ({best_por.get('por_edges', 0)} edges)")
    
    # Find best branching factor
    best_branching = max(results, key=lambda r: r.get("branching_factor", 0))
    lines.append(f"**Highest Branching Factor:** {best_branching['trial_name']} ({best_branching.get('branching_factor', 0):.2f})")
    
    lines.append("")
    
    # Healthy Growth Signature check
    lines.append("### Healthy Growth Signature Check")
    lines.append("")
    lines.append("| Trial | Depth >= 4 | Branch >= 1.5 | POR > 0 | Status |")
    lines.append("|-------|------------|---------------|---------|--------|")
    
    for r in results:
        depth_ok = r.get("max_depth", 0) >= 4
        branch_ok = r.get("branching_factor", 0) >= 1.5
        por_ok = r.get("por_edges", 0) > 0
        
        all_ok = depth_ok and branch_ok and por_ok
        status = "Healthy" if all_ok else "Growing"
        
        lines.append(
            f"| {r['trial_name']} | "
            f"{'Yes' if depth_ok else 'No'} | "
            f"{'Yes' if branch_ok else 'No'} | "
            f"{'Yes' if por_ok else 'No'} | "
            f"{status} |"
        )
    
    return "\n".join(lines)


def generate_recommendations(results: List[Dict[str, Any]]) -> str:
    """Generate recommendations based on analysis."""
    lines = [
        "## Recommendations",
        "",
    ]
    
    # Find best overall performer
    scored_results = []
    for r in results:
        score = 0
        score += r.get("final_win_rate", 0) * 100  # Win rate (0-100)
        score += r.get("max_depth", 0) * 10  # Depth bonus
        score += r.get("por_edges", 0) * 5  # POR bonus
        cycles_80 = r.get("cycles_to_80_percent")
        if cycles_80:
            score += (20 - cycles_80) * 2  # Speed bonus (faster = more points)
        
        scored_results.append((r, score))
    
    scored_results.sort(key=lambda x: -x[1])
    best = scored_results[0][0] if scored_results else None
    
    if best:
        lines.append(f"**Recommended Configuration:** {best['trial_name']}")
        lines.append("")
        
        config = best.get("config", {})
        lines.append("```")
        lines.append(f"consistency_threshold: {config.get('consistency_threshold', '?')}")
        lines.append(f"hoist_threshold: {config.get('hoist_threshold', '?')}")
        lines.append(f"enable_success_bypass: {config.get('enable_success_bypass', False)}")
        lines.append(f"enable_speculative_hoisting: {config.get('enable_speculative_hoisting', False)}")
        lines.append(f"enable_stall_recovery: {config.get('enable_stall_recovery', True)}")
        lines.append(f"enable_scent_shaping: {config.get('enable_scent_shaping', True)}")
        lines.append("```")
    
    return "\n".join(lines)


def generate_sweep_report(
    sweep_dir: Path,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate complete markdown sweep analysis report.
    
    Args:
        sweep_dir: Directory containing sweep results
        output_path: Optional path to save report
        
    Returns:
        Markdown formatted report string
    """
    results = load_sweep_results(sweep_dir)
    
    if not results:
        return f"# Sweep Analysis\n\nNo results found in {sweep_dir}"
    
    sections = [
        f"# M5.1 Hyperparameter Sweep Analysis",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Sweep Directory:** {sweep_dir}",
        f"**Trials Analyzed:** {len(results)}",
        "",
        "---",
        "",
        generate_summary_table(results),
        "",
        "---",
        "",
        generate_config_table(results),
        "",
        "---",
        "",
        generate_progression_section(results),
        "",
        "---",
        "",
        generate_learning_speed_analysis(results),
        "",
        "---",
        "",
        generate_structural_analysis(results),
        "",
        "---",
        "",
        generate_recommendations(results),
    ]
    
    report = "\n".join(sections)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze M5.1 HyperSweep results"
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Directory containing sweep results"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print report to stdout"
    )
    
    args = parser.parse_args()
    
    # Generate default output path if not specified
    if args.output is None:
        args.output = args.sweep_dir / "sweep_analysis.md"
    
    report = generate_sweep_report(args.sweep_dir, args.output)
    
    if args.print:
        print(report)


if __name__ == "__main__":
    main()

