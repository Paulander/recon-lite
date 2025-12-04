#!/usr/bin/env python3
"""
Consolidation Report Tool (M4)

Generate markdown reports summarizing consolidation state and changes.
Useful for tracking learning progress and debugging weight drift.

Usage:
    uv run python tools/report_consolidation.py weights/krk_consolidated.json --output reports/consolidation_report.md
    uv run python tools/report_consolidation.py weights/krk_consolidated.json --compare weights/krk_consolidated_old.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_consolidation_state(path: Path) -> Dict[str, Any]:
    """Load consolidation state from JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def format_edge_key(edge_key: str) -> str:
    """Format edge key for display."""
    # e.g., "p0_check->p0_move:POR" -> "p0_check → p0_move (POR)"
    if "->" in edge_key and ":" in edge_key:
        parts = edge_key.split("->")
        if len(parts) == 2:
            dst_ltype = parts[1].split(":")
            if len(dst_ltype) == 2:
                return f"`{parts[0]}` → `{dst_ltype[0]}` ({dst_ltype[1]})"
    return f"`{edge_key}`"


def generate_report(
    state: Dict[str, Any],
    compare_state: Optional[Dict[str, Any]] = None,
    state_path: Optional[Path] = None,
    compare_path: Optional[Path] = None,
) -> str:
    """Generate markdown report from consolidation state."""
    lines = []

    # Header
    lines.append("# Consolidation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # File info
    if state_path:
        checksum = compute_checksum(state_path)
        lines.append(f"**Source file:** `{state_path}`")
        lines.append(f"**Checksum:** `{checksum}`")
        lines.append("")

    # Metadata
    meta = state.get("consolidation_meta", {})
    config = state.get("config", {})

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total episodes seen:** {meta.get('total_episodes', 0)}")
    lines.append(f"- **Episodes since last apply:** {meta.get('episodes_since_apply', 0)}")
    lines.append(f"- **Last updated:** {meta.get('last_apply_time', 'N/A')}")
    lines.append(f"- **Edges tracked:** {meta.get('edges_tracked', len(state.get('w_base', {})))}")
    lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Learning rate (eta):** {config.get('eta_consolidate', 'N/A')}")
    lines.append(f"- **Min episodes:** {config.get('min_episodes', 'N/A')}")
    lines.append(f"- **Outcome weight:** {config.get('outcome_weight', 'N/A')}")
    lines.append(f"- **Max base delta:** {config.get('max_base_delta', 'N/A')}")
    lines.append(f"- **Weight bounds:** [{config.get('w_min', 'N/A')}, {config.get('w_max', 'N/A')}]")
    lines.append("")

    # w_base values
    w_base = state.get("w_base", {})
    w_init = state.get("w_init", {})

    if w_base:
        lines.append("## Edge Weights (w_base)")
        lines.append("")

        # Compute changes from initial
        changes = []
        for edge_key, w_b in w_base.items():
            w_i = w_init.get(edge_key, w_b)
            delta = w_b - w_i
            changes.append((edge_key, w_b, w_i, delta))

        # Sort by absolute delta
        changes.sort(key=lambda x: abs(x[3]), reverse=True)

        # Top changes
        lines.append("### Top Changes from Initial")
        lines.append("")
        lines.append("| Edge | w_base | w_init | Δ |")
        lines.append("|------|--------|--------|---|")

        shown = 0
        for edge_key, w_b, w_i, delta in changes:
            if abs(delta) < 0.001:
                continue
            sign = "+" if delta > 0 else ""
            lines.append(f"| {format_edge_key(edge_key)} | {w_b:.4f} | {w_i:.4f} | {sign}{delta:.4f} |")
            shown += 1
            if shown >= 15:
                break

        if shown == 0:
            lines.append("| (no significant changes) | - | - | - |")

        lines.append("")

        # All weights
        lines.append("### All Tracked Edges")
        lines.append("")
        lines.append("| Edge | w_base |")
        lines.append("|------|--------|")

        for edge_key, w_b in sorted(w_base.items()):
            lines.append(f"| {format_edge_key(edge_key)} | {w_b:.4f} |")

        lines.append("")

    # Comparison with previous state
    if compare_state:
        lines.append("## Comparison with Previous State")
        lines.append("")

        if compare_path:
            compare_checksum = compute_checksum(compare_path)
            lines.append(f"**Comparing with:** `{compare_path}` (checksum: `{compare_checksum}`)")
            lines.append("")

        compare_w_base = compare_state.get("w_base", {})
        compare_meta = compare_state.get("consolidation_meta", {})

        # Episode count change
        old_episodes = compare_meta.get("total_episodes", 0)
        new_episodes = meta.get("total_episodes", 0)
        lines.append(f"- **Episodes:** {old_episodes} → {new_episodes} (+{new_episodes - old_episodes})")
        lines.append("")

        # Weight changes
        lines.append("### Weight Changes")
        lines.append("")
        lines.append("| Edge | Old w_base | New w_base | Δ |")
        lines.append("|------|------------|------------|---|")

        all_keys = set(w_base.keys()) | set(compare_w_base.keys())
        changes = []
        for edge_key in all_keys:
            old_w = compare_w_base.get(edge_key, 1.0)
            new_w = w_base.get(edge_key, 1.0)
            delta = new_w - old_w
            if abs(delta) > 0.001:
                changes.append((edge_key, old_w, new_w, delta))

        changes.sort(key=lambda x: abs(x[3]), reverse=True)

        for edge_key, old_w, new_w, delta in changes[:20]:
            sign = "+" if delta > 0 else ""
            lines.append(f"| {format_edge_key(edge_key)} | {old_w:.4f} | {new_w:.4f} | {sign}{delta:.4f} |")

        if not changes:
            lines.append("| (no changes) | - | - | - |")

        lines.append("")

    # Statistics summary
    if w_base:
        lines.append("## Statistics")
        lines.append("")

        values = list(w_base.values())
        avg = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)

        lines.append(f"- **Average w_base:** {avg:.4f}")
        lines.append(f"- **Min w_base:** {min_v:.4f}")
        lines.append(f"- **Max w_base:** {max_v:.4f}")
        lines.append(f"- **Range:** {max_v - min_v:.4f}")

        # Count significant changes
        sig_changes = sum(1 for _, _, _, d in changes if abs(d) > 0.01) if 'changes' in dir() else 0
        lines.append(f"- **Edges with significant change (>0.01):** {sig_changes}")

        lines.append("")

        # M4: ASCII histogram of weight distribution
        lines.append("### Weight Distribution")
        lines.append("")
        lines.append(_generate_histogram(values))
        lines.append("")

        # M4: Delta distribution (changes from initial)
        if w_init:
            deltas = [w_base.get(k, 1.0) - w_init.get(k, 1.0) for k in w_base.keys()]
            if any(abs(d) > 0.001 for d in deltas):
                lines.append("### Delta Distribution (w_base - w_init)")
                lines.append("")
                lines.append(_generate_histogram(deltas, center_zero=True))
                lines.append("")

    return "\n".join(lines)


def _generate_histogram(
    values: List[float],
    bins: int = 10,
    width: int = 40,
    center_zero: bool = False,
) -> str:
    """Generate an ASCII histogram of values."""
    if not values:
        return "```\n(no data)\n```"

    min_v = min(values)
    max_v = max(values)

    # Handle edge case of all same values
    if max_v == min_v:
        return f"```\nAll values = {min_v:.4f}\n```"

    # For delta distribution, center around zero if requested
    if center_zero:
        abs_max = max(abs(min_v), abs(max_v))
        min_v = -abs_max
        max_v = abs_max

    bin_width = (max_v - min_v) / bins
    counts = [0] * bins

    for v in values:
        idx = min(int((v - min_v) / bin_width), bins - 1)
        counts[idx] += 1

    max_count = max(counts) if counts else 1

    lines = ["```"]

    for i, count in enumerate(counts):
        bin_start = min_v + i * bin_width
        bin_end = min_v + (i + 1) * bin_width

        # Normalize bar width
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len

        # Format bin range
        if center_zero and bin_start < 0 < bin_end:
            label = f"[{bin_start:+.2f}, {bin_end:+.2f})"
        else:
            label = f"[{bin_start:.2f}, {bin_end:.2f})"

        lines.append(f"{label:>16} | {bar} {count}")

    lines.append("```")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate markdown report from consolidation state"
    )
    parser.add_argument(
        "state_file",
        type=Path,
        help="Consolidation state JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output markdown file (default: stdout)",
    )
    parser.add_argument(
        "--compare", "-c",
        type=Path,
        default=None,
        help="Compare with another consolidation state file",
    )
    args = parser.parse_args()

    # Load state
    if not args.state_file.exists():
        print(f"Error: State file not found: {args.state_file}", file=sys.stderr)
        sys.exit(1)

    state = load_consolidation_state(args.state_file)

    # Load comparison state if provided
    compare_state = None
    if args.compare:
        if not args.compare.exists():
            print(f"Warning: Comparison file not found: {args.compare}", file=sys.stderr)
        else:
            compare_state = load_consolidation_state(args.compare)

    # Generate report
    report = generate_report(
        state,
        compare_state=compare_state,
        state_path=args.state_file,
        compare_path=args.compare,
    )

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

