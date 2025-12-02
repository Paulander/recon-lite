#!/usr/bin/env python3
"""
Pack Diff Tool (M4)

Compare two consolidation packs (or weight packs) and show the differences.
Useful for validating that consolidation produced expected changes.

Usage:
    uv run python tools/pack_diff.py weights/krk_old.json weights/krk_new.json
    uv run python tools/pack_diff.py weights/krk_old.json weights/krk_new.json --top 20
    uv run python tools/pack_diff.py weights/krk_old.json weights/krk_new.json --output reports/diff.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EdgeDiff:
    """Difference for a single edge."""

    edge_key: str
    old_weight: Optional[float]
    new_weight: Optional[float]
    delta: float
    pct_change: Optional[float]

    @property
    def is_new(self) -> bool:
        return self.old_weight is None

    @property
    def is_removed(self) -> bool:
        return self.new_weight is None


def load_pack(path: Path) -> Dict[str, Any]:
    """Load a consolidation or weight pack from JSON."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_weights(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract edge weights from various pack formats.

    Supports:
    - Consolidation state: {"w_base": {"edge_key": weight, ...}}
    - SWP format: {"por_edges": {"src->dst": weight, ...}}
    """
    weights = {}

    # Consolidation state format
    if "w_base" in data:
        weights.update(data["w_base"])

    # SWP format (por_edges)
    if "por_edges" in data:
        for edge_key, weight in data["por_edges"].items():
            # Convert "src->dst" to "src->dst:POR"
            if ":" not in edge_key:
                edge_key = f"{edge_key}:POR"
            weights[edge_key] = weight

    return weights


def compute_diff(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
) -> List[EdgeDiff]:
    """Compute differences between two weight sets."""
    all_edges = set(old_weights.keys()) | set(new_weights.keys())
    diffs = []

    for edge_key in all_edges:
        old_w = old_weights.get(edge_key)
        new_w = new_weights.get(edge_key)

        if old_w is None:
            delta = new_w
            pct = None
        elif new_w is None:
            delta = -old_w
            pct = None
        else:
            delta = new_w - old_w
            pct = (delta / old_w * 100) if old_w != 0 else None

        diffs.append(EdgeDiff(
            edge_key=edge_key,
            old_weight=old_w,
            new_weight=new_w,
            delta=delta,
            pct_change=pct,
        ))

    return diffs


def format_edge_key(edge_key: str) -> str:
    """Format edge key for display."""
    if "->" in edge_key and ":" in edge_key:
        parts = edge_key.split("->")
        if len(parts) == 2:
            dst_ltype = parts[1].split(":")
            if len(dst_ltype) == 2:
                return f"{parts[0]} → {dst_ltype[0]} ({dst_ltype[1]})"
    return edge_key


def format_weight(w: Optional[float]) -> str:
    """Format weight for display."""
    if w is None:
        return "—"
    return f"{w:.4f}"


def format_delta(delta: float) -> str:
    """Format delta with sign."""
    if delta >= 0:
        return f"+{delta:.4f}"
    return f"{delta:.4f}"


def format_pct(pct: Optional[float]) -> str:
    """Format percentage change."""
    if pct is None:
        return "—"
    if pct >= 0:
        return f"+{pct:.1f}%"
    return f"{pct:.1f}%"


def generate_text_report(
    old_path: Path,
    new_path: Path,
    diffs: List[EdgeDiff],
    top_n: int = 20,
) -> str:
    """Generate a text report of differences."""
    lines = []
    lines.append(f"Pack Diff Report")
    lines.append(f"================")
    lines.append(f"")
    lines.append(f"Old: {old_path}")
    lines.append(f"New: {new_path}")
    lines.append(f"")

    # Summary stats
    changed = [d for d in diffs if d.delta != 0]
    new_edges = [d for d in diffs if d.is_new]
    removed_edges = [d for d in diffs if d.is_removed]
    modified = [d for d in changed if not d.is_new and not d.is_removed]

    lines.append(f"Summary:")
    lines.append(f"  Total edges compared: {len(diffs)}")
    lines.append(f"  Modified: {len(modified)}")
    lines.append(f"  New: {len(new_edges)}")
    lines.append(f"  Removed: {len(removed_edges)}")
    lines.append(f"")

    if not changed:
        lines.append("No differences found.")
        return "\n".join(lines)

    # Sort by absolute delta
    sorted_diffs = sorted(changed, key=lambda d: abs(d.delta), reverse=True)

    lines.append(f"Top {min(top_n, len(sorted_diffs))} Changes (by absolute delta):")
    lines.append(f"")
    lines.append(f"{'Edge':<40} {'Old':>10} {'New':>10} {'Delta':>10} {'%':>8}")
    lines.append(f"{'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for diff in sorted_diffs[:top_n]:
        edge_display = format_edge_key(diff.edge_key)
        if len(edge_display) > 38:
            edge_display = edge_display[:35] + "..."
        lines.append(
            f"{edge_display:<40} "
            f"{format_weight(diff.old_weight):>10} "
            f"{format_weight(diff.new_weight):>10} "
            f"{format_delta(diff.delta):>10} "
            f"{format_pct(diff.pct_change):>8}"
        )

    return "\n".join(lines)


def generate_markdown_report(
    old_path: Path,
    new_path: Path,
    diffs: List[EdgeDiff],
    top_n: int = 20,
) -> str:
    """Generate a markdown report of differences."""
    lines = []
    lines.append(f"# Pack Diff Report")
    lines.append(f"")
    lines.append(f"- **Old**: `{old_path}`")
    lines.append(f"- **New**: `{new_path}`")
    lines.append(f"")

    # Summary stats
    changed = [d for d in diffs if d.delta != 0]
    new_edges = [d for d in diffs if d.is_new]
    removed_edges = [d for d in diffs if d.is_removed]
    modified = [d for d in changed if not d.is_new and not d.is_removed]

    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total edges compared | {len(diffs)} |")
    lines.append(f"| Modified | {len(modified)} |")
    lines.append(f"| New | {len(new_edges)} |")
    lines.append(f"| Removed | {len(removed_edges)} |")
    lines.append(f"")

    if not changed:
        lines.append("**No differences found.**")
        return "\n".join(lines)

    # Sort by absolute delta
    sorted_diffs = sorted(changed, key=lambda d: abs(d.delta), reverse=True)

    lines.append(f"## Top {min(top_n, len(sorted_diffs))} Changes")
    lines.append(f"")
    lines.append(f"| Edge | Old | New | Delta | % |")
    lines.append(f"|------|-----|-----|-------|---|")

    for diff in sorted_diffs[:top_n]:
        edge_display = format_edge_key(diff.edge_key)
        lines.append(
            f"| {edge_display} "
            f"| {format_weight(diff.old_weight)} "
            f"| {format_weight(diff.new_weight)} "
            f"| {format_delta(diff.delta)} "
            f"| {format_pct(diff.pct_change)} |"
        )

    # New edges section
    if new_edges:
        lines.append(f"")
        lines.append(f"## New Edges ({len(new_edges)})")
        lines.append(f"")
        for diff in new_edges[:10]:
            lines.append(f"- `{diff.edge_key}`: {format_weight(diff.new_weight)}")
        if len(new_edges) > 10:
            lines.append(f"- ... and {len(new_edges) - 10} more")

    # Removed edges section
    if removed_edges:
        lines.append(f"")
        lines.append(f"## Removed Edges ({len(removed_edges)})")
        lines.append(f"")
        for diff in removed_edges[:10]:
            lines.append(f"- `{diff.edge_key}`: was {format_weight(diff.old_weight)}")
        if len(removed_edges) > 10:
            lines.append(f"- ... and {len(removed_edges) - 10} more")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two consolidation/weight packs"
    )
    parser.add_argument("old", type=Path, help="Path to old pack")
    parser.add_argument("new", type=Path, help="Path to new pack")
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top changes to show"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output file (markdown if .md)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default=None,
        help="Output format (auto-detected from output extension)",
    )
    args = parser.parse_args()

    if not args.old.exists():
        print(f"Error: Old pack not found: {args.old}", file=sys.stderr)
        sys.exit(1)
    if not args.new.exists():
        print(f"Error: New pack not found: {args.new}", file=sys.stderr)
        sys.exit(1)

    # Load packs
    old_data = load_pack(args.old)
    new_data = load_pack(args.new)

    # Extract weights
    old_weights = extract_weights(old_data)
    new_weights = extract_weights(new_data)

    if not old_weights:
        print(f"Warning: No weights found in {args.old}", file=sys.stderr)
    if not new_weights:
        print(f"Warning: No weights found in {args.new}", file=sys.stderr)

    # Compute diff
    diffs = compute_diff(old_weights, new_weights)

    # Determine format
    fmt = args.format
    if fmt is None:
        if args.output and args.output.suffix == ".md":
            fmt = "markdown"
        else:
            fmt = "text"

    # Generate report
    if fmt == "markdown":
        report = generate_markdown_report(args.old, args.new, diffs, args.top)
    else:
        report = generate_text_report(args.old, args.new, diffs, args.top)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

