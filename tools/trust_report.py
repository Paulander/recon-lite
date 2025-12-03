#!/usr/bin/env python3
"""
M5.3: Generate trust reports from traces and consolidation state.

Usage:
    uv run python tools/trust_report.py \
        --traces demos/outputs/persistent/*.json \
        --consolidation weights/nightly/krk_consol.json \
        --out reports/trust.json

    # With previous report for incremental tracking
    uv run python tools/trust_report.py \
        --traces demos/outputs/persistent/*.json \
        --previous reports/trust_gen001.json \
        --out reports/trust_gen002.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_lite.trust.scoring import (
    TrustConfig,
    TrustReport,
    compute_trust_report,
)


def load_traces(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load trace frames from multiple files."""
    frames = []
    
    for path in paths:
        if not path.exists():
            continue
        
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            frames.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                frames.extend(data)
            else:
                frames.append(data)
    
    return frames


def print_report_summary(report: TrustReport) -> None:
    """Print a human-readable summary of the trust report."""
    print(f"\n{'=' * 60}")
    print(f"Trust Report - Generation {report.generation}")
    print(f"{'=' * 60}")
    
    print(f"\nNodes tracked: {len(report.node_scores)}")
    print(f"Edges tracked: {len(report.edge_scores)}")
    
    # Freeze candidates
    freeze = report.get_freeze_candidates()
    if freeze:
        print(f"\n--- Freeze Candidates ({len(freeze)}) ---")
        for item in freeze[:10]:
            print(f"  • {item}")
        if len(freeze) > 10:
            print(f"  ... and {len(freeze) - 10} more")
    
    # Remove candidates
    remove = report.get_remove_candidates()
    if remove:
        print(f"\n--- Remove Candidates ({len(remove)}) ---")
        for item in remove[:10]:
            print(f"  • {item}")
        if len(remove) > 10:
            print(f"  ... and {len(remove) - 10} more")
    
    # Promote candidates
    promote = report.get_promote_candidates()
    if promote:
        print(f"\n--- Promote Candidates ({len(promote)}) ---")
        for item in promote[:10]:
            print(f"  • {item}")
        if len(promote) > 10:
            print(f"  ... and {len(promote) - 10} more")
    
    # Top trusted nodes
    if report.node_scores:
        print(f"\n--- Top Trusted Nodes ---")
        sorted_nodes = sorted(
            report.node_scores.values(),
            key=lambda x: -x.trust_score
        )[:5]
        for score in sorted_nodes:
            print(f"  {score.node_id}: trust={score.trust_score:.3f}, "
                  f"activations={score.activation_count}, "
                  f"mean_reward={score.mean_reward:.4f}")
    
    # Top trusted edges
    if report.edge_scores:
        print(f"\n--- Top Trusted Edges ---")
        sorted_edges = sorted(
            report.edge_scores.values(),
            key=lambda x: -x.trust_score
        )[:5]
        for score in sorted_edges:
            print(f"  {score.edge_key}: trust={score.trust_score:.3f}, "
                  f"fires={score.fire_count}, "
                  f"contribution={score.mean_contribution:.4f}")
    
    # Lowest trusted (non-neutral)
    if report.node_scores:
        low_trust_nodes = [
            s for s in report.node_scores.values()
            if s.trust_score < 0.5 and s.activation_count >= 10
        ]
        if low_trust_nodes:
            print(f"\n--- Lowest Trusted Nodes ---")
            for score in sorted(low_trust_nodes, key=lambda x: x.trust_score)[:5]:
                print(f"  {score.node_id}: trust={score.trust_score:.3f}, "
                      f"activations={score.activation_count}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate trust reports from traces and consolidation state."
    )
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to trace JSON/JSONL files",
    )
    parser.add_argument(
        "--consolidation",
        type=Path,
        default=None,
        help="Path to consolidation state JSON (for w_base history)",
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Path to previous trust report (for incremental tracking)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("reports/trust.json"),
        help="Output path for trust report",
    )
    parser.add_argument(
        "--freeze-threshold",
        type=float,
        default=0.3,
        help="Trust threshold for freezing (default: 0.3)",
    )
    parser.add_argument(
        "--promote-threshold",
        type=float,
        default=0.8,
        help="Trust threshold for promotion (default: 0.8)",
    )
    parser.add_argument(
        "--min-generations",
        type=int,
        default=3,
        help="Minimum generations before action recommendations (default: 3)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output",
    )
    
    args = parser.parse_args()
    
    # Expand globs
    trace_paths = []
    for pattern in args.traces:
        if "*" in str(pattern):
            trace_paths.extend(Path(".").glob(str(pattern)))
        elif pattern.exists():
            trace_paths.append(pattern)
    
    if not trace_paths:
        print("Error: No trace files found")
        sys.exit(1)
    
    print(f"Loading traces from {len(trace_paths)} file(s)...")
    frames = load_traces(trace_paths)
    print(f"Loaded {len(frames)} frames")
    
    # Load consolidation state
    consolidation_state = None
    if args.consolidation and args.consolidation.exists():
        print(f"Loading consolidation state from {args.consolidation}")
        with open(args.consolidation) as f:
            consolidation_state = json.load(f)
    
    # Load previous report
    previous_report = None
    if args.previous and args.previous.exists():
        print(f"Loading previous report from {args.previous}")
        previous_report = TrustReport.load(args.previous)
    
    # Create config
    config = TrustConfig(
        freeze_threshold=args.freeze_threshold,
        promote_threshold=args.promote_threshold,
        min_generations=args.min_generations,
    )
    
    # Compute trust report
    print("Computing trust scores...")
    report = compute_trust_report(
        frames,
        consolidation_state=consolidation_state,
        config=config,
        previous_report=previous_report,
    )
    
    # Save report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.save(args.out)
    print(f"Saved trust report to {args.out}")
    
    # Print summary
    if not args.quiet:
        print_report_summary(report)


if __name__ == "__main__":
    main()

