#!/usr/bin/env python3
"""
M5.2: Cluster extracted motifs by type and pattern.

Usage:
    uv run python demos/experiments/cluster_motifs.py \
        --motifs reports/motifs/extracted.jsonl \
        --out reports/motifs/clusters.json

Groups motifs by:
- dtype and pattern_key (primary grouping)
- Board context similarity (secondary clustering)
- Outcome correlation (positive vs negative reward)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_lite.motifs.descriptors import BindingDescriptor, MotifDataset


def compute_context_signature(context: Dict[str, Any]) -> str:
    """
    Compute a simplified signature from context features for grouping.
    
    This creates a coarse-grained fingerprint for clustering similar positions.
    """
    parts = []
    
    # Material balance
    material = context.get("material", {})
    diff = material.get("diff", 0)
    if diff > 2:
        parts.append("mat_up")
    elif diff < -2:
        parts.append("mat_down")
    else:
        parts.append("mat_even")
    
    # Tactical features
    tactical = context.get("tactical", {})
    if tactical.get("potential_forks"):
        parts.append("has_fork")
    if tactical.get("pins"):
        parts.append("has_pin")
    if tactical.get("checks_available", 0) > 2:
        parts.append("many_checks")
    
    # Hanging pieces
    hanging = context.get("hanging_pieces", {})
    if hanging.get("white_en_prise") or hanging.get("black_en_prise"):
        parts.append("en_prise")
    
    # Pawn structure
    pawn = context.get("pawn_structure", {})
    if pawn.get("passed_pawns"):
        parts.append("passed_pawn")
    if pawn.get("isolated_pawns"):
        parts.append("isolated_pawn")
    
    return "|".join(sorted(parts)) if parts else "neutral"


def cluster_motifs(dataset: MotifDataset) -> Dict[str, Any]:
    """
    Cluster motifs into groups for analysis and script proposal.
    
    Returns:
        Dictionary with cluster information and statistics.
    """
    # Primary grouping by dtype + pattern_key
    primary_groups: Dict[str, List[BindingDescriptor]] = defaultdict(list)
    
    for motif in dataset:
        key = f"{motif.dtype}:{motif.pattern_key}"
        primary_groups[key].append(motif)
    
    clusters = {}
    
    for primary_key, motifs in primary_groups.items():
        # Secondary grouping by context signature
        context_groups: Dict[str, List[BindingDescriptor]] = defaultdict(list)
        
        for motif in motifs:
            sig = compute_context_signature(motif.context)
            context_groups[sig].append(motif)
        
        # Compute statistics for each sub-cluster
        sub_clusters = []
        for context_sig, context_motifs in context_groups.items():
            outcomes = [m.outcome_score for m in context_motifs]
            positive_count = sum(1 for o in outcomes if o > 0)
            negative_count = sum(1 for o in outcomes if o < 0)
            
            sub_clusters.append({
                "context_signature": context_sig,
                "count": len(context_motifs),
                "avg_outcome": sum(outcomes) / len(outcomes) if outcomes else 0,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "avg_confidence": sum(m.confidence for m in context_motifs) / len(context_motifs),
                "sample_fens": [m.fen for m in context_motifs[:3] if m.fen],
                "active_nodes": list(set(
                    node for m in context_motifs for node in m.active_nodes
                ))[:10],
            })
        
        # Sort sub-clusters by count
        sub_clusters.sort(key=lambda x: -x["count"])
        
        # Overall cluster statistics
        all_outcomes = [m.outcome_score for m in motifs]
        clusters[primary_key] = {
            "dtype": motifs[0].dtype if motifs else "",
            "pattern_key": motifs[0].pattern_key if motifs else "",
            "total_count": len(motifs),
            "avg_outcome": sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0,
            "positive_ratio": sum(1 for o in all_outcomes if o > 0) / len(all_outcomes) if all_outcomes else 0,
            "sub_clusters": sub_clusters,
        }
    
    return {
        "cluster_count": len(clusters),
        "total_motifs": len(dataset),
        "clusters": clusters,
    }


def filter_high_confidence_clusters(
    clusters: Dict[str, Any],
    min_count: int = 5,
    min_positive_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Filter clusters that are good candidates for script proposal.
    
    Returns:
        List of cluster summaries suitable for script generation.
    """
    candidates = []
    
    for key, cluster in clusters.get("clusters", {}).items():
        if cluster["total_count"] < min_count:
            continue
        if cluster["positive_ratio"] < min_positive_ratio:
            continue
        
        # Find the best sub-cluster
        best_sub = None
        for sub in cluster["sub_clusters"]:
            if sub["count"] >= 3 and sub["avg_outcome"] > 0:
                if best_sub is None or sub["avg_outcome"] > best_sub["avg_outcome"]:
                    best_sub = sub
        
        if best_sub:
            candidates.append({
                "cluster_id": key,
                "dtype": cluster["dtype"],
                "pattern_key": cluster["pattern_key"],
                "total_count": cluster["total_count"],
                "avg_outcome": cluster["avg_outcome"],
                "positive_ratio": cluster["positive_ratio"],
                "best_context": best_sub["context_signature"],
                "best_context_count": best_sub["count"],
                "best_context_outcome": best_sub["avg_outcome"],
                "sample_fens": best_sub["sample_fens"],
                "active_nodes": best_sub["active_nodes"],
            })
    
    # Sort by outcome * count
    candidates.sort(key=lambda x: -x["avg_outcome"] * x["total_count"])
    
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Cluster extracted motifs for M5 script proposal."
    )
    parser.add_argument(
        "--motifs",
        type=Path,
        required=True,
        help="Path to motif dataset JSONL file",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("reports/motifs/clusters.json"),
        help="Output path for cluster analysis",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum motif count for a cluster to be considered (default: 5)",
    )
    parser.add_argument(
        "--min-positive-ratio",
        type=float,
        default=0.5,
        help="Minimum positive outcome ratio for candidates (default: 0.5)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Output path for candidate clusters (filtered)",
    )
    
    args = parser.parse_args()
    
    if not args.motifs.exists():
        print(f"Error: Motif file not found: {args.motifs}")
        sys.exit(1)
    
    # Load motif dataset
    print(f"Loading motifs from {args.motifs}")
    dataset = MotifDataset.load(args.motifs)
    print(f"Loaded {len(dataset)} motifs")
    
    if len(dataset) == 0:
        print("Warning: No motifs to cluster")
        sys.exit(0)
    
    # Cluster motifs
    print("Clustering motifs...")
    clusters = cluster_motifs(dataset)
    
    # Save full cluster analysis
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"Saved cluster analysis to {args.out}")
    
    # Print summary
    print(f"\n=== Cluster Summary ===")
    print(f"Total motifs: {clusters['total_motifs']}")
    print(f"Cluster count: {clusters['cluster_count']}")
    
    print("\nTop clusters by count:")
    sorted_clusters = sorted(
        clusters["clusters"].items(),
        key=lambda x: -x[1]["total_count"]
    )[:10]
    for key, c in sorted_clusters:
        print(f"  {key}: {c['total_count']} motifs, avg_outcome={c['avg_outcome']:.3f}, pos_ratio={c['positive_ratio']:.1%}")
    
    # Filter and save candidates if requested
    candidates = filter_high_confidence_clusters(
        clusters,
        min_count=args.min_count,
        min_positive_ratio=args.min_positive_ratio,
    )
    
    print(f"\nCandidate clusters for script proposal: {len(candidates)}")
    for c in candidates[:5]:
        print(f"  {c['cluster_id']}: {c['total_count']} motifs, outcome={c['avg_outcome']:.3f}")
    
    if args.candidates:
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        with open(args.candidates, "w") as f:
            json.dump({"candidates": candidates}, f, indent=2)
        print(f"\nSaved {len(candidates)} candidates to {args.candidates}")


if __name__ == "__main__":
    main()

