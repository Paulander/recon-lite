#!/usr/bin/env python3
"""
M5.2: Generate script proposals from clustered motifs.

Usage:
    uv run python demos/experiments/propose_scripts.py \
        --clusters reports/motifs/clusters.json \
        --out proposals/

Generates YAML proposal files for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_proposal_id(cluster_id: str, version: int = 1) -> str:
    """Generate a unique proposal ID."""
    # Clean up cluster_id for filename
    clean_id = cluster_id.replace(":", "_").replace("|", "_")
    return f"{clean_id}_v{version}"


def infer_preconditions(cluster: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Infer sensor preconditions from cluster features.
    
    Returns list of sensor requirements for the proposed script.
    """
    preconditions = []
    dtype = cluster.get("dtype", "")
    pattern = cluster.get("pattern_key", "")
    context_sig = cluster.get("best_context", "")
    
    # Map patterns to sensors
    if pattern == "fork_opportunity" or "has_fork" in context_sig:
        preconditions.append({"sensor": "detect_fork", "required": True})
    
    if pattern == "hanging_piece" or "en_prise" in context_sig:
        preconditions.append({"sensor": "detect_hanging_piece", "required": True})
    
    if pattern == "pin_detected" or "has_pin" in context_sig:
        preconditions.append({"sensor": "detect_pin", "required": True})
    
    if pattern == "passed_pawn" or "passed_pawn" in context_sig:
        preconditions.append({"sensor": "detect_passed_pawn", "required": True})
    
    if dtype == "endgame":
        preconditions.append({"sensor": "is_endgame", "required": True})
    
    # Add material context
    if "mat_up" in context_sig:
        preconditions.append({"sensor": "material_advantage", "required": False})
    elif "mat_down" in context_sig:
        preconditions.append({"sensor": "material_disadvantage", "required": False})
    
    return preconditions if preconditions else [{"sensor": "position_evaluated", "required": True}]


def infer_actuator(cluster: Dict[str, Any]) -> Dict[str, Any]:
    """
    Infer suggested actuator from cluster features.
    """
    pattern = cluster.get("pattern_key", "")
    
    # Map patterns to actuators
    actuator_map = {
        "fork_opportunity": {"type": "exploit_fork", "priority": "high"},
        "hanging_piece": {"type": "capture_or_protect", "priority": "high"},
        "pin_detected": {"type": "exploit_pin", "priority": "medium"},
        "passed_pawn": {"type": "advance_passer", "priority": "medium"},
        "endgame_simple": {"type": "endgame_technique", "priority": "medium"},
        "endgame_complex": {"type": "endgame_evaluation", "priority": "low"},
    }
    
    return actuator_map.get(pattern, {"type": "generic_move_selector", "priority": "low"})


def infer_nodes(cluster: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Infer suggested nodes for the script.
    """
    pattern = cluster.get("pattern_key", "")
    dtype = cluster.get("dtype", "")
    
    nodes = []
    
    # Root node
    root_id = f"{dtype}_{pattern}_root".replace(" ", "_")
    nodes.append({
        "id": root_id,
        "type": "SCRIPT",
        "description": f"Root script for {pattern} handling",
    })
    
    # Sensor node
    sensor_id = f"detect_{pattern}".replace(" ", "_")
    nodes.append({
        "id": sensor_id,
        "type": "SENSOR",
        "description": f"Detect {pattern} pattern",
    })
    
    # Actuator node
    actuator_id = f"handle_{pattern}".replace(" ", "_")
    nodes.append({
        "id": actuator_id,
        "type": "ACTUATOR",
        "description": f"Handle {pattern} situation",
    })
    
    return nodes


def infer_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Infer edges between proposed nodes.
    """
    edges = []
    
    if len(nodes) >= 3:
        # Root -> Sensor (SUB)
        edges.append({
            "src": nodes[0]["id"],
            "dst": nodes[1]["id"],
            "ltype": "SUB",
        })
        # Sensor -> Actuator (POR)
        edges.append({
            "src": nodes[1]["id"],
            "dst": nodes[2]["id"],
            "ltype": "POR",
        })
    
    return edges


def generate_proposal(
    cluster: Dict[str, Any],
    existing_nodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a script proposal from a cluster.
    
    Returns:
        Proposal dictionary in YAML-friendly format.
    """
    proposal_id = generate_proposal_id(cluster["cluster_id"])
    
    # Infer components
    preconditions = infer_preconditions(cluster)
    actuator = infer_actuator(cluster)
    nodes = infer_nodes(cluster)
    edges = infer_edges(nodes)
    
    # Calculate confidence from cluster stats
    confidence = min(0.95, 0.3 + 0.1 * min(cluster["total_count"], 5) + 0.3 * cluster["positive_ratio"])
    
    return {
        "proposal_id": proposal_id,
        "cluster_id": cluster["cluster_id"],
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending_review",
        
        # Statistics from cluster
        "statistics": {
            "sample_count": cluster["total_count"],
            "avg_outcome": round(cluster["avg_outcome"], 4),
            "positive_ratio": round(cluster["positive_ratio"], 4),
            "confidence": round(confidence, 3),
        },
        
        # Sample positions
        "sample_fens": cluster.get("sample_fens", [])[:5],
        
        # Inferred structure
        "preconditions": preconditions,
        "suggested_actuator": actuator,
        "suggested_nodes": nodes,
        "edges": edges,
        
        # Active nodes from traces
        "trace_active_nodes": cluster.get("active_nodes", []),
        
        # Review fields
        "review": {
            "reviewer": None,
            "reviewed_at": None,
            "decision": None,
            "notes": "",
        },
    }


def save_proposal(proposal: Dict[str, Any], output_dir: Path) -> Path:
    """Save a proposal to a YAML file."""
    filename = f"{proposal['proposal_id']}.yaml"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        yaml.dump(proposal, f, default_flow_style=False, sort_keys=False)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate script proposals from clustered motifs."
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to cluster analysis JSON file",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("proposals"),
        help="Output directory for proposal files",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for proposal generation (default: 0.5)",
    )
    parser.add_argument(
        "--max-proposals",
        type=int,
        default=10,
        help="Maximum number of proposals to generate (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposals without saving",
    )
    
    args = parser.parse_args()
    
    if not args.clusters.exists():
        print(f"Error: Cluster file not found: {args.clusters}")
        sys.exit(1)
    
    # Load cluster analysis
    print(f"Loading clusters from {args.clusters}")
    with open(args.clusters) as f:
        data = json.load(f)
    
    clusters = data.get("clusters", {})
    if not clusters:
        print("No clusters found in input file")
        sys.exit(0)
    
    # Generate proposals for qualifying clusters
    proposals = []
    
    for cluster_id, cluster_data in clusters.items():
        # Skip small clusters
        if cluster_data["total_count"] < 3:
            continue
        
        # Skip low positive ratio
        if cluster_data["positive_ratio"] < 0.4:
            continue
        
        # Create cluster dict with id
        cluster = {
            "cluster_id": cluster_id,
            **cluster_data,
        }
        
        # Get best sub-cluster info
        if cluster_data["sub_clusters"]:
            best_sub = max(cluster_data["sub_clusters"], key=lambda x: x["avg_outcome"])
            cluster["best_context"] = best_sub["context_signature"]
            cluster["sample_fens"] = best_sub["sample_fens"]
            cluster["active_nodes"] = best_sub["active_nodes"]
        
        proposal = generate_proposal(cluster)
        
        # Filter by confidence
        if proposal["statistics"]["confidence"] >= args.min_confidence:
            proposals.append(proposal)
    
    # Sort by confidence and limit
    proposals.sort(key=lambda x: -x["statistics"]["confidence"])
    proposals = proposals[:args.max_proposals]
    
    print(f"\nGenerated {len(proposals)} proposals")
    
    if args.dry_run:
        print("\n[DRY RUN] Proposals:")
        for p in proposals:
            print(f"\n--- {p['proposal_id']} ---")
            print(yaml.dump(p, default_flow_style=False))
        return
    
    # Save proposals
    args.out.mkdir(parents=True, exist_ok=True)
    
    for proposal in proposals:
        path = save_proposal(proposal, args.out)
        print(f"  Saved: {path}")
    
    # Create index file
    index = {
        "generated_at": datetime.utcnow().isoformat(),
        "source_clusters": str(args.clusters),
        "proposals": [
            {
                "id": p["proposal_id"],
                "cluster": p["cluster_id"],
                "confidence": p["statistics"]["confidence"],
                "status": p["status"],
            }
            for p in proposals
        ],
    }
    
    index_path = args.out / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved proposal index to {index_path}")


if __name__ == "__main__":
    main()

