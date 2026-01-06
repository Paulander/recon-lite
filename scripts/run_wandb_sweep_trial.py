#!/usr/bin/env python3
"""W&B Sweep Trial Script for Deep-Pressure Plan.

This script is called by the W&B sweep agent for each trial.
It reads hyperparameters from wandb.config and runs evolution training.

The primary metric is `depth_win_score = final_win_rate * max_depth`,
which rewards both winning AND building hierarchical structure.

Usage (via wandb agent):
    wandb sweep configs/deep_leg_sweep.yaml
    wandb agent <sweep_id>

Direct usage (for testing):
    python scripts/run_wandb_sweep_trial.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Running in local mode.")

from scripts.evolution_driver import (
    EvolutionConfig, 
    run_evolution_training,
)
from recon_lite.models.registry import TopologyRegistry
from recon_lite.learning.m5_structure import compute_branching_metrics


def run_trial(config_dict: dict = None):
    """Run a single sweep trial with given hyperparameters.
    
    Args:
        config_dict: Optional dict of hyperparameters (for local testing).
                     If None, reads from wandb.config.
    """
    
    # Initialize W&B if available
    if HAS_WANDB:
        wandb.init(project="recon-lite-deep-pressure")
        config = wandb.config
    else:
        # Use provided config or defaults for local testing
        config = config_dict or {
            "forced_hoist_high_threshold": 0.95,
            "plasticity_eta": 0.03,
            "min_internal_ticks": 5,
            "max_trial_slots": 15,
            "inertia_prune_cycles": 20,
            "forced_hoist_low_threshold": 0.10,
            "target_stage": 12,
            "max_cycles": 20,
            "games_per_cycle": 50,
        }
    
    print("=" * 70)
    print("W&B SWEEP TRIAL - Deep-Pressure Plan")
    print("=" * 70)
    print()
    print("Hyperparameters:")
    for k, v in dict(config).items():
        print(f"  {k}: {v}")
    print()
    
    # Set environment variables for M5 configuration
    os.environ["M5_ENABLE_FORCED_HOISTING"] = "1"
    os.environ["M5_FORCED_HOIST_THRESHOLD_WIN_RATE"] = str(config.get("forced_hoist_low_threshold", 0.10))
    os.environ["M5_FORCED_HOIST_THRESHOLD_HIGH"] = str(config.get("forced_hoist_high_threshold", 0.95))
    os.environ["M5_FORCED_HOIST_INTERVAL_CYCLES"] = "3"
    os.environ["M5_INERTIA_PRUNE_CYCLES"] = str(config.get("inertia_prune_cycles", 20))
    
    # Enable heuristic suppression
    os.environ["KPK_HEURISTIC_SUPPRESSION"] = "1"
    
    # Prepare topology with gating
    output_dir = Path("snapshots/evolution/wandb_sweep_trial")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_topology = Path("topologies/kpk_legs_topology.json")
    gated_topology = output_dir / "topology_gated.json"
    
    with open(base_topology) as f:
        topo = json.load(f)
    
    # Enable gating on leg nodes
    for leg_id in ["kpk_pawn_leg", "kpk_king_leg"]:
        if leg_id in topo.get("nodes", {}):
            if "meta" not in topo["nodes"][leg_id]:
                topo["nodes"][leg_id]["meta"] = {}
            topo["nodes"][leg_id]["meta"]["require_child_confirm"] = True
    
    # Reset edge weights
    for edge_key, edge_data in topo.get("edges", {}).items():
        if isinstance(edge_data, dict) and "weight" in edge_data:
            edge_data["weight"] = 0.5
    
    with open(gated_topology, "w") as f:
        json.dump(topo, f, indent=2)
    
    # Run evolution training
    evo_config = EvolutionConfig(
        topology_path=gated_topology,
        games_per_cycle=config.get("games_per_cycle", 50),
        max_cycles=config.get("max_cycles", 20),
        max_promotions_per_cycle=3,
        
        # Directories
        snapshot_dir=output_dir / "snapshots",
        trace_dir=output_dir / "traces",
        output_dir=output_dir / "reports",
        
        # Plasticity - from sweep config
        plasticity_eta=config.get("plasticity_eta", 0.03),
        
        # Stem cell settings - SPARSITY from sweep config
        stem_cell_spawn_rate=0.10,
        stem_cell_max_cells=50,
        max_trial_slots=config.get("max_trial_slots", 15),
        
        # Curriculum
        use_curriculum=True,
        current_stage_idx=config.get("target_stage", 12),
        stage_promotion_threshold=0.90,
        
        # Deep propagation - from sweep config
        min_internal_ticks=config.get("min_internal_ticks", 5),
    )
    
    start_time = datetime.now()
    results = run_evolution_training(evo_config)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Extract final metrics
    final_win_rate = results[-1].win_rate if results else 0.0
    
    # Load final topology for structural metrics
    final_snapshots = sorted((output_dir / "snapshots").glob("cycle_*.json"))
    max_depth = 1
    hoisted_count = 0
    por_count = 0
    
    if final_snapshots:
        try:
            with open(final_snapshots[-1]) as f:
                final_topo = json.load(f)
            
            registry = TopologyRegistry(final_snapshots[-1])
            from recon_lite_chess.graph.builder import build_graph_from_topology
            graph = build_graph_from_topology(final_snapshots[-1], registry)
            metrics = compute_branching_metrics(graph)
            max_depth = metrics.get("max_depth", 1)
            
            # Count hoisted and POR
            for nid, node in final_topo.get("nodes", {}).items():
                meta = node.get("meta", {})
                if meta.get("origin") == "hoisted" or "cluster" in nid.lower():
                    hoisted_count += 1
            
            for edge_key, edge in final_topo.get("edges", {}).items():
                if isinstance(edge, dict) and edge.get("type") == "POR":
                    por_count += 1
        except Exception as e:
            print(f"Warning: Could not compute full metrics: {e}")
    
    # Calculate primary metric: depth_win_score
    # This rewards both winning AND building hierarchy
    depth_win_score = final_win_rate * max_depth
    
    # Log metrics to W&B
    metrics_dict = {
        "final_win_rate": final_win_rate,
        "max_depth": max_depth,
        "depth_win_score": depth_win_score,  # PRIMARY METRIC
        "hoisted_clusters": hoisted_count,
        "por_links": por_count,
        "duration_seconds": duration,
        "cycles_completed": len(results),
    }
    
    if HAS_WANDB:
        wandb.log(metrics_dict)
        wandb.summary.update(metrics_dict)
    
    print()
    print("=" * 70)
    print("TRIAL COMPLETE")
    print("=" * 70)
    print(f"Final Win Rate: {final_win_rate:.1%}")
    print(f"Max Depth: {max_depth}")
    print(f"Depth-Win Score: {depth_win_score:.3f}")  # PRIMARY METRIC
    print(f"Hoisted Clusters: {hoisted_count}")
    print(f"POR Links: {por_count}")
    print(f"Duration: {duration:.1f}s")
    
    if HAS_WANDB:
        wandb.finish()
    
    return depth_win_score


if __name__ == "__main__":
    run_trial()

