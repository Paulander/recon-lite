#!/usr/bin/env python3
"""
Clean Structural Spurt - KPK Training with Fresh Weights

Goal: Force hierarchical growth by starting with reset weights and
aggressive structural learning settings.

Strategy:
1. Reset all weights to baseline (0.5)
2. Use recursive_turbo persona (Success Bypass + Speculative Hoisting)
3. Force AND-gate creation even at high win rates (threshold=0.95)
4. Monitor max_depth - if still 1 at Stage 5, force-prune flat links

This creates the "struggling genius" scenario where the system must
build hierarchy to succeed, avoiding the "prodigy problem" where
flat networks vibe their way to victory.
"""

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


def reset_topology_weights(topo_path: Path, output_path: Path, baseline: float = 0.5) -> int:
    """
    Reset all edge weights to baseline value.
    
    Returns:
        Number of edges reset
    """
    with open(topo_path) as f:
        topo = json.load(f)
    
    count = 0
    for edge in topo.get("edges", []):
        if edge.get("weight", 1.0) != baseline:
            edge["weight"] = baseline
            count += 1
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(topo, f, indent=2)
    
    return count


def get_flat_links_to_legs(topo_path: Path) -> List[Tuple[str, str, float]]:
    """
    Get all direct links from sensors to legs (flat topology).
    """
    with open(topo_path) as f:
        topo = json.load(f)
    
    flat_links = []
    
    # Find leg nodes - handle both dict and list formats
    leg_nodes = set()
    nodes_data = topo.get("nodes", {})
    if isinstance(nodes_data, dict):
        nodes = list(nodes_data.values())
    else:
        nodes = nodes_data
    
    for node in nodes:
        if isinstance(node, dict) and node.get("group") == "legs":
            leg_nodes.add(node.get("id", ""))
    
    # Find direct links to legs - handle both dict and list formats
    edges_data = topo.get("edges", {})
    if isinstance(edges_data, dict):
        edges = list(edges_data.values())
    else:
        edges = edges_data
    
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        dst = edge.get("dst", edge.get("to", ""))
        src = edge.get("src", edge.get("from", ""))
        
        # Skip backbone nodes
        if src in ["kpk_detect", "kpk_root", "always_on"]:
            continue
            
        # Check if this is a direct sensor→leg link
        if dst in leg_nodes:
            weight = edge.get("weight", 1.0)
            flat_links.append((src, dst, weight))
    
    # Sort by weight descending
    flat_links.sort(key=lambda x: x[2], reverse=True)
    return flat_links


def force_prune_flat_links(topo_path: Path, output_path: Path, count: int = 5) -> List[str]:
    """
    Force-prune the strongest flat links to legs.
    """
    with open(topo_path) as f:
        topo = json.load(f)
    
    flat_links = get_flat_links_to_legs(topo_path)
    pruned = []
    
    # Find edges to remove
    edges_to_remove = set()
    for src, dst, weight in flat_links[:count]:
        edges_to_remove.add((src, dst))
        pruned.append(f"{src}->{dst} (weight={weight:.2f})")
    
    # Filter edges - handle both dict and list formats
    edges_data = topo.get("edges", {})
    if isinstance(edges_data, dict):
        # Dict format - filter by removing keys
        new_edges = {}
        for key, edge in edges_data.items():
            src = edge.get("src", edge.get("from", ""))
            dst = edge.get("dst", edge.get("to", ""))
            if (src, dst) not in edges_to_remove:
                new_edges[key] = edge
        topo["edges"] = new_edges
    else:
        # List format
        topo["edges"] = [
            e for e in edges_data
            if isinstance(e, dict) and (e.get("src", e.get("from", "")), e.get("dst", e.get("to", ""))) not in edges_to_remove
        ]
    
    with open(output_path, "w") as f:
        json.dump(topo, f, indent=2)
    
    return pruned


def run_clean_structural_spurt():
    """
    Run the full Clean Structural Spurt experiment.
    """
    print("="*60)
    print("CLEAN STRUCTURAL SPURT - KPK Fresh Weights")
    print("="*60)
    
    # Set environment variables for forced hoisting at HIGH win rates
    os.environ["M5_ENABLE_FORCED_HOISTING"] = "1"
    os.environ["M5_FORCED_HOIST_THRESHOLD_WIN_RATE"] = "0.10"  # Low threshold (crisis mode)
    os.environ["M5_FORCED_HOIST_THRESHOLD_HIGH"] = "0.95"  # High threshold (success trap)
    os.environ["M5_FORCED_HOIST_INTERVAL_CYCLES"] = "3"
    
    # MANDATORY TICK DEPTH for TRIAL node activation
    # Set to 3 to ensure TRIAL nodes have time to "speak" before Legs act
    MIN_INTERNAL_TICKS = 3
    
    print("\n📋 Configuration:")
    print("  - Weights reset to 0.5 (fresh start)")
    print("  - Success Bypass: ON")
    print("  - Speculative Hoisting: ON")
    print("  - Forced Hoist @ <10% win rate (crisis)")
    print("  - Forced Hoist @ >95% win rate (success trap)")
    print("  - Forced Pruning at Stage 5 if max_depth = 1")
    print(f"  - Mandatory Tick Depth: {MIN_INTERNAL_TICKS} (TRIAL activation)")
    print()
    
    base_topo = Path("topologies/kpk_legs_topology.json")
    output_dir = Path("snapshots/evolution/clean_structural_spurt")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Reset topology weights
    reset_topo = output_dir / "topology_reset.json"
    print("1. Resetting topology weights to baseline (0.5)...")
    count = reset_topology_weights(base_topo, reset_topo, baseline=0.5)
    print(f"   Reset {count} edge weights")
    print(f"   Saved: {reset_topo}")
    
    # Step 2: Import and run evolution driver
    from evolution_driver import run_evolution_training, EvolutionConfig
    
    # Run through curriculum stages
    stages = [
        "Pawn_One_Step",      # Stage 0
        "Guardian_E",         # Stage 1
        "Guardian_SW",        # Stage 2
        "Escort_5",           # Stage 3
        "Escort_7",           # Stage 4
        "Opposition_Lite",    # Stage 5 - check depth here
        "Opposition_Full",    # Stage 6
    ]
    
    all_results = []
    stem_cells_path = None
    
    for stage_idx, stage_name in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {stage_name}")
        print(f"{'='*60}")
        
        stage_dir = output_dir / f"stage{stage_idx}"
        
        config = EvolutionConfig(
            topology_path=reset_topo,  # Use reset topology
            games_per_cycle=50,
            max_cycles=15,
            max_promotions_per_cycle=3,
            snapshot_dir=stage_dir / "snapshots",
            trace_dir=stage_dir / "traces",
            signature_dir=stage_dir / "signatures",
            output_dir=stage_dir,
            plasticity_eta=0.08,  # Higher learning rate
            stem_cell_spawn_rate=0.15,  # Higher spawn rate
            stem_cell_max_cells=30,
            stem_cell_min_samples=20,
            stem_cells_load_path=stem_cells_path,
            use_curriculum=True,
            current_stage_idx=stage_idx,
            min_internal_ticks=MIN_INTERNAL_TICKS,  # TRIAL node activation
        )
        
        results = run_evolution_training(config)
        all_results.append({
            "stage": stage_idx,
            "name": stage_name,
            "win_rate": results[-1].win_rate if results else 0,
            "cycles": len(results),
        })
        
        # Update stem cells path for next stage
        stem_cells_path = stage_dir / "snapshots" / "stem_cells.json"
        if not stem_cells_path.exists():
            stem_cells_path = None
        
        # Check for forced pruning at Stage 5
        if stage_idx == 5:
            # Load and check max_depth
            from recon_lite.models.registry import TopologyRegistry
            from recon_lite.learning.m5_structure import compute_branching_metrics
            from recon_lite_chess.graph.builder import build_graph_from_topology
            
            final_topo = stage_dir / "snapshots" / "cycle_0015.json"
            if final_topo.exists():
                reg = TopologyRegistry(final_topo)
                # Build graph from topology to compute metrics
                graph = build_graph_from_topology(final_topo, reg)
                metrics = compute_branching_metrics(graph)
                max_depth = metrics.get("max_depth", 1)
                
                print(f"\n📊 Stage 5 Max Depth: {max_depth}")
                
                if max_depth <= 1:
                    print("\n⚠️ FORCED PRUNING TRIGGERED!")
                    print("   Max depth is 1, forcing hierarchy growth...")
                    
                    pruned_topo = stage_dir / "topology_force_pruned.json"
                    pruned = force_prune_flat_links(final_topo, pruned_topo, count=5)
                    
                    for p in pruned:
                        print(f"   🪓 Pruned: {p}")
                    
                    # Use pruned topology for remaining stages
                    reset_topo = pruned_topo
    
    # Save summary
    summary = {
        "stages": all_results,
        "settings": {
            "forced_hoist_low_threshold": 0.10,
            "forced_hoist_high_threshold": 0.95,
            "success_bypass": True,
            "speculative_hoisting": True,
        }
    }
    
    summary_path = output_dir / "clean_spurt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("CLEAN STRUCTURAL SPURT COMPLETE")
    print("="*60)
    print(f"Summary: {summary_path}")
    
    for r in all_results:
        print(f"  Stage {r['stage']} ({r['name']}): {r['win_rate']:.1%}")


if __name__ == "__main__":
    run_clean_structural_spurt()

