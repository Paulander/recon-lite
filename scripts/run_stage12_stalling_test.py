#!/usr/bin/env python3
"""Stage 12 (Zugzwang Full) Stalling Test with Deep Propagation.

This is the "Deep-Pressure" plan's final KPK gauntlet. Stage 12 requires
"waiting" moves that break simple heuristics - the agent must NOT always
approach or push.

Key Settings:
- min_internal_ticks = 5 (deep propagation)
- require_child_confirm on both legs (gating)
- max_trial_slots = 15 (sparsity sledgehammer)
- Inertia pruning enabled (20 cycle threshold)

Success Criteria:
- Observe Depth-3 chain: kpk_root -> kpk_execute -> Hoisted_Manager -> TRIAL_Sensor
- OR win rate drops, triggering Crisis Mode and forced hoisting

Usage:
    python scripts/run_stage12_stalling_test.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.evolution_driver import (
    EvolutionConfig, 
    run_evolution_training,
    CycleResult,
)
from recon_lite.models.registry import TopologyRegistry
from recon_lite.learning.m5_structure import compute_branching_metrics


def main():
    """Run Stage 12 Stalling Test with Deep-Pressure settings."""
    
    print("=" * 70)
    print("STAGE 12 STALLING TEST - Deep-Pressure Plan")
    print("=" * 70)
    print()
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    MIN_INTERNAL_TICKS = 5  # Deep propagation - forces signal to reach TRIAL sensors
    MAX_TRIAL_SLOTS = 15    # Sparsity Sledgehammer - cap TRIAL tier
    INERTIA_PRUNE_CYCLES = 20  # Remove idle TRIAL cells after 20 cycles
    TARGET_STAGE = 12       # Zugzwang Full - requires "waiting" moves
    GATED_LEGS = ["kpk_pawn_leg", "kpk_king_leg"]  # Legs that require child confirm
    
    # Set environment variables for M5 configuration
    os.environ["M5_ENABLE_FORCED_HOISTING"] = "1"
    os.environ["M5_FORCED_HOIST_THRESHOLD_WIN_RATE"] = "0.10"  # Crisis mode below 10%
    os.environ["M5_FORCED_HOIST_THRESHOLD_HIGH"] = "0.95"  # Success trap above 95%
    os.environ["M5_FORCED_HOIST_INTERVAL_CYCLES"] = "3"
    os.environ["M5_INERTIA_PRUNE_CYCLES"] = str(INERTIA_PRUNE_CYCLES)
    
    # Enable heuristic suppression (Leg Capping) to break "vibe" approach
    os.environ["KPK_HEURISTIC_SUPPRESSION"] = "1"
    
    print(f"Configuration:")
    print(f"  - Min Internal Ticks: {MIN_INTERNAL_TICKS} (deep propagation)")
    print(f"  - Max Trial Slots: {MAX_TRIAL_SLOTS} (sparsity)")
    print(f"  - Inertia Prune: {INERTIA_PRUNE_CYCLES} cycles")
    print(f"  - Target Stage: {TARGET_STAGE} (zugzwang)")
    print(f"  - Gated Legs: {GATED_LEGS}")
    print(f"  - Heuristic Suppression: ON")
    print()
    
    # =========================================================================
    # PREPARE GATED TOPOLOGY
    # =========================================================================
    
    output_dir = Path("snapshots/evolution/stage12_stalling_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Preparing gated topology...")
    
    # Copy base topology and add gating
    base_topology = Path("topologies/kpk_legs_topology.json")
    gated_topology = output_dir / "topology_gated.json"
    
    with open(base_topology) as f:
        topo = json.load(f)
    
    # Enable gating on leg nodes
    # Handle both list and dict formats for nodes
    nodes = topo.get("nodes", [])
    if isinstance(nodes, list):
        # List format: find nodes by id
        for node in nodes:
            if node.get("id") in GATED_LEGS:
                if "meta" not in node:
                    node["meta"] = {}
                node["meta"]["require_child_confirm"] = True
                print(f"  - Gating enabled: {node.get('id')}")
    else:
        # Dict format: key is node id
        for leg_id in GATED_LEGS:
            if leg_id in nodes:
                if "meta" not in nodes[leg_id]:
                    nodes[leg_id]["meta"] = {}
                nodes[leg_id]["meta"]["require_child_confirm"] = True
                print(f"  - Gating enabled: {leg_id}")
    
    # Reset edge weights to 0.5 (clean slate)
    # Handle both list and dict formats for edges
    edges = topo.get("edges", [])
    weight_reset_count = 0
    if isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, dict) and "weight" in edge:
                edge["weight"] = 0.5
                weight_reset_count += 1
    else:
        for edge_key, edge_data in edges.items():
            if isinstance(edge_data, dict) and "weight" in edge_data:
                edge_data["weight"] = 0.5
                weight_reset_count += 1
    print(f"  - Reset {weight_reset_count} edge weights to 0.5")
    
    with open(gated_topology, "w") as f:
        json.dump(topo, f, indent=2)
    print(f"  - Saved: {gated_topology}")
    print()
    
    # =========================================================================
    # RUN EVOLUTION TRAINING
    # =========================================================================
    
    print("2. Starting Evolution Training...")
    print(f"   Target: Stage {TARGET_STAGE} (zugzwang)")
    print()
    
    config = EvolutionConfig(
        topology_path=gated_topology,
        games_per_cycle=50,
        max_cycles=25,
        max_promotions_per_cycle=3,
        
        # Directories
        snapshot_dir=output_dir / "snapshots",
        trace_dir=output_dir / "traces",
        output_dir=output_dir / "reports",
        
        # Plasticity
        plasticity_eta=0.03,
        
        # Stem cell settings - SPARSITY SLEDGEHAMMER
        stem_cell_spawn_rate=0.10,
        stem_cell_max_cells=50,  # Total cells can be higher
        max_trial_slots=MAX_TRIAL_SLOTS,  # But TRIAL tier is capped
        
        # Curriculum - start at Stage 12
        use_curriculum=True,
        current_stage_idx=TARGET_STAGE,
        stage_promotion_threshold=0.90,
        
        # Deep propagation for TRIAL activation
        min_internal_ticks=MIN_INTERNAL_TICKS,
    )
    
    start_time = datetime.now()
    results = run_evolution_training(config)
    end_time = datetime.now()
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    print()
    print("=" * 70)
    print("STAGE 12 STALLING TEST COMPLETE")
    print("=" * 70)
    print(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
    print()
    
    # Load final topology
    final_snapshot = sorted((output_dir / "snapshots").glob("cycle_*.json"))[-1] if (output_dir / "snapshots").exists() else None
    
    max_depth = 1
    total_nodes = 0
    hoisted_count = 0
    por_count = 0
    
    if final_snapshot:
        with open(final_snapshot) as f:
            final_topo = json.load(f)
        
        # Build registry for metrics
        registry = TopologyRegistry(final_snapshot)
        try:
            from recon_lite_chess.graph.builder import build_graph_from_topology
            graph = build_graph_from_topology(final_snapshot, registry)
            metrics = compute_branching_metrics(graph)
            max_depth = metrics.get("max_depth", 1)
            total_nodes = len(final_topo.get("nodes", {}))
            
            # Count hoisted and POR
            for nid, node in final_topo.get("nodes", {}).items():
                meta = node.get("meta", {})
                if meta.get("origin") == "hoisted" or "cluster" in nid.lower():
                    hoisted_count += 1
            
            for edge_key, edge in final_topo.get("edges", {}).items():
                if isinstance(edge, dict) and edge.get("type") == "POR":
                    por_count += 1
        except Exception as e:
            print(f"Warning: Could not compute metrics: {e}")
    
    # Summary
    final_win_rate = results[-1].win_rate if results else 0.0
    total_games = sum(r.games_played for r in results)
    
    print(f"Final Win Rate: {final_win_rate:.1%}")
    print(f"Total Games: {total_games}")
    print()
    print("Structural Metrics:")
    print(f"  Max Depth: {max_depth}")
    print(f"  Total Nodes: {total_nodes}")
    print(f"  Hoisted Clusters: {hoisted_count}")
    print(f"  POR Links: {por_count}")
    print()
    
    # Check success criteria
    if max_depth >= 3:
        print("SUCCESS: Depth-3 chain achieved!")
        print("  kpk_root -> kpk_execute -> Hoisted_Manager -> TRIAL_Sensor")
    elif max_depth >= 2:
        print("PARTIAL: Depth-2 structure (hierarchy forming)")
    else:
        print("FLAT: max_depth=1 - hierarchy not yet formed")
        if final_win_rate < 0.50:
            print("  Crisis Mode should be triggering forced hoisting")
        else:
            print("  Still 'vibing' - may need longer training or more suppression")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "min_internal_ticks": MIN_INTERNAL_TICKS,
            "max_trial_slots": MAX_TRIAL_SLOTS,
            "inertia_prune_cycles": INERTIA_PRUNE_CYCLES,
            "target_stage": TARGET_STAGE,
            "heuristic_suppression": True,
        },
        "results": {
            "final_win_rate": final_win_rate,
            "total_games": total_games,
            "cycles_completed": len(results),
        },
        "metrics": {
            "max_depth": max_depth,
            "total_nodes": total_nodes,
            "hoisted_clusters": hoisted_count,
            "por_links": por_count,
        },
        "success": max_depth >= 3,
    }
    
    summary_path = output_dir / "stalling_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    # =========================================================================
    # ANALYZE TRACES FOR POR PATTERNS
    # =========================================================================
    
    print()
    print("3. Analyzing traces for POR patterns (Wait -> King move)...")
    
    trace_dir = output_dir / "traces"
    if trace_dir.exists():
        trace_files = list(trace_dir.glob("*.jsonl"))
        wait_to_king_count = 0
        total_episodes = 0
        
        for trace_file in trace_files[-5:]:  # Check last 5 cycles
            with open(trace_file) as f:
                for line in f:
                    try:
                        episode = json.loads(line)
                        total_episodes += 1
                        
                        # Look for wait patterns followed by king moves
                        ticks = episode.get("ticks", [])
                        for i, tick in enumerate(ticks[:-1]):
                            active = tick.get("active_nodes", [])
                            next_active = ticks[i + 1].get("active_nodes", [])
                            
                            # Check for wait-like patterns (no move suggested)
                            if not tick.get("action") and "kpk_king_leg" in next_active:
                                wait_to_king_count += 1
                                break
                    except:
                        continue
        
        if wait_to_king_count > 0:
            print(f"  Found {wait_to_king_count}/{total_episodes} episodes with Wait->King pattern")
            print("  This is a candidate for POR discovery!")
        else:
            print("  No clear Wait->King patterns detected yet")
    
    return max_depth >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

