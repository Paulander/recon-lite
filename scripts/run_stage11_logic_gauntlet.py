#!/usr/bin/env python3
"""
Stage 11 "Logic Gauntlet" - Zugzwang/Triangulation with Deep Propagation

This is the ULTIMATE test of whether the ReCoN engine can solve problems
that require hierarchical reasoning rather than flat "vibing".

Configuration:
- Tick Depth: 3 (Forces: Root → Leg → TRIAL Sensor)
- Gating: ON for kpk_pawn_leg and kpk_king_leg
- Persona: recursive_turbo (Bypass ON, Speculative Hoist ON)
- Threshold: 95% (Forces structural refinement during success)
- solidify_xp_threshold: 20 (Accelerates locking of successful nodes)

Goal: See Verticality at Depth 3+. If Hector wins Stage 11, it means
he has successfully used a TRIAL node to gate a LEG.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# EXPLORATION FALLBACK CONFIGURATION
# =============================================================================
# If win rate drops to 0% due to strict gating, we can relax temporarily
ENABLE_EXPLORATION_FALLBACK = True
FALLBACK_STUCK_THRESHOLD_MOVES = 50  # Relax gating after 50 moves without progress
FALLBACK_WIN_RATE_THRESHOLD = 0.05   # Trigger fallback if win rate < 5%


def prepare_gated_topology(
    source_path: Path, 
    output_path: Path,
    gated_nodes: List[str] = None,
    reset_weights: bool = True,
    baseline_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Prepare a topology with gating enabled on specified nodes.
    
    Gating (require_child_confirm=True) forces a node to wait for
    at least one child to confirm before it can act.
    
    Args:
        source_path: Path to source topology JSON
        output_path: Path to save modified topology
        gated_nodes: List of node IDs to enable gating on
        reset_weights: Whether to reset all edge weights
        baseline_weight: Weight to set if resetting
        
    Returns:
        Modified topology dict
    """
    gated_nodes = gated_nodes or ["kpk_pawn_leg", "kpk_king_leg"]
    
    with open(source_path) as f:
        topo = json.load(f)
    
    # Apply gating to specified nodes
    nodes_data = topo.get("nodes", [])
    gated_count = 0
    
    for node in nodes_data:
        if isinstance(node, dict) and node.get("id") in gated_nodes:
            node.setdefault("meta", {})["require_child_confirm"] = True
            gated_count += 1
            print(f"  🔒 Gating enabled: {node['id']}")
    
    # Reset weights if requested
    if reset_weights:
        edges_data = topo.get("edges", [])
        weight_count = 0
        
        if isinstance(edges_data, list):
            for edge in edges_data:
                if isinstance(edge, dict):
                    edge["weight"] = baseline_weight
                    weight_count += 1
        elif isinstance(edges_data, dict):
            for key, edge in edges_data.items():
                if isinstance(edge, dict):
                    edge["weight"] = baseline_weight
                    weight_count += 1
        
        print(f"  ⚖️  Reset {weight_count} edge weights to {baseline_weight}")
    
    # Save modified topology
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(topo, f, indent=2)
    
    print(f"  💾 Saved gated topology: {output_path}")
    return topo


def apply_gating_to_graph(graph: "Graph", gated_nodes: List[str]):
    """
    Apply gating metadata to nodes in an existing graph.
    
    This is called at runtime to ensure gating is active even if
    the topology was loaded without the metadata.
    """
    for node_id in gated_nodes:
        if node_id in graph.nodes:
            graph.nodes[node_id].meta["require_child_confirm"] = True
            print(f"    🔒 Runtime gating: {node_id}")


def run_stage11_logic_gauntlet():
    """
    Run the Stage 11 "Logic Gauntlet" experiment.
    """
    print("=" * 70)
    print("STAGE 11 LOGIC GAUNTLET - Zugzwang with Deep Propagation")
    print("=" * 70)
    print()
    
    # Configuration
    MIN_TICK_DEPTH = 3  # Forces: Root → Leg → TRIAL Sensor
    GATED_NODES = ["kpk_pawn_leg", "kpk_king_leg"]
    SOLIDIFY_XP_THRESHOLD = 20  # Accelerate locking
    
    # Stage selection - can be overridden by environment variable
    TARGET_STAGE = int(os.environ.get("TARGET_STAGE", "12"))  # Default: 12 (Zugzwang Full)
    
    # Set environment variables for recursive_turbo persona
    os.environ["M5_ENABLE_FORCED_HOISTING"] = "1"
    os.environ["M5_FORCED_HOIST_THRESHOLD_WIN_RATE"] = "0.10"  # Crisis mode
    os.environ["M5_FORCED_HOIST_THRESHOLD_HIGH"] = "0.95"  # Success trap
    os.environ["M5_FORCED_HOIST_INTERVAL_CYCLES"] = "3"
    os.environ["M5_SOLIDIFY_XP_THRESHOLD"] = str(SOLIDIFY_XP_THRESHOLD)
    
    # HEURISTIC SUPPRESSION (Leg Capping):
    # When enabled, disable the "approach" heuristic in king_leg
    # This forces the network to rely on TRIAL sensors for direction
    ENABLE_HEURISTIC_SUPPRESSION = os.environ.get("KPK_HEURISTIC_SUPPRESSION", "0") == "1"
    if ENABLE_HEURISTIC_SUPPRESSION:
        print("  - Heuristic Suppression: ON (king approach disabled)")
    
    print("📋 Configuration:")
    print(f"  - Tick Depth: {MIN_TICK_DEPTH} (Deep Propagation)")
    print(f"  - Gating: ON for {', '.join(GATED_NODES)}")
    print(f"  - Persona: recursive_turbo (Bypass ON, Speculative Hoist ON)")
    print(f"  - Hoist Threshold: 95% (Success Trap enabled)")
    print(f"  - Solidify XP: {SOLIDIFY_XP_THRESHOLD} (Accelerated locking)")
    print(f"  - Target Stage: {TARGET_STAGE} (Zugzwang/Triangulation)")
    if ENABLE_EXPLORATION_FALLBACK:
        print(f"  - Exploration Fallback: ON (trigger at <{FALLBACK_WIN_RATE_THRESHOLD:.0%} win rate)")
    print()
    
    # Paths
    base_topo = Path("topologies/kpk_legs_topology.json")
    output_dir = Path("snapshots/evolution/stage11_logic_gauntlet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare gated topology
    print("1. Preparing gated topology...")
    gated_topo_path = output_dir / "topology_gated.json"
    prepare_gated_topology(
        source_path=base_topo,
        output_path=gated_topo_path,
        gated_nodes=GATED_NODES,
        reset_weights=True,
        baseline_weight=0.5,
    )
    print()
    
    # Step 2: Import evolution driver
    print("2. Loading evolution driver...")
    from evolution_driver import run_evolution_training, EvolutionConfig
    
    # Check if we have the curriculum stages
    try:
        from recon_lite_chess.training.generators import KPK_STAGES
        max_stage = len(KPK_STAGES) - 1
        target_stage = min(TARGET_STAGE, max_stage)
        print(f"   Available stages: 0-{max_stage}")
        print(f"   Target stage: {target_stage} ({KPK_STAGES[target_stage].name})")
    except ImportError:
        target_stage = TARGET_STAGE
        print(f"   Target stage: {target_stage}")
    print()
    
    # Step 3: Run the gauntlet
    print("3. Starting Stage 11 Logic Gauntlet...")
    print("=" * 70)
    
    # Load stem cells from previous clean spurt if available
    stem_cells_path = None
    clean_spurt_stem = Path("snapshots/evolution/clean_structural_spurt/stage5/snapshots/stem_cells.json")
    if clean_spurt_stem.exists():
        stem_cells_path = clean_spurt_stem
        print(f"   Loading stem cells from: {stem_cells_path}")
    
    config = EvolutionConfig(
        topology_path=gated_topo_path,
        games_per_cycle=50,
        max_cycles=20,  # More cycles for difficult stage
        max_promotions_per_cycle=5,
        snapshot_dir=output_dir / "snapshots",
        trace_dir=output_dir / "traces",
        signature_dir=output_dir / "signatures",
        output_dir=output_dir,
        plasticity_eta=0.08,  # Higher learning rate
        stem_cell_spawn_rate=0.15,  # Higher spawn rate
        stem_cell_max_cells=50,  # More cells for complex stage
        stem_cell_min_samples=15,  # Faster promotion
        stem_cells_load_path=stem_cells_path,
        use_curriculum=True,
        current_stage_idx=target_stage,
        min_internal_ticks=MIN_TICK_DEPTH,  # CRITICAL: Deep Propagation
    )
    
    # Run training
    results = run_evolution_training(config)
    
    # Step 4: Analyze results
    print()
    print("=" * 70)
    print("STAGE 11 LOGIC GAUNTLET COMPLETE")
    print("=" * 70)
    
    if results:
        final_win_rate = results[-1].win_rate
        total_games = sum(r.games_played for r in results)
        total_promotions = sum(len(r.promotions) for r in results)
        
        print(f"Total games: {total_games}")
        print(f"Final win rate: {final_win_rate:.1%}")
        print(f"Total promotions: {total_promotions}")
        
        # Check for depth > 1
        from recon_lite.models.registry import TopologyRegistry
        from recon_lite.learning.m5_structure import compute_branching_metrics
        from recon_lite_chess.graph.builder import build_graph_from_topology
        
        final_topo = output_dir / "snapshots" / f"cycle_{config.max_cycles:04d}.json"
        if final_topo.exists():
            reg = TopologyRegistry(final_topo)
            graph = build_graph_from_topology(final_topo, reg)
            metrics = compute_branching_metrics(graph)
            max_depth = metrics.get("max_depth", 1)
            
            print(f"\n📊 Structural Metrics:")
            print(f"  Max Depth: {max_depth}")
            print(f"  Total Nodes: {metrics.get('total_nodes', 0)}")
            print(f"  Hoisted Clusters: {metrics.get('hoisted_count', 0)}")
            
            if max_depth >= 3:
                print("\n🎉 SUCCESS! Depth 3+ achieved!")
                print("   The hierarchical brain is THINKING, not just VIBING!")
            elif max_depth == 2:
                print("\n⚠️ Progress! Depth 2 achieved.")
                print("   Middle managers exist, but no deep hierarchy yet.")
            else:
                print("\n❌ Depth still 1. Flat network persists.")
                print("   Consider more aggressive forced hoisting or longer training.")
        
        # Check for exploration fallback trigger
        if final_win_rate < FALLBACK_WIN_RATE_THRESHOLD and ENABLE_EXPLORATION_FALLBACK:
            print(f"\n⚠️ EXPLORATION FALLBACK would trigger (win rate {final_win_rate:.1%} < {FALLBACK_WIN_RATE_THRESHOLD:.0%})")
            print("   Consider relaxing gating requirements temporarily.")
    
    # Save summary
    summary = {
        "experiment": "Stage 11 Logic Gauntlet",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tick_depth": MIN_TICK_DEPTH,
            "gated_nodes": GATED_NODES,
            "solidify_xp_threshold": SOLIDIFY_XP_THRESHOLD,
            "target_stage": target_stage,
            "exploration_fallback_enabled": ENABLE_EXPLORATION_FALLBACK,
        },
        "results": [
            {
                "cycle": r.cycle,
                "win_rate": r.win_rate,
                "games": r.games_played,
                "promotions": len(r.promotions),
            }
            for r in results
        ] if results else [],
    }
    
    summary_path = output_dir / "logic_gauntlet_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    return results


def analyze_trial_leg_coactivation(trace_dir: Path) -> List[Dict[str, Any]]:
    """
    Analyze trace files to find instances where a LEG and its TRIAL parent
    are active simultaneously.
    
    This is the key indicator of successful hierarchical gating.
    """
    import glob
    
    coactivations = []
    trace_files = list(trace_dir.glob("*.jsonl"))
    
    print(f"\nAnalyzing {len(trace_files)} trace files for TRIAL-LEG coactivation...")
    
    for trace_file in trace_files:
        with open(trace_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    tick = json.loads(line)
                    active_nodes = tick.get("active_nodes", [])
                    
                    # Check for LEG nodes
                    active_legs = [n for n in active_nodes if "leg" in n.lower()]
                    # Check for TRIAL nodes
                    active_trials = [n for n in active_nodes if n.startswith("TRIAL_")]
                    
                    if active_legs and active_trials:
                        coactivations.append({
                            "file": trace_file.name,
                            "line": line_num,
                            "legs": active_legs,
                            "trials": active_trials,
                            "tick": tick.get("tick", 0),
                        })
                        
                except json.JSONDecodeError:
                    continue
    
    if coactivations:
        print(f"🎯 Found {len(coactivations)} TRIAL-LEG coactivation instances!")
        for i, co in enumerate(coactivations[:5]):  # Show first 5
            print(f"   {i+1}. {co['file']}:{co['line']} - LEGs: {co['legs']}, TRIALs: {co['trials']}")
    else:
        print("   No TRIAL-LEG coactivations found yet.")
    
    return coactivations


if __name__ == "__main__":
    results = run_stage11_logic_gauntlet()
    
    # Analyze coactivation if traces exist
    trace_dir = Path("snapshots/evolution/stage11_logic_gauntlet/traces")
    if trace_dir.exists():
        analyze_trial_leg_coactivation(trace_dir)

