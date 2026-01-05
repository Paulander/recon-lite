#!/usr/bin/env python3
"""
Quick test to verify TRIAL nodes appear in active_nodes after our fix.

This runs just 3 cycles of Stage 0 and checks if TRIAL nodes are firing.
"""

import json
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Enable forced hoisting to get TRIAL nodes quickly
os.environ["M5_ENABLE_FORCED_HOISTING"] = "1"
os.environ["M5_FORCED_HOIST_THRESHOLD_WIN_RATE"] = "0.10"
os.environ["M5_FORCED_HOIST_THRESHOLD_HIGH"] = "0.95"
os.environ["M5_FORCED_HOIST_INTERVAL_CYCLES"] = "1"  # Force hoist every cycle

from evolution_driver import EvolutionConfig, run_evolution_training

def main():
    print("="*60)
    print("TRIAL NODE ACTIVATION TEST")
    print("="*60)
    print("\nThis test verifies that TRIAL nodes appear in active_nodes.")
    print()
    
    # Create test output directory
    output_dir = Path("snapshots/evolution/trial_activation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reset topology
    base_topo = Path("topologies/kpk_legs_topology.json")
    reset_topo = output_dir / "topology_reset.json"
    
    # Copy base topology with reset weights
    with open(base_topo) as f:
        topo = json.load(f)
    
    # Reset weights to 0.5
    for edge in topo.get("edges", []):
        if isinstance(edge, dict):
            edge["weight"] = 0.5
    
    with open(reset_topo, "w") as f:
        json.dump(topo, f, indent=2)
    
    print(f"1. Reset topology: {reset_topo}")
    
    # Run 3 cycles of Stage 0
    config = EvolutionConfig(
        topology_path=reset_topo,
        games_per_cycle=20,  # Small for quick test
        max_cycles=3,  # Just 3 cycles
        max_promotions_per_cycle=5,  # More promotions
        snapshot_dir=output_dir / "snapshots",
        trace_dir=output_dir / "traces",
        signature_dir=output_dir / "signatures",
        output_dir=output_dir,
        plasticity_eta=0.08,
        stem_cell_spawn_rate=0.25,  # Higher spawn rate
        stem_cell_max_cells=50,
        stem_cell_min_samples=10,  # Lower threshold for quicker TRIAL promotion
        use_curriculum=True,
        current_stage_idx=0,  # Stage 0 only
    )
    
    print(f"\n2. Running 3 cycles of Stage 0...")
    results = run_evolution_training(config)
    
    # Check traces for TRIAL nodes in active_nodes
    print(f"\n3. Analyzing traces for TRIAL node activation...")
    
    trace_files = sorted(output_dir.glob("traces/*.jsonl"))
    
    trial_activations = []
    total_games = 0
    
    for trace_file in trace_files:
        with open(trace_file) as f:
            for line in f:
                ep = json.loads(line)
                total_games += 1
                
                for tick in ep.get("ticks", []):
                    active = tick.get("active_nodes", [])
                    trials = [n for n in active if n.startswith("TRIAL") or n.startswith("cluster")]
                    if trials:
                        trial_activations.append({
                            "episode": ep.get("episode_id"),
                            "tick": tick.get("tick_id"),
                            "active_trials": trials,
                        })
    
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total games analyzed: {total_games}")
    print(f"Ticks with TRIAL node activation: {len(trial_activations)}")
    
    if trial_activations:
        print("\n✅ SUCCESS! TRIAL nodes are FIRING!")
        print("\nSample activations:")
        for act in trial_activations[:5]:
            print(f"  {act['episode']} tick {act['tick']}: {act['active_trials']}")
    else:
        print("\n❌ FAILURE: No TRIAL node activations detected!")
        
        # Check what nodes are in the topology
        final_topo = output_dir / "snapshots" / "cycle_0003.json"
        if final_topo.exists():
            with open(final_topo) as f:
                topo = json.load(f)
            nodes = topo.get("nodes", {})
            trial_nodes = [k for k in nodes.keys() if k.startswith("TRIAL")]
            print(f"\nTopology has {len(trial_nodes)} TRIAL nodes")
            if trial_nodes[:5]:
                print(f"  Sample: {trial_nodes[:5]}")
            
            # Check edges
            edges = topo.get("edges", {})
            if isinstance(edges, dict):
                edges = list(edges.values())
            trial_edges = [(e.get("src"), e.get("dst")) for e in edges 
                          if isinstance(e, dict) and e.get("dst", "").startswith("TRIAL")]
            print(f"\nEdges TO TRIAL nodes: {len(trial_edges)}")
            if trial_edges[:5]:
                print(f"  Sample: {trial_edges[:5]}")

if __name__ == "__main__":
    main()

