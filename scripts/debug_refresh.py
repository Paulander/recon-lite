#!/usr/bin/env python3
"""Debug script to check if refresh_graph_from_registry works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology, refresh_graph_from_registry

def main():
    # Use the reset topology (which should NOT have TRIAL nodes)
    reset_topo = Path("snapshots/evolution/trial_activation_test/topology_reset.json")
    
    # Use the cycle_0003 snapshot (which SHOULD have TRIAL nodes)
    cycle_topo = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
    
    print(f"1. Loading reset topology: {reset_topo}")
    base_registry = TopologyRegistry(reset_topo)
    print(f"   Nodes in base registry: {len(list(base_registry.get_all_node_ids()))}")
    
    # Check if base registry has TRIAL nodes
    base_nodes = list(base_registry.get_all_node_ids())
    base_trial = [n for n in base_nodes if n.startswith("TRIAL")]
    print(f"   TRIAL nodes in base registry: {len(base_trial)}")
    
    # Build graph from base (no TRIAL nodes expected)
    print("\n2. Building graph from base registry...")
    graph = build_graph_from_topology(reset_topo, base_registry)
    print(f"   Nodes in graph: {len(graph.nodes)}")
    graph_trial = [n for n in graph.nodes.keys() if n.startswith("TRIAL")]
    print(f"   TRIAL nodes in graph: {len(graph_trial)}")
    
    # Now load cycle_0003 registry (has TRIAL nodes)
    print(f"\n3. Loading cycle topology: {cycle_topo}")
    cycle_registry = TopologyRegistry(cycle_topo)
    cycle_nodes = list(cycle_registry.get_all_node_ids())
    cycle_trial = [n for n in cycle_nodes if n.startswith("TRIAL")]
    print(f"   Nodes in cycle registry: {len(cycle_nodes)}")
    print(f"   TRIAL nodes in cycle registry: {len(cycle_trial)}")
    
    # Try refreshing graph from cycle registry
    print("\n4. Refreshing graph from cycle registry...")
    changes = refresh_graph_from_registry(graph, cycle_registry)
    print(f"   Changes: {len(changes)}")
    trial_changes = [k for k in changes.keys() if "TRIAL" in k]
    print(f"   TRIAL changes: {len(trial_changes)}")
    if trial_changes[:5]:
        print(f"   Sample: {trial_changes[:5]}")
    
    # Check final graph
    print(f"\n5. Final graph state:")
    print(f"   Nodes in graph: {len(graph.nodes)}")
    final_trial = [n for n in graph.nodes.keys() if n.startswith("TRIAL")]
    print(f"   TRIAL nodes in graph: {len(final_trial)}")
    
    # Check children of kpk_detect after refresh
    detect_children = graph.children("kpk_detect")
    detect_trial = [c for c in detect_children if c.startswith("TRIAL")]
    print(f"   kpk_detect children: {len(detect_children)}")
    print(f"   TRIAL children of kpk_detect: {len(detect_trial)}")

if __name__ == "__main__":
    main()

