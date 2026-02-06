#!/usr/bin/env python3
"""Debug script to check if TRIAL nodes are registered as children of kpk_detect."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology

def main():
    topo_path = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
    
    if not topo_path.exists():
        print(f"Topology not found: {topo_path}")
        return
    
    print(f"Loading topology: {topo_path}")
    registry = TopologyRegistry(topo_path)
    graph = build_graph_from_topology(topo_path, registry)
    
    print(f"\nGraph has {len(graph.nodes)} nodes")
    print(f"Graph has {len(graph.edges)} edges")
    
    # Check TRIAL nodes exist
    trial_nodes = [nid for nid in graph.nodes.keys() if nid.startswith("TRIAL")]
    print(f"\nTRIAL nodes in graph: {len(trial_nodes)}")
    if trial_nodes[:5]:
        print(f"  Sample: {trial_nodes[:5]}")
    
    # Check children of kpk_detect
    print("\n=== Children of kpk_detect ===")
    detect_children = graph.children("kpk_detect")
    print(f"kpk_detect has {len(detect_children)} children")
    
    trial_children = [c for c in detect_children if c.startswith("TRIAL")]
    print(f"  TRIAL children: {len(trial_children)}")
    if trial_children[:5]:
        print(f"  Sample: {trial_children[:5]}")
    
    non_trial_children = [c for c in detect_children if not c.startswith("TRIAL")]
    print(f"  Non-TRIAL children: {len(non_trial_children)}")
    if non_trial_children:
        print(f"  Sample: {non_trial_children[:5]}")
    
    # Check edges from kpk_detect
    print("\n=== Edges from kpk_detect ===")
    from recon_lite.graph import LinkType
    detect_out = graph.out.get(("kpk_detect", LinkType.SUB), [])
    print(f"out[(kpk_detect, SUB)]: {len(detect_out)} entries")
    if detect_out[:5]:
        print(f"  Sample: {detect_out[:5]}")
    
    # Check if TRIAL nodes have correct parent
    print("\n=== TRIAL node parents ===")
    for tn in trial_nodes[:3]:
        parent = graph.parent.get(tn)
        fanin = graph.parents_fanin.get(tn, [])
        print(f"  {tn}:")
        print(f"    parent: {parent}")
        print(f"    fanin: {fanin}")

if __name__ == "__main__":
    main()

