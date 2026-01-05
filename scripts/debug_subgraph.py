#!/usr/bin/env python3
"""Debug subgraph node membership."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite.engine import ReConEngine

def main():
    topo_path = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
    
    if not topo_path.exists():
        print(f"Topology not found: {topo_path}")
        return
    
    print(f"Loading topology: {topo_path}")
    registry = TopologyRegistry(topo_path)
    graph = build_graph_from_topology(topo_path, registry)
    
    # Create engine
    engine = ReConEngine(graph)
    
    # Check TRIAL node metadata
    trial_nodes = [nid for nid in graph.nodes.keys() if nid.startswith("TRIAL")]
    print(f"\nTRIAL nodes in graph: {len(trial_nodes)}")
    
    for tn in trial_nodes[:3]:
        node = graph.nodes[tn]
        sg = node.meta.get("subgraph", "NOT SET")
        print(f"  {tn}: subgraph={sg}")
    
    # Get subgraph nodes
    print("\n=== Subgraph nodes for kpk_root ===")
    subgraph_nodes = engine._get_subgraph_nodes("kpk_root")
    print(f"Total nodes in kpk subgraph: {len(subgraph_nodes)}")
    
    trial_in_subgraph = [n for n in subgraph_nodes if n.startswith("TRIAL")]
    print(f"TRIAL nodes in subgraph: {len(trial_in_subgraph)}")
    if trial_in_subgraph[:5]:
        print(f"  Sample: {trial_in_subgraph[:5]}")
    
    # Check what nodes are NOT in subgraph
    not_in_subgraph = [nid for nid in graph.nodes.keys() if nid not in subgraph_nodes]
    print(f"\nNodes NOT in subgraph: {len(not_in_subgraph)}")
    if not_in_subgraph[:5]:
        print(f"  Sample: {not_in_subgraph[:5]}")

if __name__ == "__main__":
    main()

