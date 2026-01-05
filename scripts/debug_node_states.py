#!/usr/bin/env python3
"""Debug script to check node states during game execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite.engine import ReConEngine
from recon_lite.graph import NodeState

def main():
    topo_path = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
    
    if not topo_path.exists():
        print(f"Topology not found: {topo_path}")
        return
    
    print(f"Loading topology: {topo_path}")
    registry = TopologyRegistry(topo_path)
    graph = build_graph_from_topology(topo_path, registry)
    
    print(f"\nGraph has {len(graph.nodes)} nodes")
    
    # Get TRIAL nodes
    trial_nodes = [nid for nid in graph.nodes.keys() if nid.startswith("TRIAL")]
    
    print("\n=== Initial node states ===")
    print(f"kpk_root: {graph.nodes['kpk_root'].state}")
    print(f"kpk_detect: {graph.nodes['kpk_detect'].state}")
    for tn in trial_nodes[:3]:
        print(f"{tn}: {graph.nodes[tn].state}")
    
    # Create engine and initialize
    engine = ReConEngine(graph)
    
    # Request the root
    graph.nodes['kpk_root'].state = NodeState.REQUESTED
    
    print("\n=== After requesting kpk_root ===")
    print(f"kpk_root: {graph.nodes['kpk_root'].state}")
    
    # Run a step
    env = {"board": None}  # Minimal env
    engine.step(env)
    
    print("\n=== After first step ===")
    print(f"kpk_root: {graph.nodes['kpk_root'].state}")
    print(f"kpk_detect: {graph.nodes['kpk_detect'].state}")
    for tn in trial_nodes[:5]:
        print(f"{tn}: {graph.nodes[tn].state}")
    
    # Check which nodes are REQUESTED/ACTIVE/WAITING
    requested = [nid for nid, n in graph.nodes.items() if n.state == NodeState.REQUESTED]
    waiting = [nid for nid, n in graph.nodes.items() if n.state == NodeState.WAITING]
    active = [nid for nid, n in graph.nodes.items() if n.state == NodeState.ACTIVE]
    
    print(f"\nREQUESTED nodes: {requested}")
    print(f"WAITING nodes: {waiting}")
    print(f"ACTIVE nodes: {active}")
    
    # Check POR predecessors for TRIAL nodes
    print("\n=== POR predecessors for TRIAL nodes ===")
    for tn in trial_nodes[:3]:
        preds = graph.predecessors(tn)
        print(f"{tn}: {preds}")

if __name__ == "__main__":
    main()

