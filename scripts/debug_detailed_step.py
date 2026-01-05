#!/usr/bin/env python3
"""Debug detailed step execution to trace child requests."""

import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite.engine import ReConEngine
from recon_lite.graph import Graph, NodeState, LinkType

def main():
    topo_path = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
    
    if not topo_path.exists():
        print(f"Topology not found: {topo_path}")
        return
    
    print(f"Loading topology: {topo_path}")
    registry = TopologyRegistry(topo_path)
    graph = build_graph_from_topology(topo_path, registry)
    engine = ReConEngine(graph)
    
    # Check graph structure
    print("\n=== Graph Structure ===")
    print(f"kpk_detect children: {graph.children('kpk_detect')}")
    trial_children = [c for c in graph.children('kpk_detect') if c.startswith("TRIAL")]
    print(f"TRIAL children of kpk_detect: {len(trial_children)}")
    
    # Verify edges
    print("\n=== Edges from kpk_detect ===")
    edges_from_detect = graph.out.get(("kpk_detect", LinkType.SUB), [])
    print(f"out[(kpk_detect, SUB)]: {edges_from_detect[:10]}")
    
    # Lock subgraph
    def sentinel(env):
        return True
    engine.lock_subgraph("kpk_root", sentinel)
    
    # Check subgraph nodes
    subgraph_nodes = engine._get_subgraph_nodes("kpk_root")
    print(f"\n=== Subgraph nodes: {len(subgraph_nodes)} ===")
    trial_in_sg = [n for n in subgraph_nodes if n.startswith("TRIAL")]
    print(f"TRIAL in subgraph: {len(trial_in_sg)}")
    
    # Reset all states
    for node in graph.nodes.values():
        node.state = NodeState.INACTIVE
    
    # Set up environment
    board = chess.Board("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1")
    env = {"board": board, "our_color": chess.WHITE}
    
    print("\n=== Before step ===")
    print(f"kpk_root state: {graph.nodes['kpk_root'].state}")
    print(f"kpk_detect state: {graph.nodes['kpk_detect'].state}")
    for tn in trial_children[:3]:
        print(f"{tn} state: {graph.nodes[tn].state}")
    
    # Manually trace what _step_subgraph does
    print("\n=== Manual trace of _step_subgraph ===")
    
    # 1. Reset subgraph nodes
    print("1. Resetting subgraph nodes to INACTIVE...")
    for nid in subgraph_nodes:
        graph.nodes[nid].state = NodeState.INACTIVE
    
    # 2. Request root
    print("2. Requesting kpk_root...")
    graph.nodes['kpk_root'].state = NodeState.REQUESTED
    graph.nodes['kpk_root'].tick_entered = 1
    
    print(f"   kpk_root state now: {graph.nodes['kpk_root'].state}")
    
    # 3. Process scripts (kpk_root is SCRIPT, should request its children)
    print("3. Processing script requests...")
    
    # Find kpk_root children
    root_children = graph.children('kpk_root')
    print(f"   kpk_root children: {root_children}")
    
    # Request kpk_detect (as _process_script_requests_subset would do)
    for child_id in root_children:
        if child_id in subgraph_nodes:
            child = graph.nodes[child_id]
            if child.state == NodeState.INACTIVE:
                child.state = NodeState.REQUESTED
                print(f"   Requested: {child_id}")
    
    # Now process kpk_detect
    print("\n4. Processing kpk_detect children...")
    detect_children = graph.children('kpk_detect')
    print(f"   kpk_detect children: {len(detect_children)}")
    print(f"   TRIAL children: {[c for c in detect_children if c.startswith('TRIAL')][:5]}")
    
    for child_id in detect_children[:5]:
        if child_id in subgraph_nodes:
            child = graph.nodes[child_id]
            print(f"   {child_id}: state={child.state}, in_subgraph={child_id in subgraph_nodes}")
            if child.state == NodeState.INACTIVE:
                # Check POR gate
                preds = graph.predecessors(child_id)
                print(f"      POR predecessors: {preds}")
                if not preds:
                    child.state = NodeState.REQUESTED
                    print(f"      → REQUESTED")
    
    # Check final states
    print("\n=== After manual processing ===")
    requested = [nid for nid in subgraph_nodes if graph.nodes[nid].state == NodeState.REQUESTED]
    waiting = [nid for nid in subgraph_nodes if graph.nodes[nid].state == NodeState.WAITING]
    
    print(f"REQUESTED nodes: {requested}")
    print(f"WAITING nodes: {waiting}")
    
    trial_req = [n for n in requested if n.startswith("TRIAL")]
    print(f"TRIAL REQUESTED: {trial_req}")

if __name__ == "__main__":
    main()

