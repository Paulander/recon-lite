#!/usr/bin/env python3
"""Debug why TRIAL nodes aren't appearing in active_nodes during actual gameplay."""

import sys
import json
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite.engine import ReConEngine
from recon_lite.graph import Graph, NodeState

def main():
    # Load the topology WITH TRIAL nodes
    topo_path = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
    
    if not topo_path.exists():
        print(f"Topology not found: {topo_path}")
        return
    
    print(f"Loading topology: {topo_path}")
    registry = TopologyRegistry(topo_path)
    graph = build_graph_from_topology(topo_path, registry)
    
    # Verify TRIAL nodes
    trial_nodes = [nid for nid in graph.nodes.keys() if nid.startswith("TRIAL")]
    print(f"\nGraph has {len(trial_nodes)} TRIAL nodes")
    
    # Create engine
    engine = ReConEngine(graph)
    
    # Lock into KPK subgraph
    def kpk_sentinel(env):
        return True  # Always stay in subgraph
    
    engine.lock_subgraph("kpk_root", kpk_sentinel)
    
    # Set up a simple board
    board = chess.Board("8/8/8/4k3/8/4K3/4P3/8 w - - 0 1")
    
    print("\n=== Simulating game execution ===")
    
    for move_num in range(2):
        print(f"\n--- Move {move_num + 1} ---")
        
        # Reset node states (as done in evolution_driver.py line 559)
        print("1. Resetting all node states to INACTIVE...")
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
        
        # Check state after reset
        requested_after_reset = [nid for nid, n in graph.nodes.items() if n.state == NodeState.REQUESTED]
        print(f"   REQUESTED nodes after reset: {requested_after_reset}")
        
        # Set up environment
        env = {
            "board": board,
            "our_color": chess.WHITE,
            "move_count": move_num,
        }
        
        # Run engine step
        print("2. Running engine.step()...")
        engine.step(env)
        
        # Check state after step
        requested = [nid for nid, n in graph.nodes.items() if n.state == NodeState.REQUESTED]
        waiting = [nid for nid, n in graph.nodes.items() if n.state == NodeState.WAITING]
        active = [nid for nid, n in graph.nodes.items() if n.state == NodeState.ACTIVE]
        
        print(f"   REQUESTED: {len(requested)} nodes")
        print(f"   WAITING: {len(waiting)} nodes")
        print(f"   ACTIVE: {len(active)} nodes")
        
        # Specifically check TRIAL nodes
        trial_requested = [nid for nid in requested if nid.startswith("TRIAL")]
        print(f"   TRIAL nodes REQUESTED: {len(trial_requested)}")
        if trial_requested[:5]:
            print(f"     Sample: {trial_requested[:5]}")
        
        # What would be recorded in active_nodes
        active_nodes_list = [n.nid for n in graph.nodes.values() 
                            if n.state in (NodeState.ACTIVE, NodeState.WAITING, NodeState.REQUESTED)]
        trial_in_active = [n for n in active_nodes_list if n.startswith("TRIAL")]
        print(f"\n3. active_nodes would contain:")
        print(f"   Total: {len(active_nodes_list)}")
        print(f"   TRIAL nodes: {len(trial_in_active)}")
        if trial_in_active[:5]:
            print(f"     Sample: {trial_in_active[:5]}")
        
        # Make a random move to progress the game
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(legal_moves[0])
            print(f"\n4. Made move: {legal_moves[0]}")

if __name__ == "__main__":
    main()

