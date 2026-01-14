#!/usr/bin/env python3
"""Minimal diagnostic script to trace KPK state propagation."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite import ReConEngine, NodeState
import chess

def main():
    print("Building unified graph...")
    g = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(g)

    # Create a simple KPK position (white K on e1, white P on e2, black K on e5)
    board = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
    print(f"Position: {board.fen()}")
    
    # Print kpk_root children and POR links
    print("\n=== KPK structure ===")
    print(f"kpk_root children: {g.children('kpk_root')}")
    print(f"kpk_execute predecessors: {g.predecessors('kpk_execute')}")
    print(f"kpk_detect children: {g.children('kpk_detect')}")

    # Reset all nodes
    for n in g.nodes.values():
        n.state = NodeState.INACTIVE

    # Request kpk_root directly
    if "kpk_root" in g.nodes:
        g.nodes["kpk_root"].state = NodeState.REQUESTED
        print("\nRequested: kpk_root")
    else:
        print("ERROR: kpk_root not found!")
        return

    env = {"board": board}

    # Key nodes to trace
    trace_nodes = [
        "kpk_root", "kpk_detect", "kpk_execute", "kpk_finish", "kpk_wait",
        "kpk_material_check", "kpk_push_window", "kpk_move_selector"
    ]

    print("\n=== Initial state ===")
    for nid in trace_nodes:
        if nid in g.nodes:
            node = g.nodes[nid]
            has_pred = "YES" if node.predicate else "NO"
            print(f"  {nid}: {node.state.name} [pred:{has_pred}]")

    # Run ticks
    for t in range(10):
        engine.step(env)
        print(f"\n=== After tick {t+1} ===")
        for nid in trace_nodes:
            if nid in g.nodes:
                node = g.nodes[nid]
                print(f"  {nid}: {node.state.name}")
        
        # Check POR gate for kpk_execute
        preds = g.predecessors("kpk_execute")
        por_ok = all(g.nodes[p].state in (NodeState.TRUE, NodeState.CONFIRMED) for p in preds) if preds else True
        print(f"  [kpk_execute POR gate: {por_ok}, preds: {[f'{p}={g.nodes[p].state.name}' for p in preds]}]")
        
        suggested = env.get("kpk", {}).get("policy", {}).get("suggested_move")
        if suggested:
            print(f"\n*** MOVE FOUND: {suggested} ***")
            break
    else:
        print("\nNo move found after 10 ticks!")
        
    # Check what's in env
    print("\n=== Environment state ===")
    kpk_data = env.get("kpk", {})
    print(f"  kpk.material: {kpk_data.get('material', {})}")
    print(f"  kpk.policy: {kpk_data.get('policy', {})}")

if __name__ == "__main__":
    main()
