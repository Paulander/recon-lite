#!/usr/bin/env python3
"""Debug why pure ReCoN topology fails to produce moves."""
import sys
sys.path.insert(0, 'src')
import os
os.chdir('/home/paulander/git/recon-lite')

import chess
from recon_lite.graph import Graph
from recon_lite.engine import ReConEngine
from recon_lite_chess.graph.builder import build_graph_from_topology

# Load the pure ReCoN topology
print("=== Loading Pure ReCoN Topology ===")
graph = build_graph_from_topology("topologies/kpk_learned_topology.json")
print(f"Loaded {len(graph.nodes)} nodes:")
for nid, node in graph.nodes.items():
    print(f"  {nid}: {node.ntype.name}, predicate={node.predicate is not None}")

# Set up a simple KPK position (pawn on 7th, should promote)
print("\n=== Setting Up Position ===")
board = chess.Board()
board.clear()
board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
board.set_piece_at(chess.D6, chess.Piece(chess.KING, chess.WHITE))
board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
print(f"FEN: {board.fen()}")
print(board)
print(f"Legal moves: {list(board.legal_moves)}")

# Create engine and step through
print("\n=== Running Engine ===")
engine = ReConEngine(graph)
env = {"board": board, "our_color": chess.WHITE}

for tick in range(10):
    engine.step(env)
    
    # Check strategy outputs
    strategy_outputs = env.get("strategy_outputs", {})
    legs_proposals = env.get("legs", {})
    kpk_policy = env.get("kpk", {}).get("policy", {})
    
    print(f"\nTick {tick+1}:")
    print(f"  strategy_outputs: {strategy_outputs}")
    print(f"  legs proposals: {legs_proposals}")
    print(f"  kpk.policy: {kpk_policy}")
    
    # Check node states
    active_nodes = [n.nid for n in graph.nodes.values() if n.state.name in ("ACTIVE", "TRUE", "CONFIRMED")]
    print(f"  Active nodes: {active_nodes}")
    
    if "suggested_move" in kpk_policy:
        print(f"\n✅ Move found: {kpk_policy['suggested_move']}")
        break
else:
    print("\n❌ No move found after 10 ticks")

# Check if push_strategy was ever called
push_node = graph.nodes.get("push_strategy")
if push_node:
    print(f"\npush_strategy meta: {push_node.meta}")
