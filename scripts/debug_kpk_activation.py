#!/usr/bin/env python3
"""Quick debug script to test KPK node activation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite import ReConEngine, NodeState

# Build graph
print("Building unified graph...")
graph = build_unified_graph(include_endgames=True, include_tactics=False)
engine = ReConEngine(graph)
print(f"Graph has {len(graph.nodes)} nodes")

# Create simple KPK position (pawn on 7th rank - should promote in 1 move!)
board = chess.Board()
board.clear()
board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
board.turn = chess.WHITE

print(f"\nBoard: {board.fen()}")
print(f"Legal moves: {[m.uci() for m in board.legal_moves]}")

env = {"board": board}

# Run multiple ticks
print("\n--- Running engine ticks ---")
for i in range(10):
    engine.step(env)
    active = [nid for nid, n in graph.nodes.items() 
              if n.state in (NodeState.ACTIVE, NodeState.TRUE, NodeState.CONFIRMED)]
    kpk_nodes = [n for n in active if "kpk" in n.lower()]
    print(f"Tick {i+1}: {len(active)} active, KPK: {len(kpk_nodes)}")
    if kpk_nodes:
        print(f"  -> {kpk_nodes[:5]}")

# Check for suggested move
suggested = env.get("kpk", {}).get("policy", {}).get("suggested_move")
print(f"\n=== Suggested move: {suggested} ===")

# Also check other possible paths
if "kpk" in env:
    print(f"env['kpk'] contents: {env['kpk']}")
else:
    print("No 'kpk' key in env!")

# Try to find the move selector node
for nid, node in graph.nodes.items():
    if "move_selector" in nid.lower():
        print(f"\n{nid}: state={node.state}, meta={node.meta}")
