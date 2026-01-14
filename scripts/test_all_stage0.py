#!/usr/bin/env python3
"""Test all 5 Stage 0 positions with subgraph lock."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite import ReConEngine, NodeState

# All Stage 0 positions (fixed)
STAGE_0_FENS = [
    "k7/8/1K6/8/8/8/8/R7 w - - 0 1",  # Kc7 blocks escape
    "8/8/8/8/8/6K1/8/R6k w - - 0 1",  # Kh3#
    "8/8/8/8/5K1k/8/8/R7 w - - 0 1",  # Rh1#
    "7k/8/6K1/8/8/8/8/R7 w - - 0 1",  # Rh1#
    "4k3/8/4K3/8/8/8/8/R7 w - - 0 1", # Ra8#
]

def test_position(fen, graph, engine):
    board = chess.Board(fen)
    
    # Find actual mate moves
    mate_moves = []
    for move in board.legal_moves:
        b = board.copy()
        b.push(move)
        if b.is_checkmate():
            mate_moves.append(move.uci())
    
    # Set up env
    env = {"board": board}
    
    # Lock subgraph
    def sentinel(env):
        b = env.get("board")
        return b and not b.is_game_over()
    
    # Reset states
    for n in graph.nodes.values():
        n.state = NodeState.INACTIVE
    
    try:
        engine.lock_subgraph("krk_root", sentinel, max_internal_ticks=30, min_internal_ticks=10)
    except ValueError:
        pass
    
    # Run step
    engine.step(env)
    
    # Get suggested move
    suggested = env.get("krk_root", {}).get("policy", {}).get("suggested_move")
    
    # Check result
    is_mate = suggested in mate_moves if suggested else False
    
    return {
        "fen": fen,
        "mate_moves": mate_moves,
        "suggested": suggested,
        "is_checkmate": is_mate,
    }

def main():
    print("=== Testing All Stage 0 Positions ===\n")
    
    graph = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(graph)
    
    results = []
    for i, fen in enumerate(STAGE_0_FENS):
        result = test_position(fen, graph, engine)
        results.append(result)
        
        status = "✓ CHECKMATE" if result["is_checkmate"] else "✗ FAILED"
        print(f"Position {i+1}: {status}")
        print(f"  FEN: {fen}")
        print(f"  Mate moves: {result['mate_moves']}")
        print(f"  Suggested: {result['suggested']}")
        print()
    
    # Summary
    success = sum(1 for r in results if r["is_checkmate"])
    print(f"=== Summary: {success}/{len(results)} positions found checkmate ===")

if __name__ == "__main__":
    main()
