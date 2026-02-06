#!/usr/bin/env python3
"""Trace Stage 0 mate-in-1 execution to find why mates aren't executed."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite import ReConEngine, NodeState

# Just test one Stage 0 position
TEST_FEN = "k7/8/1K6/8/8/8/8/R7 w - - 0 1"  # Ra8# or Kc7# is mate

def main():
    print(f"=== Stage 0 Mate-in-1 Trace (with lock_subgraph) ===")
    print(f"FEN: {TEST_FEN}")
    
    board = chess.Board(TEST_FEN)
    print(f"Board:\n{board}")
    
    # Verify mate exists
    mate_moves = []
    for move in board.legal_moves:
        b = board.copy()
        b.push(move)
        if b.is_checkmate():
            mate_moves.append(move.uci())
    print(f"\nActual mate-in-1 moves: {mate_moves}")
    
    # Build graph and engine
    graph = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(graph)
    
    # Set up like curriculum does - use lock_subgraph
    env = {"board": board}
    
    def krk_sentinel(env):
        b = env.get("board")
        if b is None or b.is_game_over():
            return False
        return True
    
    # Lock subgraph like curriculum does (min_internal_ticks=10)
    try:
        engine.lock_subgraph("krk_root", krk_sentinel, max_internal_ticks=30, min_internal_ticks=10)
    except ValueError as e:
        print(f"Lock failed: {e}")
    
    print(f"\n=== Running engine.step() with subgraph lock ===")
    engine.step(env)  # This should run internal ticks
    
    # Check suggested_move
    suggested = env.get("krk_root", {}).get("policy", {}).get("suggested_move")
    suggested_krk = env.get("krk", {}).get("policy", {}).get("suggested_move")
    chosen = env.get("chosen_move")
    
    print(f"After step:")
    print(f"  krk_root.policy.suggested_move = {suggested}")
    print(f"  krk.policy.suggested_move = {suggested_krk}")
    print(f"  chosen_move = {chosen}")
    
    if suggested:
        print(f"\n*** Move suggested: {suggested} ***")
        if suggested in mate_moves:
            print("*** SUCCESS: This is checkmate! ***")
        else:
            print(f"*** NOT checkmate (mates were: {mate_moves}) ***")
    else:
        print(f"\n*** NO MOVE SUGGESTED! ***")
    
    # Final env
    print(f"\n=== Final env (non-board keys) ===")
    for key in sorted(env.keys()):
        if key not in ("board", "__graph__"):
            print(f"  {key}: {env[key]}")

if __name__ == "__main__":
    main()

