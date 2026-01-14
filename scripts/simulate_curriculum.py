#!/usr/bin/env python3
"""Simulate curriculum's exact game loop for Stage 0 to trace win counting."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import random
from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite import ReConEngine, NodeState
from recon_lite_chess.predicates import box_min_side

# All Stage 0 positions
STAGE_0_FENS = [
    "k7/8/1K6/8/8/8/8/R7 w - - 0 1",
    "8/8/8/8/8/6K1/8/R6k w - - 0 1",
    "8/8/8/8/5K1k/8/8/R7 w - - 0 1",
    "7k/8/6K1/8/8/8/8/R7 w - - 0 1",
    "4k3/8/4K3/8/8/8/8/R7 w - - 0 1",
]

def play_game(fen, graph, engine, max_moves=5, verbose=True):
    """Play a game exactly like curriculum does."""
    board = chess.Board(fen)
    initial_fen = fen
    move_count = 0
    
    # Sentinel for subgraph lock
    def sentinel(env):
        b = env.get("board")
        if not b:
            return False
        pieces = list(b.piece_map().values())
        has_rook = any(p.piece_type == chess.ROOK for p in pieces)
        king_count = sum(1 for p in pieces if p.piece_type == chess.KING)
        return has_rook and king_count == 2 and len(pieces) == 3
    
    # Reset and lock
    for n in graph.nodes.values():
        n.state = NodeState.INACTIVE
    try:
        engine.lock_subgraph("krk_root", sentinel, max_internal_ticks=30, min_internal_ticks=10)
    except ValueError:
        pass
    
    while move_count < max_moves:
        if board.is_game_over():
            break
        
        # Engine ticks (like curriculum lines 364-402)
        env = {"board": board}
        engine.reset_states()
        
        suggested = None
        for tick in range(10):  # max_ticks_per_move
            engine.step(env)
            suggested = env.get("krk_root", {}).get("policy", {}).get("suggested_move")
            if suggested:
                break
        
        # Make move (curriculum lines 404-447)
        legal_ucis = [m.uci() for m in board.legal_moves]
        if suggested and suggested in legal_ucis:
            move = chess.Move.from_uci(suggested)
            if verbose:
                print(f"  Move {move_count+1}: {suggested} (ReCoN)")
        else:
            # Fallback - random
            move = random.choice(list(board.legal_moves))
            if verbose:
                print(f"  Move {move_count+1}: {move.uci()} (FALLBACK)")
        
        board.push(move)
        move_count += 1
        
        if verbose:
            print(f"    -> After move: is_checkmate={board.is_checkmate()}, is_game_over={board.is_game_over()}")
        
        # Check if we just won
        if board.is_checkmate():
            break
        
        # Opponent move (curriculum lines 518-532)
        if not board.is_game_over():
            legal = list(board.legal_moves)
            if legal:
                opp_best = None
                opp_best_score = -1000
                for m in legal[:10]:
                    board.push(m)
                    opp_score = box_min_side(board)
                    board.pop()
                    if opp_score > opp_best_score:
                        opp_best_score = opp_score
                        opp_best = m
                opp_move = opp_best if opp_best else random.choice(legal)
                board.push(opp_move)
                if verbose:
                    print(f"    Opponent: {opp_move.uci()}")
    
    # Determine result (curriculum lines 535-542)
    if board.is_checkmate():
        result = "win" if board.turn == chess.BLACK else "loss"
    elif board.is_stalemate():
        result = "stalemate"
    else:
        result = "draw"
    
    try:
        engine.unlock_subgraph("krk_root")
    except:
        pass
    
    return result, move_count

def main():
    print("=== Simulating Curriculum Game Loop for Stage 0 ===\n")
    
    graph = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(graph)
    
    wins = 0
    for i, fen in enumerate(STAGE_0_FENS):
        print(f"Position {i+1}: {fen}")
        result, moves = play_game(fen, graph, engine, max_moves=3, verbose=True)
        print(f"Result: {result} in {moves} moves\n")
        if result == "win":
            wins += 1
    
    print(f"=== Summary: {wins}/{len(STAGE_0_FENS)} wins ({100*wins/len(STAGE_0_FENS):.0f}%) ===")

if __name__ == "__main__":
    main()
