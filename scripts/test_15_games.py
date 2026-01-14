#!/usr/bin/env python3
"""Run 15 Stage 0 games to match curriculum's --quick mode and verify win rate."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import random
from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite_chess.training.krk_curriculum import KRK_STAGES
from recon_lite import ReConEngine, NodeState
from recon_lite_chess.predicates import box_min_side

def play_game(fen, graph, engine, max_moves=50, verbose=False):
    """Play a game with full curriculum logic."""
    board = chess.Board(fen)
    move_count = 0
    
    def sentinel(env):
        b = env.get("board")
        if not b:
            return False
        pieces = list(b.piece_map().values())
        has_rook = any(p.piece_type == chess.ROOK for p in pieces)
        king_count = sum(1 for p in pieces if p.piece_type == chess.KING)
        return has_rook and king_count == 2 and len(pieces) == 3
    
    try:
        engine.lock_subgraph("krk_root", sentinel, max_internal_ticks=30, min_internal_ticks=10)
    except ValueError:
        pass
    
    while move_count < max_moves:
        if board.is_game_over():
            break
        
        # Engine get move
        env = {"board": board}
        engine.reset_states()
        
        suggested = None
        for tick in range(10):
            engine.step(env)
            suggested = env.get("krk_root", {}).get("policy", {}).get("suggested_move")
            if suggested:
                break
        
        # Make move
        legal_ucis = [m.uci() for m in board.legal_moves]
        if suggested and suggested in legal_ucis:
            move = chess.Move.from_uci(suggested)
        else:
            # Random fallback (pure mode)
            legal = list(board.legal_moves)
            if not legal:
                break
            move = random.choice(legal)
        
        board.push(move)
        move_count += 1
        
        if board.is_checkmate():
            break
        
        # Opponent move
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
                board.push(opp_best if opp_best else random.choice(legal))
    
    # Determine result
    if board.is_checkmate():
        result = "win" if board.turn == chess.BLACK else "loss"
    elif board.is_stalemate():
        result = "stalemate"
    elif board.is_insufficient_material():
        result = "loss"
    else:
        result = "draw"
    
    try:
        engine.unlock_subgraph("krk_root")
    except:
        pass
    
    return result, move_count

def main():
    print("=== Running 15 Stage 0 Games (matching --quick) ===\n")
    
    graph = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(graph)
    
    stage = KRK_STAGES[0]  # Stage 0: Mate_In_1
    print(f"Stage: {stage.name}")
    print(f"Positions: {[p.fen for p in stage.positions]}\n")
    
    wins = 0
    losses = 0
    draws = 0
    
    for game in range(15):
        # Replicate curriculum's position selection
        fen = random.choice(stage.positions).fen
        result, moves = play_game(fen, graph, engine, max_moves=50)
        
        status = "✓" if result == "win" else "✗"
        print(f"Game {game+1}: {status} {result} in {moves} moves | FEN: {fen[:25]}...")
        
        if result == "win":
            wins += 1
        elif result in ("loss", "stalemate"):
            losses += 1
        else:
            draws += 1
    
    print(f"\n=== Summary ===")
    print(f"Wins: {wins} ({100*wins/15:.1f}%)")
    print(f"Losses/Stalemates: {losses}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    main()
