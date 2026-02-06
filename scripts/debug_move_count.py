#!/usr/bin/env python3
"""Debug script to trace move counting in KRK curriculum."""
import sys
sys.path.insert(0, "src")

import chess
from recon_lite_chess.training.krk_curriculum import (
    KRK_STAGES,
    box_min_side,
)

def test_mate_in_2():
    """Test Mate_In_2 stage positions."""
    stage = KRK_STAGES[1]  # Mate_In_2
    print(f"Testing: {stage.name}")
    print(f"Positions: {len(stage.positions)}")
    
    for i, pos in enumerate(stage.positions):
        board = pos.to_board()
        print(f"\n=== Position {i}: {pos.description} ===")
        print(f"FEN: {pos.fen}")
        print(f"Optimal moves: {pos.optimal_moves}")
        print(board)
        
        # Play with simple heuristic
        move_count = 0
        initial_box = box_min_side(board)
        
        for _ in range(10):  # Max 10 iterations
            if board.is_game_over():
                print(f"Game over after {move_count} WHITE moves")
                if board.is_checkmate():
                    print(f"  -> CHECKMATE! {'Win' if board.turn == chess.BLACK else 'Loss'}")
                elif board.is_stalemate():
                    print(f"  -> Stalemate")
                break
            
            # White's move
            legal = list(board.legal_moves)
            best_move = None
            best_score = -1000
            
            for move in legal:
                score = 0
                board.push(move)
                if board.is_checkmate():
                    score = 1000
                elif board.is_check():
                    score = 50
                elif board.is_stalemate():
                    score = -500
                else:
                    new_box = box_min_side(board)
                    if new_box < initial_box:
                        score = 20
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            move = best_move if best_move else legal[0]
            print(f"  White move {move_count + 1}: {move} (score={best_score})")
            board.push(move)
            move_count += 1
            
            if board.is_game_over():
                print(f"Game over after {move_count} WHITE moves")
                if board.is_checkmate():
                    print(f"  -> CHECKMATE! {'Win' if board.turn == chess.BLACK else 'Loss'}")
                break
            
            # Black's move (escape attempt)
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
                black_move = opp_best if opp_best else legal[0]
                print(f"  Black reply: {black_move}")
                board.push(black_move)
        
        print(f"Final move_count: {move_count}")

if __name__ == "__main__":
    test_mate_in_2()
