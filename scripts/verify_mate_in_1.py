
import chess
from recon_lite_chess.training.krk_curriculum import STAGE_0_MATE_IN_1

def verify_mate_in_1():
    print(f"Verifying {len(STAGE_0_MATE_IN_1.positions)} positions for Stage 0 (Mate-In-1)...")
    all_valid = True
    
    for i, pos in enumerate(STAGE_0_MATE_IN_1.positions):
        board = chess.Board(pos.fen)
        legal_moves = list(board.legal_moves)
        mate_found = False
        winning_move = None
        
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                mate_found = True
                winning_move = move.uci()
                board.pop()
                break
            board.pop()
            
        if mate_found:
            print(f"[PASS] Position {i+1}: {pos.description} -> Winner: {winning_move}")
        else:
            print(f"[FAIL] Position {i+1}: {pos.description} ({pos.fen}) -> NO MATE FOUND!")
            all_valid = False
            
    if all_valid:
        print("\nSUCCESS: All Stage 0 positions are valid Mate-in-1.")
    else:
        print("\nFAILURE: Some positions are invalid.")

if __name__ == "__main__":
    verify_mate_in_1()
