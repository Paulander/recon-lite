#!/usr/bin/env python3
"""Quick verification that Stage 0 positions are mate-in-1."""
import chess

# Stage 0 positions from curriculum (FIXED)
STAGE_0_FENS = [
    "k7/8/1K6/8/8/8/8/R7 w - - 0 1",  # Kc7 blocks escape, Ra8 or king move is mate
    "8/8/8/8/8/6K1/8/R6k w - - 0 1",  # Kh3#
    "8/8/8/8/5K1k/8/8/R7 w - - 0 1",  # Rh1#
    "7k/8/6K1/8/8/8/8/R7 w - - 0 1",  # FIXED: Rh1# (was R7/8/6Kk which was NOT mate-in-1)
    "4k3/8/4K3/8/8/8/8/R7 w - - 0 1", # Ra8#
]

print("Verifying Stage 0 positions are mate-in-1:")
print("=" * 60)

all_valid = True
for fen in STAGE_0_FENS:
    board = chess.Board(fen)
    
    # Find mate-in-1 move
    mate_move = None
    for move in board.legal_moves:
        b = board.copy()
        b.push(move)
        if b.is_checkmate():
            mate_move = move
            break
    
    if mate_move:
        print(f"✓ {fen}")
        print(f"  Mate: {mate_move.uci()}")
    else:
        print(f"✗ {fen}")
        print(f"  NO MATE-IN-1 FOUND!")
        all_valid = False
    print()

print("=" * 60)
print(f"Result: {'ALL VALID' if all_valid else 'SOME INVALID'}")
