"""Test with verified mate-in-1 positions provided by user"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import numpy as np
from recon_lite_chess.baseline_teacher import KRKTeacher

teacher = KRKTeacher()

# User-provided verified mate-in-1 positions
VERIFIED_POSITIONS = [
    "4k3/7R/4K3/8/8/8/8/8 w - - 0 1",
    "3k4/7R/3K4/8/8/8/8/8 w - - 0 1",
    "3k4/6R1/3K4/8/8/8/8/8 w - - 0 1",
    "8/8/8/k1K5/8/8/1R6/8 w - - 0 1",
    "8/8/8/k1K5/8/8/8/1R6 w - - 0 1",
    "8/8/8/8/8/6K1/R7/6k1 w - - 0 1",
    "8/8/8/8/8/4K3/7R/4k3 w - - 0 1",
    "8/8/8/5K1k/8/8/6R1/8 w - - 0 1",
    "8/8/5K1k/8/8/8/6R1/8 w - - 0 1",
    "5K1k/8/8/8/8/8/6R1/8 w - - 0 1",
    "k7/3R4/1K6/8/8/8/8/8 w - - 0 1",
]

print("=" * 70)
print("Testing with User-Verified Mate-in-1 Positions")
print("=" * 70)

for i, fen in enumerate(VERIFIED_POSITIONS, 1):
    board = chess.Board(fen)
    v0 = teacher.features(board)
    
    print(f"\nPosition {i}: {fen}")
    print(f"  is_check (v0[11]):        {v0[11]:.0f}")
    print(f"  can_deliver_mate (v0[12]): {v0[12]:.0f}")
    
    # Find mate move
    mate_move = None
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        if b2.is_checkmate():
            mate_move = move
            break
    
    if mate_move is None:
        print("  *** ERROR: No mate move found! ***")
        continue
    
    print(f"  Mate move: {mate_move.uci()}")
    
    # Analyze features for mate move
    b_mate = board.copy()
    b_mate.push(mate_move)
    v1_mate = teacher.features(b_mate)
    
    print(f"  v1 is_check:        {v1_mate[11]:.0f}")
    print(f"  v1 can_deliver_mate: {v1_mate[12]:.0f}")
    
    delta = v1_mate - v0
    print(f"  Δ is_check:        {delta[11]:+.0f}")
    print(f"  Δ can_deliver_mate: {delta[12]:+.0f}")
    
    # Check which move would be selected by maximizing is_check delta
    best_move_by_check = None
    best_delta_check = -999
    
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        v1 = teacher.features(b2)
        delta_check = v1[11] - v0[11]
        
        if delta_check > best_delta_check:
            best_delta_check = delta_check
            best_move_by_check = move
    
    if best_move_by_check == mate_move:
        print(f"  ✓ Check-delta actuator would select mate move")
    else:
        print(f"  ✗ Check-delta would select: {best_move_by_check.uci()} (delta={best_delta_check})")
        # Analyze why
        for move in board.legal_moves:
            b2 = board.copy()
            b2.push(move)
            v1 = teacher.features(b2)
            is_mate = b2.is_checkmate()
            print(f"    {move.uci()}: is_check={v1[11]:.0f}, delta={v1[11]-v0[11]:+.0f}, mate={is_mate}")
