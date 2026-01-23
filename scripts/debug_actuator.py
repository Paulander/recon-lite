"""Simple debug: compare features before/after for mate move vs others"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import numpy as np
from recon_lite_chess.baseline_teacher import generate_krk_mate_in_1_position, KRKTeacher

teacher = KRKTeacher()

print("=" * 60)
print("Feature Analysis: Mate Move vs Others")
print("=" * 60)

# Test 5 positions
for pos_idx in range(5):
    board = generate_krk_mate_in_1_position()
    print(f"\nPosition {pos_idx+1}: {board.fen()}")
    
    v0 = teacher.features(board)
    print(f"Features v0: {v0}")
    print(f"  is_check:        {v0[11]:.0f}")
    print(f"  can_deliver_mate:{v0[12]:.0f}")
    
    # Find mate move
    mate_move = None
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        if b2.is_checkmate():
            mate_move = move
            break
    
    if mate_move is None:
        print("  No mate move found!")
        continue
    
    print(f"\nMate move: {mate_move.uci()}")
    
    # Analyze mate move
    b_mate = board.copy()
    b_mate.push(mate_move)
    v1_mate = teacher.features(b_mate)
    delta_mate = v1_mate - v0
    
    print(f"  v1 (after mate): {v1_mate}")
    print(f"  delta:           {delta_mate}")
    print(f"  Key deltas:")
    print(f"    box_area:    {delta_mate[0]:+.3f}")
    print(f"    king_dist:   {delta_mate[1]:+.3f}")
    print(f"    is_check:    {delta_mate[11]:+.0f}")
    print(f"    can_mate:    {delta_mate[12]:+.0f}")
    
    # Analyze one non-mate move
    for move in board.legal_moves:
        if move != mate_move:
            b_other = board.copy()
            b_other.push(move)
            v1_other = teacher.features(b_other)
            delta_other = v1_other - v0
            
            print(f"\nOther move: {move.uci()}")
            print(f"  delta:           {delta_other}")
            print(f"  Key deltas:")
            print(f"    box_area:    {delta_other[0]:+.3f}")
            print(f"    king_dist:   {delta_other[1]:+.3f}")
            print(f"    is_check:    {delta_other[11]:+.0f}")
            print(f"    can_mate:    {delta_other[12]:+.0f}")
            break
    
    print("-" * 60)
