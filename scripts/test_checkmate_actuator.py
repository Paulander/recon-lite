"""Test actuator that ONLY rewards checkmate (is_checkmate: 0→1)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import numpy as np
from recon_lite_chess.baseline_teacher import KRKTeacher

teacher = KRKTeacher()

# Verify feature dimension now includes is_checkmate
print(f"Feature dimension: {teacher.feature_dim}")
assert teacher.feature_dim == 14, f"Expected 14 features, got {teacher.feature_dim}"

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

def select_move_by_checkmate_delta(board: chess.Board) -> chess.Move:
    """Select move that maximizes is_checkmate delta (feature 13)"""
    v0 = teacher.features(board)
    
    best_move = None
    best_delta = -999
    
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        v1 = teacher.features(b2)
        
        delta_checkmate = v1[13] - v0[13]  # is_checkmate delta
        
        if delta_checkmate > best_delta:
            best_delta = delta_checkmate
            best_move = move
    
    return best_move, best_delta

print("=" * 60)
print("Testing Actuator: Maximize Δis_checkmate")
print("(ONLY checkmate gets delta +1, all other moves get 0)")
print("=" * 60)

correct = 0
total = len(VERIFIED_POSITIONS)

for i, fen in enumerate(VERIFIED_POSITIONS, 1):
    board = chess.Board(fen)
    
    # Find mate move for verification
    mate_move = None
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        if b2.is_checkmate():
            mate_move = move
            break
    
    # Select using checkmate delta
    selected, delta = select_move_by_checkmate_delta(board)
    
    is_correct = (selected == mate_move)
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    print(f"{i:2d}. {status} Mate: {mate_move.uci()}, Selected: {selected.uci()}, Delta: {delta:+.0f}")

print(f"\n{'='*60}")
print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
print("=" * 60)

if correct == total:
    print("✓ SUCCESS: Checkmate-delta actuator finds ALL mates!")
else:
    print("✗ Some positions still failing")
