"""
Simple KRK Actuator that uses is_check feature for mate finding.

Since we know mate moves cause is_check: 0→1, we can create a simple
actuator that selects the move with highest Δis_check.

This bypasses the learned patterns to verify the infrastructure works.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
import numpy as np
from recon_lite_chess.baseline_teacher import generate_krk_mate_in_1_position, KRKTeacher

teacher = KRKTeacher()

def select_move_by_check_delta(board: chess.Board) -> chess.Move:
    """Select move that maximizes is_check delta (feature 11)"""
    v0 = teacher.features(board)
    
    best_move = None
    best_delta = -999
    
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        v1 = teacher.features(b2)
        
        delta_check = v1[11] - v0[11]  # is_check delta
        
        if delta_check > best_delta:
            best_delta = delta_check
            best_move = move
    
    return best_move

# Test
print("=" * 60)
print("Simple Check-Delta Actuator Test")
print("=" * 60)

# Generate unique positions by fixing random seed
import random
random.seed(42)

stats = {"total": 0, "mate": 0, "wrong": 0}

for i in range(100):
    board = generate_krk_mate_in_1_position()
    
    # Check if position has mate-in-1
    has_mate = False
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        if b2.is_checkmate():
            has_mate = True
            break
    
    if not has_mate:
        continue  # Skip positions without mate-in-1
    
    stats["total"] += 1
    
    selected = select_move_by_check_delta(board)
    if selected:
        b2 = board.copy()
        b2.push(selected)
        if b2.is_checkmate():
            stats["mate"] += 1
        else:
            stats["wrong"] += 1

print(f"\nResults (positions with mate-in-1 only):")
print(f"  Total:  {stats['total']}")
print(f"  Mates:  {stats['mate']} ({100*stats['mate']/stats['total']:.1f}%)")
print(f"  Wrong:  {stats['wrong']} ({100*stats['wrong']/stats['total']:.1f}%)")

if stats["mate"] / stats["total"] >= 0.9:
    print("\n✓ SUCCESS: Simple check-delta actuator finds >90% mates!")
    print("  This proves the infrastructure works.")
    print("  The issue was the placeholder learned patterns.")
else:
    print("\n✗ Even simple check-delta doesn't work. Need more investigation.")
