"""
Test full graph execution with simplified checkmate actuator.

Uses verified mate-in-1 positions from user.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite.graph import Graph
from recon_lite_chess.krk_checkmate_actuator import create_checkmate_actuator, _teacher

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
print("Full Graph Test: Checkmate Actuator")
print("=" * 70)

# Create simple graph with just the checkmate actuator
graph = Graph()
actuator = create_checkmate_actuator("krk_checkmate")
graph.add_node(actuator)

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
    
    # Execute actuator
    env = {"board": board}
    success, _ = actuator.predicate(actuator, graph, env)
    
    selected = env.get("suggested_move")
    confidence = env.get("move_confidence", 0)
    
    is_correct = (selected == mate_move)
    if is_correct:
        correct += 1
    
    status = "✓" if is_correct else "✗"
    print(f"{i:2d}. {status} Selected: {selected.uci()}, Mate: {mate_move.uci()}, Conf: {confidence:+.0f}")

print(f"\n{'='*70}")
print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
print("=" * 70)

if correct == total:
    print("\n✓ SUCCESS: Phase 2 complete - full graph execution works!")
    print("  - is_checkmate feature correctly identifies goal state")
    print("  - Actuator finds 100% of mate moves")
else:
    print("\n✗ FAILED: Some positions incorrect")
