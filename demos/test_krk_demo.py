#!/usr/bin/env python3
"""
Quick test of KRK demo with different board positions
"""

import chess
from recon_lite import Graph, ReConEngine, LinkType, NodeState
from recon_lite_chess import (
    create_krk_root, create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator, create_stalemate_detector,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate
)

def test_position(board_fen, description):
    print(f"\n=== {description} ===")
    board = chess.Board(board_fen)
    print(f"FEN: {board_fen}")
    print(f"Board:\n{board}")

    # Test individual evaluators
    from recon_lite_chess.krk_nodes import KingAtEdgeDetector, BoxShrinkEvaluator

    king_detector = KingAtEdgeDetector("test")
    box_evaluator = BoxShrinkEvaluator("test2")

    _, king_at_edge = king_detector._king_at_edge(None, {"board": board})
    _, can_shrink = box_evaluator._can_shrink_box(None, {"board": board})

    print(f"King at edge: {king_at_edge}")
    print(f"Can shrink box: {can_shrink}")

# Test different positions
# Dumb, hardcoded positions for now. Add external board input later. (incluiding images of boards)
test_position("4k3/6K1/8/8/8/8/R7/8 w - - 0 1", "King on e8 (8th rank edge)")
test_position("4k3/6K1/8/8/8/3K4/R7/8 w - - 0 1", "King on e5 (center)")
test_position("k7/6K1/8/8/8/8/R7/8 w - - 0 1", "King on a8 (corner)")
