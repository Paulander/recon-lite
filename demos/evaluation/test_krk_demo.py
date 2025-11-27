#!/usr/bin/env python3
"""
Quick sanity checks for KRK evaluators on representative positions.
"""

import chess
import pytest

from recon_lite_chess.krk_nodes import BoxShrinkEvaluator, KingAtEdgeDetector


@pytest.mark.parametrize(
    ("board_fen", "description", "expected_edge"),
    [
        ("4k3/6K1/8/8/8/8/R7/8 w - - 0 1", "King on e8 (8th rank edge)", True),
        ("4k3/6K1/8/8/8/3K4/R7/8 w - - 0 1", "King on e5 (center)", True),
        ("k7/6K1/8/8/8/8/R7/8 w - - 0 1", "King on a8 (corner)", True),
    ],
)
def test_position(board_fen: str, description: str, expected_edge: bool):
    board = chess.Board(board_fen)

    king_detector = KingAtEdgeDetector("edge")
    box_evaluator = BoxShrinkEvaluator("shrink")

    _, king_at_edge = king_detector._king_at_edge(None, {"board": board})
    _, can_shrink = box_evaluator._can_shrink_box(None, {"board": board})

    assert king_at_edge is expected_edge, f"Unexpected edge detection for {description}"
    assert isinstance(can_shrink, bool)

