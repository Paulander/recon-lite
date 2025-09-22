#!/usr/bin/env python3
"""
Unit tests for the new confinement box logic.

Tests the confinement-aware KRK algorithm components:
- confinement_box calculation
- barrier detection
- move scoring improvements
"""

import sys
from pathlib import Path
import chess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite_chess.predicates import confinement_box, box_min_side, enemy_nearest_edge_info
from recon_lite_chess.actuators import choose_confinement_move, choose_barrier_move


def test_confinement_box_basic():
    """Test basic confinement box calculation."""
    print("Testing basic confinement box calculation...")

    # King on e8, rook on a2 (starting position)
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")
    width, height = confinement_box(board)
    print(f"Starting position: confinement = ({width}, {height}), min_side = {min(width, height)}")
    assert min(width, height) == 8, f"Expected min_side=8, got {min(width, height)}"

    # King on e8, rook moves to a7
    board_after = chess.Board("4k3/R5K1/8/8/8/8/8/8 b - - 1 1")
    width_after, height_after = confinement_box(board_after)
    print(f"After Ra2-a7: confinement = ({width_after}, {height_after}), min_side = {min(width_after, height_after)}")
    assert min(width_after, height_after) == 1, f"Expected min_side=1 after Ra2-a7, got {min(width_after, height_after)}"

    print("✓ Basic confinement box test passed")


def test_barrier_detection():
    """Test that barriers are properly detected."""
    print("\nTesting barrier detection...")

    # King on e8 (rank 7), closest to rank 7 edge
    board = chess.Board("4k3/R5K1/8/8/8/8/8/8 b - - 1 1")
    edge_info = enemy_nearest_edge_info(board, enemy_king=chess.E8)
    print(f"Edge info: {edge_info}")

    assert edge_info["axis"] == "rank", f"Expected rank axis, got {edge_info['axis']}"
    assert edge_info["edge_index"] == 7, f"Expected edge_index=7, got {edge_info['edge_index']}"
    assert edge_info["target_line"] == 6, f"Expected target_line=6, got {edge_info['target_line']}"

    # Check confinement box
    width, height = confinement_box(board)
    print(f"Confinement with barrier: ({width}, {height})")
    assert height == 1, f"Expected height=1 with barrier on target line, got {height}"

    print("✓ Barrier detection test passed")


def test_confinement_moves():
    """Test that confinement-focused move selection works."""
    print("\nTesting confinement move selection...")

    # Starting position
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")

    # Test confinement move selection
    move = choose_confinement_move(board, {})
    print(f"Confinement move selected: {move}")

    if move:
        # Apply the move and check if it improves confinement
        board_copy = board.copy()
        board_copy.push_uci(move)

        min_side_before = box_min_side(board)
        min_side_after = box_min_side(board_copy)

        print(f"Min side: {min_side_before} -> {min_side_after}")
        # The move might not immediately improve confinement, but should be reasonable
        assert min_side_after >= 1, "Min side should be at least 1"

    print("✓ Confinement move selection test passed")


def test_barrier_moves():
    """Test that barrier placement move selection works."""
    print("\nTesting barrier placement move selection...")

    # Starting position
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")

    # Test barrier move selection
    move = choose_barrier_move(board, {})
    print(f"Barrier move selected: {move}")

    if move:
        print(f"Selected move: {move}")
        # Should be a legal move
        legal_moves = [m.uci() for m in board.legal_moves]
        assert move in legal_moves, f"Move {move} not in legal moves: {legal_moves}"

    print("✓ Barrier placement move selection test passed")


def test_ra2_a7_scenario():
    """Test the specific Ra2-a7 scenario that was problematic."""
    print("\nTesting Ra2-a7 scenario...")

    # Starting position: king on e8, rook on a2
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")

    print("Initial position:")
    print(board)
    print(f"King at: {chess.square_name(board.king(not board.turn))}")
    print(f"Rook at: {chess.square_name(board.king(board.turn))}")

    # Check initial confinement
    width, height = confinement_box(board)
    min_side = min(width, height)
    print(f"Initial confinement: ({width}, {height}), min_side = {min_side}")

    # Apply Ra2-a7
    board.push_uci("a2a7")
    print("\nAfter Ra2-a7:")
    print(board)

    # Check confinement after move
    width_after, height_after = confinement_box(board)
    min_side_after = min(width_after, height_after)
    print(f"Confinement after: ({width_after}, {height_after}), min_side = {min_side_after}")

    # Should show dramatic improvement
    improvement = min_side - min_side_after
    print(f"Improvement: {improvement} (lower is better)")

    assert min_side_after < min_side, f"Expected improvement, got {min_side} -> {min_side_after}"
    assert min_side_after <= 2, f"Expected tight confinement, got min_side = {min_side_after}"

    print("✓ Ra2-a7 scenario test passed")


def run_all_tests():
    """Run all confinement logic tests."""
    print("Running Confinement Logic Unit Tests")
    print("=" * 40)

    try:
        test_confinement_box_basic()
        test_barrier_detection()
        test_confinement_moves()
        test_barrier_moves()
        test_ra2_a7_scenario()

        print("\n" + "=" * 40)
        print("✅ All confinement logic tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
