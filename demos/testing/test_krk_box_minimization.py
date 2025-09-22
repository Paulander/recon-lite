#!/usr/bin/env python3
"""
Unit test for KRK box minimization mechanism.

Tests that a minimal ReCoN (sub) graph can suggest moves that minimize
the confinement box for the enemy king in KRK positions.

This test creates 20 diverse KRK positions and verifies that the ReCoN
network can propose reasonable moves that work towards box minimization.
"""

import sys
from pathlib import Path
import chess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite import Graph, ReConEngine, LinkType, NodeState
from recon_lite_chess import (
    create_box_shrink_moves,
    create_krk_root
)
from recon_lite.graph import Node, NodeType
from recon_lite_chess.predicates import box_area, box_min_side, enemy_nearest_edge_info


def create_minimal_box_minimization_graph() -> Graph:
    """
    Create a minimal ReCoN graph for box minimization in KRK positions.

    Structure:
    ROOT (script)
    └── Move Selector (script)
        ├── Box Shrink Moves (terminal)
        ├── King Drive Moves (terminal)
        ├── Confinement Moves (terminal)
        └── Barrier Placement Moves (terminal)
    """
    g = Graph()

    # Create root script node
    root = Node(nid="box_min_root", ntype=NodeType.SCRIPT)

    # Create a script node to orchestrate move selection
    move_selector = Node(nid="move_selector", ntype=NodeType.SCRIPT)

    # Create the phase-2 box shrink move generator.
    box_shrink_moves = create_box_shrink_moves("box_shrink_moves")

    # Add nodes to graph
    for node in [root, move_selector, box_shrink_moves]:
        g.add_node(node)

    # Connect root to move selector
    g.add_edge("box_min_root", "move_selector", LinkType.SUB)

    # Connect move selector to move generators (SUB relationships for parallel execution)
    # No POR relationships - let them all execute in parallel
    g.add_edge("move_selector", "box_shrink_moves", LinkType.SUB)

    return g


def _make_position(enemy_square: chess.Square,
                   rook_square: chess.Square,
                   our_king_square: chess.Square = chess.F5) -> chess.Board:
    board = chess.Board(None)
    board.set_piece_at(enemy_square, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(our_king_square, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
    board.turn = chess.WHITE
    board.castling_rights = 0
    board.halfmove_clock = 0
    board.fullmove_number = 1
    if not board.is_valid():
        raise ValueError(f"Generated illegal KRK position: enemy={enemy_square}, our_king={our_king_square}, rook={rook_square}")
    return board


def generate_test_positions() -> list[chess.Board]:
    """
    Generate curated KRK positions where safe box-shrinking moves exist.

    The dataset mixes:
      - Enemy kings already on the H-file edge with different rook/king supports
      - Enemy kings one file inside (G-file) where our king guards the target fence
      - Mirrored A-file cases to verify coordinate symmetry
    Each position is vetted to ensure the phase-2 move generator can reduce the
    confinement box against best defense (no optimistic opponent).
    """
    combos = [
        # Right edge – rook needs to slide to the g-file to fence
        (chess.H8, chess.A2, chess.C4),
        (chess.H7, chess.A2, chess.D4),
        (chess.H6, chess.B2, chess.E4),
        (chess.H5, chess.C2, chess.F4),
        (chess.H4, chess.B2, chess.E3),
        (chess.H3, chess.A2, chess.F3),
        (chess.H8, chess.A3, chess.E4),
        (chess.H7, chess.B3, chess.D4),
        (chess.H6, chess.A3, chess.F4),
        (chess.H5, chess.B3, chess.E4),
        (chess.H4, chess.A3, chess.F4),
        (chess.H6, chess.C2, chess.D4),

        # Near edge – our king sits on g3 so Rg2 is defended
        (chess.G5, chess.C2, chess.G3),
        (chess.G5, chess.D2, chess.G3),
        (chess.G5, chess.E2, chess.G3),

        # Left edge mirror cases to confirm coordinate handling
        (chess.A8, chess.H2, chess.F4),
        (chess.A7, chess.G2, chess.E4),
        (chess.A6, chess.F2, chess.D4),
        (chess.A5, chess.E2, chess.E3),
        (chess.A4, chess.F2, chess.E3),
    ]

    positions: list[chess.Board] = []
    for enemy_sq, rook_sq, king_sq in combos:
        positions.append(_make_position(enemy_sq, rook_sq, king_sq))

    return positions


def test_box_minimization_mechanism():
    """Test that the ReCoN graph can suggest box-minimizing moves."""
    print("Testing KRK Box Minimization Mechanism")
    print("=" * 50)

    # Create the minimal graph
    print("Building minimal ReCoN graph...")
    graph = create_minimal_box_minimization_graph()
    engine = ReConEngine(graph)
    print(f"Graph created with {len(graph.nodes)} nodes")

    # Generate test positions
    positions = generate_test_positions()
    print(f"Generated {len(positions)} test positions")

    successful_tests = 0
    total_tests = len(positions)

    for i, board in enumerate(positions, 1):
        enemy_sq = board.king(chess.BLACK)
        file_char = chr(ord('A') + chess.square_file(enemy_sq))
        rank_num = chess.square_rank(enemy_sq) + 1
        print(f"\nTest Position {i} - Enemy king on {file_char}{rank_num}")
        print(board)

        # Calculate initial box metrics
        initial_area = box_area(board)
        initial_min_side = box_min_side(board)
        print(f"Box area: {initial_area}, min_side: {initial_min_side}")

        # Validate that geometric helpers see the enemy king on the right edge.
        edge_info = enemy_nearest_edge_info(board)
        enemy_square = enemy_sq
        ef = chess.square_file(enemy_square)
        er = chess.square_rank(enemy_square)
        if edge_info["ef"] != ef or edge_info["er"] != er:
            raise AssertionError(f"Edge info reported inconsistent king coordinates: {edge_info}")

        if ef == 7:  # enemy already on H file
            if edge_info["axis"] != "file" or edge_info["edge_index"] != 7:
                raise AssertionError(f"Expected file-edge target for h-file king, got {edge_info}")
            if edge_info["target_line"] != 6:
                raise AssertionError(f"Expected fence one file inside (6) for h-file king, got {edge_info}")
        elif ef == 6:  # g-file
            right_dist = 7 - ef
            rank_dist = min(er, 7 - er)
            expected_axis = "file" if right_dist <= rank_dist else "rank"
            if edge_info["axis"] != expected_axis:
                raise AssertionError(f"Unexpected axis for g-file king: expected {expected_axis}, got {edge_info}")
            if expected_axis == "file":
                if edge_info["edge_index"] != 7 or edge_info["target_line"] != 6:
                    raise AssertionError(f"Expected target fence on file 6 for g-file king, got {edge_info}")
            else:
                expected_rank_edge = 0 if er <= 3 else 7
                if edge_info["edge_index"] != expected_rank_edge:
                    raise AssertionError(f"Expected rank edge {expected_rank_edge} for g-file king, got {edge_info}")
        elif ef == 0:  # a-file mirror
            if edge_info["axis"] != "file" or edge_info["edge_index"] != 0:
                raise AssertionError(f"Expected file-edge target for a-file king, got {edge_info}")
            if edge_info["target_line"] != 1:
                raise AssertionError(f"Expected fence one file inside (1) for a-file king, got {edge_info}")
        else:
            raise AssertionError(f"Unexpected enemy file {ef} in test setup")

        # Reset graph state and cached metadata so each position is independent
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
            node.meta.pop("suggested_moves", None)
            node.meta.pop("phase", None)
            node.meta.pop("reason", None)
        graph.nodes["box_min_root"].state = NodeState.REQUESTED

        # Create environment with board
        env = {"board": board, "last_reason": None}

        # Run the engine to get move suggestions
        move_found = False
        try:
            # Run steps until a move is found or we give up
            for step in range(20):
                new_requests = engine.step(env)

                # Check if any move was chosen
                chosen_move = env.get("chosen_move")
                if chosen_move:
                    # Convert UCI to algebraic notation
                    try:
                        move_obj = chess.Move.from_uci(chosen_move)
                        algebraic_move = board.san(move_obj)
                        # Identify which node produced the move for diagnostics
                        producer = None
                        for nid, node in graph.nodes.items():
                            suggested = node.meta.get("suggested_moves")
                            if suggested and chosen_move in suggested:
                                producer = node.meta.get("phase", nid)
                                env["phase"] = producer
                                break
                        phase = env.get('phase', 'unknown')
                        print(f"🎯 Suggested move: {algebraic_move} (from {phase})")
                    except Exception as e:
                        print(f"✓ Move suggested: {chosen_move} (UCI)")
                        phase = env.get('phase', 'unknown')

                    print(f"   Reason: {env.get('last_reason', 'none')}")

                    # Verify the move is legal
                    if chosen_move in [m.uci() for m in board.legal_moves]:
                        print("✓ Move is legal")

                        # Apply move and check box metrics
                        board_copy = board.copy()
                        board_copy.push_uci(chosen_move)

                        new_area = box_area(board_copy)
                        new_min_side = box_min_side(board_copy)
                        print(f"   Result (immediate): Box {initial_area}→{new_area}, min_side {initial_min_side}→{new_min_side}")

                        # Ensure we are not giving an unnecessary check during confinement phase
                        if board.gives_check(move_obj):
                            print("✗ Move gives check prematurely; expected quiet confinement move")
                            move_found = True
                            break

                        # Evaluate worst-case enemy reply to confirm confinement actually tightens
                        worst_area = new_area
                        worst_min_side = new_min_side
                        if board_copy.legal_moves:
                            worst_area = 0
                            worst_min_side = 0
                            for reply in board_copy.legal_moves:
                                reply_board = board_copy.copy()
                                reply_board.push(reply)
                                area_after_reply = box_area(reply_board)
                                min_side_after_reply = box_min_side(reply_board)
                                worst_area = max(worst_area, area_after_reply)
                                worst_min_side = max(worst_min_side, min_side_after_reply)
                        print(f"   Result (worst-case reply): Box ≤{worst_area}, min_side ≤{worst_min_side}")

                        if worst_min_side > initial_min_side:
                            print("✗ Worst-case min_side regresses; confinement failed")
                            move_found = True
                            break
                        if initial_min_side > 1 and worst_min_side >= initial_min_side:
                            print("✗ No min_side improvement achieved")
                            move_found = True
                            break
                        if initial_area > 1 and worst_area >= initial_area:
                            print("✗ No area improvement against best defense")
                            move_found = True
                            break

                        successful_tests += 1
                        print("✓ Test passed")
                        move_found = True
                    else:
                        print("✗ Move is not legal")
                        move_found = True
                    break

                if not new_requests and step > 2:  # Give it a few steps to get started
                    break

            if not move_found:
                print("✗ No move suggested within 20 steps")

        except Exception as e:
            print(f"✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Test Results: {successful_tests}/{total_tests} positions produced valid confinement moves")

    return successful_tests == total_tests


def run_all_tests():
    """Run all box minimization tests."""
    print("KRK Box Minimization Unit Tests")
    print("=" * 50)

    try:
        success = test_box_minimization_mechanism()

        if success:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed")
            return False

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
