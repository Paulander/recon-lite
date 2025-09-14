#!/usr/bin/env python3
"""
Example demonstrating move tracking for efficient chess visualization.
This shows how to track moves instead of full FEN snapshots.
"""

import chess
from recon_lite import Graph, ReConEngine, LinkType, NodeState
from recon_lite.logger import RunLogger

def demo_move_tracking():
    """
    Example of how moves would be tracked when they're actually made during ReCoN execution.
    """
    print("Move Tracking Example")
    print("=" * 40)

    # Start with a position
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")
    initial_fen = board.fen()
    moves_made = []

    logger = RunLogger()

    print(f"Initial position: {initial_fen}")
    print("Making some example moves to demonstrate tracking...")

    # Example: Make some moves and track them (proper UCI format)
    example_moves = ["a1a2", "g7f6", "a2a6", "f6f5"]  # Simple rook maneuvers

    for i, move_uci in enumerate(example_moves):
        # Make the move on the board
        move = board.parse_uci(move_uci)
        board.push(move)
        moves_made.append(move_uci)

        print(f"Move {i+1}: {move_uci}")
        print(f"Position after move: {board.fen()}")

        # Log with move-based format (efficient!)
        logger.snapshot(
            engine=None,  # Would be real engine in actual implementation
            note=f"After move {move_uci}",
            env={
                "initial_fen": initial_fen,
                "moves": moves_made.copy(),
                "fen": board.fen()  # Keep for backward compatibility
            },
            thoughts=f"Executed move {move_uci} in KRK strategy",
            new_requests=[]
        )

    # Export the move-based data
    logger.to_json("demos/move_tracking_example.json")

    print("\nMove-based logging benefits:")
    print(f"- Initial FEN: {len(initial_fen)} characters")
    print(f"- Moves: {len(moves_made)} moves = {sum(len(m) for m in moves_made)} characters")
    print(f"- Total efficient storage: {len(initial_fen) + sum(len(m) for m in moves_made)} characters")
    print("- Each additional move only adds a few characters!")
    print("- Chess board can be reconstructed by replaying moves from initial position")

    # Show how to reconstruct position from moves
    print("\nReconstructing position from moves:")
    reconstructed_board = chess.Board(initial_fen)
    for move_uci in moves_made:
        move = reconstructed_board.parse_uci(move_uci)
        reconstructed_board.push(move)
        print(f"Applied {move_uci}: {reconstructed_board.fen()}")

    print(f"\nOriginal FEN matches reconstructed: {board.fen() == reconstructed_board.fen()}")

if __name__ == "__main__":
    demo_move_tracking()
