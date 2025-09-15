#!/usr/bin/env python3
"""
Interactive KRK Chess Demo - Per-Move "Dumb" Loop

This demo shows ReCoN playing chess interactively against a random opponent.
Uses the "dumb" approach: rebuilds KRK graph from scratch each move (stateless).

Strategy:
1. Build fresh KRK ReCoN graph each turn
2. Tick engine until actuator sets env["chosen_move"]
3. Apply move to board
4. Random opponent plays legal move
5. Repeat with fresh graph

This produces clean JSON logs per move and demonstrates basic ReCoN chess playing.
"""

import random
import chess
from recon_lite import Graph, ReConEngine, LinkType, NodeState
from recon_lite.logger import RunLogger
from recon_lite_chess import (
    # Evaluators (sensors)
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_stalemate_detector,

    # Move generators (actuators)
    create_king_drive_moves, create_random_legal_moves,

    # Script phases
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_krk_root
)

logger = RunLogger()


def build_krk_play_graph() -> Graph:
    """
    Build KRK graph with move generators for interactive play.

    Structure:
    ROOT: krk_root (KRK mate procedure)
    ├── PHASE1: phase1_drive_edge (drive black king to edge)
    │   ├── king_at_edge_detector (evaluator)
    │   └── king_drive_moves (actuator - generates moves)
    ├── PHASE2: phase2_shrink_box (shrink the box)
    │   ├── box_shrink_evaluator (evaluator)
    │   └── random_legal_moves (fallback actuator)
    ├── PHASE3: phase3_take_opposition (take opposition)
    │   ├── opposition_evaluator (evaluator)
    │   └── random_legal_moves (fallback actuator)
    └── PHASE4: phase4_deliver_mate (deliver mate)
        ├── mate_deliver_evaluator (evaluator)
        ├── stalemate_detector (evaluator)
        └── random_legal_moves (fallback actuator)
    """
    g = Graph()

    # Create all nodes
    root = create_krk_root("krk_root")

    # Phase 1: Drive to edge
    phase1 = create_phase1_drive_to_edge("phase1_drive_edge")
    king_detector = create_king_edge_detector("king_at_edge_detector")
    king_moves = create_king_drive_moves("king_drive_moves")

    # Phase 2: Shrink box
    phase2 = create_phase2_shrink_box("phase2_shrink_box")
    box_evaluator = create_box_shrink_evaluator("box_shrink_evaluator")
    box_moves = create_random_legal_moves("box_random_moves")

    # Phase 3: Take opposition
    phase3 = create_phase3_take_opposition("phase3_take_opposition")
    opp_evaluator = create_opposition_evaluator("opposition_evaluator")
    opp_moves = create_random_legal_moves("opp_random_moves")

    # Phase 4: Deliver mate
    phase4 = create_phase4_deliver_mate("phase4_deliver_mate")
    mate_evaluator = create_mate_deliver_evaluator("mate_deliver_evaluator")
    stalemate_detector = create_stalemate_detector("stalemate_detector")
    mate_moves = create_random_legal_moves("mate_random_moves")

    # Add all nodes to graph
    for node in [root, phase1, phase2, phase3, phase4,
                 king_detector, king_moves, box_evaluator, box_moves,
                 opp_evaluator, opp_moves, mate_evaluator, stalemate_detector, mate_moves]:
        g.add_node(node)

    # Create edges (SUB for hierarchy)
    g.add_edge("krk_root", "phase1_drive_edge", LinkType.SUB)
    g.add_edge("krk_root", "phase2_shrink_box", LinkType.SUB)
    g.add_edge("krk_root", "phase3_take_opposition", LinkType.SUB)
    g.add_edge("krk_root", "phase4_deliver_mate", LinkType.SUB)

    # Phase 1 connections
    g.add_edge("phase1_drive_edge", "king_at_edge_detector", LinkType.SUB)
    g.add_edge("phase1_drive_edge", "king_drive_moves", LinkType.SUB)

    # Phase 2 connections
    g.add_edge("phase2_shrink_box", "box_shrink_evaluator", LinkType.SUB)
    g.add_edge("phase2_shrink_box", "box_random_moves", LinkType.SUB)

    # Phase 3 connections
    g.add_edge("phase3_take_opposition", "opposition_evaluator", LinkType.SUB)
    g.add_edge("phase3_take_opposition", "opp_random_moves", LinkType.SUB)

    # Phase 4 connections
    g.add_edge("phase4_deliver_mate", "mate_deliver_evaluator", LinkType.SUB)
    g.add_edge("phase4_deliver_mate", "stalemate_detector", LinkType.SUB)
    g.add_edge("phase4_deliver_mate", "mate_random_moves", LinkType.SUB)

    return g


def make_random_opponent_move(board: chess.Board) -> str:
    """Make a random legal move for the opponent."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    random_move = random.choice(legal_moves)
    board.push(random_move)
    return random_move.uci()


def get_chosen_move(env: dict) -> str:
    """
    Get the chosen move from the environment.
    Actuators set env["chosen_move"] when they find good moves.
    """
    return env.get("chosen_move")


def play_interactive_krk():
    """
    Main interactive KRK game loop using "dumb" per-move approach.
    """
    print("🎮 Interactive KRK Chess Demo")
    print("=" * 50)
    print("ReCoN (White) vs Random Opponent (Black)")
    print("Using 'dumb' approach: fresh graph each move\n")

    # Initialize board
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")
    move_count = 0

    print(f"Initial position:\n{board}\n")

    while not board.is_game_over() and move_count < 50:  # Safety limit
        move_count += 1
        print(f"Move {move_count} - ReCoN's turn (White to move)")
        print(f"Current position:\n{board}")

        # Create environment for this move
        env = {
            "board": board,
            "chosen_move": None
        }

        # Build fresh KRK graph for this move
        print("Building fresh KRK graph...")
        graph = build_krk_play_graph()
        engine = ReConEngine(graph)

        # Set root to requested to start evaluation
        graph.nodes["krk_root"].state = NodeState.REQUESTED

        # Run evaluation until we get a move or hit safety limit
        max_ticks = 100
        ticks = 0

        print("Evaluating position...")

        while ticks < max_ticks and env.get("chosen_move") is None:
            ticks += 1
            now_requested = engine.step(env)

            if ticks % 10 == 0:
                print(f"  Tick {ticks}: evaluating...")

        # Apply the chosen move
        chosen_move = get_chosen_move(env)
        if chosen_move:
            print(f"✅ ReCoN chooses: {chosen_move}")
            try:
                board.push_uci(chosen_move)
                print(f"Position after ReCoN's move:\n{board}")

                # Log this move
                logger.snapshot(
                    engine=engine,
                    note=f"ReCoN move {move_count}: {chosen_move}",
                    env={
                        "initial_fen": "4k3/6K1/8/8/8/8/R7/8 w - - 0 1",
                        "moves": [],  # Would track full game history
                        "fen": board.fen(),
                        "move_number": move_count,
                        "recons_move": chosen_move
                    },
                    thoughts=f"Chose {chosen_move} on move {move_count}",
                    new_requests=[]
                )

            except Exception as e:
                print(f"❌ Error applying move {chosen_move}: {e}")
                break
        else:
            print("❌ No move chosen by ReCoN - ending game")
            break

        # Check if game ended
        if board.is_game_over():
            break

        # Opponent's turn
        print(f"\nMove {move_count} - Opponent's turn (Black to move)")
        opponent_move = make_random_opponent_move(board)

        if opponent_move:
            print(f"🎲 Opponent chooses: {opponent_move}")
            print(f"Position after opponent's move:\n{board}")

            # Log opponent move
            logger.snapshot(
                engine=engine,
                note=f"Opponent move {move_count}: {opponent_move}",
                env={
                    "initial_fen": "4k3/6K1/8/8/8/8/R7/8 w - - 0 1",
                    "moves": [],  # Would track full game history
                    "fen": board.fen(),
                    "move_number": move_count,
                    "opponents_move": opponent_move
                },
                thoughts=f"Opponent chose {opponent_move}",
                new_requests=[]
            )
        else:
            print("❌ No legal moves for opponent")
            break

        print("-" * 50)

    # Game end
    print("\n🏁 Game Over!")
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            print("🎉 ReCoN (White) wins by checkmate!")
        else:
            print("💔 ReCoN (White) loses by checkmate!")
    elif board.is_stalemate():
        print("🤝 Game ends in stalemate")
    elif board.is_insufficient_material():
        print("🤝 Game ends due to insufficient material")
    else:
        print("Game ended for other reasons")

    print(f"\nFinal position:\n{board}")
    print(f"Total moves: {move_count}")

    # Save game log
    logger.to_json("demos/krk_interactive_game.json")
    print("\n💾 Game log saved to: demos/krk_interactive_game.json")


if __name__ == "__main__":
    play_interactive_krk()
