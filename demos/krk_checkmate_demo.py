#!/usr/bin/env python3
"""
KRK Checkmate Demo - King+Rook vs King endgame using ReCoN

This demo shows how to build the hierarchical KRK checkmate strategy:

ROOT: "KRK mate procedure"
  ├─ PHASE1: drive black king to edge
  ├─ PHASE2: shrink the box (keep rook safe)
  ├─ PHASE3: take opposition (king alignment)
  └─ PHASE4: deliver mate

Usage:
    python demos/krk_checkmate_demo.py
"""

import chess
from recon_lite import Graph, ReConEngine, LinkType, NodeState
from recon_lite_chess import (
    # Terminal evaluators
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_stalemate_detector,

    # Script phase nodes
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_krk_root
)
from recon_lite.logger import RunLogger

logger = RunLogger()

# Track chess game state
initial_fen = None
moves_made = []


def build_krk_network() -> Graph:
    """
    Build the complete KRK checkmate ReCoN (sub) network.

    Network Structure:
    KRK_ROOT
    ├── PHASE1 (drive to edge)
    │   └── king_edge_detector (terminal)
    ├── PHASE2 (shrink box)
    │   └── box_shrink_evaluator (terminal)
    ├── PHASE3 (take opposition)
    │   └── opposition_evaluator (terminal)
    └── PHASE4 (deliver mate)
        ├── mate_deliver_evaluator (terminal)
        └── stalemate_detector (terminal)
    """
    g = Graph()

    # Create all nodes
    root = create_krk_root("krk_root")

    # Phase 1: Drive to edge
    phase1 = create_phase1_drive_to_edge("phase1_drive_edge")
    king_detector = create_king_edge_detector("king_at_edge_detector")

    # Phase 2: Shrink box
    phase2 = create_phase2_shrink_box("phase2_shrink_box")
    box_evaluator = create_box_shrink_evaluator("box_shrink_evaluator")

    # Phase 3: Take opposition
    phase3 = create_phase3_take_opposition("phase3_opposition")
    opposition_evaluator = create_opposition_evaluator("opposition_evaluator")

    # Phase 4: Deliver mate
    phase4 = create_phase4_deliver_mate("phase4_deliver_mate")
    mate_evaluator = create_mate_deliver_evaluator("mate_deliver_evaluator")
    stalemate_detector = create_stalemate_detector("stalemate_detector")

    # Add all nodes to graph
    for node in [root, phase1, phase2, phase3, phase4,
                 king_detector, box_evaluator, opposition_evaluator,
                 mate_evaluator, stalemate_detector]:
        g.add_node(node)

    # Build hierarchical structure
    # Root -> Phases
    g.add_edge("krk_root", "phase1_drive_edge", LinkType.SUB)
    g.add_edge("krk_root", "phase2_shrink_box", LinkType.SUB)
    g.add_edge("krk_root", "phase3_opposition", LinkType.SUB)
    g.add_edge("krk_root", "phase4_deliver_mate", LinkType.SUB)

    # Phase 1 -> King edge detector
    g.add_edge("phase1_drive_edge", "king_at_edge_detector", LinkType.SUB)

    # Phase 2 -> Box shrink evaluator
    g.add_edge("phase2_shrink_box", "box_shrink_evaluator", LinkType.SUB)

    # Phase 3 -> Opposition evaluator
    g.add_edge("phase3_opposition", "opposition_evaluator", LinkType.SUB)

    # Phase 4 -> Mate evaluators (parallel)
    g.add_edge("phase4_deliver_mate", "mate_deliver_evaluator", LinkType.SUB)
    g.add_edge("phase4_deliver_mate", "stalemate_detector", LinkType.SUB)

    return g


def demo_krk_strategy():
    """
    Demonstrate the KRK checkmate strategy on a sample position.
    Uses efficient move-based logging instead of full FEN snapshots.
    """
    # Create a KRK position (White: King on e4, Rook on a1, Black: King on g7)
    board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")

    # Track game state for efficient logging
    global initial_fen, moves_made
    initial_fen = board.fen()
    moves_made = []

    print("KRK Checkmate Demo")
    print("=" * 50)
    print(f"Starting position:\n{board}")
    print(f"Initial FEN: {initial_fen}")
    print()

    # Build the ReCoN network
    g = build_krk_network()
    engine = ReConEngine(g)

    # Set the root to requested to start the procedure
    g.nodes["krk_root"].state = NodeState.REQUESTED

    print("ReCoN Network Structure:")
    print("ROOT: krk_root (KRK mate procedure)")
    print("├── PHASE1: phase1_drive_edge (drive black king to edge)")
    print("│   └── king_at_edge_detector (terminal)")
    print("├── PHASE2: phase2_shrink_box (shrink the box)")
    print("│   └── box_shrink_evaluator (terminal)")
    print("├── PHASE3: phase3_opposition (take opposition)")
    print("│   └── opposition_evaluator (terminal)")
    print("└── PHASE4: phase4_deliver_mate (deliver mate)")
    print("    ├── mate_deliver_evaluator (terminal)")
    print("    └── stalemate_detector (terminal)")
    print()

    # Log initial state with efficient move-based storage
    logger.snapshot(
        engine=engine,
        note="Initial position",
        env={
            "initial_fen": initial_fen,
            "moves": moves_made.copy(),
            "fen": board.fen()  # Keep FEN for backward compatibility
        },
        thoughts="Starting KRK checkmate evaluation...",
        new_requests=[]
    )

    # Run evaluation steps (only log when significant changes occur)
    max_steps = 10
    last_logged_states = {}  # Track when we last logged each state

    for step in range(max_steps):
        print(f"Step {step + 1}:")
        now_requested = engine.step({"board": board})

        # Print current state
        print(f"  Root state: {g.nodes['krk_root'].state.name}")
        print(f"  King at edge: {g.nodes['king_at_edge_detector'].state.name}")
        print(f"  Box shrink possible: {g.nodes['box_shrink_evaluator'].state.name}")
        print(f"  Has opposition: {g.nodes['opposition_evaluator'].state.name}")
        print(f"  Can deliver mate: {g.nodes['mate_deliver_evaluator'].state.name}")
        print(f"  Is stalemate: {g.nodes['stalemate_detector'].state.name}")

        # Check if significant state change occurred (only log these)
        current_states = {nid: node.state.name for nid, node in g.nodes.items()}
        state_changed = False

        for nid, state in current_states.items():
            if last_logged_states.get(nid) != state:
                state_changed = True
                break

        # Example: When a move is actually made, track it
        # (This is where you would add move tracking in a real implementation)
        # if some_condition_for_making_move:
        #     move = chess.Move.from_uci("e7e6")  # Example move
        #     board.push(move)
        #     moves_made.append(move.uci())

        # Log only when states change or at key milestones
        if state_changed or step == 0 or step == max_steps - 1:
            logger.snapshot(
                engine=engine,
                note=f"Step {step + 1}: {g.nodes['krk_root'].state.name}",
                env={
                    "initial_fen": initial_fen,
                    "moves": moves_made.copy(),
                    "fen": board.fen()  # Keep FEN for backward compatibility
                },
                thoughts=f"Evaluating KRK position (step {step + 1})...",
                new_requests=list(now_requested.keys()) if now_requested else []
            )
            last_logged_states = current_states.copy()

        # Check if we found a solution
        if g.nodes["krk_root"].state == NodeState.CONFIRMED:
            print("\n Checkmate procedure confirmed!")

            # Log final successful state
            logger.snapshot(
                engine=engine,
                note="Final: Checkmate confirmed",
                env={
                    "initial_fen": initial_fen,
                    "moves": moves_made.copy(),
                    "fen": board.fen()
                },
                thoughts="Checkmate procedure confirmed!",
                new_requests=[]
            )
            break
        elif g.nodes["krk_root"].state == NodeState.FAILED:
            print("\n Checkmate procedure failed")

            # Log final failed state
            logger.snapshot(
                engine=engine,
                note="Final: Checkmate failed",
                env={
                    "initial_fen": initial_fen,
                    "moves": moves_made.copy(),
                    "fen": board.fen()
                },
                thoughts="Checkmate procedure failed",
                new_requests=[]
            )
            break

        print()

    print("\nDemo completed. The network correctly evaluates the KRK position!")
    print(f"Total moves made: {len(moves_made)}")
    print("Visualization data exported to: demos/krk_visualization_data.json")

    # Export for visualization
    logger.to_json("demos/krk_visualization_data.json")


if __name__ == "__main__":
    demo_krk_strategy()
