"""
KRK (King+Rook vs King) ReCoN Network Implementation

This module provides a complete, modular KRK checkmate network that can be used
across different demo types (evaluation, gameplay, testing).
"""

import chess
from typing import Dict, Any
from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite.graph import NodeState
from recon_lite_chess import (
    # Terminal evaluators
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_stalemate_detector, create_wait_for_board_change,
    create_cut_established_detector, create_rook_lost_detector,

    # Script phase nodes
    create_phase0_establish_cut, create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_krk_root,

    # Move generators (actuators)
    create_phase0_choose_moves, create_king_drive_moves, create_box_shrink_moves, create_opposition_moves,
    create_mate_moves, create_random_legal_moves, create_no_progress_watch,

    # New confinement-aware nodes
    create_confinement_evaluator, create_barrier_ready_evaluator,
    create_confinement_moves, create_barrier_placement_moves
)
from recon_lite_chess.actuators import choose_any_safe_move


def build_krk_network() -> Graph:
    """
    Build the complete KRK checkmate ReCoN network.

    Returns:
        Graph: The complete KRK ReCoN network ready for execution
    """
    g = Graph()

    # ===== TERMINAL NODES (Evaluators) =====
    # Phase 1: Drive king to edge
    g.add_node(create_king_edge_detector("king_at_edge"))
    # New confinement evaluators
    g.add_node(create_confinement_evaluator("king_confined", target_size=2))
    g.add_node(create_barrier_ready_evaluator("barrier_ready"))

    # Phase 2: Shrink the box
    g.add_node(create_box_shrink_evaluator("box_can_shrink"))

    # Phase 3: Take opposition
    g.add_node(create_opposition_evaluator("can_take_opposition"))

    # Phase 4: Deliver mate
    g.add_node(create_mate_deliver_evaluator("can_deliver_mate"))
    g.add_node(create_stalemate_detector("is_stalemate"))
    g.add_node(create_cut_established_detector("cut_established"))

    # ===== SCRIPT NODES (Phase Orchestrators) =====
    # Phase 0: Establish cut / rendezvous
    g.add_node(create_phase0_establish_cut("phase0_establish_cut"))
    # Phase 1: Drive to edge
    g.add_node(create_phase1_drive_to_edge("phase1_drive_to_edge"))

    # Phase 2: Shrink box
    g.add_node(create_phase2_shrink_box("phase2_shrink_box"))

    # Phase 3: Take opposition
    g.add_node(create_phase3_take_opposition("phase3_take_opposition"))

    # Phase 4: Deliver mate
    g.add_node(create_phase4_deliver_mate("phase4_deliver_mate"))

    # ===== ROOT NODE =====
    g.add_node(create_krk_root("krk_root"))

    # ===== MOVE GENERATOR NODES (Actuators) =====
    g.add_node(create_phase0_choose_moves("choose_phase0"))
    g.add_node(create_king_drive_moves("king_drive_moves"))
    # New confinement-aware move generators
    g.add_node(create_confinement_moves("confinement_moves"))
    g.add_node(create_barrier_placement_moves("barrier_placement_moves"))
    g.add_node(create_box_shrink_moves("box_shrink_moves"))
    g.add_node(create_opposition_moves("opposition_moves"))
    g.add_node(create_mate_moves("mate_moves"))
    g.add_node(create_random_legal_moves("random_legal_moves"))
    # Supervisor / Watchdog
    g.add_node(create_no_progress_watch("no_progress_watch"))

    # ===== PER-PHASE SCRIPT SUBNODES (for article-compliant sequences) =====
    # Phase 0: [CheckCut] -> [ChooseMove] -> [Wait]
    p0_check = Node(nid="p0_check", ntype=NodeType.SCRIPT)
    p0_move  = Node(nid="p0_move",  ntype=NodeType.SCRIPT)
    p0_wait  = Node(nid="p0_wait",  ntype=NodeType.SCRIPT)
    g.add_node(p0_check); g.add_node(p0_move); g.add_node(p0_wait)

    # Phase 1: [CheckEdge] -> [DriveMoves] -> [Wait]
    p1_check = Node(nid="p1_check", ntype=NodeType.SCRIPT)
    p1_move  = Node(nid="p1_move",  ntype=NodeType.SCRIPT)
    p1_wait  = Node(nid="p1_wait",  ntype=NodeType.SCRIPT)
    g.add_node(p1_check); g.add_node(p1_move); g.add_node(p1_wait)

    # Phase 2: [CheckShrink] -> [ShrinkMoves] -> [Wait]
    p2_check = Node(nid="p2_check", ntype=NodeType.SCRIPT)
    p2_move  = Node(nid="p2_move",  ntype=NodeType.SCRIPT)
    p2_wait  = Node(nid="p2_wait",  ntype=NodeType.SCRIPT)
    g.add_node(p2_check); g.add_node(p2_move); g.add_node(p2_wait)

    # Phase 3: [CheckOpposition] -> [OppositionMoves] -> [Wait]
    p3_check = Node(nid="p3_check", ntype=NodeType.SCRIPT)
    p3_move  = Node(nid="p3_move",  ntype=NodeType.SCRIPT)
    p3_wait  = Node(nid="p3_wait",  ntype=NodeType.SCRIPT)
    g.add_node(p3_check); g.add_node(p3_move); g.add_node(p3_wait)

    # Phase 4: [CheckMate] -> [MateMoves] -> [Wait]
    p4_check = Node(nid="p4_check", ntype=NodeType.SCRIPT)
    p4_move  = Node(nid="p4_move",  ntype=NodeType.SCRIPT)
    p4_wait  = Node(nid="p4_wait",  ntype=NodeType.SCRIPT)
    g.add_node(p4_check); g.add_node(p4_move); g.add_node(p4_wait)

    # Distinct wait terminals per phase (to satisfy single-parent SUB constraint)
    g.add_node(create_wait_for_board_change("wait_after_p0"))
    g.add_node(create_wait_for_board_change("wait_after_p1"))
    g.add_node(create_wait_for_board_change("wait_after_p2"))
    g.add_node(create_wait_for_board_change("wait_after_p3"))
    g.add_node(create_wait_for_board_change("wait_after_p4"))

    # ===== CONNECTIONS =====

    # Root connections (hierarchical - SUB)
    g.add_edge("krk_root", "phase0_establish_cut", LinkType.SUB)
    g.add_edge("krk_root", "phase1_drive_to_edge", LinkType.SUB)
    g.add_edge("krk_root", "phase2_shrink_box", LinkType.SUB)
    g.add_edge("krk_root", "phase3_take_opposition", LinkType.SUB)
    g.add_edge("krk_root", "phase4_deliver_mate", LinkType.SUB)
    # Attach supervisor under root without POR edges
    g.add_edge("krk_root", "no_progress_watch", LinkType.SUB)
    # Root-level sentinels (parallel, do not POR)
    g.add_edge("krk_root", "is_stalemate", LinkType.SUB)
    g.add_node(create_rook_lost_detector("rook_lost"))
    g.add_edge("krk_root", "rook_lost", LinkType.SUB)

    # Phase 0 internal sequence: CheckCut -> ChooseP0 -> Wait
    g.add_edge("phase0_establish_cut", "p0_check", LinkType.SUB)
    g.add_edge("phase0_establish_cut", "p0_move",  LinkType.SUB)
    g.add_edge("phase0_establish_cut", "p0_wait",  LinkType.SUB)
    g.add_edge("p0_check", "p0_move", LinkType.POR)
    g.add_edge("p0_move",  "p0_wait", LinkType.POR)
    g.add_edge("p0_check", "cut_established", LinkType.SUB)
    g.add_edge("p0_move",  "choose_phase0",   LinkType.SUB)
    g.add_edge("p0_wait",  "wait_after_p0",   LinkType.SUB)

    # Phase sequencing P0 -> P1 -> P2 -> P3 -> P4 (scripts only)
    g.add_edge("phase0_establish_cut", "phase1_drive_to_edge", LinkType.POR)
    g.add_edge("phase1_drive_to_edge", "phase2_shrink_box",   LinkType.POR)
    g.add_edge("phase2_shrink_box", "phase3_take_opposition", LinkType.POR)
    g.add_edge("phase3_take_opposition", "phase4_deliver_mate", LinkType.POR)

    # Phase 1 internal sequence - now supports multiple approaches
    g.add_edge("phase1_drive_to_edge", "p1_check", LinkType.SUB)
    g.add_edge("phase1_drive_to_edge", "p1_move",  LinkType.SUB)
    g.add_edge("phase1_drive_to_edge", "p1_wait",  LinkType.SUB)
    # Mark OR root for immediate phase completion if already satisfied
    g.nodes["p1_check"].meta["alt"] = True
    g.add_edge("p1_check", "p1_move", LinkType.POR)
    g.add_edge("p1_move",  "p1_wait", LinkType.POR)

    # Phase 1 check: Multiple completion criteria
    g.add_edge("p1_check", "king_at_edge",      LinkType.SUB)
    g.add_edge("p1_check", "king_confined",     LinkType.SUB)  # New: confinement achieved
    g.add_edge("p1_check", "barrier_ready",     LinkType.SUB)  # New: barrier in place

    # Phase 1 move: Multiple move generation strategies (competing alternatives)
    g.add_edge("p1_move",  "king_drive_moves",      LinkType.SUB)  # Traditional approach
    g.add_edge("p1_move",  "confinement_moves",     LinkType.SUB)  # New: confinement-focused
    g.add_edge("p1_move",  "barrier_placement_moves", LinkType.SUB)  # New: barrier placement

    g.add_edge("p1_wait",  "wait_after_p1",     LinkType.SUB)

    # Phase 2 internal sequence
    g.add_edge("phase2_shrink_box", "p2_check", LinkType.SUB)
    g.add_edge("phase2_shrink_box", "p2_move",  LinkType.SUB)
    g.add_edge("phase2_shrink_box", "p2_wait",  LinkType.SUB)
    g.nodes["p2_check"].meta["alt"] = True
    g.add_edge("p2_check", "p2_move", LinkType.POR)
    g.add_edge("p2_move",  "p2_wait", LinkType.POR)
    g.add_edge("p2_check", "box_can_shrink",   LinkType.SUB)
    g.add_edge("p2_move",  "box_shrink_moves", LinkType.SUB)
    g.add_edge("p2_wait",  "wait_after_p2",    LinkType.SUB)

    # Phase 3 internal sequence
    g.add_edge("phase3_take_opposition", "p3_check", LinkType.SUB)
    g.add_edge("phase3_take_opposition", "p3_move",  LinkType.SUB)
    g.add_edge("phase3_take_opposition", "p3_wait",  LinkType.SUB)
    g.nodes["p3_check"].meta["alt"] = True
    g.add_edge("p3_check", "p3_move", LinkType.POR)
    g.add_edge("p3_move",  "p3_wait", LinkType.POR)
    g.add_edge("p3_check", "can_take_opposition", LinkType.SUB)
    g.add_edge("p3_move",  "opposition_moves",   LinkType.SUB)
    g.add_edge("p3_wait",  "wait_after_p3",      LinkType.SUB)

    # Phase 4 internal sequence (do not POR-gate on opposition; early mates allowed)
    g.add_edge("phase4_deliver_mate", "p4_check", LinkType.SUB)
    g.add_edge("phase4_deliver_mate", "p4_move",  LinkType.SUB)
    g.add_edge("phase4_deliver_mate", "p4_wait",  LinkType.SUB)
    g.nodes["p4_check"].meta["alt"] = True
    g.add_edge("p4_check", "p4_move", LinkType.POR)
    g.add_edge("p4_move",  "p4_wait", LinkType.POR)
    g.add_edge("p4_check", "can_deliver_mate", LinkType.SUB)
    g.add_edge("p4_move",  "mate_moves",       LinkType.SUB)
    g.add_edge("p4_wait",  "wait_after_p4",    LinkType.SUB)

    # Root-level fallback removed to encourage strategic actions

    # Note: Stall prevention is handled by:
    # 1. Fallback mechanisms in each phase chooser (choose_any_safe_move)
    # 2. Watchdog timer in gameplay demo (50 ticks)
    # No separate last resort terminal needed

    # Optional: validate article compliance
    g.validate_article_compliance()
    return g


def create_random_krk_board(white_to_move: bool = True) -> str:
    """
    Create a random KRK position for testing.

    Args:
        white_to_move: Whether white should move first

    Returns:
        str: FEN string of the random KRK position
    """
    import random

    # Place white king randomly
    wk_squares = ['g7', 'h7', 'g8', 'h8']  # Corner area
    wk_square = random.choice(wk_squares)

    # Place white rook randomly (not on king square)
    all_squares = [f"{file}{rank}" for rank in '12345678' for file in 'abcdefgh']
    available_squares = [sq for sq in all_squares if sq != wk_square]
    wr_square = random.choice(available_squares)

    # Place black king randomly (not on white pieces)
    bk_squares = [sq for sq in all_squares if sq not in [wk_square, wr_square]]
    bk_square = random.choice(bk_squares)

    # Create FEN
    board_fen = ""
    for rank in '87654321':
        empty_count = 0
        for file in 'abcdefgh':
            square = f"{file}{rank}"
            if square == wk_square:
                if empty_count > 0:
                    board_fen += str(empty_count)
                    empty_count = 0
                board_fen += 'K'
            elif square == wr_square:
                if empty_count > 0:
                    board_fen += str(empty_count)
                    empty_count = 0
                board_fen += 'R'
            elif square == bk_square:
                if empty_count > 0:
                    board_fen += str(empty_count)
                    empty_count = 0
                board_fen += 'k'
            else:
                empty_count += 1
        if empty_count > 0:
            board_fen += str(empty_count)
        if rank != '1':
            board_fen += '/'

    turn = 'w' if white_to_move else 'b'
    fen = f"{board_fen} {turn} - - 0 1"

    return fen


def evaluate_krk_position(board: chess.Board, max_ticks: int = 100) -> Dict[str, Any]:
    """
    Evaluate a KRK position using the ReCoN network.

    Args:
        board: The chess board position to evaluate
        max_ticks: Maximum ticks to run the network

    Returns:
        Dict with evaluation results
    """
    from recon_lite import ReConEngine
    from recon_lite.logger import RunLogger

    g = build_krk_network()
    engine = ReConEngine(g)
    logger = RunLogger()

    # Set root as requested
    g.nodes["krk_root"].state = g.nodes["krk_root"].NodeState.REQUESTED

    env = {"board": board}
    ticks = 0

    while ticks < max_ticks:
        ticks += 1
        now_requested = engine.step(env)

        if ticks % 10 == 0:  # Log periodically
            logger.snapshot(
                engine=engine,
                note=f"Evaluation tick {ticks}",
                env={"fen": board.fen(), "evaluation_tick": ticks},
                thoughts=f"Evaluating KRK position (tick {ticks})",
                new_requests=list(now_requested.keys()) if now_requested else []
            )

        # Check if network has reached a conclusion
        if not now_requested:
            break

    return {
        "ticks": ticks,
        "final_fen": board.fen(),
        "network_state": {nid: n.state.name for nid, n in g.nodes.items()},
        "logs": logger.events
    }


def play_krk_game(max_plies: int = 50, initial_fen: str = None) -> Dict[str, Any]:
    """
    Play a complete KRK game with ReCoN vs random opponent.

    Args:
        max_plies: Maximum number of plies to play
        initial_fen: Starting FEN (random if None)

    Returns:
        Dict with game results
    """
    import random
    from recon_lite import ReConEngine
    from recon_lite.logger import RunLogger

    # Setup board
    if initial_fen:
        board = chess.Board(initial_fen)
    else:
        board = chess.Board(create_random_krk_board(white_to_move=True))

    logger = RunLogger()
    stalls = 0
    rook_lost = False
    ply = 0
    game_moves = []

    while not board.is_game_over() and ply < max_plies:
        ply += 1

        # ReCoN's turn (White)
        g = build_krk_network()
        engine = ReConEngine(g)
        g.nodes["krk_root"].state = NodeState.REQUESTED

        env = {"board": board}
        chosen_move = None
        ticks = 0
        max_ticks = 100

        # Run ReCoN evaluation
        while ticks < max_ticks and chosen_move is None:
            ticks += 1
            now_requested = engine.step(env)
            chosen_move = env.get("chosen_move")

        # Apply ReCoN's move
        if chosen_move and chosen_move in [mv.uci() for mv in board.legal_moves]:
            board.push_uci(chosen_move)
            game_moves.append(chosen_move)

            # Log the move
            logger.snapshot(
                engine=engine,
                note=f"ReCoN move {ply}: {chosen_move}",
                env={"fen": board.fen(), "ply": ply, "recons_move": chosen_move},
                thoughts=f"Chose {chosen_move}",
                new_requests=[]
            )
        else:
            stalls += 1
            break

        if board.is_game_over():
            break

        # Random opponent's turn (Black)
        if board.legal_moves:
            opp_move = random.choice(list(board.legal_moves)).uci()
            board.push_uci(opp_move)
            game_moves.append(opp_move)

            logger.snapshot(
                engine=None,
                note=f"Opponent move {ply}: {opp_move}",
                env={"fen": board.fen(), "ply": ply, "opponents_move": opp_move},
                thoughts="Random defense",
                new_requests=[]
            )

        # Check for rook loss
        if not any(p.piece_type == chess.ROOK and p.color == chess.WHITE
                  for p in board.piece_map().values()):
            rook_lost = True

    # Determine outcome
    outcome = {
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate(),
        "insufficient_material": board.is_insufficient_material(),
        "fifty_moves": board.is_fifty_moves(),
        "threefold_repetition": board.is_repetition(),
    }

    return {
        "initial_fen": board.fen().split(' ')[0] + " w - - 0 1",  # Simplified
        "final_fen": board.fen(),
        "moves": game_moves,
        "plies": ply,
        "stalls": stalls,
        "rook_lost": rook_lost,
        "outcome": outcome,
        "logs": logger.events
    }
