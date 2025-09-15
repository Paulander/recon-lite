"""
Chess move selection actuators for ReCoN KRK system.

These functions implement the "filter-first, then score" approach:
1. Hard filters: Eliminate unsafe/illegal moves
2. Soft scoring: Rank remaining moves by desirability
"""

import chess
from typing import List, Tuple, Optional
from .predicates import (
    is_stalemate, rook_safe_after, box_area, box_area_after,
    shrinks_or_preserves_box, our_king_progress, has_opposition_after,
    gives_safe_check
)


def choose_move_phase1(board: chess.Board) -> Optional[str]:
    """
    Choose best move for Phase 1: Drive enemy king toward edge.

    Focus: King safety, box shrinking, edge proximity.
    """
    return choose_move_with_filters(board, phase="phase1")


def choose_move_phase2(board: chess.Board) -> Optional[str]:
    """
    Choose best move for Phase 2: Shrink enemy king's box.

    Focus: Maximum box shrinking while maintaining safety.
    """
    return choose_move_with_filters(board, phase="phase2")


def choose_move_phase3(board: chess.Board) -> Optional[str]:
    """
    Choose best move for Phase 3: Take opposition.

    Focus: King positioning for opposition and mate setup.
    """
    return choose_move_with_filters(board, phase="phase3")


def choose_move_phase4(board: chess.Board) -> Optional[str]:
    """
    Choose best move for Phase 4: Deliver mate.

    Focus: Checkmate moves, safe checks.
    """
    return choose_move_with_filters(board, phase="phase4")


def choose_move_with_filters(board: chess.Board, phase: str = "general") -> Optional[str]:
    """
    Core move selection with filter-first, then score approach.

    Args:
        board: Current chess position
        phase: Phase context ("phase1", "phase2", etc.) for scoring weights

    Returns:
        Best move in UCI format, or None if no good moves
    """
    legal_moves = list(board.legal_moves)

    # PHASE 1: HARD FILTERS (must pass all)
    candidates = []
    base_area = box_area(board)

    for move in legal_moves:
        # Filter 1: No stalemate
        if is_stalemate_after(board, move):
            continue

        # Filter 2: Rook safety (critical!)
        if not rook_safe_after(board, move):
            continue

        # Filter 3: No box expansion (monotonic constraint)
        if not shrinks_or_preserves_box(board, move):
            continue

        # Passed all filters - add to candidates with area info
        area_next = box_area_after(board, move)
        candidates.append((move, area_next))

    if not candidates:
        return None

    # PHASE 2: SCORING (rank remaining moves)
    scored_moves = []

    for move, area_next in candidates:
        score = 0.0

        # Base box shrinking score (always positive factor)
        area_reduction = base_area - area_next
        score += area_reduction * 3.0

        # King progress score
        king_score = our_king_progress(board, move)
        score += king_score

        # Phase-specific bonuses
        if phase == "phase1":
            # Phase 1: Prioritize edge-driving
            score += edge_driving_bonus(board, move) * 2.0
        elif phase == "phase2":
            # Phase 2: Maximize box shrinking
            score += area_reduction * 5.0  # Extra weight for shrinking
        elif phase == "phase3":
            # Phase 3: Opposition and positioning
            if has_opposition_after(board, move):
                score += 1.0
            score += king_score * 1.5  # Extra king positioning weight
        elif phase == "phase4":
            # Phase 4: Checkmate focus
            if gives_safe_check(board, move):
                score += 2.0

        scored_moves.append((score, move))

    # Sort by score (highest first) and return best move
    scored_moves.sort(reverse=True, key=lambda x: x[0])

    if scored_moves:
        best_move = scored_moves[0][1]
        return best_move.uci()

    return None


def is_stalemate_after(board: chess.Board, move: chess.Move) -> bool:
    """Check if move leads to stalemate."""
    board_copy = board.copy()
    board_copy.push(move)
    return board_copy.is_stalemate()


def edge_driving_bonus(board: chess.Board, move: chess.Move) -> float:
    """
    Calculate bonus for moves that drive enemy king toward edge.

    Higher bonus for moves that reduce king's distance to edge.
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.0

    # Current edge distance
    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)
    current_dist = min(king_file, 7 - king_file, king_rank, 7 - king_rank)

    # Edge distance after move
    board_copy = board.copy()
    board_copy.push(move)
    new_enemy_king = board_copy.king(not board_copy.turn)

    if new_enemy_king is None:
        return 0.0

    new_file = chess.square_file(new_enemy_king)
    new_rank = chess.square_rank(new_enemy_king)
    new_dist = min(new_file, 7 - new_file, new_rank, 7 - new_rank)

    # Bonus for reducing edge distance
    distance_reduction = current_dist - new_dist
    return max(0, distance_reduction) * 0.5


# Legacy function for backward compatibility
def choose_move_phase(board: chess.Board) -> Optional[str]:
    """Legacy function - use phase-specific functions instead."""
    return choose_move_with_filters(board)
