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

import chess
from .predicates import (
    is_stalemate_after,
    box_area, box_area_after,
    rook_safe_after,
    gives_safe_check,
    our_king_progress,
    creates_stable_cut,

)


def _rook_distance_travel(move: chess.Move) -> float:
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    return float(abs(f1 - f2) + abs(r1 - r2))


def _rook_distance_travel(move: chess.Move) -> float:
    # small penalty: longer rook drags are slightly worse
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    return float(abs(f1 - f2) + abs(r1 - r2))

# --- Add near the top if not present ---
import chess
from typing import Optional
from .predicates import (
    is_stalemate_after,
    box_area, box_area_after,
    rook_safe_after,
    shrinks_or_preserves_box,
    our_king_progress,
    gives_safe_check,
    # If you already implemented this, keep it; otherwise omit and the code below won’t rely on it.
    # creates_stable_cut,
)

def _rook_distance_travel(move: chess.Move) -> float:
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    return float(abs(f1 - f2) + abs(r1 - r2))


# -------- NEW: Phase 0 chooser (establish box & rendezvous) --------
def choose_move_p0(board: chess.Board) -> Optional[str]:
    """
    Phase 0: establish a first 'box' and bring our king toward supporting the rook.
    Filter-first (must pass):
      - not stalemate
      - rook_safe_after
      - box_nonincreasing (allow equal if king_progress or safe_check)
    Score:
      + 3.0 * (base_area - area_next)
      + 1.5 * our_king_progress
      + 0.5 if gives_safe_check
      - 0.1 * rook_distance_travel
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    base = box_area(board)
    best_mv, best_score = None, float("-inf")

    for mv in legal:
        if is_stalemate_after(board, mv):
            continue
        if not rook_safe_after(board, mv):
            continue

        area_next = box_area_after(board, mv)

        # Phase-0 rule: allow equal area if we get king progress OR a safe check
        preserves = (area_next == base)
        if area_next > base:
            # expanding the box is only OK in Phase 0 if you implement a stricter 'creates_stable_cut'
            # which we skip here to stay compatible with your current predicates.
            continue

        kprog = our_king_progress(board, mv)
        safe_chk = gives_safe_check(board, mv)

        if (area_next < base) or (preserves and (kprog > 0 or safe_chk)):
            score = 0.0
            score += 3.0 * (base - area_next)     # prefer shrinking
            score += 1.5 * kprog                  # bring king closer
            if safe_chk:
                score += 0.5
            score -= 0.1 * _rook_distance_travel(mv)

            if score > best_score:
                best_score, best_mv = score, mv

    return best_mv.uci() if best_mv else None



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

#If we can't find a good move for the strategy. 
def choose_any_safe_move(board: chess.Board) -> str | None:
    """
    Fallback: try any non-stalemate, rook-safe move; else any legal move.
    Ensures we *always* act, so the engine never stalls.
    """
    for mv in board.legal_moves:
        if not is_stalemate_after(board, mv) and rook_safe_after(board, mv):
            return mv.uci()
    # desperate: literally any legal move
    any_mv = next(iter(board.legal_moves), None)
    return any_mv.uci() if any_mv else None