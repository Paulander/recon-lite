
"""
Chess move selection actuators for ReCoN KRK system.

These functions implement the "filter-first, then score" approach:
1. Hard filters: Eliminate unsafe/illegal moves
2. Soft scoring: Rank remaining moves by desirability
3. Fallback: If no moves pass filters, use safe legal move

IMPROVEMENTS (Sept 15, 2025):
- Global stall prevention with choose_any_safe_move() fallback
- Enhanced P0 rendezvous logic
- Rim promotion and monotonicity constraints
- Simplified scoring: 3.0*king_progress + 2.0*box_shrink + 1.0*safe_check − 0.2*rook_drag
"""

import chess
from typing import List, Tuple, Optional, Dict, Any
from .predicates import (
    is_stalemate_after, rook_safe_after, box_area, box_area_after,
    shrinks_or_preserves_box, our_king_progress, gives_safe_check,
    has_opposition_after, creates_stable_cut
)


def choose_any_safe_move(board: chess.Board) -> Optional[str]:
    """
    Global fallback: Choose any legal move that doesn't lose the rook immediately.
    This prevents stalls by ensuring there's always a valid move selection.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # Prefer moves that don't lose the rook
    safe_moves = []
    for move in legal_moves:
        if rook_safe_after(board, move):
            safe_moves.append(move)

    if safe_moves:
        # Choose randomly from safe moves
        chosen = safe_moves[0]  # Could randomize, but deterministic for now
        return chosen.uci()

    # If no safe moves, take any legal move (better than stalling)
    return legal_moves[0].uci()


def king_to_rook_distance(board: chess.Board) -> float:
    """Calculate distance between our king and rook."""
    wk_square = board.king(board.turn)
    wr_square = None

    for square in chess.SQUARES:
        if board.piece_at(square) and board.piece_at(square).piece_type == chess.ROOK and board.piece_at(square).color == board.turn:
            wr_square = square
            break

    if wr_square is None:
        return 8.0  # Max distance if no rook found

    wk_file, wk_rank = chess.square_file(wk_square), chess.square_rank(wk_square)
    wr_file, wr_rank = chess.square_file(wr_square), chess.square_rank(wr_square)
    return float(abs(wk_file - wr_file) + abs(wk_rank - wr_rank))


def is_rim_square(square: int) -> bool:
    """Check if square is on the board rim (edge)."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return file in [0, 7] or rank in [0, 7]


def is_cornered(board: chess.Board) -> bool:
    """Check if enemy king is cornered (in corner + kings in opposition)."""
    ek_square = board.king(not board.turn)
    wk_square = board.king(board.turn)

    # Check if enemy king is in corner
    corners = [chess.A1, chess.H1, chess.A8, chess.H8]
    if ek_square not in corners:
        return False

    # Check if kings are in opposition
    ek_file, ek_rank = chess.square_file(ek_square), chess.square_rank(ek_square)
    wk_file, wk_rank = chess.square_file(wk_square), chess.square_rank(wk_square)

    # Same file or rank with odd squares between
    if ek_file == wk_file:
        squares_between = abs(ek_rank - wk_rank) - 1
        return squares_between % 2 == 1
    elif ek_rank == wk_rank:
        squares_between = abs(ek_file - wk_file) - 1
        return squares_between % 2 == 1

    return False


def _rook_distance_travel(move: chess.Move) -> float:
    """Calculate how far the rook moves (for penalty scoring)."""
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    return float(abs(f1 - f2) + abs(r1 - r2))


def _calculate_score(board: chess.Board, move: chess.Move, phase: int) -> float:
    """
    Unified scoring function with simplified weights.
    Score = 3.0*king_progress + 2.0*box_shrink + 1.0*safe_check − 0.2*rook_drag
    """
    score = 0.0

    # King progress (most important)
    score += 2.0 * our_king_progress(board, move)

    # Box shrinking
    old_area = box_area(board)
    new_area = box_area_after(board, move)
    if new_area < old_area:
        score += 2.0 * (old_area - new_area)

    # Safe check bonus
    if gives_safe_check(board, move):
        score += 1.0

    # Rook drag penalty
    score -= 0.2 * _rook_distance_travel(move)

    return score


def choose_move_phase0(board: chess.Board) -> Optional[str]:
    """
    Phase 0: Rendezvous king and rook, create safe cut.
    Enhanced with proper distance-based king movement.
    """
    legal_moves = list(board.legal_moves)
    candidates = []
    distance = king_to_rook_distance(board)

    for move in legal_moves:
        # Hard filters
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        if not shrinks_or_preserves_box(board, move):
            # Allow preservation only if king progresses or gives safe check
            if our_king_progress(board, move) <= 0 and not gives_safe_check(board, move):
                continue

        # P0-specific logic: focus on rendezvous until distance ≤ 2
        if distance > 2:
            # Prefer king moves that reduce distance to rook or enemy king
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                # Calculate distance reduction
                wk_square = board.king(board.turn)
                ek_square = board.king(not board.turn)
                wr_square = None

                for sq in chess.SQUARES:
                    if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.ROOK and board.piece_at(sq).color == board.turn:
                        wr_square = sq
                        break

                if wr_square:
                    old_dist_rook = abs(chess.square_file(wk_square) - chess.square_file(wr_square)) + abs(chess.square_rank(wk_square) - chess.square_rank(wr_square))
                    new_dist_rook = abs(chess.square_file(move.to_square) - chess.square_file(wr_square)) + abs(chess.square_rank(move.to_square) - chess.square_rank(wr_square))

                    if new_dist_rook < old_dist_rook:
                        candidates.append((move, _calculate_score(board, move, 0) + 1.0))  # Bonus for reducing distance
                    else:
                        candidates.append((move, _calculate_score(board, move, 0)))
                else:
                    candidates.append((move, _calculate_score(board, move, 0)))
            else:
                # Rook moves: only allow if creating safe cut
                candidates.append((move, _calculate_score(board, move, 0) - 0.5))  # Small penalty for rook moves in P0
        else:
            # Distance ≤ 2: normal scoring
            candidates.append((move, _calculate_score(board, move, 0)))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase1(board: chess.Board) -> Optional[str]:
    """
    Phase 1: Drive enemy king to edge.
    Enhanced with rim promotion logic and monotonicity.
    """
    legal_moves = list(board.legal_moves)
    candidates = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        if not shrinks_or_preserves_box(board, move):
            continue

        score = _calculate_score(board, move, 1)

        # Rim promotion: bonus if enemy king moves toward rim
        ek_square = board.king(not board.turn)
        if is_rim_square(ek_square):
            score += 2.0  # Bonus for being on rim

        candidates.append((move, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase2(board: chess.Board) -> Optional[str]:
    """
    Phase 2: Shrink the box.
    Enhanced with strict monotonicity requirements.
    """
    legal_moves = list(board.legal_moves)
    candidates = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        if not shrinks_or_preserves_box(board, move):
            continue

        score = _calculate_score(board, move, 2)
        candidates.append((move, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase3(board: chess.Board) -> Optional[str]:
    """
    Phase 3: Take opposition.
    Enhanced with opposition detection and P4 promotion.
    """
    legal_moves = list(board.legal_moves)
    candidates = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        if not shrinks_or_preserves_box(board, move):
            continue

        score = _calculate_score(board, move, 3)

        # P4 promotion: bonus if enemy king is cornered with opposition
        if is_cornered(board):
            score += 3.0  # Strong bonus for mate setup

        candidates.append((move, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase4(board: chess.Board) -> Optional[str]:
    """
    Phase 4: Deliver mate.
    Enhanced with mate detection.
    """
    legal_moves = list(board.legal_moves)
    candidates = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue

        score = _calculate_score(board, move, 4)

        # Mate bonus
        b = board.copy()
        b.push(move)
        if b.is_checkmate():
            score += 10.0  # Huge bonus for mate

        candidates.append((move, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)
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
    """



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
            score += edge_driving_bonus(board, move) * 2.0
        elif phase == "phase2":
            score += area_reduction * 5.0
        elif phase == "phase3":
            if has_opposition_after(board, move):
                score += 1.0
            score += king_score * 1.5
        elif phase == "phase4":
            if gives_safe_check(board, move):
                score += 1.0

        # Global: bonus for forming a stable rook cut
        if creates_stable_cut(board, move):
            score += 1.5

        # Penalize rook drag to avoid shuffling
        score -= 0.2 * _rook_distance_travel(move)

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
    "Calculate bonus for moves that drive enemy king toward edge."
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
    "Fallback: try any non-stalemate, rook-safe move; else any legal move."
    for mv in board.legal_moves:
        if not is_stalemate_after(board, mv) and rook_safe_after(board, mv):
            return mv.uci()
    # desperate: literally any legal move
    any_mv = next(iter(board.legal_moves), None)
    return any_mv.uci() if any_mv else None