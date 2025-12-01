"""
Heuristic evaluation for chess positions.

Provides a simple, fast evaluation function suitable for reward signal
computation in M3 plasticity. All values are in pawn units (positive = white
advantage).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import chess

if TYPE_CHECKING:
    import chess.engine


# Piece values in centipawns (standard)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King has no material value
}


def _material_score(board: chess.Board) -> float:
    """
    Compute material balance in pawn units (positive = white advantage).
    """
    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    return score / 100.0  # Convert to pawn units


def _king_safety_score(board: chess.Board) -> float:
    """
    Simple king safety heuristic based on king position and attackers.

    Returns score in pawn units (positive = white safer).
    """
    score = 0.0

    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue

        # Penalty for king in center (files d-e, ranks 4-5)
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)

        # Bonus for castled position (king on g/h or a/b files)
        if file in (0, 1, 6, 7):
            safety = 0.2
        elif file in (2, 5):
            safety = 0.1
        else:
            safety = -0.1

        # Penalty for exposed king (high rank for white, low for black)
        if color == chess.WHITE:
            if rank > 1:
                safety -= 0.1 * (rank - 1)
        else:
            if rank < 6:
                safety -= 0.1 * (6 - rank)

        # Count attackers on king's neighborhood
        for neighbor in board.attacks(king_sq):
            attackers = board.attackers(not color, neighbor)
            safety -= 0.05 * len(attackers)

        if color == chess.WHITE:
            score += safety
        else:
            score -= safety

    return score


def _mobility_score(board: chess.Board) -> float:
    """
    Simple mobility heuristic based on legal move count.

    Returns score in pawn units (positive = white more mobile).
    """
    # Count legal moves for current side
    current_moves = len(list(board.legal_moves))

    # Temporarily switch sides to count opponent moves
    board_copy = board.copy()
    board_copy.turn = not board.turn
    opponent_moves = len(list(board_copy.legal_moves))

    # Normalize: ~0.01 pawns per move difference
    if board.turn == chess.WHITE:
        return (current_moves - opponent_moves) * 0.01
    else:
        return (opponent_moves - current_moves) * 0.01


def _endgame_factor(board: chess.Board) -> float:
    """
    Compute endgame factor (0 = opening, 1 = pure endgame).

    Based on total material on board.
    """
    total_material = 0
    for piece in board.piece_map().values():
        if piece.piece_type != chess.KING:
            total_material += PIECE_VALUES.get(piece.piece_type, 0)

    # Full material ~= 7800 (2*Q + 4*R + 4*B + 4*N + 16*P)
    # Endgame starts around 2000
    if total_material >= 4000:
        return 0.0
    elif total_material <= 1000:
        return 1.0
    else:
        return 1.0 - (total_material - 1000) / 3000.0


def _krk_specific_score(board: chess.Board) -> float:
    """
    KRK-specific evaluation: distance of enemy king to corner/edge.

    Only applies in KRK endgames (K+R vs K).
    """
    # Check if this is a KRK position
    pieces = list(board.piece_map().values())
    piece_types = [(p.piece_type, p.color) for p in pieces]

    white_pieces = [pt for pt, c in piece_types if c == chess.WHITE]
    black_pieces = [pt for pt, c in piece_types if c == chess.BLACK]

    is_krk = (
        sorted(white_pieces) == [chess.KING, chess.ROOK]
        and sorted(black_pieces) == [chess.KING]
    )

    if not is_krk:
        return 0.0

    # Find enemy king
    enemy_king = board.king(chess.BLACK)
    if enemy_king is None:
        return 0.0

    # Distance to center (we want enemy king away from center)
    file = chess.square_file(enemy_king)
    rank = chess.square_rank(enemy_king)

    # Distance from center (3.5, 3.5)
    center_dist = max(abs(file - 3.5), abs(rank - 3.5))

    # Bonus for enemy king near edge (center_dist ranges 0-3.5)
    return center_dist * 0.3


def eval_position(board: chess.Board) -> float:
    """
    Evaluate a chess position using heuristics.

    Args:
        board: The chess board to evaluate

    Returns:
        Evaluation in pawn units (positive = white advantage)
    """
    # Check for terminal positions
    if board.is_checkmate():
        return -100.0 if board.turn == chess.WHITE else 100.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    # Compute components
    material = _material_score(board)
    king_safety = _king_safety_score(board)
    mobility = _mobility_score(board)
    krk_bonus = _krk_specific_score(board)

    # Weight by endgame factor
    eg_factor = _endgame_factor(board)

    # In endgame, king safety matters less, KRK bonus matters more
    score = (
        material
        + king_safety * (1.0 - eg_factor * 0.5)
        + mobility * (1.0 - eg_factor * 0.3)
        + krk_bonus * eg_factor
    )

    return score


def compute_reward_tick(
    eval_before: float,
    eval_after: float,
    r_max: float = 2.0,
) -> float:
    """
    Compute clipped reward from evaluation delta.

    Args:
        eval_before: Evaluation before the action
        eval_after: Evaluation after the action
        r_max: Maximum absolute reward value

    Returns:
        Clipped reward in [-r_max, +r_max]
    """
    delta = eval_after - eval_before
    return max(-r_max, min(r_max, delta))


def eval_position_stockfish(
    board: chess.Board,
    engine: "chess.engine.SimpleEngine",
    depth: int = 3,
    time_limit: float = 0.1,
) -> float:
    """
    Evaluate a position using Stockfish.

    Args:
        board: The chess board to evaluate
        engine: An initialized Stockfish engine
        depth: Search depth
        time_limit: Time limit in seconds

    Returns:
        Evaluation in pawn units (positive = white advantage)
    """
    try:
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=time_limit),
        )
        score = info.get("score")
        if score is None:
            return eval_position(board)  # Fallback

        pov_score = score.white()

        if pov_score.is_mate():
            mate_in = pov_score.mate()
            if mate_in is not None:
                # Convert mate score to large value
                if mate_in > 0:
                    return 100.0 - mate_in * 0.1
                else:
                    return -100.0 - mate_in * 0.1
            return 0.0

        cp = pov_score.score()
        if cp is not None:
            return cp / 100.0  # Convert centipawns to pawns

        return eval_position(board)  # Fallback

    except Exception:
        return eval_position(board)  # Fallback on any error

