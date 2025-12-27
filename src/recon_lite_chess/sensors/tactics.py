"""Tactical KPK sensors (push safety, opposition readiness)."""

from __future__ import annotations

from typing import Optional

import chess

from .structure import _basic_kpk_signature
from recon_lite_chess.endgame_utils import chebyshev as _chebyshev




def can_push_pawn_safely(board: chess.Board, *, attacker_color: Optional[bool] = None) -> bool:
    summary = _basic_kpk_signature(board)
    if not summary.is_kpk or summary.pawn_square is None:
        return False
    color = summary.attacker_color if attacker_color is None else attacker_color
    pawn_sq = summary.pawn_square
    if color is None:
        return False

    direction = 8 if color else -8
    to_sq = pawn_sq + direction
    move = chess.Move(pawn_sq, to_sq)
    if move not in board.legal_moves:
        return False

    trial = board.copy(stack=False)
    trial.push(move)
    defender_king = trial.king(not color)
    attacker_king = trial.king(color)
    if defender_king is None:
        return True
    if attacker_king is None:
        return False
    if _chebyshev(defender_king, to_sq) <= 1 and _chebyshev(attacker_king, to_sq) > 1:
        return False
    return True


def has_opposition_alignment(board: chess.Board, *, attacker_color: Optional[bool] = None) -> bool:
    summary = _basic_kpk_signature(board)
    if not summary.is_kpk:
        return False
    color = summary.attacker_color if attacker_color is None else attacker_color
    attacker_king = summary.attacker_king
    defender_king = summary.defender_king
    if color is None or attacker_king is None or defender_king is None:
        return False

    same_file = chess.square_file(attacker_king) == chess.square_file(defender_king)
    same_rank = chess.square_rank(attacker_king) == chess.square_rank(defender_king)
    if not (same_file or same_rank):
        return False
    distance = _chebyshev(attacker_king, defender_king)
    return distance % 2 == 1 and distance >= 1


def opposition_after_push(board: chess.Board, *, attacker_color: Optional[bool] = None) -> bool:
    summary = _basic_kpk_signature(board)
    if not summary.is_kpk or summary.pawn_square is None:
        return False
    color = summary.attacker_color if attacker_color is None else attacker_color
    if color is None:
        return False
    direction = 8 if color else -8
    to_sq = summary.pawn_square + direction
    move = chess.Move(summary.pawn_square, to_sq)
    if move not in board.legal_moves:
        return False
    trial = board.copy(stack=False)
    trial.push(move)
    return has_opposition_alignment(trial, attacker_color=color)
