"""Structural sensors for endgame evaluation (KPK focus)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import chess


@dataclass(frozen=True)
class KPKMaterialSummary:
    is_kpk: bool
    attacker_color: Optional[bool]
    pawn_square: Optional[chess.Square]
    attacker_king: Optional[chess.Square]
    defender_king: Optional[chess.Square]

    def as_dict(self) -> Dict[str, Optional[object]]:
        return {
            "is_kpk": self.is_kpk,
            "attacker_color": self.attacker_color,
            "pawn_square": self.pawn_square,
            "attacker_king": self.attacker_king,
            "defender_king": self.defender_king,
        }


def _basic_kpk_signature(board: chess.Board) -> KPKMaterialSummary:
    if board is None:
        return KPKMaterialSummary(False, None, None, None, None)

    piece_map = board.piece_map()
    white_pawns = [sq for sq, piece in piece_map.items() if piece.color and piece.piece_type == chess.PAWN]
    black_pawns = [sq for sq, piece in piece_map.items() if (not piece.color) and piece.piece_type == chess.PAWN]
    white_other = [piece for piece in piece_map.values() if piece.color and piece.piece_type not in (chess.KING, chess.PAWN)]
    black_other = [piece for piece in piece_map.values() if (not piece.color) and piece.piece_type not in (chess.KING, chess.PAWN)]

    if white_other or black_other:
        return KPKMaterialSummary(False, None, None, None, None)
    if len(white_pawns) + len(black_pawns) != 1:
        return KPKMaterialSummary(False, None, None, None, None)

    attacker_color = True if white_pawns else False
    pawn_square = white_pawns[0] if white_pawns else black_pawns[0]
    attacker_king = board.king(attacker_color)
    defender_king = board.king(not attacker_color)

    if attacker_king is None or defender_king is None:
        return KPKMaterialSummary(False, None, pawn_square, attacker_king, defender_king)

    return KPKMaterialSummary(True, attacker_color, pawn_square, attacker_king, defender_king)


def summarize_kpk_material(board: chess.Board) -> Dict[str, Optional[object]]:
    """Return a dictionary describing whether the position is basic KPK and key squares."""
    return _basic_kpk_signature(board).as_dict()


def pawn_distance_to_promotion(board: chess.Board, *, attacker_color: Optional[bool] = None) -> int:
    """Number of ranks remaining for the pawn to promote (>=0); returns 8 if not applicable."""
    summary = _basic_kpk_signature(board)
    if not summary.is_kpk:
        return 8
    color = summary.attacker_color if attacker_color is None else attacker_color
    if color is None or summary.pawn_square is None:
        return 8
    rank = chess.square_rank(summary.pawn_square)
    if color:
        return 7 - rank
    return rank


def pawn_has_clear_path(board: chess.Board, *, attacker_color: Optional[bool] = None) -> bool:
    """True if the pawn's file is unobstructed towards promotion."""
    summary = _basic_kpk_signature(board)
    if not summary.is_kpk:
        return False
    color = summary.attacker_color if attacker_color is None else attacker_color
    pawn_sq = summary.pawn_square
    if color is None or pawn_sq is None:
        return False

    step = 1 if color else -1
    file_index = chess.square_file(pawn_sq)
    rank = chess.square_rank(pawn_sq)
    target_rank = 7 if color else 0

    for r in range(rank + step, target_rank + step, step):
        sq = chess.square(file_index, r)
        piece = board.piece_at(sq)
        if piece is not None:
            # Allow capturing the defender king on the promotion square, but nothing else
            if sq == summary.defender_king:
                continue
            return False
    return True
