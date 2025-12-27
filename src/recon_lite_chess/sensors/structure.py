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


def is_kpk_theoretical_draw(board: chess.Board) -> bool:
    """
    Check if a KPK position is a theoretical draw.
    
    This function is designed to be called on STARTING positions only,
    not positions reached during play. It uses conservative heuristics
    to avoid false positives.
    
    Drawing scenarios detected:
    1. Rook pawn (a/h file) with defending king in the corner
    2. Rule of the square: defending king can reach queening square in time
    3. Defending king controls the key squares ahead of the pawn
    
    Args:
        board: A KPK position (K+P vs K)
        
    Returns:
        True if the position is likely a theoretical draw, False otherwise
    """
    summary = _basic_kpk_signature(board)
    if not summary.is_kpk:
        return False
    
    pawn_sq = summary.pawn_square
    attacker_king = summary.attacker_king
    defender_king = summary.defender_king
    attacker_color = summary.attacker_color
    
    if pawn_sq is None or attacker_king is None or defender_king is None or attacker_color is None:
        return False
    
    pawn_file = chess.square_file(pawn_sq)
    pawn_rank = chess.square_rank(pawn_sq)
    
    # Promotion square
    promo_rank = 7 if attacker_color else 0
    promo_sq = chess.square(pawn_file, promo_rank)
    
    # Distance calculations
    pawn_to_promo = abs(promo_rank - pawn_rank)
    defender_to_promo = _chebyshev_distance(defender_king, promo_sq)
    attacker_king_to_pawn = _chebyshev_distance(attacker_king, pawn_sq)
    
    # =========================================================================
    # Rule 1: Rook pawn (a/h file) with defending king in the corner
    # =========================================================================
    if pawn_file in (0, 7):  # a-file or h-file
        # Corner squares for this rook pawn
        corner_sq = chess.square(pawn_file, promo_rank)
        adjacent_corner = chess.square(pawn_file, promo_rank - 1 if attacker_color else promo_rank + 1)
        
        # If defending king is in/near the corner and can reach it, it's a draw
        defender_to_corner = _chebyshev_distance(defender_king, corner_sq)
        if defender_to_corner <= 1:
            return True
        
        # Rook pawn draw: if defender can reach the corner before pawn promotes
        # and attacking king can't approach from the side
        if defender_to_corner <= pawn_to_promo:
            return True
    
    # =========================================================================
    # Rule 2: Rule of the Square (simple version)
    # =========================================================================
    # If it's the defender's turn (or will be soon), and they can reach the 
    # queening square before the pawn, it's a draw
    
    # Whose turn is it?
    defender_turn = (board.turn != attacker_color)
    
    # Effective distance for defender (subtract 1 if their turn)
    effective_defender_dist = defender_to_promo - (1 if defender_turn else 0)
    
    # If defender can reach promo square in time AND attacking king is far
    if effective_defender_dist <= pawn_to_promo and attacker_king_to_pawn > 2:
        return True
    
    # =========================================================================
    # Rule 3: Key squares / Opposition
    # =========================================================================
    # The key squares are the squares directly ahead of the pawn
    # If defender controls these and attacking king can't help, it's a draw
    
    # Key squares for the pawn (simplified: 3 squares in front of pawn on rank+1/+2)
    direction = 1 if attacker_color else -1
    key_squares = []
    for dr in [1, 2]:
        for df in [-1, 0, 1]:
            key_rank = pawn_rank + dr * direction
            key_file = pawn_file + df
            if 0 <= key_rank <= 7 and 0 <= key_file <= 7:
                key_squares.append(chess.square(key_file, key_rank))
    
    # Check if defender has opposition (controls key squares)
    defender_on_key = defender_king in key_squares
    
    # If pawn is on ranks 2-4 (early) and defender has opposition with distant attacker king, likely draw
    if attacker_color:  # White pawn
        if pawn_rank <= 3 and defender_on_key and attacker_king_to_pawn >= 3:
            return True
    else:  # Black pawn
        if pawn_rank >= 4 and defender_on_key and attacker_king_to_pawn >= 3:
            return True
    
    return False


# Import shared helper (replaces local duplicate)
from recon_lite_chess.endgame_utils import chebyshev as _chebyshev_distance


