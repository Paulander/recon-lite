"""
Shared utility functions for chess endgame evaluation.

This module consolidates commonly-used helper functions that were previously
duplicated across predicates.py, sensors/, and scripts/.
"""

from __future__ import annotations

import chess
from typing import Optional, List, Tuple
from enum import Enum, auto


# =============================================================================
# Distance & Position Helpers
# =============================================================================

def chebyshev(sq1: chess.Square, sq2: chess.Square) -> int:
    """
    Chebyshev (king) distance between two squares.
    
    This is the minimum number of king moves to get from sq1 to sq2.
    """
    return max(
        abs(chess.square_file(sq1) - chess.square_file(sq2)),
        abs(chess.square_rank(sq1) - chess.square_rank(sq2))
    )


def dist_to_edge(sq: chess.Square) -> int:
    """Distance from square to nearest board edge (0-3)."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return min(f, 7 - f, r, 7 - r)


def is_on_edge(sq: chess.Square) -> bool:
    """Check if square is on any board edge (rim)."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return f in (0, 7) or r in (0, 7)


# Alias for backward compatibility
on_rim = is_on_edge


def is_in_corner(sq: chess.Square) -> bool:
    """Check if square is a corner square (a1, a8, h1, h8)."""
    return sq in (chess.A1, chess.A8, chess.H1, chess.H8)


def is_near_corner(sq: chess.Square) -> bool:
    """Check if square is in or adjacent to a corner (2x2 region)."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return (
        (f <= 1 and r <= 1) or  # a1 corner
        (f <= 1 and r >= 6) or  # a8 corner
        (f >= 6 and r <= 1) or  # h1 corner
        (f >= 6 and r >= 6)     # h8 corner
    )


# =============================================================================
# Escape Square / Mobility Analysis
# =============================================================================

def count_escape_squares(
    board: chess.Board,
    king_color: chess.Color,
    attacker_color: chess.Color,
) -> int:
    """
    Count the number of escape squares available to a king.
    
    An escape square is:
    - Adjacent to the king
    - Not occupied by a friendly piece
    - Not attacked by the opponent
    
    Args:
        board: Current board position
        king_color: Color of the king to analyze
        attacker_color: Color of the attacking side
        
    Returns:
        Number of available escape squares (0-8)
    """
    king_sq = board.king(king_color)
    if king_sq is None:
        return 0
    
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    escape_count = 0
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            
            nf, nr = king_file + df, king_rank + dr
            if not (0 <= nf <= 7 and 0 <= nr <= 7):
                continue
            
            sq = chess.square(nf, nr)
            
            # Skip if occupied by friendly piece
            occupant = board.piece_at(sq)
            if occupant and occupant.color == king_color:
                continue
            
            # Skip if attacked by opponent
            if board.is_attacked_by(attacker_color, sq):
                continue
            
            escape_count += 1
    
    return escape_count


# =============================================================================
# Endgame Classification
# =============================================================================

class EndgameType(Enum):
    """Classification of common endgame types."""
    UNKNOWN = auto()
    KRK = auto()     # King + Rook vs King
    KQK = auto()     # King + Queen vs King
    KPK = auto()     # King + Pawn vs King
    KBBK = auto()    # King + 2 Bishops vs King
    KBNK = auto()    # King + Bishop + Knight vs King


def classify_endgame(board: chess.Board) -> Tuple[EndgameType, Optional[chess.Color]]:
    """
    Classify the endgame type and identify the attacker.
    
    Returns:
        (EndgameType, attacker_color) - attacker_color is None if not a recognized endgame
    """
    pieces = list(board.piece_map().values())
    
    white_pieces = [p for p in pieces if p.color == chess.WHITE]
    black_pieces = [p for p in pieces if p.color == chess.BLACK]
    
    white_types = sorted([p.piece_type for p in white_pieces])
    black_types = sorted([p.piece_type for p in black_pieces])
    
    # Check for KRK
    if white_types == [chess.KING, chess.ROOK] and black_types == [chess.KING]:
        return EndgameType.KRK, chess.WHITE
    if black_types == [chess.KING, chess.ROOK] and white_types == [chess.KING]:
        return EndgameType.KRK, chess.BLACK
    
    # Check for KQK
    if white_types == [chess.KING, chess.QUEEN] and black_types == [chess.KING]:
        return EndgameType.KQK, chess.WHITE
    if black_types == [chess.KING, chess.QUEEN] and white_types == [chess.KING]:
        return EndgameType.KQK, chess.BLACK
    
    # Check for KPK
    if white_types == [chess.KING, chess.PAWN] and black_types == [chess.KING]:
        return EndgameType.KPK, chess.WHITE
    if black_types == [chess.KING, chess.PAWN] and white_types == [chess.KING]:
        return EndgameType.KPK, chess.BLACK
    
    return EndgameType.UNKNOWN, None


# =============================================================================
# Piece Finding Helpers
# =============================================================================

def find_piece_square(
    board: chess.Board,
    piece_type: chess.PieceType,
    color: chess.Color,
) -> Optional[chess.Square]:
    """Find the first square containing a specific piece type and color."""
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == piece_type and piece.color == color:
            return sq
    return None


def find_all_piece_squares(
    board: chess.Board,
    piece_type: chess.PieceType,
    color: chess.Color,
) -> List[chess.Square]:
    """Find all squares containing a specific piece type and color."""
    squares = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == piece_type and piece.color == color:
            squares.append(sq)
    return squares


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Distance helpers
    "chebyshev",
    "dist_to_edge",
    "is_on_edge",
    "on_rim",
    "is_in_corner",
    "is_near_corner",
    # Mobility
    "count_escape_squares",
    # Classification
    "EndgameType",
    "classify_endgame",
    # Piece finding
    "find_piece_square",
    "find_all_piece_squares",
]
