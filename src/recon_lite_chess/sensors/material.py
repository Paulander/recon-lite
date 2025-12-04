"""Material sensor terminal for M6 goal hierarchy.

This sensor provides material balance assessment that can be queried
by multiple strategic plans (fan-in terminal).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple

import chess


class MaterialCategory(Enum):
    """Material balance categories for strategic decision-making."""
    EQUAL = auto()           # Within 0.5 pawns
    MINOR_UP = auto()        # +3 (bishop/knight advantage)
    MINOR_DOWN = auto()      # -3
    EXCHANGE_UP = auto()     # +2 (rook for minor)
    EXCHANGE_DOWN = auto()   # -2
    PIECE_UP = auto()        # +5 (full rook or more)
    PIECE_DOWN = auto()      # -5
    WINNING = auto()         # +9 or more (queen advantage)
    LOSING = auto()          # -9 or more
    # Specific endgame patterns
    KRK = auto()             # King + Rook vs King
    KQK = auto()             # King + Queen vs King  
    KPK = auto()             # King + Pawn vs King
    KBNK = auto()            # King + Bishop + Knight vs King
    ROOK_ENDING = auto()     # Rooks + pawns only
    QUEEN_ENDING = auto()    # Queens + pawns only


# Standard piece values
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass
class MaterialAssessment:
    """Full material assessment for a position."""
    white_material: float
    black_material: float
    balance: float  # Positive favors white
    category: MaterialCategory
    piece_counts: Dict[str, int]
    is_endgame: bool
    pattern: Optional[str] = None  # e.g., "KRK", "KBNK"
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "white_material": self.white_material,
            "black_material": self.black_material,
            "balance": self.balance,
            "category": self.category.name,
            "piece_counts": self.piece_counts,
            "is_endgame": self.is_endgame,
            "pattern": self.pattern,
        }


def count_material(board: chess.Board, color: bool) -> Tuple[float, Dict[chess.PieceType, int]]:
    """Count material value and piece counts for a color."""
    total = 0.0
    counts: Dict[chess.PieceType, int] = {}
    
    for pt in chess.PIECE_TYPES:
        if pt == chess.KING:
            continue
        count = len(board.pieces(pt, color))
        counts[pt] = count
        total += count * PIECE_VALUES[pt]
    
    return total, counts


def detect_endgame_pattern(board: chess.Board) -> Optional[str]:
    """Detect specific endgame patterns."""
    white_pieces = {}
    black_pieces = {}
    
    for pt in chess.PIECE_TYPES:
        if pt == chess.KING:
            continue
        wc = len(board.pieces(pt, chess.WHITE))
        bc = len(board.pieces(pt, chess.BLACK))
        if wc > 0:
            white_pieces[pt] = wc
        if bc > 0:
            black_pieces[pt] = bc
    
    # KRK: King + Rook vs King
    if white_pieces == {chess.ROOK: 1} and not black_pieces:
        return "KRK"
    if black_pieces == {chess.ROOK: 1} and not white_pieces:
        return "KRK"
    
    # KQK: King + Queen vs King
    if white_pieces == {chess.QUEEN: 1} and not black_pieces:
        return "KQK"
    if black_pieces == {chess.QUEEN: 1} and not white_pieces:
        return "KQK"
    
    # KPK: King + Pawn vs King
    if white_pieces == {chess.PAWN: 1} and not black_pieces:
        return "KPK"
    if black_pieces == {chess.PAWN: 1} and not white_pieces:
        return "KPK"
    
    # KBNK: King + Bishop + Knight vs King
    if white_pieces == {chess.BISHOP: 1, chess.KNIGHT: 1} and not black_pieces:
        return "KBNK"
    if black_pieces == {chess.BISHOP: 1, chess.KNIGHT: 1} and not white_pieces:
        return "KBNK"
    
    # Rook ending: Only rooks and pawns
    white_non_rp = {k: v for k, v in white_pieces.items() if k not in (chess.ROOK, chess.PAWN)}
    black_non_rp = {k: v for k, v in black_pieces.items() if k not in (chess.ROOK, chess.PAWN)}
    if not white_non_rp and not black_non_rp:
        if chess.ROOK in white_pieces or chess.ROOK in black_pieces:
            return "ROOK_ENDING"
    
    # Queen ending: Only queens and pawns
    white_non_qp = {k: v for k, v in white_pieces.items() if k not in (chess.QUEEN, chess.PAWN)}
    black_non_qp = {k: v for k, v in black_pieces.items() if k not in (chess.QUEEN, chess.PAWN)}
    if not white_non_qp and not black_non_qp:
        if chess.QUEEN in white_pieces or chess.QUEEN in black_pieces:
            return "QUEEN_ENDING"
    
    return None


def categorize_balance(balance: float, pattern: Optional[str]) -> MaterialCategory:
    """Convert material balance to a strategic category."""
    # Check for specific endgame patterns first
    if pattern == "KRK":
        return MaterialCategory.KRK
    if pattern == "KQK":
        return MaterialCategory.KQK
    if pattern == "KPK":
        return MaterialCategory.KPK
    if pattern == "KBNK":
        return MaterialCategory.KBNK
    if pattern == "ROOK_ENDING":
        return MaterialCategory.ROOK_ENDING
    if pattern == "QUEEN_ENDING":
        return MaterialCategory.QUEEN_ENDING
    
    # General categories based on balance
    if balance >= 9.0:
        return MaterialCategory.WINNING
    if balance <= -9.0:
        return MaterialCategory.LOSING
    if balance >= 5.0:
        return MaterialCategory.PIECE_UP
    if balance <= -5.0:
        return MaterialCategory.PIECE_DOWN
    if balance >= 2.5:
        return MaterialCategory.MINOR_UP
    if balance <= -2.5:
        return MaterialCategory.MINOR_DOWN
    if balance >= 1.5:
        return MaterialCategory.EXCHANGE_UP
    if balance <= -1.5:
        return MaterialCategory.EXCHANGE_DOWN
    
    return MaterialCategory.EQUAL


def assess_material(board: chess.Board) -> MaterialAssessment:
    """Full material assessment for strategic planning."""
    white_mat, white_counts = count_material(board, chess.WHITE)
    black_mat, black_counts = count_material(board, chess.BLACK)
    
    balance = white_mat - black_mat
    pattern = detect_endgame_pattern(board)
    
    # Determine if it's an endgame (few pieces, no queens, or <15 material total)
    total_material = white_mat + black_mat
    has_queens = (white_counts.get(chess.QUEEN, 0) + black_counts.get(chess.QUEEN, 0)) > 0
    is_endgame = total_material < 15.0 or (not has_queens and total_material < 25.0)
    
    category = categorize_balance(balance, pattern)
    
    # Build piece count dict
    piece_counts = {
        "white_pawns": white_counts.get(chess.PAWN, 0),
        "white_knights": white_counts.get(chess.KNIGHT, 0),
        "white_bishops": white_counts.get(chess.BISHOP, 0),
        "white_rooks": white_counts.get(chess.ROOK, 0),
        "white_queens": white_counts.get(chess.QUEEN, 0),
        "black_pawns": black_counts.get(chess.PAWN, 0),
        "black_knights": black_counts.get(chess.KNIGHT, 0),
        "black_bishops": black_counts.get(chess.BISHOP, 0),
        "black_rooks": black_counts.get(chess.ROOK, 0),
        "black_queens": black_counts.get(chess.QUEEN, 0),
    }
    
    return MaterialAssessment(
        white_material=white_mat,
        black_material=black_mat,
        balance=balance,
        category=category,
        piece_counts=piece_counts,
        is_endgame=is_endgame,
        pattern=pattern,
    )


# Terminal predicate for use in ReCoN graphs
def material_sensor_predicate(node: Any, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Terminal predicate that assesses material balance.
    
    Stores assessment in node.activation.meta and env.
    Returns (done=True, success=True) immediately since this is a sensor.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    assessment = assess_material(board)
    
    # Store in node activation meta for retrieval
    node.activation.meta["material"] = assessment.as_dict()
    node.activation.meta["category"] = assessment.category.name
    node.activation.meta["balance"] = assessment.balance
    
    # Also store in env for other nodes to access
    env["material_assessment"] = assessment.as_dict()
    
    # Activation value encodes normalized balance (-1 to 1)
    node.activation.value = max(-1.0, min(1.0, assessment.balance / 10.0))
    
    return True, True


def create_material_sensor_node():
    """Factory to create a material sensor terminal node."""
    from recon_lite.graph import Node, NodeType
    
    return Node(
        nid="MaterialSensor",
        ntype=NodeType.TERMINAL,
        predicate=material_sensor_predicate,
        meta={"sensor_type": "material", "fan_in_allowed": True},
    )

