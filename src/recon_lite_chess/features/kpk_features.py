"""KPK Feature Extraction with Relative Coordinates.

Extracts features centered on the pawn position to enable pattern discovery.
Stem cells can learn relationships like opposition without explicit coding.

Usage:
    from recon_lite_chess.features.kpk_features import extract_kpk_features
    
    features = extract_kpk_features(board)  # Returns 6-element list
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chess

from recon_lite.graph import Node, NodeType


def get_kpk_pieces(board: chess.Board) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract piece positions for KPK endgame.
    
    Returns:
        (white_king_sq, black_king_sq, pawn_sq) or None if not found
    """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    
    # Find the pawn (should be only one in KPK)
    pawn = None
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        pawn = sq
        break
    
    # Also check for black pawn (in case we're playing as black)
    if pawn is None:
        for sq in board.pieces(chess.PAWN, chess.BLACK):
            pawn = sq
            break
    
    return wk, bk, pawn


def extract_kpk_features(board: chess.Board) -> List[float]:
    """
    Extract relative features centered on pawn position.
    
    This enables pattern generalization - a stem cell can learn
    that wk_rel=(+1, 0) and bk_rel=(+2, 0) correlates with wins,
    which IS opposition, but discovered not hard-coded.
    
    Returns:
        6-element feature vector:
        [pawn_rank, wk_rel_rank, wk_rel_file, bk_rel_rank, bk_rel_file, is_rook_pawn]
    """
    wk, bk, pawn = get_kpk_pieces(board)
    
    # Handle missing pieces gracefully
    if wk is None or bk is None or pawn is None:
        return [0.0] * 6
    
    # Get coordinates
    pawn_rank = chess.square_rank(pawn)
    pawn_file = chess.square_file(pawn)
    
    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)
    
    bk_rank = chess.square_rank(bk)
    bk_file = chess.square_file(bk)
    
    # Relative positions (centered on pawn)
    wk_rel_rank = (wk_rank - pawn_rank) / 7.0
    wk_rel_file = (wk_file - pawn_file) / 7.0
    bk_rel_rank = (bk_rank - pawn_rank) / 7.0
    bk_rel_file = (bk_file - pawn_file) / 7.0
    
    # Rook pawn flag (a or h file = potentially drawn)
    is_rook_pawn = float(pawn_file in (0, 7))
    
    return [
        pawn_rank / 7.0,      # Absolute: how far advanced (0-1)
        wk_rel_rank,          # Relative: white king rank offset
        wk_rel_file,          # Relative: white king file offset
        bk_rel_rank,          # Relative: black king rank offset
        bk_rel_file,          # Relative: black king file offset
        is_rook_pawn,         # Flag: rook pawn (a/h file)
    ]


def create_feature_extractor(nid: str) -> Node:
    """
    Factory for feature extraction node.
    
    This node runs early in the detection phase and populates
    env["features"] for all other sensors to use.
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board:
            env["features"] = extract_kpk_features(board)
            env["kpk_features"] = env["features"]  # Alias for clarity
        return True, True  # Always succeeds
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


# Feature names for debugging/visualization
FEATURE_NAMES = [
    "pawn_rank",
    "wk_rel_rank",
    "wk_rel_file", 
    "bk_rel_rank",
    "bk_rel_file",
    "is_rook_pawn",
]


def features_to_dict(features: List[float]) -> Dict[str, float]:
    """Convert feature vector to named dict for debugging."""
    return {name: val for name, val in zip(FEATURE_NAMES, features)}
