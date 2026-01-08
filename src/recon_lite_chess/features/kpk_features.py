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
    Extract pawn-centric relative features for geometric invariance.
    
    This enables pattern generalization - a stem cell can learn
    that wk_rel=(+1, 0) and bk_rel=(+2, 0) correlates with wins,
    which IS opposition, but discovered not hard-coded.
    
    All positions are relative to pawn, enabling file-agnostic learning.
    
    Returns:
        10-element feature vector:
        [pawn_progress, wk_rank_delta, wk_file_delta, bk_rank_delta, bk_file_delta,
         wk_bk_rank_delta, wk_bk_file_delta, wk_pawn_dist, bk_pawn_dist, is_edge_pawn]
    """
    wk, bk, pawn = get_kpk_pieces(board)
    
    # Handle missing pieces gracefully
    if wk is None or bk is None or pawn is None:
        return [0.0] * 10
    
    # Get coordinates
    pawn_rank = chess.square_rank(pawn)
    pawn_file = chess.square_file(pawn)
    
    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)
    
    bk_rank = chess.square_rank(bk)
    bk_file = chess.square_file(bk)
    
    # === Pawn-centric features ===
    # Pawn progress: 0 = rank 1, 1 = rank 7 (about to promote)
    pawn_progress = pawn_rank / 7.0
    
    # King positions relative to pawn (normalized to [-1, 1])
    wk_rank_delta = (wk_rank - pawn_rank) / 7.0
    wk_file_delta = (wk_file - pawn_file) / 7.0
    bk_rank_delta = (bk_rank - pawn_rank) / 7.0
    bk_file_delta = (bk_file - pawn_file) / 7.0
    
    # === Opposition detection features ===
    # Direct WK vs BK comparison (crucial for opposition/shouldering)
    wk_bk_rank_delta = (wk_rank - bk_rank) / 7.0  # Positive = WK ahead of BK
    wk_bk_file_delta = (wk_file - bk_file) / 7.0  # Positive = WK right of BK
    
    # === Distance features (for Square Rule) ===
    # Chebyshev distance from kings to pawn
    wk_pawn_dist = max(abs(wk_rank - pawn_rank), abs(wk_file - pawn_file)) / 7.0
    bk_pawn_dist = max(abs(bk_rank - pawn_rank), abs(bk_file - pawn_file)) / 7.0
    
    # === Edge pawn flag (a/h file = potentially drawn) ===
    is_edge_pawn = float(pawn_file in (0, 7))
    
    # === Tempo feature (CRUCIAL for opposition and zugzwang) ===
    side_to_move = 1.0 if board.turn == chess.WHITE else 0.0
    
    return [
        pawn_progress,      # How far advanced (0-1)
        wk_rank_delta,      # WK rank offset from pawn
        wk_file_delta,      # WK file offset from pawn
        bk_rank_delta,      # BK rank offset from pawn
        bk_file_delta,      # BK file offset from pawn
        wk_bk_rank_delta,   # Opposition: WK vs BK vertical
        wk_bk_file_delta,   # Opposition: WK vs BK horizontal
        wk_pawn_dist,       # WK distance to pawn (for escort)
        bk_pawn_dist,       # BK distance to pawn (for Square Rule)
        is_edge_pawn,       # Rook pawn flag (draw risk)
        side_to_move,       # ADDED: whose turn (crucial for zugzwang!)
    ]


def extract_kpk_features_v1(board: chess.Board) -> List[float]:
    """
    Legacy 6-element feature vector for backward compatibility.
    
    Returns:
        6-element feature vector:
        [pawn_rank, wk_rel_rank, wk_rel_file, bk_rel_rank, bk_rel_file, is_rook_pawn]
    """
    wk, bk, pawn = get_kpk_pieces(board)
    
    if wk is None or bk is None or pawn is None:
        return [0.0] * 6
    
    pawn_rank = chess.square_rank(pawn)
    pawn_file = chess.square_file(pawn)
    wk_rank = chess.square_rank(wk)
    wk_file = chess.square_file(wk)
    bk_rank = chess.square_rank(bk)
    bk_file = chess.square_file(bk)
    
    wk_rel_rank = (wk_rank - pawn_rank) / 7.0
    wk_rel_file = (wk_file - pawn_file) / 7.0
    bk_rel_rank = (bk_rank - pawn_rank) / 7.0
    bk_rel_file = (bk_file - pawn_file) / 7.0
    is_rook_pawn = float(pawn_file in (0, 7))
    
    return [pawn_rank / 7.0, wk_rel_rank, wk_rel_file, bk_rel_rank, bk_rel_file, is_rook_pawn]


def create_feature_extractor(nid: str) -> Node:
    """
    Factory for feature extraction node.
    
    This node runs early in the detection phase and populates
    env["features"] for all other sensors to use.
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board:
            features = extract_kpk_features(board)
            env["features"] = features
            env["kpk_features"] = features
            # Also store legacy 6-element for compatibility
            env["kpk_features_v1"] = extract_kpk_features_v1(board)
        return True, True  # Always succeeds
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


# Feature names for debugging/visualization (11 elements now)
FEATURE_NAMES = [
    "pawn_progress",
    "wk_rank_delta",
    "wk_file_delta", 
    "bk_rank_delta",
    "bk_file_delta",
    "wk_bk_rank_delta",
    "wk_bk_file_delta",
    "wk_pawn_dist",
    "bk_pawn_dist",
    "is_edge_pawn",
    "side_to_move",  # ADDED: tempo indicator
]

# Legacy feature names (6 elements)
FEATURE_NAMES_V1 = [
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
