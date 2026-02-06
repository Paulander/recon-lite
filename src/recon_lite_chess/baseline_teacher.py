"""
KRK Teacher for Baseline Architecture

Provides domain-specific feature extraction and transition labeling for KRK endgame.
All chess-specific logic lives HERE, not in the baseline module.
"""

import chess
import numpy as np
from typing import List
from recon_lite.learning.baseline import TransitionData


# ============================================================================
# KRK Feature Extraction (Domain-Specific)
# ============================================================================

def box_area(board: chess.Board) -> int:
    """Compute the box area (number of squares black king can reach)"""
    bk = board.king(chess.BLACK)
    if bk is None:
        return 64
    
    # Find rook position
    rooks = board.pieces(chess.ROOK, chess.WHITE)
    if not rooks:
        return 64
    rook = rooks.pop()
    
    # Box is bounded by rook's file/rank
    rook_file = chess.square_file(rook)
    rook_rank = chess.square_rank(rook)
    bk_file = chess.square_file(bk)
    bk_rank = chess.square_rank(bk)
    
    # Count squares in the box
    if bk_file < rook_file:
        width = rook_file
    else:
        width = 7 - rook_file
    
    if bk_rank < rook_rank:
        height = rook_rank
    else:
        height = 7 - rook_rank
    
    return width * height


def king_distance(board: chess.Board) -> int:
    """Chebyshev distance between kings"""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 7
    
    wk_file, wk_rank = chess.square_file(wk), chess.square_rank(wk)
    bk_file, bk_rank = chess.square_file(bk), chess.square_rank(bk)
    
    return max(abs(wk_file - bk_file), abs(wk_rank - bk_rank))


def edge_distance(square: int) -> float:
    """Distance from square to nearest edge"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return min(file, 7 - file, rank, 7 - rank)


def has_opposition(board: chess.Board) -> bool:
    """Check if white has opposition"""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return False
    
    dist = king_distance(board)
    if dist != 2:
        return False
    
    # Opposition if kings are on same file or rank and 2 squares apart
    wk_file, wk_rank = chess.square_file(wk), chess.square_rank(wk)
    bk_file, bk_rank = chess.square_file(bk), chess.square_rank(bk)
    
    return (wk_file == bk_file) or (wk_rank == bk_rank)


def can_deliver_mate(board: chess.Board) -> bool:
    """Check if there exists a move that delivers checkmate"""
    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        if test_board.is_checkmate():
            return True
    return False


# ============================================================================
# KRK Teacher
# ============================================================================

class KRKTeacher:
    """
    Teacher for KRK endgame bootstrap.
    
    Provides:
    - Feature extraction (all domain logic here)
    - Transition labeling (mate-in-1 for Stage 0)
    """
    
    # Feature dimension for KRK
    FEATURE_DIM = 15  # Added side_to_move feature
    
    def __init__(self):
        """Initialize KRK teacher"""
        pass
    
    @property
    def feature_dim(self) -> int:
        """Return feature dimension"""
        return self.FEATURE_DIM
    
    def features(self, board: chess.Board) -> np.ndarray:
        """
        Extract full designer feature vector for KRK.
        
        Features (13 total):
        0. box_area (normalized 0-1)
        1. king_distance (normalized 0-1)
        2. white_king_file (normalized 0-1)
        3. white_king_rank (normalized 0-1)
        4. black_king_file (normalized 0-1)
        5. black_king_rank (normalized 0-1)
        6. rook_file (normalized 0-1)
        7. rook_rank (normalized 0-1)
        8. opposition (binary 0 or 1)
        9. edge_distance_white_king (normalized 0-1)
        10. edge_distance_black_king (normalized 0-1)
        11. is_check (binary 0 or 1)
        12. can_deliver_mate (binary 0 or 1)
        13. is_checkmate (binary 0 or 1)
        14. side_to_move (1 if White to move, 0 if Black)
        
        All domain-specific logic (e.g., "dist_to_edge") is computed HERE,
        not in baseline readouts.
        
        Args:
            board: Chess board state
        
        Returns:
            Feature vector of length FEATURE_DIM
        """
        features = []
        
        # Box area (0-64, normalized)
        features.append(box_area(board) / 64.0)
        
        # King distance (0-7, normalized)
        features.append(king_distance(board) / 7.0)
        
        # Positions (0-7, normalized)
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        
        if wk is not None:
            features.extend([
                chess.square_file(wk) / 7.0,
                chess.square_rank(wk) / 7.0,
            ])
        else:
            features.extend([0.0, 0.0])
        
        if bk is not None:
            features.extend([
                chess.square_file(bk) / 7.0,
                chess.square_rank(bk) / 7.0,
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Rook position
        rooks = board.pieces(chess.ROOK, chess.WHITE)
        if rooks:
            rook_sq = rooks.pop()
            features.extend([
                chess.square_file(rook_sq) / 7.0,
                chess.square_rank(rook_sq) / 7.0,
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Binary features
        features.append(1.0 if has_opposition(board) else 0.0)
        features.append(edge_distance(wk) / 3.5 if wk is not None else 0.0)  # max dist to edge is 3.5
        features.append(edge_distance(bk) / 3.5 if bk is not None else 0.0)
        features.append(1.0 if board.is_check() else 0.0)
        features.append(1.0 if can_deliver_mate(board) else 0.0)
        features.append(1.0 if board.is_checkmate() else 0.0)  # THE GOAL STATE
        features.append(1.0 if board.turn == chess.WHITE else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def label_transitions(self, x0: chess.Board) -> List[TransitionData]:
        """
        Generate labeled transitions from starting position.
        
        For KRK Stage 0 (mate-in-1):
        - label=1 if move delivers checkmate
        - label=0 otherwise
        
        Args:
            x0: Starting board position
        
        Returns:
            List of TransitionData with v0, v1, label, action
        """
        transitions = []
        v0 = self.features(x0)
        
        for move in x0.legal_moves:
            x1 = x0.copy()
            x1.push(move)
            
            v1 = self.features(x1)
            label = 1 if x1.is_checkmate() else 0
            reward = 1.0 if label == 1 else 0.0
            
            transitions.append(TransitionData(
                v0=v0,
                v1=v1,
                label=label,
                action=move,
                reward=reward
            ))
        
        return transitions


# ============================================================================
# Position Generation
# ============================================================================

def generate_krk_mate_in_1_position(target_corner: int | None = None) -> chess.Board:
    """
    Generate a random legal KRK position where white can deliver mate in 1.

    This brute-forces random legal KRK boards until it finds one with a
    mate-in-1 move for White. This avoids invalid "mate-in-1" templates.
    
    Returns:
        Chess board with a mate-in-1 position
    """
    import random

    squares = list(chess.SQUARES)
    max_attempts = 10000

    for _ in range(max_attempts):
        wk, bk, wr = random.sample(squares, 3)
        board = chess.Board(None)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
        board.turn = chess.WHITE

        # Basic legality filters
        if chess.square_distance(wk, bk) <= 1:
            continue
        if not board.is_valid():
            continue
        if board.is_check():
            continue

        if target_corner is not None and bk != target_corner:
            continue

        # Check if any legal move is checkmate
        for move in board.legal_moves:
            b2 = board.copy()
            b2.push(move)
            if b2.is_checkmate():
                return board

    raise RuntimeError("Failed to find KRK mate-in-1 position after many attempts")
