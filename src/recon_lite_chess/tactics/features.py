"""Feature extraction for MLP-based tactics detection.

This module extracts board features relevant to detecting:
- Back rank mate patterns
- Double check patterns  
- Smothered mate patterns

Features are designed to be computed quickly and capture the essential
positional elements that indicate each tactic.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import chess


# Total feature count for combined extraction
FEATURE_COUNT = 208  # 64 squares * 3 features + attack/defense features


@dataclass
class TacticsFeatureVector:
    """Container for extracted features."""
    features: List[float]
    tactic_type: str
    metadata: Dict[str, any] = None
    
    def to_numpy(self):
        """Convert to numpy array (if numpy available)."""
        try:
            import numpy as np
            return np.array(self.features, dtype=np.float32)
        except ImportError:
            return self.features


def _square_to_index(square: chess.Square) -> int:
    """Convert square to 0-63 index."""
    return square


def _get_piece_value(piece_type: chess.PieceType) -> float:
    """Get piece value for encoding."""
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 100.0,
    }
    return values.get(piece_type, 0.0)


def _encode_board_pieces(board: chess.Board) -> List[float]:
    """
    Encode piece positions on board.
    
    Returns 128 features: 64 for white pieces (value or 0), 64 for black pieces.
    """
    features = [0.0] * 128
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = _get_piece_value(piece.piece_type)
            if piece.color == chess.WHITE:
                features[square] = value
            else:
                features[64 + square] = value
    
    return features


def _encode_attacks(board: chess.Board) -> List[float]:
    """
    Encode attack patterns.
    
    Returns 64 features: for each square, net attack value 
    (positive = white attacking, negative = black attacking)
    """
    features = [0.0] * 64
    
    for square in chess.SQUARES:
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))
        features[square] = (white_attackers - black_attackers) / 5.0  # Normalize
    
    return features


def _encode_king_safety(board: chess.Board, color: bool) -> List[float]:
    """
    Encode king safety features for one side.
    
    Returns 16 features about king position and surrounding squares.
    """
    features = [0.0] * 16
    king_sq = board.king(color)
    
    if king_sq is None:
        return features
    
    # King position encoding (rank and file)
    features[0] = chess.square_rank(king_sq) / 7.0
    features[1] = chess.square_file(king_sq) / 7.0
    
    # Is king on back rank?
    back_rank = 0 if color == chess.WHITE else 7
    features[2] = 1.0 if chess.square_rank(king_sq) == back_rank else 0.0
    
    # Count escape squares
    escape_count = 0
    adjacent_squares = board.attacks(king_sq)
    for sq in adjacent_squares:
        if board.piece_at(sq) is None or board.piece_at(sq).color != color:
            # Check if square is safe
            if not board.is_attacked_by(not color, sq):
                escape_count += 1
    features[3] = escape_count / 8.0
    
    # Surrounding pieces (own and enemy)
    own_pieces_around = 0
    enemy_pieces_around = 0
    for sq in adjacent_squares:
        piece = board.piece_at(sq)
        if piece:
            if piece.color == color:
                own_pieces_around += 1
            else:
                enemy_pieces_around += 1
    features[4] = own_pieces_around / 8.0
    features[5] = enemy_pieces_around / 8.0
    
    # Is king in check?
    features[6] = 1.0 if board.is_check() and board.turn == color else 0.0
    
    # Attackers on king
    attackers = len(board.attackers(not color, king_sq))
    features[7] = min(1.0, attackers / 3.0)
    
    # Heavy pieces attacking king zone
    heavy_attackers = 0
    king_zone = [king_sq] + list(adjacent_squares)
    for sq in king_zone:
        for attacker_sq in board.attackers(not color, sq):
            piece = board.piece_at(attacker_sq)
            if piece and piece.piece_type in [chess.QUEEN, chess.ROOK]:
                heavy_attackers += 1
    features[8] = min(1.0, heavy_attackers / 4.0)
    
    return features


# === Tactic-Specific Feature Extraction ===

def extract_back_rank_features(board: chess.Board) -> TacticsFeatureVector:
    """
    Extract features for back rank mate detection.
    
    Key patterns:
    - Enemy king on back rank
    - Our heavy pieces (Q/R) on adjacent files or same rank
    - Blocked escape squares (pawns, own pieces)
    """
    features = []
    
    # Basic board encoding
    features.extend(_encode_board_pieces(board))  # 128 features
    features.extend(_encode_attacks(board))  # 64 features
    
    # King safety for defender (opponent)
    opponent = not board.turn
    features.extend(_encode_king_safety(board, opponent))  # 16 features
    
    # Back rank specific features
    back_rank_features = [0.0] * 16
    enemy_king = board.king(opponent)
    
    if enemy_king is not None:
        back_rank = 7 if opponent == chess.WHITE else 0
        king_rank = chess.square_rank(enemy_king)
        
        # Is enemy king on back rank?
        back_rank_features[0] = 1.0 if king_rank == back_rank else 0.0
        
        # Heavy pieces on same rank as enemy king
        our_heavy_on_rank = 0
        for sq in chess.SQUARES:
            if chess.square_rank(sq) == king_rank:
                piece = board.piece_at(sq)
                if piece and piece.color == board.turn:
                    if piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        our_heavy_on_rank += 1
        back_rank_features[1] = min(1.0, our_heavy_on_rank / 2.0)
        
        # Pawns blocking king escape
        pawns_blocking = 0
        for file in range(max(0, chess.square_file(enemy_king) - 1), 
                         min(8, chess.square_file(enemy_king) + 2)):
            sq = chess.square(file, back_rank + (1 if opponent == chess.WHITE else -1))
            if 0 <= sq < 64:
                piece = board.piece_at(sq)
                if piece and piece.color == opponent and piece.piece_type == chess.PAWN:
                    pawns_blocking += 1
        back_rank_features[2] = pawns_blocking / 3.0
        
        # Open file to king
        king_file = chess.square_file(enemy_king)
        open_file = True
        for rank in range(8):
            sq = chess.square(king_file, rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                open_file = False
                break
        back_rank_features[3] = 1.0 if open_file else 0.0
    
    features.extend(back_rank_features)
    
    # Pad to FEATURE_COUNT
    while len(features) < FEATURE_COUNT:
        features.append(0.0)
    
    return TacticsFeatureVector(
        features=features[:FEATURE_COUNT],
        tactic_type="backRankMate",
    )


def extract_double_check_features(board: chess.Board) -> TacticsFeatureVector:
    """
    Extract features for double check detection.
    
    Key patterns:
    - Discovered check potential (piece can move to reveal attack)
    - Knight positioned to give check
    - Multiple attacking lines to king
    """
    features = []
    
    # Basic board encoding
    features.extend(_encode_board_pieces(board))  # 128 features
    features.extend(_encode_attacks(board))  # 64 features
    
    # King safety for defender
    opponent = not board.turn
    features.extend(_encode_king_safety(board, opponent))  # 16 features
    
    # Double check specific features
    dc_features = [0.0] * 16
    enemy_king = board.king(opponent)
    our_king = board.king(board.turn)
    
    if enemy_king is not None:
        # Pieces that could give discovered check
        discovered_potential = 0
        
        # Check each of our pieces for discovered attack potential
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                # Is this piece blocking an attack on enemy king?
                # Remove piece and check if king is attacked
                board_copy = board.copy()
                board_copy.remove_piece_at(sq)
                if board_copy.is_attacked_by(board.turn, enemy_king):
                    discovered_potential += 1
        
        dc_features[0] = min(1.0, discovered_potential / 3.0)
        
        # Knights that can check
        knights_checking = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn and piece.piece_type == chess.KNIGHT:
                # Can this knight reach a checking square?
                for move in board.legal_moves:
                    if move.from_square == sq:
                        board_copy = board.copy()
                        board_copy.push(move)
                        if board_copy.is_check():
                            knights_checking += 1
                            break
        dc_features[1] = min(1.0, knights_checking / 2.0)
        
        # Bishop/Rook alignment with king
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                if piece.piece_type == chess.BISHOP:
                    # Check diagonal alignment
                    if abs(chess.square_file(sq) - chess.square_file(enemy_king)) == \
                       abs(chess.square_rank(sq) - chess.square_rank(enemy_king)):
                        dc_features[2] += 0.25
                elif piece.piece_type == chess.ROOK:
                    # Check file/rank alignment
                    if chess.square_file(sq) == chess.square_file(enemy_king) or \
                       chess.square_rank(sq) == chess.square_rank(enemy_king):
                        dc_features[3] += 0.25
    
    features.extend(dc_features)
    
    # Pad to FEATURE_COUNT
    while len(features) < FEATURE_COUNT:
        features.append(0.0)
    
    return TacticsFeatureVector(
        features=features[:FEATURE_COUNT],
        tactic_type="doubleCheck",
    )


def extract_smothered_mate_features(board: chess.Board) -> TacticsFeatureVector:
    """
    Extract features for smothered mate detection.
    
    Key patterns:
    - Enemy king surrounded by own pieces (trapped)
    - Our knight in position to deliver mate
    - Queen sacrifice potential to force smothered position
    """
    features = []
    
    # Basic board encoding
    features.extend(_encode_board_pieces(board))  # 128 features
    features.extend(_encode_attacks(board))  # 64 features
    
    # King safety for defender
    opponent = not board.turn
    features.extend(_encode_king_safety(board, opponent))  # 16 features
    
    # Smothered mate specific features
    sm_features = [0.0] * 16
    enemy_king = board.king(opponent)
    
    if enemy_king is not None:
        # How trapped is the enemy king?
        adjacent = list(board.attacks(enemy_king))
        blocked_by_own = 0
        total_adjacent = len(adjacent)
        
        for sq in adjacent:
            piece = board.piece_at(sq)
            if piece and piece.color == opponent:
                blocked_by_own += 1
        
        sm_features[0] = blocked_by_own / max(1, total_adjacent)  # Smother ratio
        
        # King in corner?
        corners = [chess.A1, chess.H1, chess.A8, chess.H8]
        sm_features[1] = 1.0 if enemy_king in corners else 0.0
        
        # Our knights near enemy king
        knights_near_king = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn and piece.piece_type == chess.KNIGHT:
                # Knight distance to king
                dist = abs(chess.square_file(sq) - chess.square_file(enemy_king)) + \
                       abs(chess.square_rank(sq) - chess.square_rank(enemy_king))
                if dist <= 3:
                    knights_near_king += 1
        sm_features[2] = min(1.0, knights_near_king / 2.0)
        
        # Can any knight give check?
        knight_checks = 0
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KNIGHT:
                board_copy = board.copy()
                board_copy.push(move)
                if board_copy.is_check():
                    knight_checks += 1
        sm_features[3] = min(1.0, knight_checks / 3.0)
        
        # Queen available for sacrifice?
        has_queen = any(
            board.piece_at(sq) and board.piece_at(sq).piece_type == chess.QUEEN 
            and board.piece_at(sq).color == board.turn
            for sq in chess.SQUARES
        )
        sm_features[4] = 1.0 if has_queen else 0.0
    
    features.extend(sm_features)
    
    # Pad to FEATURE_COUNT
    while len(features) < FEATURE_COUNT:
        features.append(0.0)
    
    return TacticsFeatureVector(
        features=features[:FEATURE_COUNT],
        tactic_type="smotheredMate",
    )


def extract_tactics_features(
    board: chess.Board,
    tactic_type: str,
) -> TacticsFeatureVector:
    """
    Extract features for a specific tactic type.
    
    Args:
        board: Current board position
        tactic_type: One of "backRankMate", "doubleCheck", "smotheredMate"
        
    Returns:
        TacticsFeatureVector with extracted features
    """
    extractors = {
        "backRankMate": extract_back_rank_features,
        "doubleCheck": extract_double_check_features,
        "smotheredMate": extract_smothered_mate_features,
    }
    
    extractor = extractors.get(tactic_type)
    if extractor is None:
        raise ValueError(f"Unknown tactic type: {tactic_type}. "
                        f"Supported: {list(extractors.keys())}")
    
    return extractor(board)

