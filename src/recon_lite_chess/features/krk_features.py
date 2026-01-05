"""
KRK-specific feature extraction for stem cell pattern matching.

These features drive the M5 structural learning for King+Rook vs King endgames.
Features are divided into:
- UNIVERSAL: Transfer directly from KPK (king proximity, opposition)
- KRK-SPECIFIC: Rook-based features (box area, fence distance)
"""

import chess
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class KRKFeatures:
    """Feature vector for KRK position analysis."""
    # Universal features (shared with KPK)
    king_distance: int  # Chebyshev distance between kings
    opposition_status: int  # 1 if we have opposition, 0 otherwise
    enemy_king_edge_distance: int  # Distance from enemy king to nearest edge
    
    # KRK-specific features
    box_area: int  # Confinement box area (smaller = better)
    box_min_side: int  # Minimum side of confinement box
    rook_fence_distance: int  # Distance from rook to target fence line
    cut_established: int  # 1 if stable cut exists, 0 otherwise
    rook_safe: int  # 1 if rook is safe, 0 otherwise
    
    # Derived features for pattern matching
    king_rook_distance: int  # Distance between our king and rook
    can_mate_now: int  # 1 if mate in one exists, 0 otherwise
    stalemate_danger: int  # 1 if near stalemate, 0 otherwise
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "king_distance": self.king_distance,
            "opposition_status": self.opposition_status,
            "enemy_king_edge_distance": self.enemy_king_edge_distance,
            "box_area": self.box_area,
            "box_min_side": self.box_min_side,
            "rook_fence_distance": self.rook_fence_distance,
            "cut_established": self.cut_established,
            "rook_safe": self.rook_safe,
            "king_rook_distance": self.king_rook_distance,
            "can_mate_now": self.can_mate_now,
            "stalemate_danger": self.stalemate_danger,
        }
    
    def to_vector(self) -> list:
        """Convert to numeric vector for pattern matching."""
        return [
            self.king_distance,
            self.opposition_status,
            self.enemy_king_edge_distance,
            self.box_area,
            self.box_min_side,
            self.rook_fence_distance,
            self.cut_established,
            self.rook_safe,
            self.king_rook_distance,
            self.can_mate_now,
            self.stalemate_danger,
        ]
    
    @staticmethod
    def feature_names() -> list:
        """Return ordered list of feature names."""
        return [
            "king_distance",
            "opposition_status", 
            "enemy_king_edge_distance",
            "box_area",
            "box_min_side",
            "rook_fence_distance",
            "cut_established",
            "rook_safe",
            "king_rook_distance",
            "can_mate_now",
            "stalemate_danger",
        ]
    
    @staticmethod
    def universal_feature_indices() -> list:
        """Indices of features that transfer from KPK."""
        return [0, 1, 2]  # king_distance, opposition_status, enemy_king_edge_distance


def extract_krk_features(board: chess.Board) -> KRKFeatures:
    """
    Extract KRK features from a chess position.
    
    These features are used by stem cells to learn patterns
    and can be matched against KPK-learned patterns for transfer.
    
    Args:
        board: Chess board in KRK position (White has King+Rook, Black has King)
        
    Returns:
        KRKFeatures dataclass with all feature values
    """
    from ..predicates import (
        chebyshev, dist_to_edge, box_area, box_min_side,
        rook_distance_to_target_fence, has_stable_cut, rook_safe_now,
        king_to_rook_distance, has_opposition, can_deliver_mate,
    )
    
    color = board.turn
    our_king = board.king(color)
    enemy_king = board.king(not color)
    
    # Handle missing kings gracefully
    if our_king is None or enemy_king is None:
        return KRKFeatures(
            king_distance=8,
            opposition_status=0,
            enemy_king_edge_distance=4,
            box_area=64,
            box_min_side=8,
            rook_fence_distance=8,
            cut_established=0,
            rook_safe=0,
            king_rook_distance=8,
            can_mate_now=0,
            stalemate_danger=0,
        )
    
    # Universal features
    king_distance = chebyshev(our_king, enemy_king)
    opposition_status = 1 if has_opposition(board) else 0
    enemy_king_edge_distance = dist_to_edge(enemy_king)
    
    # KRK-specific features
    current_box_area = box_area(board)
    current_box_min_side = box_min_side(board)
    fence_distance = rook_distance_to_target_fence(board)
    cut_est = 1 if has_stable_cut(board) else 0
    rook_safe = 1 if rook_safe_now(board) else 0
    
    # Derived features
    kr_distance = king_to_rook_distance(board)
    mate_possible = 1 if can_deliver_mate(board) else 0
    stalemate_risk = 1 if board.is_stalemate() else 0
    
    return KRKFeatures(
        king_distance=king_distance,
        opposition_status=opposition_status,
        enemy_king_edge_distance=enemy_king_edge_distance,
        box_area=current_box_area,
        box_min_side=current_box_min_side,
        rook_fence_distance=fence_distance,
        cut_established=cut_est,
        rook_safe=rook_safe,
        king_rook_distance=kr_distance,
        can_mate_now=mate_possible,
        stalemate_danger=stalemate_risk,
    )


def extract_krk_feature_dict(board: chess.Board) -> Dict[str, Any]:
    """
    Extract KRK features as a dictionary.
    
    Convenience wrapper for integration with existing systems.
    """
    return extract_krk_features(board).to_dict()


def krk_feature_similarity(features_a: KRKFeatures, features_b: KRKFeatures) -> float:
    """
    Compute similarity between two KRK feature vectors.
    
    Used for stem cell pattern matching during knowledge transfer.
    Returns value 0.0 to 1.0 (1.0 = identical).
    """
    vec_a = features_a.to_vector()
    vec_b = features_b.to_vector()
    
    # Normalize each feature and compute cosine similarity
    total_diff = 0.0
    max_diff = 0.0
    
    # Feature-specific normalization factors
    norm_factors = [
        8,   # king_distance (0-7)
        1,   # opposition_status (0-1)
        4,   # enemy_king_edge_distance (0-3)
        64,  # box_area (1-64)
        8,   # box_min_side (1-8)
        8,   # rook_fence_distance (0-8)
        1,   # cut_established (0-1)
        1,   # rook_safe (0-1)
        8,   # king_rook_distance (0-7)
        1,   # can_mate_now (0-1)
        1,   # stalemate_danger (0-1)
    ]
    
    for a, b, norm in zip(vec_a, vec_b, norm_factors):
        diff = abs(a - b) / max(norm, 1)
        total_diff += diff
        max_diff += 1.0
    
    if max_diff == 0:
        return 1.0
    
    return 1.0 - (total_diff / max_diff)


def universal_feature_match(krk_features: KRKFeatures, kpk_pattern: Dict[str, Any]) -> float:
    """
    Check if KRK features match a KPK-learned pattern on universal features.
    
    Used for knowledge transfer: if a KPK stem cell fires on certain
    king_distance/opposition patterns, we check if those same patterns
    appear in KRK positions.
    
    Args:
        krk_features: Current KRK position features
        kpk_pattern: Pattern from a KPK stem cell (stored signature)
        
    Returns:
        Match score 0.0 to 1.0
    """
    universal_indices = KRKFeatures.universal_feature_indices()
    feature_names = KRKFeatures.feature_names()
    
    match_score = 0.0
    count = 0
    
    for idx in universal_indices:
        name = feature_names[idx]
        if name in kpk_pattern:
            krk_val = krk_features.to_vector()[idx]
            kpk_val = kpk_pattern[name]
            
            # Fuzzy match: within 1 step = partial match
            diff = abs(krk_val - kpk_val)
            if diff == 0:
                match_score += 1.0
            elif diff == 1:
                match_score += 0.5
            count += 1
    
    if count == 0:
        return 0.0
    
    return match_score / count


# Feature extraction for move evaluation
def extract_move_features(board: chess.Board, move: chess.Move) -> Dict[str, Any]:
    """
    Extract features before and after a move for learning.
    
    Returns dict with 'before', 'after', and 'delta' feature sets.
    """
    from ..predicates import move_features as base_move_features
    
    # Base move features from predicates
    base_features = base_move_features(board, move)
    
    # KRK features before
    before = extract_krk_features(board)
    
    # KRK features after
    board_copy = board.copy()
    board_copy.push(move)
    after = extract_krk_features(board_copy)
    
    # Compute deltas
    delta = {
        "king_distance_delta": after.king_distance - before.king_distance,
        "box_area_delta": after.box_area - before.box_area,
        "box_min_side_delta": after.box_min_side - before.box_min_side,
        "edge_distance_delta": after.enemy_king_edge_distance - before.enemy_king_edge_distance,
        "fence_distance_delta": after.rook_fence_distance - before.rook_fence_distance,
        "cut_gained": 1 if (after.cut_established and not before.cut_established) else 0,
        "opposition_gained": 1 if (after.opposition_status and not before.opposition_status) else 0,
    }
    
    return {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "delta": delta,
        "base": base_features,
    }

