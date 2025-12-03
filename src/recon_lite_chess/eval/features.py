"""Feature extraction for distillation (M7).

Extracts a fixed-size feature vector from a chess position
for training the distilled evaluation model.

Features include:
- Material counts (12 features: 6 piece types × 2 colors)
- Piece positions (compressed, not full bitboards)
- Pawn structure (isolated, doubled, passed pawns)
- King safety (attacks near king, pawn shield)
- Mobility (legal moves count approximation)
- Phase indicators

Total: ~150 features, designed for lightweight model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import math

import chess


# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass
class FeatureVector:
    """Feature vector for a chess position."""
    features: List[float]
    feature_names: List[str]
    
    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.features))
    
    def __len__(self) -> int:
        return len(self.features)


def extract_material_features(board: chess.Board) -> List[float]:
    """Extract material count features (12 features)."""
    features = []
    
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(piece_type, color))
            # Normalize: max 8 pawns, 2 knights, 2 bishops, 2 rooks, 1 queen
            max_counts = {
                chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
                chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1,
            }
            features.append(count / max_counts[piece_type])
    
    return features


def extract_material_balance_features(board: chess.Board) -> List[float]:
    """Extract material balance features (6 features)."""
    features = []
    
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                      chess.ROOK, chess.QUEEN, chess.KING]:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        # Balance: positive = white ahead
        balance = (white_count - black_count) / 10.0  # Normalize
        features.append(balance)
    
    # Total material balance
    white_total = sum(
        len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt]
        for pt in chess.PIECE_TYPES if pt != chess.KING
    )
    black_total = sum(
        len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]
        for pt in chess.PIECE_TYPES if pt != chess.KING
    )
    features.append((white_total - black_total) / 40.0)  # Normalize
    
    return features


def extract_piece_position_features(board: chess.Board) -> List[float]:
    """Extract piece position features (compressed).
    
    For each piece type, encode average file and rank.
    32 features: 8 piece types × 2 colors × 2 (avg_file, avg_rank)
    """
    features = []
    
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            squares = list(board.pieces(piece_type, color))
            
            if squares:
                avg_file = sum(chess.square_file(sq) for sq in squares) / len(squares)
                avg_rank = sum(chess.square_rank(sq) for sq in squares) / len(squares)
                # Normalize to [0, 1]
                features.append(avg_file / 7.0)
                features.append(avg_rank / 7.0)
            else:
                features.append(0.5)  # Center as default
                features.append(0.5)
    
    return features


def extract_king_position_features(board: chess.Board) -> List[float]:
    """Extract king position features (8 features)."""
    features = []
    
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            features.extend([0.5, 0.5, 0.0, 0.0])
            continue
        
        # King file and rank (normalized)
        features.append(chess.square_file(king_sq) / 7.0)
        features.append(chess.square_rank(king_sq) / 7.0)
        
        # Distance from center
        center_dist = abs(3.5 - chess.square_file(king_sq)) + abs(3.5 - chess.square_rank(king_sq))
        features.append(center_dist / 7.0)
        
        # Has castled (heuristic: king on g or c file, not starting square)
        start_sq = chess.E1 if color else chess.E8
        castled_sqs = [chess.G1, chess.C1] if color else [chess.G8, chess.C8]
        has_castled = 1.0 if king_sq in castled_sqs else 0.0
        features.append(has_castled)
    
    return features


def extract_pawn_structure_features(board: chess.Board) -> List[float]:
    """Extract pawn structure features (12 features)."""
    features = []
    
    for color in [chess.WHITE, chess.BLACK]:
        pawns = list(board.pieces(chess.PAWN, color))
        
        # Count pawns per file
        file_counts = [0] * 8
        for sq in pawns:
            file_counts[chess.square_file(sq)] += 1
        
        # Doubled pawns (files with 2+ pawns)
        doubled = sum(1 for c in file_counts if c >= 2)
        features.append(doubled / 4.0)
        
        # Isolated pawns (no friendly pawns on adjacent files)
        isolated = 0
        for f in range(8):
            if file_counts[f] > 0:
                has_neighbor = False
                if f > 0 and file_counts[f-1] > 0:
                    has_neighbor = True
                if f < 7 and file_counts[f+1] > 0:
                    has_neighbor = True
                if not has_neighbor:
                    isolated += file_counts[f]
        features.append(isolated / 8.0)
        
        # Passed pawns (no enemy pawns ahead on same or adjacent files)
        passed = 0
        enemy_pawns = list(board.pieces(chess.PAWN, not color))
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            
            is_passed = True
            for enemy_sq in enemy_pawns:
                ef = chess.square_file(enemy_sq)
                er = chess.square_rank(enemy_sq)
                
                if abs(ef - f) <= 1:
                    if color and er > r:  # White pawn, enemy ahead
                        is_passed = False
                        break
                    elif not color and er < r:  # Black pawn, enemy ahead
                        is_passed = False
                        break
            
            if is_passed:
                passed += 1
        
        features.append(passed / 8.0)
        
        # Pawn chain length (connected pawns)
        connected = 0
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            # Check diagonally adjacent pawns
            for df in [-1, 1]:
                for dr in [-1, 1]:
                    nf, nr = f + df, r + dr
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        nsq = chess.square(nf, nr)
                        piece = board.piece_at(nsq)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            connected += 1
                            break
        features.append(connected / 8.0)
        
        # Central pawns (on d/e files)
        central = sum(1 for sq in pawns if chess.square_file(sq) in [3, 4])
        features.append(central / 4.0)
        
        # Advanced pawns
        if color:  # White
            advanced = sum(1 for sq in pawns if chess.square_rank(sq) >= 4)
        else:  # Black
            advanced = sum(1 for sq in pawns if chess.square_rank(sq) <= 3)
        features.append(advanced / 8.0)
    
    return features


def extract_king_safety_features(board: chess.Board) -> List[float]:
    """Extract king safety features (8 features)."""
    features = []
    
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        
        # Pawn shield (pawns in front of king)
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        shield_rank = king_rank + 1 if color else king_rank - 1
        
        shield_pawns = 0
        if 0 <= shield_rank <= 7:
            for f in [max(0, king_file-1), king_file, min(7, king_file+1)]:
                sq = chess.square(f, shield_rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_pawns += 1
        features.append(shield_pawns / 3.0)
        
        # Attackers in king zone
        king_zone = []
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                f = king_file + df
                r = king_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    king_zone.append(chess.square(f, r))
        
        attackers = 0
        for sq in king_zone:
            attackers += len(board.attackers(not color, sq))
        features.append(min(1.0, attackers / 10.0))
        
        # Open files near king
        open_files = 0
        for f in [max(0, king_file-1), king_file, min(7, king_file+1)]:
            has_pawn = False
            for r in range(8):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
        features.append(open_files / 3.0)
        
        # King in check
        features.append(1.0 if board.is_check() and board.turn == color else 0.0)
    
    return features


def extract_mobility_features(board: chess.Board) -> List[float]:
    """Extract mobility features (4 features)."""
    features = []
    
    # Legal moves count (approximation - actual count is expensive)
    if board.turn:
        white_moves = len(list(board.legal_moves))
        board.turn = False
        black_moves = len(list(board.legal_moves))
        board.turn = True
    else:
        black_moves = len(list(board.legal_moves))
        board.turn = True
        white_moves = len(list(board.legal_moves))
        board.turn = False
    
    features.append(white_moves / 40.0)
    features.append(black_moves / 40.0)
    features.append((white_moves - black_moves) / 40.0)
    
    # Center control (attacks on e4, d4, e5, d5)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_center = sum(len(board.attackers(chess.WHITE, sq)) for sq in center_squares)
    black_center = sum(len(board.attackers(chess.BLACK, sq)) for sq in center_squares)
    features.append((white_center - black_center) / 16.0)
    
    return features


def extract_phase_features(board: chess.Board) -> List[float]:
    """Extract phase indicator features (4 features)."""
    features = []
    
    # Total material (excluding kings)
    total_material = sum(
        len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt] +
        len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]
        for pt in chess.PIECE_TYPES if pt != chess.KING
    )
    
    # Max material is about 78 (2×39)
    features.append(total_material / 78.0)
    
    # Queen presence
    has_queens = (len(board.pieces(chess.QUEEN, chess.WHITE)) + 
                  len(board.pieces(chess.QUEEN, chess.BLACK)))
    features.append(has_queens / 2.0)
    
    # Minor pieces remaining
    minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
              len(board.pieces(chess.KNIGHT, chess.BLACK)) +
              len(board.pieces(chess.BISHOP, chess.WHITE)) + 
              len(board.pieces(chess.BISHOP, chess.BLACK)))
    features.append(minors / 8.0)
    
    # Move number (normalized, caps at 50)
    features.append(min(board.fullmove_number, 50) / 50.0)
    
    return features


def extract_tactical_features(board: chess.Board) -> List[float]:
    """Extract tactical features (6 features)."""
    features = []
    
    # Hanging pieces (undefended pieces under attack)
    for color in [chess.WHITE, chess.BLACK]:
        hanging = 0
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(pt, color):
                attackers = board.attackers(not color, sq)
                defenders = board.attackers(color, sq)
                if attackers and not defenders:
                    hanging += PIECE_VALUES[pt]
        features.append(hanging / 10.0)
    
    # Pieces giving check possibilities (simplified)
    check_threats_white = 0
    check_threats_black = 0
    
    # Just check if any piece can reach near enemy king
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if black_king:
        for sq in board.pieces(chess.QUEEN, chess.WHITE):
            if chess.square_distance(sq, black_king) <= 2:
                check_threats_white += 1
        for sq in board.pieces(chess.ROOK, chess.WHITE):
            if (chess.square_file(sq) == chess.square_file(black_king) or
                chess.square_rank(sq) == chess.square_rank(black_king)):
                check_threats_white += 1
    
    if white_king:
        for sq in board.pieces(chess.QUEEN, chess.BLACK):
            if chess.square_distance(sq, white_king) <= 2:
                check_threats_black += 1
        for sq in board.pieces(chess.ROOK, chess.BLACK):
            if (chess.square_file(sq) == chess.square_file(white_king) or
                chess.square_rank(sq) == chess.square_rank(white_king)):
                check_threats_black += 1
    
    features.append(check_threats_white / 4.0)
    features.append(check_threats_black / 4.0)
    
    # Is position in check
    features.append(1.0 if board.is_check() else 0.0)
    
    # Side to move
    features.append(1.0 if board.turn else 0.0)
    
    return features


def extract_features(board: chess.Board) -> FeatureVector:
    """Extract full feature vector from a chess position.
    
    Args:
        board: Chess position to analyze
        
    Returns:
        FeatureVector with all features
    """
    features: List[float] = []
    names: List[str] = []
    
    # Material counts (12)
    mat = extract_material_features(board)
    features.extend(mat)
    for color in ["w", "b"]:
        for pt in ["P", "N", "B", "R", "Q", "K"]:
            names.append(f"mat_{color}_{pt}")
    
    # Material balance (7)
    bal = extract_material_balance_features(board)
    features.extend(bal)
    for pt in ["P", "N", "B", "R", "Q", "K"]:
        names.append(f"bal_{pt}")
    names.append("bal_total")
    
    # Piece positions (16)
    pos = extract_piece_position_features(board)
    features.extend(pos)
    for color in ["w", "b"]:
        for pt in ["P", "N", "B", "R"]:
            names.append(f"pos_{color}_{pt}_file")
            names.append(f"pos_{color}_{pt}_rank")
    
    # King positions (8)
    king = extract_king_position_features(board)
    features.extend(king)
    for color in ["w", "b"]:
        names.extend([f"king_{color}_file", f"king_{color}_rank",
                     f"king_{color}_center_dist", f"king_{color}_castled"])
    
    # Pawn structure (12)
    pawn = extract_pawn_structure_features(board)
    features.extend(pawn)
    for color in ["w", "b"]:
        names.extend([f"pawn_{color}_doubled", f"pawn_{color}_isolated",
                     f"pawn_{color}_passed", f"pawn_{color}_connected",
                     f"pawn_{color}_central", f"pawn_{color}_advanced"])
    
    # King safety (8)
    safety = extract_king_safety_features(board)
    features.extend(safety)
    for color in ["w", "b"]:
        names.extend([f"safety_{color}_shield", f"safety_{color}_attackers",
                     f"safety_{color}_open_files", f"safety_{color}_check"])
    
    # Mobility (4)
    mob = extract_mobility_features(board)
    features.extend(mob)
    names.extend(["mob_white", "mob_black", "mob_diff", "mob_center"])
    
    # Phase (4)
    phase = extract_phase_features(board)
    features.extend(phase)
    names.extend(["phase_material", "phase_queens", "phase_minors", "phase_move"])
    
    # Tactical (6)
    tact = extract_tactical_features(board)
    features.extend(tact)
    names.extend(["tact_hanging_w", "tact_hanging_b", "tact_check_w", 
                 "tact_check_b", "tact_in_check", "tact_turn"])
    
    return FeatureVector(features=features, feature_names=names)


def features_to_tensor(feature_vector: FeatureVector):
    """Convert feature vector to numpy array or tensor.
    
    Returns numpy array if numpy is available, else list.
    """
    try:
        import numpy as np
        return np.array(feature_vector.features, dtype=np.float32)
    except ImportError:
        return feature_vector.features


def batch_extract_features(boards: List[chess.Board]) -> List[FeatureVector]:
    """Extract features from multiple boards."""
    return [extract_features(board) for board in boards]


# Feature count for model configuration
FEATURE_COUNT = 77  # Update if adding/removing features

