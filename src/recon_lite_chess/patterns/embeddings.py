"""M9.1: Position Embedding Model.

Encodes chess board positions into fixed-size vectors for pattern matching.
Reuses features from M7 distillation where possible.

Usage:
    from recon_lite_chess.patterns import encode_position, PositionEncoder
    
    # Simple encoding
    embedding = encode_position(board)
    
    # With trained encoder
    encoder = PositionEncoder.load("weights/position_encoder.pt")
    embedding = encoder.encode(board)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json
import numpy as np
import chess

# Reuse M7 features for base encoding
try:
    from recon_lite_chess.eval.features import extract_features, FeatureVector
    HAS_M7_FEATURES = True
except ImportError:
    HAS_M7_FEATURES = False


@dataclass
class PositionEmbedding:
    """A fixed-size vector representation of a chess position."""
    vector: np.ndarray
    fen: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def dim(self) -> int:
        return len(self.vector)
    
    def cosine_similarity(self, other: "PositionEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        dot = np.dot(self.vector, other.vector)
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self == 0 or norm_other == 0:
            return 0.0
        return float(dot / (norm_self * norm_other))
    
    def euclidean_distance(self, other: "PositionEmbedding") -> float:
        """Compute Euclidean distance to another embedding."""
        return float(np.linalg.norm(self.vector - other.vector))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.tolist(),
            "fen": self.fen,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionEmbedding":
        return cls(
            vector=np.array(data["vector"]),
            fen=data["fen"],
            metadata=data.get("metadata", {}),
        )


def _piece_plane(board: chess.Board, piece_type: chess.PieceType, color: chess.Color) -> np.ndarray:
    """Create 64-element binary plane for a piece type and color."""
    plane = np.zeros(64, dtype=np.float32)
    for sq in board.pieces(piece_type, color):
        plane[sq] = 1.0
    return plane


def _encode_board_planes(board: chess.Board) -> np.ndarray:
    """
    Encode board as piece planes (12 x 64 = 768 features).
    Standard representation used by neural chess engines.
    """
    planes = []
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]:
            planes.append(_piece_plane(board, piece_type, color))
    return np.concatenate(planes)


def _encode_additional_features(board: chess.Board) -> np.ndarray:
    """
    Encode additional positional features beyond piece placement.
    """
    features = []
    
    # Castling rights (4)
    features.extend([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ])
    
    # En passant (1 if available, 0 otherwise)
    features.append(1.0 if board.ep_square is not None else 0.0)
    
    # Turn (1 = white, 0 = black)
    features.append(1.0 if board.turn == chess.WHITE else 0.0)
    
    # Material balance (normalized)
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values() if p.color == chess.WHITE
    )
    black_material = sum(
        piece_values.get(p.piece_type, 0) 
        for p in board.piece_map().values() if p.color == chess.BLACK
    )
    balance = (white_material - black_material) / 40.0  # Normalize by max material
    features.append(balance)
    
    # Game phase (0 = endgame, 1 = opening)
    total_material = white_material + black_material
    phase = min(1.0, total_material / 60.0)
    features.append(phase)
    
    # King positions (normalized 0-7)
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is not None:
        features.append(chess.square_file(white_king) / 7.0)
        features.append(chess.square_rank(white_king) / 7.0)
    else:
        features.extend([0.0, 0.0])
    if black_king is not None:
        features.append(chess.square_file(black_king) / 7.0)
        features.append(chess.square_rank(black_king) / 7.0)
    else:
        features.extend([0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


def encode_position(
    board: chess.Board,
    include_planes: bool = True,
    include_features: bool = True,
    use_m7_features: bool = True,
) -> PositionEmbedding:
    """
    Encode a chess position into a fixed-size embedding.
    
    Args:
        board: Chess board to encode
        include_planes: Include 12x64 piece planes
        include_features: Include additional features
        use_m7_features: Use M7 feature extraction if available
        
    Returns:
        PositionEmbedding with the encoded vector
    """
    parts = []
    
    # Use M7 features if available (preferred - already optimized)
    if use_m7_features and HAS_M7_FEATURES:
        fv = extract_features(board)
        parts.append(np.array(fv.values, dtype=np.float32))
    else:
        # Fall back to basic encoding
        if include_planes:
            parts.append(_encode_board_planes(board))
        if include_features:
            parts.append(_encode_additional_features(board))
    
    vector = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    
    return PositionEmbedding(
        vector=vector,
        fen=board.fen(),
        metadata={
            "encoding": "m7_features" if (use_m7_features and HAS_M7_FEATURES) else "basic",
            "dim": len(vector),
        },
    )


def encode_positions_batch(
    boards: List[chess.Board],
    **kwargs,
) -> List[PositionEmbedding]:
    """Encode multiple positions in batch."""
    return [encode_position(board, **kwargs) for board in boards]


class PositionEncoder:
    """
    Trained position encoder using a small MLP.
    
    Takes raw features and projects to a lower-dimensional embedding space
    suitable for similarity search.
    """
    
    def __init__(
        self,
        input_dim: int = 77,  # M7 feature dimension
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._model = None
        
        # Initialize simple linear projection (can be replaced with trained MLP)
        self._w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self._b1 = np.zeros(hidden_dim, dtype=np.float32)
        self._w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.01
        self._b2 = np.zeros(output_dim, dtype=np.float32)
    
    def encode(self, board: chess.Board) -> PositionEmbedding:
        """Encode a single position."""
        # Get raw features
        raw_embedding = encode_position(board, use_m7_features=True)
        
        # Project through MLP
        x = raw_embedding.vector
        
        # Pad or truncate to expected input dim
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        elif len(x) > self.input_dim:
            x = x[:self.input_dim]
        
        # Forward pass (ReLU activation)
        h = np.maximum(0, x @ self._w1 + self._b1)
        out = h @ self._w2 + self._b2
        
        # L2 normalize
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        
        return PositionEmbedding(
            vector=out,
            fen=board.fen(),
            metadata={"encoder": "mlp", "raw_dim": len(raw_embedding.vector)},
        )
    
    def encode_batch(self, boards: List[chess.Board]) -> List[PositionEmbedding]:
        """Encode multiple positions."""
        return [self.encode(board) for board in boards]
    
    def save(self, path: Path) -> None:
        """Save encoder weights to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "w1": self._w1.tolist(),
            "b1": self._b1.tolist(),
            "w2": self._w2.tolist(),
            "b2": self._b2.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "PositionEncoder":
        """Load encoder weights from file."""
        with open(path) as f:
            data = json.load(f)
        encoder = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            output_dim=data["output_dim"],
        )
        encoder._w1 = np.array(data["w1"], dtype=np.float32)
        encoder._b1 = np.array(data["b1"], dtype=np.float32)
        encoder._w2 = np.array(data["w2"], dtype=np.float32)
        encoder._b2 = np.array(data["b2"], dtype=np.float32)
        return encoder

