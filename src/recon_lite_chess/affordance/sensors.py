"""
Affordance Sensors for ReCoN Chess Subgraphs.

Computes continuous [0.0, 1.0] "distance to applicability" signals for each
subgraph. These signals enable the planning layer to sense when strategies
are becoming relevant, creating an implicit lookahead capability.

Key Concept:
    Unlike binary gates (is_applicable or not), affordance signals provide
    a gradient. A KRK affordance of 0.6 means "we're 60% of the way to a
    pure K+R vs K position". This allows the M3 Bandit to select actions
    that "climb the hill" toward a strategy even before it's fully active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import chess


@dataclass
class AffordanceConfig:
    """Configuration for affordance signal computation."""
    
    # Sigmoid steepness for distance-to-pattern curves
    sigmoid_k: float = 3.0
    
    # Thresholds for considering endgame relevant
    endgame_material_threshold: float = 15.0  # Total material
    
    # Minimum affordance to report (filter noise)
    min_affordance: float = 0.05


@dataclass
class AffordanceSignal:
    """
    Continuous affordance signal for a subgraph.
    
    Attributes:
        subgraph: Name of the subgraph (e.g., "krk", "kpk")
        value: Affordance value in [0.0, 1.0]
        components: Breakdown of contributing factors
        is_exact_match: True if this is the exact pattern (value should be 1.0)
    """
    subgraph: str
    value: float  # [0.0, 1.0]
    components: Dict[str, float] = field(default_factory=dict)
    is_exact_match: bool = False
    
    def __post_init__(self):
        # Clamp value to valid range
        self.value = max(0.0, min(1.0, self.value))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgraph": self.subgraph,
            "value": round(self.value, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "is_exact_match": self.is_exact_match,
        }


def _sigmoid(x: float, k: float = 3.0, midpoint: float = 0.5) -> float:
    """Sigmoid function for smooth transitions."""
    return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))


def _count_material(board: chess.Board) -> Dict[str, int]:
    """Count pieces by type and color."""
    counts = {
        "white_pawns": 0, "white_knights": 0, "white_bishops": 0,
        "white_rooks": 0, "white_queens": 0,
        "black_pawns": 0, "black_knights": 0, "black_bishops": 0,
        "black_rooks": 0, "black_queens": 0,
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.piece_type == chess.KING:
            continue
        
        color_prefix = "white" if piece.color == chess.WHITE else "black"
        piece_name = {
            chess.PAWN: "pawns", chess.KNIGHT: "knights", chess.BISHOP: "bishops",
            chess.ROOK: "rooks", chess.QUEEN: "queens",
        }.get(piece.piece_type)
        
        if piece_name:
            counts[f"{color_prefix}_{piece_name}"] += 1
    
    return counts


def _compute_material_values(counts: Dict[str, int]) -> Dict[str, float]:
    """Compute material values from piece counts."""
    piece_values = {
        "pawns": 1, "knights": 3, "bishops": 3, "rooks": 5, "queens": 9,
    }
    
    white_material = sum(
        counts[f"white_{pt}"] * pv for pt, pv in piece_values.items()
    )
    black_material = sum(
        counts[f"black_{pt}"] * pv for pt, pv in piece_values.items()
    )
    
    return {
        "white": white_material,
        "black": black_material,
        "total": white_material + black_material,
        "balance": white_material - black_material,
    }


def compute_krk_affordance(
    board: chess.Board,
    config: Optional[AffordanceConfig] = None,
) -> AffordanceSignal:
    """
    Compute KRK (King + Rook vs King) affordance signal.
    
    Returns high values when approaching K+R vs K:
    - 1.0: Exact KRK position
    - 0.7-0.9: One side has only K+R, other has 1-2 minor pieces
    - 0.4-0.6: Significant material advantage with rook, few pieces
    - 0.2-0.3: Material imbalance trending toward simplification
    - 0.0-0.1: Full material, no KRK relevance
    
    Components tracked:
    - rook_advantage: Do we have rook advantage?
    - piece_scarcity: How few pieces are on the board?
    - opponent_material: How little material does opponent have?
    - pawn_absence: Are pawns gone? (favors KRK over KPK)
    """
    config = config or AffordanceConfig()
    counts = _count_material(board)
    mat = _compute_material_values(counts)
    
    components: Dict[str, float] = {}
    
    # Check for exact KRK pattern (either side)
    white_is_krk = (
        counts["white_rooks"] == 1 and
        counts["white_queens"] == 0 and
        counts["white_bishops"] == 0 and
        counts["white_knights"] == 0 and
        counts["white_pawns"] == 0 and
        mat["black"] == 0
    )
    black_is_krk = (
        counts["black_rooks"] == 1 and
        counts["black_queens"] == 0 and
        counts["black_bishops"] == 0 and
        counts["black_knights"] == 0 and
        counts["black_pawns"] == 0 and
        mat["white"] == 0
    )
    
    if white_is_krk or black_is_krk:
        return AffordanceSignal(
            subgraph="krk",
            value=1.0,
            components={"exact_match": 1.0},
            is_exact_match=True,
        )
    
    # Component 1: Rook advantage
    # Having a rook while opponent doesn't is key for KRK
    rook_diff = (counts["white_rooks"] - counts["black_rooks"])
    has_rook = counts["white_rooks"] > 0 or counts["black_rooks"] > 0
    rook_advantage = _sigmoid(abs(rook_diff), k=2.0, midpoint=0.5) if has_rook else 0.0
    components["rook_advantage"] = rook_advantage
    
    # Component 2: Piece scarcity (fewer pieces = closer to endgame)
    total_pieces = sum(
        counts[f"{c}_{p}"] for c in ["white", "black"]
        for p in ["pawns", "knights", "bishops", "rooks", "queens"]
    )
    # Map 0-16 pieces to 1.0-0.0 scarcity
    piece_scarcity = max(0.0, 1.0 - total_pieces / 10.0)
    components["piece_scarcity"] = piece_scarcity
    
    # Component 3: Opponent material (less = closer to KRK)
    # Consider the side with less material as "opponent"
    opponent_mat = min(mat["white"], mat["black"])
    # 0 material = 1.0, 10+ material = 0.0
    opponent_weakness = max(0.0, 1.0 - opponent_mat / 8.0)
    components["opponent_weakness"] = opponent_weakness
    
    # Component 4: Pawn absence (KRK needs no pawns)
    total_pawns = counts["white_pawns"] + counts["black_pawns"]
    pawn_absence = max(0.0, 1.0 - total_pawns / 4.0)
    components["pawn_absence"] = pawn_absence
    
    # Component 5: Queen absence (KRK needs no queens)
    total_queens = counts["white_queens"] + counts["black_queens"]
    queen_absence = 1.0 if total_queens == 0 else 0.3
    components["queen_absence"] = queen_absence
    
    # Weighted combination
    weights = {
        "rook_advantage": 0.25,
        "piece_scarcity": 0.25,
        "opponent_weakness": 0.25,
        "pawn_absence": 0.15,
        "queen_absence": 0.10,
    }
    
    affordance = sum(components[k] * weights[k] for k in weights)
    
    # Apply sigmoid to create sharper transitions
    affordance = _sigmoid(affordance, k=config.sigmoid_k, midpoint=0.4)
    
    # Filter noise
    if affordance < config.min_affordance:
        affordance = 0.0
    
    return AffordanceSignal(
        subgraph="krk",
        value=affordance,
        components=components,
        is_exact_match=False,
    )


def compute_kpk_affordance(
    board: chess.Board,
    config: Optional[AffordanceConfig] = None,
) -> AffordanceSignal:
    """
    Compute KPK (King + Pawn vs King) affordance signal.
    
    Returns high values when approaching K+P vs K:
    - 1.0: Exact KPK position
    - 0.7-0.9: One side has only K+P(s), other has no pieces
    - 0.4-0.6: Few pieces, pawn endgame likely
    - 0.2-0.3: Trending toward pawn endgame
    - 0.0-0.1: Many pieces, no KPK relevance
    
    Components tracked:
    - pawn_presence: Do we have pawns?
    - piece_absence: Are heavy pieces gone?
    - material_scarcity: How little total material?
    - passed_pawn_advancement: How advanced are passed pawns?
    """
    config = config or AffordanceConfig()
    counts = _count_material(board)
    mat = _compute_material_values(counts)
    
    components: Dict[str, float] = {}
    
    # Check for exact KPK pattern (either side)
    white_is_kpk = (
        counts["white_pawns"] >= 1 and
        counts["white_queens"] == 0 and
        counts["white_rooks"] == 0 and
        counts["white_bishops"] == 0 and
        counts["white_knights"] == 0 and
        mat["black"] == 0
    )
    black_is_kpk = (
        counts["black_pawns"] >= 1 and
        counts["black_queens"] == 0 and
        counts["black_rooks"] == 0 and
        counts["black_bishops"] == 0 and
        counts["black_knights"] == 0 and
        mat["white"] == 0
    )
    
    if white_is_kpk or black_is_kpk:
        return AffordanceSignal(
            subgraph="kpk",
            value=1.0,
            components={"exact_match": 1.0},
            is_exact_match=True,
        )
    
    # Component 1: Pawn presence
    total_pawns = counts["white_pawns"] + counts["black_pawns"]
    pawn_presence = min(1.0, total_pawns / 4.0) if total_pawns > 0 else 0.0
    components["pawn_presence"] = pawn_presence
    
    # Component 2: Heavy piece absence (no queens/rooks = closer to KPK)
    heavy_pieces = (
        counts["white_queens"] + counts["black_queens"] +
        counts["white_rooks"] + counts["black_rooks"]
    )
    heavy_absence = 1.0 if heavy_pieces == 0 else max(0.0, 1.0 - heavy_pieces / 3.0)
    components["heavy_absence"] = heavy_absence
    
    # Component 3: Minor piece scarcity
    minor_pieces = (
        counts["white_knights"] + counts["black_knights"] +
        counts["white_bishops"] + counts["black_bishops"]
    )
    minor_scarcity = max(0.0, 1.0 - minor_pieces / 4.0)
    components["minor_scarcity"] = minor_scarcity
    
    # Component 4: Material scarcity
    material_scarcity = max(0.0, 1.0 - mat["total"] / 15.0)
    components["material_scarcity"] = material_scarcity
    
    # Component 5: Passed pawn advancement (check for advanced pawns)
    max_pawn_advancement = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(sq)
            if piece.color == chess.WHITE:
                advancement = rank / 7.0  # 0-7 -> 0-1
            else:
                advancement = (7 - rank) / 7.0
            max_pawn_advancement = max(max_pawn_advancement, advancement)
    components["pawn_advancement"] = max_pawn_advancement
    
    # Weighted combination
    weights = {
        "pawn_presence": 0.20,
        "heavy_absence": 0.30,
        "minor_scarcity": 0.20,
        "material_scarcity": 0.20,
        "pawn_advancement": 0.10,
    }
    
    affordance = sum(components[k] * weights[k] for k in weights)
    
    # Apply sigmoid
    affordance = _sigmoid(affordance, k=config.sigmoid_k, midpoint=0.4)
    
    # Filter noise
    if affordance < config.min_affordance:
        affordance = 0.0
    
    return AffordanceSignal(
        subgraph="kpk",
        value=affordance,
        components=components,
        is_exact_match=False,
    )


def compute_kqk_affordance(
    board: chess.Board,
    config: Optional[AffordanceConfig] = None,
) -> AffordanceSignal:
    """
    Compute KQK (King + Queen vs King) affordance signal.
    
    Returns high values when approaching K+Q vs K:
    - 1.0: Exact KQK position
    - 0.7-0.9: One side has only K+Q, other has minimal pieces
    - 0.4-0.6: Queen advantage with few other pieces
    - 0.2-0.3: Material trending toward queen endgame
    - 0.0-0.1: Complex position, no KQK relevance
    
    Components tracked:
    - queen_advantage: Do we have queen advantage?
    - piece_scarcity: How few pieces are on the board?
    - opponent_weakness: How little material does opponent have?
    - pawn_absence: Are pawns gone?
    """
    config = config or AffordanceConfig()
    counts = _count_material(board)
    mat = _compute_material_values(counts)
    
    components: Dict[str, float] = {}
    
    # Check for exact KQK pattern (either side)
    white_is_kqk = (
        counts["white_queens"] == 1 and
        counts["white_rooks"] == 0 and
        counts["white_bishops"] == 0 and
        counts["white_knights"] == 0 and
        counts["white_pawns"] == 0 and
        mat["black"] == 0
    )
    black_is_kqk = (
        counts["black_queens"] == 1 and
        counts["black_rooks"] == 0 and
        counts["black_bishops"] == 0 and
        counts["black_knights"] == 0 and
        counts["black_pawns"] == 0 and
        mat["white"] == 0
    )
    
    if white_is_kqk or black_is_kqk:
        return AffordanceSignal(
            subgraph="kqk",
            value=1.0,
            components={"exact_match": 1.0},
            is_exact_match=True,
        )
    
    # Component 1: Queen advantage
    queen_diff = counts["white_queens"] - counts["black_queens"]
    has_queen = counts["white_queens"] > 0 or counts["black_queens"] > 0
    queen_advantage = _sigmoid(abs(queen_diff), k=2.0, midpoint=0.5) if has_queen else 0.0
    components["queen_advantage"] = queen_advantage
    
    # Component 2: Piece scarcity
    total_pieces = sum(
        counts[f"{c}_{p}"] for c in ["white", "black"]
        for p in ["pawns", "knights", "bishops", "rooks", "queens"]
    )
    piece_scarcity = max(0.0, 1.0 - total_pieces / 10.0)
    components["piece_scarcity"] = piece_scarcity
    
    # Component 3: Opponent weakness
    opponent_mat = min(mat["white"], mat["black"])
    opponent_weakness = max(0.0, 1.0 - opponent_mat / 8.0)
    components["opponent_weakness"] = opponent_weakness
    
    # Component 4: Pawn absence
    total_pawns = counts["white_pawns"] + counts["black_pawns"]
    pawn_absence = max(0.0, 1.0 - total_pawns / 4.0)
    components["pawn_absence"] = pawn_absence
    
    # Component 5: Rook absence (KQK needs no rooks)
    total_rooks = counts["white_rooks"] + counts["black_rooks"]
    rook_absence = 1.0 if total_rooks == 0 else 0.3
    components["rook_absence"] = rook_absence
    
    # Weighted combination
    weights = {
        "queen_advantage": 0.30,
        "piece_scarcity": 0.20,
        "opponent_weakness": 0.25,
        "pawn_absence": 0.15,
        "rook_absence": 0.10,
    }
    
    affordance = sum(components[k] * weights[k] for k in weights)
    
    # Apply sigmoid
    affordance = _sigmoid(affordance, k=config.sigmoid_k, midpoint=0.4)
    
    # Filter noise
    if affordance < config.min_affordance:
        affordance = 0.0
    
    return AffordanceSignal(
        subgraph="kqk",
        value=affordance,
        components=components,
        is_exact_match=False,
    )


def compute_all_affordances(
    board: chess.Board,
    config: Optional[AffordanceConfig] = None,
) -> Dict[str, AffordanceSignal]:
    """
    Compute all affordance signals for a board position.
    
    Returns:
        Dict mapping subgraph name to AffordanceSignal
    """
    config = config or AffordanceConfig()
    
    return {
        "krk": compute_krk_affordance(board, config),
        "kpk": compute_kpk_affordance(board, config),
        "kqk": compute_kqk_affordance(board, config),
    }


def get_dominant_affordance(
    affordances: Dict[str, AffordanceSignal],
    threshold: float = 0.3,
) -> Optional[str]:
    """
    Get the subgraph with the highest affordance above threshold.
    
    Returns:
        Subgraph name or None if no affordance above threshold
    """
    best_name = None
    best_value = threshold
    
    for name, signal in affordances.items():
        if signal.value > best_value:
            best_value = signal.value
            best_name = name
    
    return best_name


def affordances_to_dict(
    affordances: Dict[str, AffordanceSignal],
) -> Dict[str, Any]:
    """Convert affordance signals to a serializable dict."""
    return {
        name: signal.to_dict() for name, signal in affordances.items()
    }

