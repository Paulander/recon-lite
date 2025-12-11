"""
Global Feature Hub for ReCoN Chess.

The FeatureHub centralizes tactical and geometric pattern detection,
allowing multiple subgraphs to share computed features without duplication.

Key Benefits:
1. Compute features once per tick, share across subscribers
2. Enable M5 Structure Learning to detect patterns in novel contexts
3. Provide consistent feature naming for logging and analysis
4. Support runtime feature registration for discovered patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import chess


class FeatureCategory(Enum):
    """Categories for organizing features."""
    TACTICAL = auto()      # Forks, pins, skewers, etc.
    GEOMETRIC = auto()     # Mating nets, batteries, alignments
    MATERIAL = auto()      # Material counts, imbalances
    POSITIONAL = auto()    # Pawn structure, king safety
    DYNAMIC = auto()       # Mobility, activity
    PHASE = auto()         # Game phase indicators


@dataclass
class FeatureDefinition:
    """
    Definition of a feature that can be computed.
    
    Attributes:
        name: Unique feature identifier
        category: Feature category for organization
        compute_fn: Function that computes the feature value
        description: Human-readable description
        dependencies: Other features this depends on
        is_binary: If True, output is 0.0 or 1.0; otherwise continuous
    """
    name: str
    category: FeatureCategory
    compute_fn: Callable[[chess.Board, Dict[str, float]], float]
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    is_binary: bool = False
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class FeatureResult:
    """
    Result of computing a feature.
    
    Attributes:
        name: Feature identifier
        value: Computed value (typically [0.0, 1.0])
        components: Optional breakdown of contributing factors
        metadata: Additional information about the computation
    """
    name: str
    value: float
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "metadata": self.metadata,
        }


class FeatureHub:
    """
    Global feature registry and computation hub.
    
    Features are registered with compute functions, and nodes can subscribe
    to be notified when features are computed. This enables:
    
    1. Compute-once, use-many: Features computed once per tick
    2. Feature hoisting: Tactical patterns available globally
    3. Subscription model: Subgraphs declare interest in features
    4. Runtime registration: New features can be added dynamically
    """
    
    def __init__(self):
        # Feature definitions by name
        self._definitions: Dict[str, FeatureDefinition] = {}
        
        # Computed feature values (cleared each tick)
        self._values: Dict[str, FeatureResult] = {}
        
        # Subscribers per feature (node IDs interested in each feature)
        self._subscribers: Dict[str, Set[str]] = {}
        
        # Last board FEN to detect cache invalidation
        self._last_fen: Optional[str] = None
        
        # Computation order (topologically sorted)
        self._compute_order: List[str] = []
        self._order_dirty: bool = True
    
    def register(self, definition: FeatureDefinition) -> None:
        """
        Register a feature definition.
        
        Args:
            definition: The feature to register
        """
        self._definitions[definition.name] = definition
        self._subscribers.setdefault(definition.name, set())
        self._order_dirty = True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a feature.
        
        Args:
            name: Feature name to remove
            
        Returns:
            True if feature was removed, False if not found
        """
        if name in self._definitions:
            del self._definitions[name]
            self._subscribers.pop(name, None)
            self._values.pop(name, None)
            self._order_dirty = True
            return True
        return False
    
    def subscribe(self, feature_name: str, subscriber_id: str) -> bool:
        """
        Subscribe a node to a feature.
        
        Args:
            feature_name: Name of feature to subscribe to
            subscriber_id: ID of the subscribing node
            
        Returns:
            True if subscription added, False if feature not found
        """
        if feature_name in self._subscribers:
            self._subscribers[feature_name].add(subscriber_id)
            return True
        return False
    
    def unsubscribe(self, feature_name: str, subscriber_id: str) -> bool:
        """
        Unsubscribe a node from a feature.
        
        Args:
            feature_name: Name of feature
            subscriber_id: ID of the node to unsubscribe
            
        Returns:
            True if removed, False if not found
        """
        if feature_name in self._subscribers:
            self._subscribers[feature_name].discard(subscriber_id)
            return True
        return False
    
    def get_subscribers(self, feature_name: str) -> Set[str]:
        """Get all subscribers for a feature."""
        return self._subscribers.get(feature_name, set()).copy()
    
    def compute_all(self, board: chess.Board, force: bool = False) -> Dict[str, float]:
        """
        Compute all registered features for the current board.
        
        Args:
            board: Current board position
            force: If True, recompute even if board hasn't changed
            
        Returns:
            Dict mapping feature name to computed value
        """
        current_fen = board.fen()
        
        # Check cache validity
        if not force and current_fen == self._last_fen:
            return {name: result.value for name, result in self._values.items()}
        
        # Clear previous values
        self._values.clear()
        self._last_fen = current_fen
        
        # Ensure compute order is valid
        if self._order_dirty:
            self._compute_order = self._topological_sort()
            self._order_dirty = False
        
        # Compute features in dependency order
        computed: Dict[str, float] = {}
        for name in self._compute_order:
            definition = self._definitions[name]
            try:
                value = definition.compute_fn(board, computed)
                self._values[name] = FeatureResult(
                    name=name,
                    value=value,
                )
                computed[name] = value
            except Exception as e:
                # On error, set to 0.0 and continue
                self._values[name] = FeatureResult(
                    name=name,
                    value=0.0,
                    metadata={"error": str(e)},
                )
                computed[name] = 0.0
        
        return computed
    
    def get(self, feature_name: str, default: float = 0.0) -> float:
        """
        Get a computed feature value.
        
        Args:
            feature_name: Name of the feature
            default: Value to return if not found
            
        Returns:
            Feature value or default
        """
        result = self._values.get(feature_name)
        return result.value if result else default
    
    def get_result(self, feature_name: str) -> Optional[FeatureResult]:
        """Get full feature result including components."""
        return self._values.get(feature_name)
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all computed feature values."""
        return {name: result.value for name, result in self._values.items()}
    
    def get_all_results(self) -> Dict[str, FeatureResult]:
        """Get all computed feature results."""
        return dict(self._values)
    
    def get_by_category(self, category: FeatureCategory) -> Dict[str, float]:
        """Get all features in a category."""
        result = {}
        for name, definition in self._definitions.items():
            if definition.category == category:
                value = self._values.get(name)
                result[name] = value.value if value else 0.0
        return result
    
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._definitions.keys())
    
    def list_features_by_category(self) -> Dict[str, List[str]]:
        """List features organized by category."""
        result: Dict[str, List[str]] = {}
        for name, definition in self._definitions.items():
            cat_name = definition.category.name
            if cat_name not in result:
                result[cat_name] = []
            result[cat_name].append(name)
        return result
    
    def _topological_sort(self) -> List[str]:
        """
        Sort features by dependencies.
        
        Returns features in order such that dependencies are computed first.
        """
        # Build dependency graph
        visited: Set[str] = set()
        order: List[str] = []
        
        def visit(name: str, path: Set[str]) -> None:
            if name in visited:
                return
            if name in path:
                # Cycle detected - just add and continue
                visited.add(name)
                order.append(name)
                return
            
            path.add(name)
            definition = self._definitions.get(name)
            if definition:
                for dep in definition.dependencies:
                    if dep in self._definitions:
                        visit(dep, path)
            
            path.remove(name)
            visited.add(name)
            order.append(name)
        
        for name in self._definitions:
            visit(name, set())
        
        return order
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize hub state to dict."""
        return {
            "features": [
                {
                    "name": d.name,
                    "category": d.category.name,
                    "description": d.description,
                    "is_binary": d.is_binary,
                    "dependencies": d.dependencies,
                }
                for d in self._definitions.values()
            ],
            "subscribers": {
                name: list(subs) for name, subs in self._subscribers.items()
            },
            "values": {
                name: result.to_dict() for name, result in self._values.items()
            },
        }


# ============================================================================
# Default Feature Definitions
# ============================================================================

def _feature_fork_available(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect if a fork opportunity exists."""
    from recon_lite_chess.scripts.tactics import detect_forks
    forks = detect_forks(board)
    if not forks:
        return 0.0
    # Scale by total value of forked pieces
    max_value = max(f.get("total_value", 0) for f in forks)
    return min(1.0, max_value / 10.0)


def _feature_pin_present(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect if pins exist."""
    from recon_lite_chess.scripts.tactics import detect_pins
    pins = detect_pins(board)
    return min(1.0, len(pins) / 3.0)


def _feature_hanging_piece(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect hanging pieces."""
    from recon_lite_chess.scripts.tactics import detect_hanging_pieces
    hanging = detect_hanging_pieces(board)
    enemy_hanging = len(hanging.get("enemy_hanging", []))
    return min(1.0, enemy_hanging / 2.0)


def _feature_back_rank_vulnerable(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect back rank vulnerability."""
    from recon_lite_chess.scripts.tactics import detect_back_rank_weakness
    weakness = detect_back_rank_weakness(board)
    if weakness.get("has_weakness"):
        # Scale by number of attacking moves
        attacks = len(weakness.get("attacking_moves", []))
        return min(1.0, 0.5 + attacks * 0.2)
    return 0.0


def _feature_discovered_attack(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect discovered attack opportunities."""
    from recon_lite_chess.scripts.tactics import detect_discovered_attacks
    discovered = detect_discovered_attacks(board)
    if not discovered:
        return 0.0
    # Prioritize discovered checks
    has_check = any(d.get("is_discovered_check") for d in discovered)
    return 1.0 if has_check else min(1.0, len(discovered) / 3.0)


def _feature_double_check(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect double check opportunities."""
    from recon_lite_chess.scripts.tactics import detect_double_check
    double_checks = detect_double_check(board)
    return 1.0 if double_checks else 0.0


def _feature_skewer(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect skewer opportunities."""
    from recon_lite_chess.scripts.tactics import detect_skewers
    skewers = detect_skewers(board)
    return min(1.0, len(skewers) / 2.0)


def _feature_material_advantage(board: chess.Board, computed: Dict[str, float]) -> float:
    """Compute material advantage for side to move."""
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9,
    }
    white_material = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.WHITE
    )
    black_material = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.BLACK
    )
    balance = white_material - black_material
    if not board.turn:  # Black to move
        balance = -balance
    # Normalize to [0, 1] where 0.5 is equal
    return max(0.0, min(1.0, 0.5 + balance / 20.0))


def _feature_king_safety(board: chess.Board, computed: Dict[str, float]) -> float:
    """Compute king safety for side to move."""
    king_sq = board.king(board.turn)
    if king_sq is None:
        return 0.5
    
    # Count attackers near king
    attackers = 0
    for sq in chess.SQUARES:
        if chess.square_distance(sq, king_sq) <= 2:
            piece = board.piece_at(sq)
            if piece and piece.color != board.turn:
                attackers += 1
    
    # Count pawn shield
    shield = 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_rank = king_rank + (1 if board.turn else -1)
    
    if 0 <= shield_rank <= 7:
        for df in [-1, 0, 1]:
            f = king_file + df
            if 0 <= f <= 7:
                sq = chess.square(f, shield_rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == board.turn:
                    shield += 1
    
    # Safety = shield bonus - attacker penalty
    safety = 0.5 + shield * 0.15 - attackers * 0.1
    return max(0.0, min(1.0, safety))


def _feature_center_control(board: chess.Board, computed: Dict[str, float]) -> float:
    """Compute center control for side to move."""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                       chess.D3, chess.D6, chess.E3, chess.E6,
                       chess.F3, chess.F4, chess.F5, chess.F6]
    
    control = 0.0
    for sq in center_squares:
        if board.is_attacked_by(board.turn, sq):
            control += 0.15
        piece = board.piece_at(sq)
        if piece and piece.color == board.turn:
            control += 0.1
    
    for sq in extended_center:
        if board.is_attacked_by(board.turn, sq):
            control += 0.03
    
    return min(1.0, control)


def _feature_mobility(board: chess.Board, computed: Dict[str, float]) -> float:
    """Compute piece mobility for side to move."""
    legal_moves = len(list(board.legal_moves))
    # Normalize: 20 moves = 0.5, 40+ moves = 1.0
    return min(1.0, legal_moves / 40.0)


def _feature_pawn_structure(board: chess.Board, computed: Dict[str, float]) -> float:
    """Evaluate pawn structure quality."""
    # Count doubled pawns (bad)
    doubled = 0
    for file in range(8):
        white_pawns = 0
        black_pawns = 0
        for rank in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    white_pawns += 1
                else:
                    black_pawns += 1
        if white_pawns > 1:
            doubled += white_pawns - 1
        if black_pawns > 1:
            doubled += black_pawns - 1
    
    # Count isolated pawns (bad)
    isolated = 0
    # Simplified: just penalize doubled for now
    
    # Quality score: start at 1.0, reduce for structural weaknesses
    quality = max(0.0, 1.0 - doubled * 0.15)
    return quality


def _feature_phase_opening(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect if position is in opening phase."""
    # Simplified: count developed pieces
    pieces = board.piece_map()
    total = len([p for p in pieces.values() if p.piece_type != chess.KING])
    
    # Check if knights/bishops are on starting squares
    starting = 0
    for sq in [chess.B1, chess.G1, chess.C1, chess.F1,
               chess.B8, chess.G8, chess.C8, chess.F8]:
        if board.piece_at(sq):
            starting += 1
    
    # Opening if many pieces still in place
    if total >= 14 and starting >= 4:
        return 1.0
    elif total >= 12 and starting >= 2:
        return 0.6
    return 0.0


def _feature_phase_endgame(board: chess.Board, computed: Dict[str, float]) -> float:
    """Detect if position is in endgame phase."""
    pieces = board.piece_map()
    total = len([p for p in pieces.values() if p.piece_type != chess.KING])
    queens = len([p for p in pieces.values() if p.piece_type == chess.QUEEN])
    
    if total <= 6:
        return 1.0
    elif total <= 10 and queens == 0:
        return 0.8
    elif total <= 12:
        return 0.4
    return 0.0


# ============================================================================
# Affordance / "Scent" Features for Bridge Discovery
# ============================================================================

def _feature_affordance_krk(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    KRK affordance - continuous distance to clean K+R vs K state.
    
    This is the "scent" signal for liquidation strategies.
    Spikes when trading pieces toward a KRK endgame.
    
    Returns:
        0.0 = far from KRK, 1.0 = exact KRK position
    """
    from recon_lite_chess.affordance import compute_krk_affordance
    signal = compute_krk_affordance(board)
    return signal.value


def _feature_affordance_kpk(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    KPK affordance - continuous distance to K+P vs K state.
    
    Scent signal for pawn endgame conversion.
    
    Returns:
        0.0 = far from KPK, 1.0 = exact KPK position
    """
    from recon_lite_chess.affordance import compute_kpk_affordance
    signal = compute_kpk_affordance(board)
    return signal.value


def _feature_affordance_kqk(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    KQK affordance - continuous distance to K+Q vs K state.
    
    Scent signal for queen endgame conversion.
    
    Returns:
        0.0 = far from KQK, 1.0 = exact KQK position
    """
    from recon_lite_chess.affordance import compute_kqk_affordance
    signal = compute_kqk_affordance(board)
    return signal.value


def _feature_opposition_status(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Opposition detection - universal geometric key for king play.
    
    Hoisted from KRK subgraph so middlegame planner can learn
    to use the King aggressively.
    
    Returns:
        1.0 = direct opposition (kings 2 squares apart on file/rank)
        0.8 = diagonal direct opposition (2 squares diagonally)
        0.5 = distant opposition (even distance apart)
        0.4 = diagonal distant opposition
        0.0 = no opposition
    """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0.0
    
    wk_file, wk_rank = chess.square_file(wk), chess.square_rank(wk)
    bk_file, bk_rank = chess.square_file(bk), chess.square_rank(bk)
    
    file_diff = abs(wk_file - bk_file)
    rank_diff = abs(wk_rank - bk_rank)
    
    # Direct opposition: same file or rank, exactly 2 squares apart
    if file_diff == 0 and rank_diff == 2:
        return 1.0  # Vertical direct opposition
    if rank_diff == 0 and file_diff == 2:
        return 1.0  # Horizontal direct opposition
    
    # Diagonal direct opposition (important for KPK)
    if file_diff == rank_diff == 2:
        return 0.8
    
    # Distant opposition: same file/rank, even distance apart
    if file_diff == 0 and rank_diff % 2 == 0 and rank_diff <= 6:
        return 0.5
    if rank_diff == 0 and file_diff % 2 == 0 and file_diff <= 6:
        return 0.5
    
    # Diagonal distant opposition
    if file_diff == rank_diff and file_diff % 2 == 0 and file_diff <= 6:
        return 0.4
    
    return 0.0


def _feature_color_complex_weakness(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if enemy King and pieces are stuck on one color complex.
    
    Crucial for Knight strategies and Outpost discovery.
    Helps M5 discover "Outpost" strategies by detecting when
    pieces are color-bound.
    
    Returns:
        0.0-1.0 where higher = more color-bound weakness
    """
    # Determine side to move's enemy
    enemy = not board.turn
    enemy_king = board.king(enemy)
    if enemy_king is None:
        return 0.0
    
    # King's square color (light=True, dark=False)
    king_light = (chess.square_file(enemy_king) + chess.square_rank(enemy_king)) % 2 == 0
    
    # Count enemy pieces on each color
    light_pieces = 0
    dark_pieces = 0
    total_pieces = 0
    
    for sq, piece in board.piece_map().items():
        if piece.color == enemy and piece.piece_type != chess.KING:
            sq_light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            if sq_light:
                light_pieces += 1
            else:
                dark_pieces += 1
            total_pieces += 1
    
    if total_pieces == 0:
        return 0.0
    
    # Calculate imbalance toward king's color
    king_color_pieces = light_pieces if king_light else dark_pieces
    
    # Weakness = pieces concentrated on king's color / total
    concentration = king_color_pieces / total_pieces if total_pieces > 0 else 0
    
    # Check if enemy has only same-colored bishop (big weakness)
    enemy_bishops = [
        sq for sq, p in board.piece_map().items()
        if p.piece_type == chess.BISHOP and p.color == enemy
    ]
    if len(enemy_bishops) == 1:
        bishop_sq = enemy_bishops[0]
        bishop_light = (chess.square_file(bishop_sq) + chess.square_rank(bishop_sq)) % 2 == 0
        if bishop_light == king_light:
            concentration += 0.2  # Bonus: bishop can't defend opposite color
    
    return min(1.0, concentration)


def create_default_hub() -> FeatureHub:
    """
    Create a FeatureHub with default tactical and positional features.
    
    Returns:
        FeatureHub with standard features registered
    """
    hub = FeatureHub()
    
    # Tactical features
    hub.register(FeatureDefinition(
        name="fork_available",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_fork_available,
        description="Fork opportunity exists",
    ))
    
    hub.register(FeatureDefinition(
        name="pin_present",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_pin_present,
        description="Pin(s) on the board",
    ))
    
    hub.register(FeatureDefinition(
        name="hanging_piece",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_hanging_piece,
        description="Enemy hanging pieces",
    ))
    
    hub.register(FeatureDefinition(
        name="back_rank_vulnerable",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_back_rank_vulnerable,
        description="Back rank weakness detected",
    ))
    
    hub.register(FeatureDefinition(
        name="discovered_attack",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_discovered_attack,
        description="Discovered attack possible",
    ))
    
    hub.register(FeatureDefinition(
        name="double_check",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_double_check,
        description="Double check possible",
        is_binary=True,
    ))
    
    hub.register(FeatureDefinition(
        name="skewer",
        category=FeatureCategory.TACTICAL,
        compute_fn=_feature_skewer,
        description="Skewer opportunity",
    ))
    
    # Material features
    hub.register(FeatureDefinition(
        name="material_advantage",
        category=FeatureCategory.MATERIAL,
        compute_fn=_feature_material_advantage,
        description="Material balance (0.5 = equal)",
    ))
    
    # Positional features
    hub.register(FeatureDefinition(
        name="king_safety",
        category=FeatureCategory.POSITIONAL,
        compute_fn=_feature_king_safety,
        description="King safety score",
    ))
    
    hub.register(FeatureDefinition(
        name="center_control",
        category=FeatureCategory.POSITIONAL,
        compute_fn=_feature_center_control,
        description="Center control measure",
    ))
    
    hub.register(FeatureDefinition(
        name="mobility",
        category=FeatureCategory.DYNAMIC,
        compute_fn=_feature_mobility,
        description="Piece mobility",
    ))
    
    hub.register(FeatureDefinition(
        name="pawn_structure",
        category=FeatureCategory.POSITIONAL,
        compute_fn=_feature_pawn_structure,
        description="Pawn structure quality",
    ))
    
    # Phase features
    hub.register(FeatureDefinition(
        name="phase_opening",
        category=FeatureCategory.PHASE,
        compute_fn=_feature_phase_opening,
        description="Opening phase indicator",
    ))
    
    hub.register(FeatureDefinition(
        name="phase_endgame",
        category=FeatureCategory.PHASE,
        compute_fn=_feature_phase_endgame,
        description="Endgame phase indicator",
    ))
    
    # Affordance / "Scent" features for bridge discovery
    hub.register(FeatureDefinition(
        name="affordance_krk",
        category=FeatureCategory.PHASE,
        compute_fn=_feature_affordance_krk,
        description="Distance to KRK endgame state (0=far, 1=exact)",
    ))
    
    hub.register(FeatureDefinition(
        name="affordance_kpk",
        category=FeatureCategory.PHASE,
        compute_fn=_feature_affordance_kpk,
        description="Distance to KPK endgame state (0=far, 1=exact)",
    ))
    
    hub.register(FeatureDefinition(
        name="affordance_kqk",
        category=FeatureCategory.PHASE,
        compute_fn=_feature_affordance_kqk,
        description="Distance to KQK endgame state (0=far, 1=exact)",
    ))
    
    # Geometric features (hoisted from subgraphs)
    hub.register(FeatureDefinition(
        name="opposition_status",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=_feature_opposition_status,
        description="King opposition (1=direct, 0.5=distant, 0=none)",
    ))
    
    hub.register(FeatureDefinition(
        name="color_complex_weakness",
        category=FeatureCategory.POSITIONAL,
        compute_fn=_feature_color_complex_weakness,
        description="Enemy pieces stuck on one color complex",
    ))
    
    return hub

