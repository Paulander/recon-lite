"""M10.1: Stem Cell Terminal Implementation with Three-Tier Lifecycle.

Stem cells are special terminal nodes that can discover new patterns
and potentially specialize into full sensors through a graduated promotion system.

Three-Tier Lifecycle:
    1. EXPLORING (Tier 1): Collecting samples, low consistency OK
    2. TRIAL (Tier 2): Transient vertex with XP system, must prove utility
    3. MATURE (Tier 3): Solidified permanent node in topology.json

XP System (for TRIAL tier):
    - Success (positive affordance delta): +10 XP
    - Failure (negative affordance delta): -10 XP
    - Background decay: -1 XP per cycle
    - Solidify threshold: XP >= 100 → MATURE
    - Demotion threshold: XP <= 0 → back to EXPLORING

Usage:
    from recon_lite.nodes import StemCellTerminal, StemCellState
    
    stem = StemCellTerminal("stem_001", exploration_budget=100)
    
    # During game play, stem cell collects samples
    stem.observe(board, reward_tick)
    
    # Promote to trial when consistency > 0.25 (lowered from 0.35)
    if stem.can_enter_trial():
        stem.promote_to_trial(registry, parent_id)
    
    # Update XP based on move outcomes
    stem.update_xp(affordance_delta)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Iterable
import json
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import chess
    HAS_CHESS = True
except ImportError:
    chess = None
    HAS_CHESS = False


def _random_goal_value() -> float:
    """Generate a bounded random goal target for actuator vectors."""
    return random.uniform(-1.0, 1.0)


class StemCellState(Enum):
    """Three-tier lifecycle states for stem cells.
    
    Tier 1 (Exploratory):
        DORMANT: Not yet activated
        EXPLORING: Actively collecting samples
        CANDIDATE: Has enough samples, awaiting trial evaluation
    
    Tier 2 (Probationary):
        TRIAL: Transient vertex in graph, earning XP to prove utility
    
    Tier 3 (Solidified):
        MATURE: Permanent node in topology.json, fully trusted
    
    Terminal:
        SPECIALIZED: Legacy alias for MATURE
        PRUNED: Removed due to low value or XP depletion
    """
    DORMANT = auto()      # Not yet activated
    EXPLORING = auto()    # Tier 1: Actively collecting samples
    CANDIDATE = auto()    # Tier 1: Ready for trial consideration
    TRIAL = auto()        # Tier 2: Transient vertex with XP system
    MATURE = auto()       # Tier 3: Permanent node, solidified
    SPECIALIZED = auto()  # Legacy alias for MATURE
    PRUNED = auto()       # Removed/deleted


class CellTier(Enum):
    """Desperation tiers for adaptive exploration/exploitation.
    
    Cells are classified by their XP variance over recent observations:
    - VOLATILE: High XP variance OR very low XP → aggressive exploration
    - MEDIUM: Moderate XP variance → balanced exploration/exploitation
    - INERT: Low XP variance, stable XP → conservative refinement
    
    When the system is "desperate" (win_delta < threshold), VOLATILE cells
    get boosted UCB exploration bonus to try new strategies.
    """
    VOLATILE = auto()   # High variance / unstable → aggressive exploration
    MEDIUM = auto()     # Balanced → normal UCB
    INERT = auto()      # Low variance / stable → conservative refinement

@dataclass
class StemCellConfig:
    """Configuration for stem cell behavior."""
    min_samples: int = 50  # Minimum samples before specialization attempt
    max_samples: int = 500  # Maximum samples to collect
    reward_threshold: float = 0.3  # Only store samples with |reward| > this
    specialization_threshold: float = 0.50  # Lowered from 0.70 for easier specialization
    exploration_budget: int = 100  # Ticks allowed in exploring state
    decay_rate: float = 0.99  # How fast samples decay in importance


@dataclass
class StemCellSample:
    """A sample collected by a stem cell."""
    fen: str
    features: Optional[List[float]] = None
    reward: float = 0.0
    tick: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Delta-based learning: store before/after features for transitions
    features_before: Optional[List[float]] = None
    features_after: Optional[List[float]] = None
    is_transition: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fen": self.fen,
            "features": self.features,
            "reward": self.reward,
            "tick": self.tick,
            "metadata": self.metadata,
            "features_before": self.features_before,
            "features_after": self.features_after,
            "is_transition": self.is_transition,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StemCellSample":
        return cls(
            fen=data["fen"],
            features=data.get("features"),
            reward=data.get("reward", 0.0),
            tick=data.get("tick", 0),
            metadata=data.get("metadata", {}),
            features_before=data.get("features_before"),
            features_after=data.get("features_after"),
            is_transition=data.get("is_transition", False),
        )


class StemCellTerminal:
    """
    A stem cell terminal with three-tier lifecycle and XP system.
    
    Stem cells are exploratory nodes that:
    1. EXPLORING: Collect samples during high-reward moments
    2. TRIAL: Operate as transient vertex, earning XP to prove utility
    3. MATURE: Solidified as permanent node in topology.json
    
    XP System (TRIAL tier only):
        - Base XP on promotion: 50
        - Success: +10 XP (positive affordance delta)
        - Failure: -10 XP (negative affordance delta)
        - Decay: -1 XP per cycle
        - Solidify: XP >= 100 → MATURE
        - Demote: XP <= 0 → back to EXPLORING
    """
    
    # XP Configuration
    XP_INITIAL = 50        # Starting XP when entering TRIAL
    XP_SOLIDIFY = 100      # XP needed to become MATURE
    XP_SUCCESS = 10        # XP for positive affordance delta
    XP_FAILURE = -10       # XP for negative affordance delta
    XP_DECAY = -1          # XP lost per cycle (cost of living)
    
    def __init__(
        self,
        cell_id: str,
        config: Optional[StemCellConfig] = None,
        feature_extractor: Optional[Callable] = None,
    ):
        self.cell_id = cell_id
        self.config = config or StemCellConfig()
        self.feature_extractor = feature_extractor
        
        # Core state
        self.state = StemCellState.DORMANT
        self.samples: List[StemCellSample] = []
        self.exploration_ticks = 0
        self.pattern_signature: Optional[List[float]] = None
        self.trust_score = 0.0
        self.metadata: Dict[str, Any] = {}
        
        # XP System (Tier 2: TRIAL)
        self.xp: int = 0
        self.xp_successes: int = 0
        self.xp_failures: int = 0
        self.trial_node_id: Optional[str] = None  # Graph node ID when solidified
        self.trial_edge_key: Optional[str] = None  # Edge key to parent
        
        # Trial preparation data (stored when entering TRIAL, used when solidifying)
        self.trial_parent_id: Optional[str] = None
        self.trial_consistency: float = 0.0
        self.trial_signature: Optional[List[float]] = None
        self.trial_tick: int = 0
        
        # Pattern centroid for similarity comparison
        self.pattern_centroid: Optional[List[float]] = None
        
        # Inertia Pruning: Track when cell last contributed to CONFIRM signal
        # Used to prune TRIAL cells that haven't been useful for N cycles
        self.last_confirm_cycle: Optional[int] = None
        
        # =========== M5.1 COMPOSITION FIELDS ===========
        # Parent XP delegation: If set, this cell's XP is managed by parent
        self.parent_xp_owner: Optional[str] = None  # Parent cell_id for XP delegation
        
        # Grace period: New compositions get N games before XP decay starts
        self.grace_games: int = 20  # Default grace period
        self.total_exposures: int = 0  # Total episodes this cell was exposed to
        self.min_exposure_threshold: int = 50  # Can't prune before this many exposures
        
        # Recursive composition depth: 0=raw sensor, 1=AND/OR of raw, 2=AND of ANDs, etc.
        self.depth: int = 0
        
        # Children for composition cells (AND/OR gates)
        self.children: List[str] = []  # Child cell_ids (empty for raw sensors)
        self.is_composition: bool = False  # True if this is an AND/OR gate
        
        # =========== DESPERATION TIERS ===========
        # Track XP history for variance-based tier classification
        self.xp_history: List[int] = []  # Last N XP values for variance calculation
        self.xp_history_window: int = 10  # Window size for variance
        self.cell_tier: CellTier = CellTier.MEDIUM  # Current tier classification
        
        # Tier thresholds (tuned for XP scale 0-100+)
        self.tier_variance_high: float = 100.0  # XP variance for VOLATILE
        self.tier_variance_low: float = 25.0    # XP variance for INERT
        self.tier_xp_low: int = 20              # XP threshold for forced VOLATILE
    
    def activate(self) -> None:
        """Start exploration."""
        if self.state == StemCellState.DORMANT:
            self.state = StemCellState.EXPLORING
            self.exploration_ticks = 0
    
    def observe(
        self,
        board: Any,  # chess.Board
        reward: float,
        tick: int = 0,
        **metadata,
    ) -> bool:
        """
        Observe a position during exploration.
        
        Args:
            board: Chess board (or any domain object)
            reward: Reward signal at this tick
            tick: Current tick number
            **metadata: Additional data to store
            
        Returns:
            True if sample was stored, False if filtered out
        """
        if self.state not in (StemCellState.EXPLORING, StemCellState.CANDIDATE):
            return False
        
        # Filter by reward threshold
        if abs(reward) < self.config.reward_threshold:
            return False
        
        # Extract features if extractor available
        features = None
        if self.feature_extractor and board is not None:
            try:
                features = self.feature_extractor(board)
                if hasattr(features, "to_vector"):
                    features = features.to_vector()
                elif hasattr(features, "tolist"):
                    features = features.tolist()
                elif isinstance(features, tuple):
                    features = list(features)
            except Exception:
                features = None
        
        # Get FEN if it's a chess board
        fen = ""
        if HAS_CHESS and hasattr(board, 'fen'):
            fen = board.fen()
        elif isinstance(board, str):
            fen = board
        
        sample = StemCellSample(
            fen=fen,
            features=features,
            reward=reward,
            tick=tick,
            metadata=metadata,
        )
        
        self.samples.append(sample)
        self.exploration_ticks += 1
        
        # Check state transitions
        if len(self.samples) >= self.config.min_samples:
            if self.state == StemCellState.EXPLORING:
                self.state = StemCellState.CANDIDATE
        
        # Prune if too many samples
        if len(self.samples) > self.config.max_samples:
            self._prune_samples()
        
        # Check if exploration budget exhausted
        if self.exploration_ticks >= self.config.exploration_budget:
            if len(self.samples) < self.config.min_samples:
                self.state = StemCellState.PRUNED
        
        return True
    
    def should_specialize(self) -> bool:
        """Check if the stem cell should attempt specialization."""
        if self.state != StemCellState.CANDIDATE:
            return False
        if len(self.samples) < self.config.min_samples:
            return False
        return True
    
    def analyze_pattern(self) -> Tuple[float, Optional[List[float]]]:
        """
        Analyze collected samples for patterns.
        
        Returns:
            (consistency_score, pattern_signature)
            consistency_score: How consistent the pattern is (0-1)
            pattern_signature: Representative feature vector
        """
        if not self.samples:
            return 0.0, None
        
        # Get samples with features
        featured = [s for s in self.samples if s.features is not None]
        if len(featured) < 10:
            return 0.0, None
        
        # Stack features
        try:
            feature_rows = [list(s.features) for s in featured]
        except Exception:
            return 0.0, None

        original_len = len(feature_rows[0]) if feature_rows else 0

        # Optionally mask features to a sparse subset
        mask = self.metadata.get("feature_mask")
        if mask:
            try:
                if isinstance(mask[0], bool):
                    feature_rows = [
                        [val for val, keep in zip(row, mask) if keep]
                        for row in feature_rows
                    ]
                else:
                    feature_rows = [
                        [row[idx] for idx in mask if idx < len(row)]
                        for row in feature_rows
                    ]
            except Exception:
                pass

        pattern_mode = self.metadata.get("pattern_mode", "centroid")
        
        if HAS_NUMPY:
            try:
                features = np.array(feature_rows)
            except Exception:
                return 0.0, None

            # Compute pattern signature
            if pattern_mode == "medoid":
                try:
                    diffs = features[:, None, :] - features[None, :, :]
                    dists = np.linalg.norm(diffs, axis=2)
                    avg_dists = np.mean(dists, axis=1)
                    medoid_idx = int(np.argmin(avg_dists))
                    signature_vec = features[medoid_idx]
                except Exception:
                    signature_vec = np.mean(features, axis=0)
            else:
                signature_vec = np.mean(features, axis=0)

            # Compute consistency (inverse of average distance to signature)
            distances = np.linalg.norm(features - signature_vec, axis=1)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
        else:
            # Manual fallback without numpy
            feature_rows = [row for row in feature_rows if row]
            if not feature_rows:
                return 0.0, None

            if pattern_mode == "medoid":
                try:
                    avg_dists = []
                    for row in feature_rows:
                        dist_sum = 0.0
                        for other in feature_rows:
                            dist_sum += sum((a - b) ** 2 for a, b in zip(row, other)) ** 0.5
                        avg_dists.append(dist_sum / max(1, len(feature_rows)))
                    medoid_idx = min(range(len(avg_dists)), key=avg_dists.__getitem__)
                    signature_vec = feature_rows[medoid_idx]
                except Exception:
                    signature_vec = [
                        sum(row[i] for row in feature_rows) / len(feature_rows)
                        for i in range(len(feature_rows[0]))
                    ]
            else:
                signature_vec = [
                    sum(row[i] for row in feature_rows) / len(feature_rows)
                    for i in range(len(feature_rows[0]))
                ]

            distances = [
                sum((a - b) ** 2 for a, b in zip(row, signature_vec)) ** 0.5
                for row in feature_rows
            ]
            avg_distance = sum(distances) / max(1, len(distances))
            max_distance = max(distances) if distances else 1.0

        # Consistency: 1.0 if all samples are identical, lower if spread out
        consistency = 1.0 - min(1.0, avg_distance / (max_distance + 1e-6))

        # Store signature for pattern matching (expand to full length if masked)
        if hasattr(signature_vec, "tolist"):
            signature_list = signature_vec.tolist()
        else:
            signature_list = list(signature_vec)
        if mask:
            full_len = original_len
            full_signature = [0.0] * full_len
            if isinstance(mask[0], bool):
                sig_idx = 0
                for idx, keep in enumerate(mask):
                    if keep and sig_idx < len(signature_list):
                        full_signature[idx] = signature_list[sig_idx]
                        sig_idx += 1
            else:
                for sig_idx, idx in enumerate(mask):
                    if idx < full_len and sig_idx < len(signature_list):
                        full_signature[idx] = signature_list[sig_idx]
            signature_list = full_signature

        self.pattern_signature = signature_list
        self.pattern_centroid = self.pattern_signature
        return float(consistency), self.pattern_signature
    
    def analyze_pattern_delta_based(self) -> Tuple[float, Optional[List[float]]]:
        """
        Compute consistency based on terminal output DELTAS, not centroids.
        
        This is the CORRECT approach per the spec:
        - For each transition (v0, v1), compute terminal output s0 = T(v0), s1 = T(v1)
        - Compute delta Δs = s1 - s0
        - Consistency = how stable Δs is across different transitions (low variance)
        
        Returns:
            (consistency_score, median_delta_value)
        """
        if not self.samples:
            return 0.0, None
        
        # Filter to transition samples only
        transitions = [s for s in self.samples if s.is_transition and s.features_before and s.features_after]
        if len(transitions) < 3:
            # Not enough transitions, fall back to centroid method
            return self.analyze_pattern()
        
        # Compute deltas for each transition
        deltas = []
        for sample in transitions:
            try:
                # Compute terminal output as cosine similarity to current pattern
                if self.pattern_centroid is None:
                    # No pattern yet, use simple magnitude difference
                    before_mag = sum(f ** 2 for f in sample.features_before) ** 0.5
                    after_mag = sum(f ** 2 for f in sample.features_after) ** 0.5
                    delta = after_mag - before_mag
                else:
                    # Compute similarity-based terminal output
                    s_before = self._cosine_similarity(sample.features_before, self.pattern_centroid)
                    s_after = self._cosine_similarity(sample.features_after, self.pattern_centroid)
                    delta = s_after - s_before
                
                deltas.append(delta)
            except Exception:
                continue
        
        if len(deltas) < 3:
            return 0.0, None
        
        # Measure consistency: how stable are the deltas?
        mean_delta = sum(deltas) / len(deltas)
        variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
        std_dev = variance ** 0.5
        
        # Consistency = inverse of coefficient of variation
        # High consistency when deltas are stable (low std relative to mean)
        if abs(mean_delta) < 1e-6:
            # Deltas are near zero - not discriminative
            consistency = 0.0
        else:
            # coefficient_of_variation = std_dev / abs(mean_delta)
            # consistency = high when CV is low
            cv = std_dev / abs(mean_delta)
            consistency = 1.0 / (1.0 + cv)
        
        # Store mean delta as the "signature" for this terminal
        return consistency, mean_delta
    
    def observe_transition(
        self,
        features_before: List[float],
        features_after: List[float],
        reward: float,
        fen_before: str,
        tick: int
    ) -> bool:
        """
        Observe a state transition for delta-based learning.
        
        Args:
            features_before: Feature vector before the transition
            features_after: Feature vector after the transition
            reward: Reward signal (1.0 for mate, 0.0 otherwise)
            fen_before: FEN string before the transition
            tick: Current tick
            
        Returns:
            True if sample was stored
        """
        if self.state not in (StemCellState.EXPLORING, StemCellState.CANDIDATE, StemCellState.TRIAL):
            return False
        
        sample = StemCellSample(
            fen=fen_before,
            features=features_after,  # For backward compatibility
            reward=reward,
            tick=tick,
            features_before=features_before,
            features_after=features_after,
            is_transition=True
        )
        
        self.samples.append(sample)
        self._prune_samples()
        return True
    
    # =========================================================================
    # XP SYSTEM - THREE-TIER LIFECYCLE MANAGEMENT
    # =========================================================================
    
    def can_enter_trial(self, min_samples: int = 100, min_consistency: float = 0.25) -> bool:
        """
        Check if cell is ready to enter TRIAL tier.
        
        Requirements:
            - In CANDIDATE state
            - At least min_samples collected
            - Pattern consistency >= min_consistency
        """
        if self.state != StemCellState.CANDIDATE:
            return False
        
        if len(self.samples) < min_samples:
            return False
        
        consistency, _ = self.analyze_pattern()
        return consistency >= min_consistency
    
    def promote_to_trial(
        self,
        registry: Any,  # TopologyRegistry
        parent_id: str = "kpk_detect",
        current_tick: int = 0,
        min_consistency: float = 0.30,  # Lowered from 0.50 for easier TRIAL promotion
        wire_to_legs: bool = True,  # NEW: Also wire as child of legs for gating
        leg_node_ids: Optional[List[str]] = None,  # NEW: Which legs to wire to
        node_factory: Optional[str] = None,  # Optional factory override
        extra_parent_ids: Optional[List[str]] = None,  # Optional extra SUB parents
        node_meta_extra: Optional[Dict[str, Any]] = None,  # Extra meta to merge
    ) -> bool:
        """
        Promote cell to TRIAL tier as a transient vertex.
        
        MANIFEST: Now injects TRIAL nodes directly into the TopologyRegistry
        so they appear in the runtime graph and can actually activate.
        
        NEW: When wire_to_legs=True, TRIAL nodes are ALSO added as SUB children
        of the leg nodes. This enables gating (require_child_confirm) to work:
        Legs will wait for TRIAL children to confirm before suggesting moves.
        
        Args:
            registry: TopologyRegistry to add the TRIAL node to
            parent_id: Parent node to wire to (kpk_detect for sensors, kpk_execute for managers)
            current_tick: Current tick for metadata
            min_consistency: Minimum pattern consistency (default 0.50)
            wire_to_legs: If True, also wire TRIAL as child of leg nodes for gating
            leg_node_ids: List of leg node IDs to wire to (default: ["kpk_pawn_leg", "kpk_king_leg"])
            
        Returns:
            True if promotion successful
        """
        if self.state not in (StemCellState.CANDIDATE, StemCellState.EXPLORING):
            return False
        
        consistency, signature = self.analyze_pattern()
        if consistency < min_consistency:
            return False
        
        # Default leg nodes for KPK
        if leg_node_ids is None:
            leg_node_ids = ["kpk_pawn_leg", "kpk_king_leg"]
        
        # Generate trial node ID
        self.trial_node_id = f"TRIAL_{self.cell_id}_{current_tick}"
        
        # Store trial preparation data
        self.trial_parent_id = parent_id
        self.trial_consistency = consistency
        self.trial_signature = signature
        self.trial_tick = current_tick
        
        # =========================================================================
        # MANIFEST: Inject TRIAL node into registry immediately
        # This ensures the node appears in the runtime graph and can activate.
        # =========================================================================
        
        # Extract subgraph from parent_id (e.g., "kpk_detect" -> "kpk")
        # This is CRITICAL: nodes must belong to the subgraph to be processed
        subgraph_name = parent_id.split("_")[0] if "_" in parent_id else parent_id
        
        factory = node_factory or self.metadata.get("node_factory")
        if factory is None:
            factory = "recon_lite.learning.m5_structure:create_pattern_sensor"

        node_meta = {
            "cell_id": self.cell_id,
            "tier": "trial",
            "consistency": consistency,
            "promoted_tick": current_tick,
            "subgraph": subgraph_name,  # CRITICAL: Required for subgraph execution
        }
        feature_mask = self.metadata.get("feature_mask")
        if feature_mask:
            node_meta["feature_mask"] = list(feature_mask)
        pattern_mode = self.metadata.get("pattern_mode")
        if pattern_mode:
            node_meta["pattern_mode"] = pattern_mode
        threshold = self.metadata.get("threshold")
        if threshold is not None:
            node_meta["threshold"] = threshold
        goal_features = self.metadata.get("goal_features")
        if goal_features:
            node_meta["goal_features"] = list(goal_features)
        goal_weights = self.metadata.get("goal_weights")
        if goal_weights:
            node_meta["goal_weights"] = list(goal_weights)

        node_spec = {
            "id": self.trial_node_id,
            "type": "TERMINAL",
            "group": "trial",
            "factory": factory,
            "pattern_signature": signature,
            "meta": node_meta,
        }
        if node_meta_extra:
            node_spec["meta"].update(node_meta_extra)

        if factory.endswith("create_goal_actuator"):
            goal_features = node_spec["meta"].get(
                "goal_features",
                ["box_area_delta", "king_distance_delta", "opposition_gain", "safe_check"],
            )
            goal_vector = node_spec["meta"].get("goal_vector")
            if goal_vector is None:
                goal_vector = [_random_goal_value() for _ in goal_features]
                node_spec["meta"]["goal_vector"] = goal_vector
            node_spec["meta"].setdefault("goal_features", goal_features)
            node_spec["meta"].setdefault("goal_weights", [1.0] * len(goal_features))
        
        try:
            registry.add_node(node_spec, tick=current_tick)
            
            # PRIMARY WIRING: Connect to sensor parent (e.g., kpk_detect)
            # Weight 0.5 allows activation without hijacking the strategy
            registry.add_edge(
                parent_id,
                self.trial_node_id,
                "SUB",
                weight=0.5,  # Exploratory - not hijacking
                tick=current_tick
            )
            
            # LEG WIRING: Also connect as child of legs for GATING support
            # This enables require_child_confirm to work on legs:
            # Legs will wait for TRIAL children to confirm before acting.
            if wire_to_legs:
                for leg_id in leg_node_ids:
                    try:
                        # Check if leg exists in registry
                        if registry.get_node(leg_id) is not None:
                            registry.add_edge(
                                leg_id,
                                self.trial_node_id,
                                "SUB",
                                weight=0.5,  # Exploratory weight
                                tick=current_tick
                            )
                    except Exception:
                        pass  # Skip if leg doesn't exist

            if extra_parent_ids:
                for extra_parent in extra_parent_ids:
                    try:
                        registry.add_edge(
                            extra_parent,
                            self.trial_node_id,
                            "SUB",
                            weight=0.5,
                            tick=current_tick,
                        )
                    except Exception:
                        pass
            
            registry.save()
        except ValueError:
            pass  # Node already exists (e.g., from previous cycle)
        
        # Update state to TRIAL
        self.state = StemCellState.TRIAL
        self.xp = self.XP_INITIAL
        self.xp_successes = 0
        self.xp_failures = 0
        
        # Store in metadata for serialization
        self.metadata["trial_prep"] = {
            "node_id": self.trial_node_id,
            "parent_id": parent_id,
            "consistency": consistency,
            "promoted_tick": current_tick,
            "sample_count": len(self.samples),
        }
        
        return True
    
    def update_xp(self, affordance_delta: float) -> Tuple[int, str]:
        """
        Update XP based on affordance delta from a move.
        
        Args:
            affordance_delta: Change in affordance (positive = good, negative = bad)
            
        Returns:
            (xp_change, result) where result is 'success', 'failure', or 'neutral'
        """
        if self.state != StemCellState.TRIAL:
            return 0, "not_trial"
        
        if affordance_delta >= 0.05:
            # Success: positive affordance delta
            xp_change = self.XP_SUCCESS
            self.xp_successes += 1
            result = "success"
        elif affordance_delta <= -0.05:
            # Failure: negative affordance delta
            xp_change = self.XP_FAILURE
            self.xp_failures += 1
            result = "failure"
        else:
            # Neutral: no significant change
            xp_change = 0
            result = "neutral"
        
        self.xp += xp_change
        
        # Record XP history for tier classification
        self.xp_history.append(self.xp)
        if len(self.xp_history) > self.xp_history_window:
            self.xp_history.pop(0)
        
        # Reclassify tier based on updated history
        self.classify_tier()
        
        return xp_change, result

    def update_actuator_xp(self, affordance_delta: float) -> Tuple[int, str]:
        """
        Update actuator XP separately from sensor XP.

        Uses the same thresholds as sensor XP but stores values in metadata.
        """
        if self.state != StemCellState.TRIAL:
            return 0, "not_trial"

        actuator_xp = self.metadata.get("actuator_xp", self.XP_INITIAL)
        actuator_history = self.metadata.get("actuator_xp_history", [])

        if affordance_delta >= 0.05:
            xp_change = self.XP_SUCCESS
            result = "success"
        elif affordance_delta <= -0.05:
            xp_change = self.XP_FAILURE
            result = "failure"
        else:
            xp_change = 0
            result = "neutral"

        actuator_xp += xp_change
        actuator_history.append(actuator_xp)
        if len(actuator_history) > self.xp_history_window:
            actuator_history.pop(0)

        self.metadata["actuator_xp"] = actuator_xp
        self.metadata["actuator_xp_history"] = actuator_history

        return xp_change, result
    
    def classify_tier(self) -> CellTier:
        """
        Classify cell into desperation tier based on XP variance.
        
        VOLATILE: High XP variance OR very low XP → aggressive exploration
        MEDIUM: Moderate XP variance → balanced exploration/exploitation  
        INERT: Low XP variance, stable XP → conservative refinement
        
        Returns:
            The current CellTier classification
        """
        import math
        
        # Not enough history? Default to MEDIUM
        if len(self.xp_history) < 3:
            self.cell_tier = CellTier.MEDIUM
            return self.cell_tier
        
        # Force VOLATILE if XP is dangerously low
        if self.xp < self.tier_xp_low:
            self.cell_tier = CellTier.VOLATILE
            return self.cell_tier
        
        # Compute XP variance
        mean_xp = sum(self.xp_history) / len(self.xp_history)
        variance = sum((x - mean_xp) ** 2 for x in self.xp_history) / len(self.xp_history)
        
        # Classify based on variance thresholds
        if variance >= self.tier_variance_high:
            self.cell_tier = CellTier.VOLATILE
        elif variance <= self.tier_variance_low:
            self.cell_tier = CellTier.INERT
        else:
            self.cell_tier = CellTier.MEDIUM
        
        return self.cell_tier
    
    def get_xp(self, manager: Optional["StemCellManager"] = None) -> int:
        """
        Get effective XP, delegating to parent if part of a composition.
        
        M5.1 XP DELEGATION: If this cell is part of a composition (AND/OR gate),
        XP is tracked at the parent level. This cell's xp field is ignored
        in favor of the parent's XP.
        
        Args:
            manager: StemCellManager to look up parent cell (optional)
            
        Returns:
            Effective XP value (parent's if delegated, own otherwise)
        """
        if self.parent_xp_owner and manager:
            parent = manager.cells.get(self.parent_xp_owner)
            if parent:
                return parent.xp
        return self.xp
    
    def decay_xp(self) -> int:
        """
        Apply XP decay (cost of living).
        
        Called once per cycle for all TRIAL cells.
        
        M5.1 GRACE PERIOD: New cells/compositions are protected from decay
        for the first `grace_games` exposures to give them time to prove value.
        
        XP FLOOR: Prevents decay below 10 to avoid "death spiral" in 0% stages.
        This gives cells time to accumulate engagement XP and partial signals.
        
        Returns:
            XP after decay
        """
        if self.state != StemCellState.TRIAL:
            return self.xp
        
        # M5.1: Track total exposures for min_exposure threshold
        self.total_exposures += 1
        
        # M5.1 GRACE PERIOD: Skip decay during grace period
        if self.total_exposures <= self.grace_games:
            return self.xp  # Protected - no decay
        
        XP_FLOOR = 10  # Minimum XP to prevent death spiral
        
        self.xp += self.XP_DECAY
        
        # Apply floor to prevent death spiral
        if self.xp < XP_FLOOR:
            self.xp = XP_FLOOR
        
        return self.xp
    
    # =========================================================================
    # M5.1 ENGAGEMENT XP: Gradual growth during failure states
    # =========================================================================
    
    # Engagement XP rewards (capped per game)
    ENGAGEMENT_XP_ACTIVATION = 0.5  # Node participates in game
    ENGAGEMENT_XP_CONSISTENCY = 0.2  # Pattern matches reliably
    ENGAGEMENT_XP_DEPTH_BONUS = 0.1  # Activation propagates > 2 levels
    ENGAGEMENT_XP_MAX_PER_GAME = 2.0  # Cap to prevent inflation
    
    def accumulate_engagement_xp(
        self,
        was_active: bool = False,
        pattern_matched: bool = False,
        propagation_depth: int = 0,
        stage_idx: Optional[int] = None,
    ) -> float:
        """
        Award XP for engagement, not just wins.
        
        HYBRID FIX: Allows gradual XP accumulation during failure states,
        breaking the "no wins → no XP → no growth" deadlock.
        
        Args:
            was_active: Node activated this game
            pattern_matched: Pattern signature matched reliably
            propagation_depth: How deep activation propagated
            stage_idx: Current stage (for stage-specific filtering)
            
        Returns:
            XP change this game
        """
        if self.state != StemCellState.TRIAL:
            return 0.0
        
        # Stage-specific filtering: only accumulate XP in relevant stages
        relevant_stages = self.metadata.get("relevant_stages")
        if relevant_stages is not None and stage_idx is not None:
            if stage_idx not in relevant_stages:
                return 0.0
        
        xp_gain = 0.0
        
        # +0.5 for activation (node participates)
        if was_active:
            xp_gain += self.ENGAGEMENT_XP_ACTIVATION
            self.metadata["engagement_activations"] = self.metadata.get("engagement_activations", 0) + 1
        
        # +0.2 for consistent pattern matching
        if pattern_matched:
            xp_gain += self.ENGAGEMENT_XP_CONSISTENCY
            self.metadata["engagement_consistency_hits"] = self.metadata.get("engagement_consistency_hits", 0) + 1
        
        # +0.1 * levels for deep propagation (hierarchical reasoning)
        if propagation_depth > 2:
            depth_bonus = self.ENGAGEMENT_XP_DEPTH_BONUS * (propagation_depth - 2)
            xp_gain += min(depth_bonus, 0.5)  # Cap depth bonus at 0.5
        
        # Cap at +2 XP/game to prevent inflation
        xp_gain = min(xp_gain, self.ENGAGEMENT_XP_MAX_PER_GAME)
        
        # Apply XP gain
        self.xp += int(xp_gain)  # Convert to int for consistency with XP system
        self.metadata["total_engagement_xp"] = self.metadata.get("total_engagement_xp", 0.0) + xp_gain
        
        return xp_gain
    
    def check_solidification(self) -> Tuple[bool, str]:
        """
        Check if cell should be solidified (MATURE) or demoted (EXPLORING).
        
        Returns:
            (should_change, new_state) where new_state is 'mature', 'demoted', or 'stay'
        """
        if self.state != StemCellState.TRIAL:
            return False, "not_trial"
        
        if self.xp >= self.XP_SOLIDIFY:
            return True, "mature"
        elif self.xp <= 0:
            return True, "demoted"
        else:
            return False, "stay"
    
    # =========================================================================
    # INERTIA PRUNING - Track CONFIRM Signal Contributions
    # =========================================================================
    
    def mark_confirmed(self, cycle: int) -> None:
        """
        Mark that this cell contributed to a CONFIRM signal.
        
        Called when the cell's corresponding graph node reaches CONFIRMED state.
        This resets the inertia timer.
        
        Args:
            cycle: Current training cycle number
        """
        self.last_confirm_cycle = cycle
    
    def is_inert(self, current_cycle: int, max_inactive: int = 20) -> bool:
        """
        Check if cell hasn't contributed to CONFIRM signal in max_inactive cycles.
        
        Used for "Inertia Pruning" - removing TRIAL cells that aren't
        contributing to actual network behavior despite being fired.
        
        Args:
            current_cycle: Current training cycle number
            max_inactive: Maximum cycles without CONFIRM before considered inert
            
        Returns:
            True if cell is inert (hasn't confirmed in max_inactive cycles)
        """
        if self.state != StemCellState.TRIAL:
            return False  # Only TRIAL cells can be inert
        
        if self.last_confirm_cycle is None:
            # Never confirmed - consider inert if past max_inactive cycles
            return current_cycle > max_inactive
        
        return (current_cycle - self.last_confirm_cycle) > max_inactive
    
    def solidify_to_mature(self, registry: Any, current_tick: int = 0) -> bool:
        """
        Solidify trial node to permanent MATURE node.
        
        Args:
            registry: TopologyRegistry for updating node
            current_tick: Current tick for metadata
            
        Returns:
            True if successful
        """
        if self.state != StemCellState.TRIAL or not self.trial_node_id:
            return False
        
        if not self.trial_parent_id:
            self.metadata["solidify_error"] = "No trial parent ID stored"
            return False
        
        try:
            # Create permanent mature node in registry
            node_spec = {
                "id": self.trial_node_id,
                "type": "TERMINAL",
                "group": "mature",  # Permanent mature node
                "factory": "recon_lite.learning.m5_structure:create_pattern_sensor",
                "pattern_signature": self.trial_signature,
                "transient": False,  # Permanent!
                "meta": {
                    "cell_id": self.cell_id,
                    "promoted_tick": self.trial_tick,
                    "solidified_tick": current_tick,
                    "consistency": self.trial_consistency,
                    "sample_count": len(self.samples),
                    "final_xp": self.xp,
                    "xp_successes": self.xp_successes,
                    "xp_failures": self.xp_failures,
                    "tier": "mature",
                }
            }
            
            # Add mature node to registry
            registry.add_node(node_spec, tick=current_tick)
            
            # Wire to parent with full weight
            registry.add_edge(
                self.trial_parent_id, 
                self.trial_node_id, 
                "SUB", 
                weight=1.0, 
                tick=current_tick
            )
            self.trial_edge_key = f"{self.trial_parent_id}->{self.trial_node_id}:SUB"
            
            # Update state
            self.state = StemCellState.MATURE
            registry.save()
            return True
            
        except Exception as e:
            self.metadata["solidify_error"] = str(e)
            return False
    
    def demote_to_exploring(self, registry: Any) -> bool:
        """
        Demote cell back to EXPLORING (XP depleted).
        
        Since TRIAL state is tracked internally (no registry node yet),
        we just reset the cell state and keep samples for continued learning.
        
        Args:
            registry: TopologyRegistry (unused, kept for API compat)
            
        Returns:
            True if successful
        """
        if self.state != StemCellState.TRIAL:
            return False
        
        # Reset to EXPLORING state
        self.state = StemCellState.EXPLORING
        self.xp = 0
        self.xp_successes = 0
        self.xp_failures = 0
        self.trial_node_id = None
        self.trial_edge_key = None
        self.trial_parent_id = None
        self.trial_signature = None
        self.trial_tick = 0
        self.trial_consistency = 0.0
        
        # Keep some samples for continued learning
        if len(self.samples) > 50:
            # Keep most recent 50 samples
            self.samples = self.samples[-50:]
        
        self.exploration_ticks = 0
        return True
    
    def get_xp_stats(self) -> Dict[str, Any]:
        """Get XP statistics for reporting."""
        return {
            "xp": self.xp,
            "xp_successes": self.xp_successes,
            "xp_failures": self.xp_failures,
            "xp_ratio": self.xp_successes / max(1, self.xp_failures),
            "trial_node_id": self.trial_node_id,
            "state": self.state.name,
        }
    
    # =========================================================================
    # M5 Intelligent Growth - Success-Triggered Spawning
    # =========================================================================
    
    def spawn_neighbors(
        self,
        manager: "StemCellManager",
        high_utility_features: Optional[List[str]] = None,
        spawn_count: int = 2,
        target_leg: Optional[str] = None,
    ) -> List[str]:
        """
        Spawn neighbor cells on successful solidification (100 XP).
        
        RECURSIVE OUTGROWTH RULE: Only called when cell reaches MATURE (100 XP).
        Children are Specialized Variations of the parent's successful pattern.
        
        M5 VERTICAL PARENTING: Children link to THIS node as their "local_root",
        NOT to backbone (kpk_detect). This breaks the flat topology and creates
        hierarchical tactical reasoning trees.
        
        For Sensors: Spawn AND gates combining this sensor with high-utility features.
        For Scripts: Spawn actuator legs to explore alternative strategies.
        
        Args:
            manager: StemCellManager to spawn new cells
            high_utility_features: Feature names to combine with (for sensors)
            spawn_count: Number of neighbors to spawn (default 2-3)
            target_leg: Specific leg to target ('kpk_pawn_leg' or 'kpk_king_leg')
            
        Returns:
            List of spawned cell IDs
        """
        # RECURSIVE OUTGROWTH: Only spawn at 100 XP solidification
        if self.state != StemCellState.MATURE:
            return []
        
        spawned_ids = []
        
        # Determine if this is a sensor or script based on trial signature
        is_sensor = self.pattern_signature is not None
        
        for i in range(spawn_count):
            new_cell = manager.spawn_cell()
            if new_cell is None:
                break
            
            # =====================================================================
            # M5 VERTICAL PARENTING - KEY CHANGE: local_root_id
            # Children link to THIS SOLID node as their local root, not backbone.
            # This enables hierarchical growth: Sensor -> Sub-goal -> Leg -> Backbone
            # =====================================================================
            new_cell.metadata["local_root_id"] = self.trial_node_id  # VERTICAL PARENT!
            new_cell.metadata["parent_cell_id"] = self.cell_id
            new_cell.metadata["parent_node_id"] = self.trial_node_id
            new_cell.metadata["parent_xp"] = self.xp  # For sparsity comparison
            new_cell.metadata["spawn_reason"] = "recursive_outgrowth"
            new_cell.metadata["spawn_type"] = "sensor_and_gate" if is_sensor else "script_leg"
            
            # LEG-TARGETED SPAWNING: Children should target specific legs
            if target_leg:
                new_cell.metadata["target_leg"] = target_leg
            
            # =====================================================================
            # INHERIT TACTICAL CONTEXT - Specialized Variations
            # Children inherit parent's pattern signature as a prior for faster
            # convergence. They become specialized variations, not random mutations.
            # =====================================================================
            if self.pattern_signature:
                # Copy parent signature as starting context
                new_cell.metadata["inherit_context"] = self.pattern_signature.copy()
                new_cell.metadata["initial_signature"] = self.pattern_signature.copy()
                # Seed pattern_centroid with parent's pattern for faster learning
                new_cell.pattern_centroid = self.pattern_signature.copy()
                
                if high_utility_features:
                    new_cell.metadata["combine_with"] = high_utility_features[:2]
            
            spawned_ids.append(new_cell.cell_id)
        
        self.metadata["spawned_neighbors"] = spawned_ids
        return spawned_ids
    
    # =========================================================================
    # M5.1 FAILURE-DRIVEN SPAWNING: Spawn during failure states
    # =========================================================================
    
    # Exploration spawn parameters (penalized children)
    EXPLORATION_CHILD_START_XP = 30  # Lower than normal (50)
    EXPLORATION_DECAY_MULTIPLIER = 1.5  # Faster decay
    
    def spawn_exploration_children(
        self,
        manager: "StemCellManager",
        spawn_count: int = 2,
        target_leg: Optional[str] = None,
    ) -> List[str]:
        """
        Spawn children during failure states (exploration mode).
        
        HYBRID FIX: Unlike spawn_neighbors (requires MATURE), this spawns from
        TRIAL nodes when win_rate < 10%. Children have penalties:
        - Start at 30 XP (not 50)
        - 1.5x decay rate
        
        Args:
            manager: StemCellManager to spawn new cells
            spawn_count: Number of children to spawn
            target_leg: Specific leg to target
            
        Returns:
            List of spawned cell IDs
        """
        # Can spawn from TRIAL nodes during failure (not just MATURE)
        if self.state != StemCellState.TRIAL:
            return []
        
        # Require minimum activation count to ensure node is "active but failing"
        activation_count = self.metadata.get("engagement_activations", 0)
        if activation_count < 30:
            return []
        
        spawned_ids = []
        
        for i in range(spawn_count):
            new_cell = manager.spawn_cell()
            if new_cell is None:
                break
            
            # Vertical parenting: link to this TRIAL node
            new_cell.metadata["local_root_id"] = self.trial_node_id
            new_cell.metadata["parent_cell_id"] = self.cell_id
            new_cell.metadata["parent_node_id"] = self.trial_node_id
            
            # EXPLORATION PENALTIES
            new_cell.xp = self.EXPLORATION_CHILD_START_XP  # 30 XP instead of 50
            new_cell.metadata["spawn_reason"] = "failure_exploration"
            new_cell.metadata["decay_multiplier"] = self.EXPLORATION_DECAY_MULTIPLIER  # 1.5x
            new_cell.metadata["speculative"] = True  # Track as speculation
            
            # Leg targeting
            if target_leg:
                new_cell.metadata["target_leg"] = target_leg
            
            # Inherit pattern context
            if self.pattern_signature:
                new_cell.metadata["inherit_context"] = self.pattern_signature.copy()
                new_cell.pattern_centroid = self.pattern_signature.copy()
            
            spawned_ids.append(new_cell.cell_id)
        
        self.metadata["exploration_children"] = self.metadata.get("exploration_children", []) + spawned_ids
        return spawned_ids
    
    def spawn_with_lottery(
        self,
        manager: "StemCellManager",
        graph: Any,  # Graph from registry
        current_tick: int,
        win_rate: float = 0.0,  # Current stage win rate for adaptive bias
        backward_chain_sentinel: Optional[Dict[str, Any]] = None,  # Previous mastered stage sensor
    ) -> Dict[str, Any]:
        """
        Probabilistically spawn template pack, single cell, or variant.
        
        DYNAMIC LOTTERY BIAS: Adapts probabilities based on win_rate.
        - win_rate < 0.10: 60% pack, 30% single, 10% variant (favor structure in stalls)
        - win_rate < 0.30: 50% pack, 40% single, 10% variant (balanced)
        - win_rate >= 0.30: 30% pack, 50% single, 20% variant (favor creativity in success)
        
        BACKWARD CHAINING: If backward_chain_sentinel is provided, use it as the
        pack's sentinel_fn - success = reaching the previous mastered stage config.
        
        Args:
            manager: StemCellManager to spawn new cells
            graph: ReCoN graph for pack injection
            current_tick: Current tick for naming
            win_rate: Current stage win rate for adaptive bias
            backward_chain_sentinel: Info about previous mastered stage for chaining
            
        Returns:
            {"type": "pack"|"single"|"variant", "ids": [...]}
        """
        import os
        
        # Dynamic lottery bias based on win_rate
        if win_rate < 0.10:
            # Heavy stall - favor structure (packs) to bootstrap learning
            pack_prob = 0.60
            single_prob = 0.30
            # variant_prob = 0.10
        elif win_rate < 0.30:
            # Mid-range - balanced exploration
            pack_prob = float(os.environ.get("M5_PACK_PROB", "0.50"))
            single_prob = float(os.environ.get("M5_SINGLE_PROB", "0.40"))
        else:
            # Success mode - still spawn packs for structure building
            # Increased from 0.30 to enable deeper topology growth
            pack_prob = 0.50
            single_prob = 0.40
            # variant_prob = 0.10
        
        roll = random.random()
        
        if roll < pack_prob and graph is not None:
            # Spawn template pack - randomly select type!
            try:
                import os
                from recon_lite.nodes.pack_template import (
                    spawn_goal_delegation_pack,
                    spawn_and_gate_pack,
                    spawn_sequence_pack,
                    spawn_phase_triad_pack,
                )
                
                # Choose pack type based on situation
                pack_roll = random.random()
                pack_type = os.environ.get("M5_PACK_TYPE", "auto")
                
                # Make condition/action functions
                condition_fn = self._make_pattern_sensor_fn()
                sentinel_fn = lambda env: env.get("reward", 0) > 0.3
                actuator_fn = self._make_exploration_actuator_fn()
                depth = self.metadata.get("depth", 0) + 1
                
                # VERTICAL GROWTH: Try to spawn from TRIAL nodes for deeper hierarchies
                # 50% chance to use a TRIAL node as parent (if available)
                # This creates depth 3+ structures instead of everything at depth 2
                parent_id = None
                trial_parent_candidates = []
                
                # Find TRIAL/SOLID nodes that can be parents (SCRIPT type only)
                for nid, node in graph.nodes.items():
                    if nid.startswith("TRIAL_") or nid.startswith("SOLID_"):
                        # Only SCRIPT nodes can have SUB children
                        if node.ntype.name == "SCRIPT":
                            trial_parent_candidates.append(nid)
                    # Also include promoted pack gates as candidates
                    elif "_gate" in nid and node.ntype.name == "SCRIPT":
                        trial_parent_candidates.append(nid)
                
                # DUAL-DEPTH SPAWNING: Spawn at BOTH backbone AND deeper level
                # This ensures we always have level 1 sensors while also building hierarchies
                backbone_parent = "kpk_execute" if "kpk_execute" in graph.nodes else "kpk_detect"
                
                pack_ids = {}
                
                # FORCED: Only AND-gate for now (other pack types have POR→TERMINAL bugs)
                pack_type = "and"
                
                # 1. ALWAYS spawn one pack at backbone (level 1 sensor)
                if pack_type == "and":
                    conditions = [condition_fn, lambda env: env.get("can_progress", True)]
                    pack_ids = spawn_and_gate_pack(
                        gate_name=f"and_{self.cell_id}_{current_tick}",
                        parent_id=backbone_parent,
                        graph=graph,
                        conditions=conditions,
                        then_action=actuator_fn,
                        is_trial=True,
                    )
                    if pack_ids:
                        self.metadata["pack_type"] = "and_gate"
                        print(f"      🔗 AND-gate spawned (L1): {pack_ids.get('gate')}")
                
                # 2. ADDITIONALLY spawn one deeper pack if vertical candidates exist
                if trial_parent_candidates:
                    deep_parent = random.choice(trial_parent_candidates)
                    deep_pack_ids = spawn_and_gate_pack(
                        gate_name=f"and_{self.cell_id}_{current_tick}_deep",
                        parent_id=deep_parent,
                        graph=graph,
                        conditions=[condition_fn, lambda env: env.get("can_progress", True)],
                        then_action=actuator_fn,
                        is_trial=True,
                    )
                    if deep_pack_ids:
                        print(f"      🌲 AND-gate spawned (deep under {deep_parent}): {deep_pack_ids.get('gate')}")
                        
                elif pack_type == "sequence" or (pack_type == "auto" and pack_roll < 0.60):
                    # SEQUENCE: 25% chance - for multi-step tactics
                    steps = [
                        {"sensor": condition_fn, "actuator": None},
                        {"sensor": None, "actuator": actuator_fn},
                    ]
                    pack_ids = spawn_sequence_pack(
                        seq_name=f"seq_{self.cell_id}_{current_tick}",
                        parent_id=parent_id,
                        graph=graph,
                        steps=steps,
                        final_sentinel=sentinel_fn,
                        is_trial=True,
                    )
                    if pack_ids:
                        self.metadata["pack_type"] = "sequence"
                        print(f"      🔗 Sequence spawned: {pack_ids.get('root')}")
                        
                elif pack_type == "triad" or (pack_type == "auto" and pack_roll < 0.75):
                    # PHASE TRIAD: 15% chance - check→move→wait
                    wait_fn = lambda env: env.get("game_over", False) or env.get("made_progress", False)
                    pack_ids = spawn_phase_triad_pack(
                        triad_name=f"triad_{self.cell_id}_{current_tick}",
                        parent_id=parent_id,
                        graph=graph,
                        check_fn=condition_fn,
                        move_fn=actuator_fn,
                        wait_fn=wait_fn,
                        is_trial=True,
                    )
                    if pack_ids:
                        self.metadata["pack_type"] = "phase_triad"
                        print(f"      🔗 Triad spawned: {pack_ids.get('root')}")
                        
                else:
                    # GOAL DELEGATION: 25% chance - original full pack
                    pack_ids = spawn_goal_delegation_pack(
                        goal_name=f"explore_{self.cell_id}_{current_tick}",
                        parent_id=parent_id,
                        graph=graph,
                        condition_sensor_fn=condition_fn,
                        sentinel_fn=sentinel_fn,
                        actuator_fn=actuator_fn,
                        depth=depth,
                        is_trial=True,
                        parent_signature=self.pattern_signature,
                        attach_stem_cells=1,
                        mutate_edges=True,
                    )
                    if pack_ids:
                        self.metadata["pack_type"] = "goal_delegation"
                        print(f"      🔗 Goal pack spawned: {pack_ids.get('root')}")
                
                # FORCED TEST: Also spawn OR-gate for comparison
                if pack_ids and graph is not None:
                    try:
                        from recon_lite.nodes.pack_template import spawn_or_gate_pack
                        or_pack = spawn_or_gate_pack(
                            gate_name=f"or_{self.cell_id}_{current_tick}",
                            parent_id=parent_id,
                            graph=graph,
                            conditions=[condition_fn, lambda env: env.get("alternative", False)],
                            then_action=actuator_fn,
                            is_trial=True,
                        )
                        if or_pack:
                            print(f"      🔗 OR-gate also spawned: {or_pack.get('gate')}")
                    except Exception as e:
                        print(f"      ⚠ OR-gate spawn failed: {e}")
                
                if pack_ids:
                    self.metadata["spawned_pack_root"] = pack_ids.get("root") or pack_ids.get("gate")
                    self.metadata["spawned_pack"] = True
                    return {"type": "pack", "pack_type": self.metadata.get("pack_type"), "ids": pack_ids}
                    
            except ImportError as e:
                print(f"[M5] Pack import error: {e}")
                pass  # Fall through to single
        
        if roll < pack_prob + single_prob:
            # Spawn single TERMINAL (legacy behavior)
            new_cells = self.spawn_exploration_children(manager, spawn_count=1)
            return {"type": "single", "ids": new_cells}
        
        else:
            # Spawn variant (mutated pattern signature)
            new_cells = self.spawn_exploration_children(manager, spawn_count=1)
            for cell_id in new_cells:
                if cell_id in manager.cells:
                    cell = manager.cells[cell_id]
                    # Apply heavy mutation (50%)
                    if cell.pattern_centroid:
                        mutated = []
                        for v in cell.pattern_centroid:
                            if random.random() < 0.5:
                                mutated.append(v + random.gauss(0, 0.2))
                            else:
                                mutated.append(v)
                        cell.pattern_centroid = mutated
                    cell.metadata["spawn_type"] = "variant"
            return {"type": "variant", "ids": new_cells}
    
    def _make_pattern_sensor_fn(self) -> Callable[[Dict[str, Any]], bool]:
        """Create condition sensor function based on pattern signature."""
        signature = self.pattern_signature or []
        threshold = 0.6  # Similarity threshold for activation
        
        def sensor_fn(env: Dict[str, Any]) -> bool:
            # Check if current features match pattern signature
            features = env.get("features", [])
            if not features or not signature:
                return True  # Default active if no pattern
            
            # Simple dot product similarity
            if len(features) != len(signature):
                return True
            
            similarity = sum(f * s for f, s in zip(features, signature))
            norm = (sum(f**2 for f in features) * sum(s**2 for s in signature)) ** 0.5
            if norm > 0:
                similarity /= norm
            
            return similarity > threshold
        
        return sensor_fn
    
    def _make_exploration_actuator_fn(self) -> Callable[[Dict[str, Any]], Optional[str]]:
        """Create exploration actuator (placeholder for domain-specific logic)."""
        def actuator_fn(env: Dict[str, Any]) -> Optional[str]:
            # Return suggested move from environment if available
            return env.get("suggested_move")
        
        return actuator_fn

    
    # =========================================================================
    # M5 Survival Bond - Hierarchical Dependency
    # =========================================================================
    
    def accelerated_decay(self, multiplier: float = 2.0) -> int:
        """
        Apply accelerated XP decay (survival bond: parent death).
        
        When a parent node is pruned, children experience 2x decay rate.
        
        Args:
            multiplier: Decay rate multiplier (default 2x)
            
        Returns:
            XP after accelerated decay
        """
        if self.state != StemCellState.TRIAL:
            return self.xp
        
        # Apply multiplied decay
        accelerated_decay = int(self.XP_DECAY * multiplier)
        self.xp += accelerated_decay  # XP_DECAY is negative
        self.metadata["orphan_decay_applied"] = True
        return self.xp
    
    def is_orphaned(self, live_node_ids: Set[str]) -> bool:
        """
        Check if this cell's parent has been pruned.
        
        Args:
            live_node_ids: Set of currently active node IDs
            
        Returns:
            True if parent was pruned (orphaned)
        """
        parent_id = self.metadata.get("parent_node_id")
        if not parent_id:
            return False  # No parent = not orphaned
        return parent_id not in live_node_ids

    
    def specialize(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to specialize into a full sensor.
        
        Returns:
            Sensor specification dict if successful, None if failed
        """
        if self.state != StemCellState.CANDIDATE:
            return None
        
        consistency, signature = self.analyze_pattern()
        
        if consistency < self.config.specialization_threshold:
            # Not consistent enough
            self.state = StemCellState.PRUNED
            return None
        
        # Successful specialization
        self.state = StemCellState.SPECIALIZED
        self.trust_score = consistency
        
        # Create sensor specification
        sensor_spec = {
            "sensor_id": f"sensor_{self.cell_id}",
            "type": "pattern_detector",
            "pattern_signature": signature,
            "threshold": self.config.specialization_threshold,
            "sample_count": len(self.samples),
            "consistency": consistency,
            "source_cell": self.cell_id,
            "metadata": {
                "avg_reward": sum(s.reward for s in self.samples) / len(self.samples),
                "sample_fens": [s.fen for s in self.samples[:10]],  # Example positions
            },
        }
        
        return sensor_spec
    
    def _prune_samples(self) -> None:
        """Remove low-value samples to stay under max."""
        if len(self.samples) <= self.config.max_samples:
            return
        
        # Keep samples with highest absolute reward
        self.samples.sort(key=lambda s: abs(s.reward), reverse=True)
        self.samples = self.samples[:self.config.max_samples]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize stem cell to dict."""
        return {
            "cell_id": self.cell_id,
            "state": self.state.name,
            "samples": [s.to_dict() for s in self.samples],
            "exploration_ticks": self.exploration_ticks,
            "pattern_signature": self.pattern_signature,
            "trust_score": self.trust_score,
            "config": {
                "min_samples": self.config.min_samples,
                "max_samples": self.config.max_samples,
                "reward_threshold": self.config.reward_threshold,
                "specialization_threshold": self.config.specialization_threshold,
                "exploration_budget": self.config.exploration_budget,
            },
            "metadata": self.metadata,
            # XP System fields
            "xp": self.xp,
            "xp_successes": self.xp_successes,
            "xp_failures": self.xp_failures,
            "trial_node_id": self.trial_node_id,
            "trial_edge_key": self.trial_edge_key,
            "pattern_centroid": self.pattern_centroid,
            # Inertia Pruning tracking
            "last_confirm_cycle": self.last_confirm_cycle,
            # M5.1 Composition fields
            "parent_xp_owner": self.parent_xp_owner,
            "grace_games": self.grace_games,
            "total_exposures": self.total_exposures,
            "min_exposure_threshold": self.min_exposure_threshold,
            "depth": self.depth,
            "children": self.children,
            "is_composition": self.is_composition,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StemCellTerminal":
        """Deserialize stem cell from dict."""
        config_data = data.get("config", {})
        config = StemCellConfig(
            min_samples=config_data.get("min_samples", 50),
            max_samples=config_data.get("max_samples", 500),
            reward_threshold=config_data.get("reward_threshold", 0.3),
            specialization_threshold=config_data.get("specialization_threshold", 0.7),
            exploration_budget=config_data.get("exploration_budget", 100),
        )
        
        cell = cls(cell_id=data["cell_id"], config=config)
        cell.state = StemCellState[data.get("state", "DORMANT")]
        cell.samples = [StemCellSample.from_dict(s) for s in data.get("samples", [])]
        cell.exploration_ticks = data.get("exploration_ticks", 0)
        cell.pattern_signature = data.get("pattern_signature")
        cell.trust_score = data.get("trust_score", 0.0)
        cell.metadata = data.get("metadata", {})
        
        # XP System fields
        cell.xp = data.get("xp", 0)
        cell.xp_successes = data.get("xp_successes", 0)
        cell.xp_failures = data.get("xp_failures", 0)
        cell.trial_node_id = data.get("trial_node_id")
        cell.trial_edge_key = data.get("trial_edge_key")
        cell.pattern_centroid = data.get("pattern_centroid")
        # Inertia Pruning tracking
        cell.last_confirm_cycle = data.get("last_confirm_cycle")
        # M5.1 Composition fields
        cell.parent_xp_owner = data.get("parent_xp_owner")
        cell.grace_games = data.get("grace_games", 20)
        cell.total_exposures = data.get("total_exposures", 0)
        cell.min_exposure_threshold = data.get("min_exposure_threshold", 50)
        cell.depth = data.get("depth", 0)
        cell.children = data.get("children", [])
        cell.is_composition = data.get("is_composition", False)
        
        return cell


# =============================================================================
# LAG META-TERMINALS: Temporal Gates for Trend Detection
# =============================================================================
# Design:
# - LagMetaTerminal wraps a source terminal and provides t-1 (previous) value
# - TemporalComparator compares current vs previous (LESS/GREATER)
# - Emergent chaining: t-k delays via recursive lags (depth limited)
# - Integrates with M5.1 spawning for discovery via co-occurrence
# =============================================================================

# Maximum lag recursion depth to prevent explosion
MAX_LAG_DEPTH = 3

# Comparator modes
class ComparatorMode(Enum):
    """Comparator types for temporal gates."""
    LESS = "less"          # Current < Previous (decreasing/shrinking)
    GREATER = "greater"    # Current > Previous (increasing/growing)
    EQUAL = "equal"        # Within tolerance
    DELTA = "delta"        # Just compute difference (continuous output)


@dataclass
class LagMetaTerminal:
    """
    Meta-terminal that tracks t-1 values of a source terminal.
    
    Enables temporal reasoning without bloating the feature vector.
    The network can discover trends (e.g., "shrinking box area") through
    co-occurrence of LagMetaTerminals with win outcomes.
    
    Design principles:
    - References source terminal via child_id (no full decoupling)
    - Mini-pipeline bubbles values: current → previous each tick
    - Recursive chaining for t-k delays (up to MAX_LAG_DEPTH)
    - XP/grace period for RL-guided pruning
    """
    
    lag_id: str                          # Unique ID for this lag terminal
    source_id: str                       # ID of the source terminal we're lagging
    feature_index: int                   # Which feature in the vector we track
    depth: int = 1                       # Lag depth (1=t-1, 2=t-2, etc.)
    
    # Value pipeline (mini-buffer)
    current_value: float = 0.0           # Value at current tick
    previous_value: float = 0.0          # Value at t-1
    
    # For recursive lags (e.g., lag of a lag for t-2)
    parent_lag_id: Optional[str] = None  # If this is a lag of another lag
    
    # State tracking
    tick_count: int = 0                  # Total ticks observed
    valid: bool = False                  # True after first update cycle
    
    # Metadata for M5.1 integration
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, new_value: float) -> float:
        """
        Update the lag pipeline with a new value.
        
        Returns the delta (current - previous) for comparator consumption.
        """
        # Bubble values
        self.previous_value = self.current_value
        self.current_value = new_value
        self.tick_count += 1
        
        # Valid after at least 2 observations
        if self.tick_count >= 2:
            self.valid = True
        
        return self.current_value - self.previous_value
    
    def get_delta(self) -> float:
        """Get the change since last tick (positive = increasing)."""
        if not self.valid:
            return 0.0
        return self.current_value - self.previous_value
    
    def is_decreasing(self, tolerance: float = 0.01) -> bool:
        """Returns True if value decreased since last tick."""
        return self.valid and self.get_delta() < -tolerance
    
    def is_increasing(self, tolerance: float = 0.01) -> bool:
        """Returns True if value increased since last tick."""
        return self.valid and self.get_delta() > tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "lag_id": self.lag_id,
            "source_id": self.source_id,
            "feature_index": self.feature_index,
            "depth": self.depth,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "parent_lag_id": self.parent_lag_id,
            "tick_count": self.tick_count,
            "valid": self.valid,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LagMetaTerminal":
        """Deserialize from persistence."""
        return cls(
            lag_id=data["lag_id"],
            source_id=data["source_id"],
            feature_index=data["feature_index"],
            depth=data.get("depth", 1),
            current_value=data.get("current_value", 0.0),
            previous_value=data.get("previous_value", 0.0),
            parent_lag_id=data.get("parent_lag_id"),
            tick_count=data.get("tick_count", 0),
            valid=data.get("valid", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TemporalComparator:
    """
    Script-like wrapper that compares current vs previous values.
    
    Fires like a SCRIPT node: (succeed, done) based on comparison result.
    Integrates with POR chains for multi-tick confirmation.
    
    Modes:
    - LESS: Fires when current < previous - tolerance (decreasing)
    - GREATER: Fires when current > previous + tolerance (increasing)
    - EQUAL: Fires when within tolerance (stable)
    - DELTA: Always fires, passes delta value (continuous)
    
    Example use case:
    - LESS(box_area) → fires when enemy king's confinement box is shrinking
    - GREATER(edge_distance) → fires when enemy king escaping edge
    """
    
    comparator_id: str                   # Unique ID
    lag_terminal_id: str                 # ID of LagMetaTerminal we compare
    mode: ComparatorMode                 # Type of comparison
    tolerance: float = 0.02              # Fuzzy tolerance for continuous signals
    
    # State
    last_result: bool = False            # Result of last comparison
    fire_count: int = 0                  # Times this comparator fired True
    total_count: int = 0                 # Total evaluations
    
    # XP system (for M5.1 pruning)
    xp: int = 0                          # Reinforcement signal
    
    # Metadata for registry patterns
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, delta: float) -> Tuple[bool, bool]:
        """
        Evaluate the comparator.
        
        Returns (succeed, done) like a SCRIPT node:
        - succeed: True if comparison matches
        - done: Always True (single-tick evaluation)
        """
        self.total_count += 1
        
        if self.mode == ComparatorMode.LESS:
            result = delta < -self.tolerance
        elif self.mode == ComparatorMode.GREATER:
            result = delta > self.tolerance
        elif self.mode == ComparatorMode.EQUAL:
            result = abs(delta) <= self.tolerance
        elif self.mode == ComparatorMode.DELTA:
            result = True  # Always fires for delta mode
        else:
            result = False
        
        self.last_result = result
        if result:
            self.fire_count += 1
        
        return (result, True)
    
    def get_activation_rate(self) -> float:
        """Fraction of evaluations where comparator fired."""
        if self.total_count == 0:
            return 0.0
        return self.fire_count / self.total_count
    
    def record_reward(self, reward: float) -> None:
        """Update XP based on reward correlation."""
        if self.last_result and reward > 0:
            self.xp += int(reward * 10)
        elif self.last_result and reward < 0:
            self.xp -= int(abs(reward) * 5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "comparator_id": self.comparator_id,
            "lag_terminal_id": self.lag_terminal_id,
            "mode": self.mode.value,
            "tolerance": self.tolerance,
            "last_result": self.last_result,
            "fire_count": self.fire_count,
            "total_count": self.total_count,
            "xp": self.xp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalComparator":
        """Deserialize from persistence."""
        return cls(
            comparator_id=data["comparator_id"],
            lag_terminal_id=data["lag_terminal_id"],
            mode=ComparatorMode(data["mode"]),
            tolerance=data.get("tolerance", 0.02),
            last_result=data.get("last_result", False),
            fire_count=data.get("fire_count", 0),
            total_count=data.get("total_count", 0),
            xp=data.get("xp", 0),
            metadata=data.get("metadata", {}),
        )


# Registry pattern for pool transfer
LAG_REGISTRY_PATTERN = {
    "type": "lag_meta_terminal",
    "template": {
        "lag_id": None,
        "source_id": None,
        "feature_index": None,
        "depth": 1,
    }
}

COMPARATOR_REGISTRY_PATTERN = {
    "type": "temporal_comparator",
    "modes": ["less", "greater", "equal", "delta"],
    "template": {
        "comparator_id": None,
        "lag_terminal_id": None,
        "mode": "less",
        "tolerance": 0.02,
    }
}


class StemCellManager:
    """
    Manages a pool of stem cells for pattern discovery.
    """
    
    def __init__(
        self,
        max_cells: int = 20,
        spawn_rate: float = 0.1,  # Probability of spawning new cell per tick
        config: Optional[StemCellConfig] = None,
        feature_extractor: Optional[Callable] = None,
        max_trial_slots: int = 15,  # SPARSITY SLEDGEHAMMER: Max TRIAL cells (reduced to force competition)
    ):
        self.max_cells = max_cells
        self.max_trial_slots = max_trial_slots  # Separate cap on TRIAL tier
        self.spawn_rate = spawn_rate
        self.default_config = config or StemCellConfig()
        self.cells: Dict[str, StemCellTerminal] = {}
        self._next_id = 0

        # Optional sparse feature mask policy (used for targeted pattern discovery)
        self.feature_mask_indices: Optional[List[int]] = None
        self.feature_mask_size_range: Tuple[int, int] = (2, 4)
        self.pattern_mode: Optional[str] = None
        self.stage_goal_features: Optional[List[str]] = None
        self.stage_goal_weights: Optional[List[float]] = None
        self.stage_match_mode: Optional[str] = None
        self.stage_goal_match_mode: Optional[str] = None
        self.stage_match_threshold: Optional[float] = None
        
        # Win-coactivation tracking for AND-gate discovery (M5 Recursive Branching)
        self.win_coactivation: Dict[Tuple[str, str], int] = {}  # (a,b) -> co-fire count
        self.win_active_counts: Dict[str, int] = {}  # cell_id -> wins where active
        
        # =========== LAG META-TERMINAL STORAGE ===========
        # Store lag terminals and comparators for temporal trend detection
        self.lag_terminals: Dict[str, LagMetaTerminal] = {}
        self.comparators: Dict[str, TemporalComparator] = {}
        self._next_lag_id = 0
        self._next_comparator_id = 0
        
        # Feature variance tracking (for proactive lag spawning on high-variance features)
        self.feature_variance: Dict[int, List[float]] = {}  # feature_index -> recent values
        self._feature_variance_window = 20  # Track last N values per feature
        
        # =========== DESPERATION TRACKING ===========
        # Track win rate changes to detect stalls and ramp exploration
        self.prev_win_rate: float = 0.0          # Previous cycle's win rate
        self.current_win_rate: float = 0.0       # Current cycle's win rate
        self.desperation: float = 0.0            # 0.0 = not desperate, 1.0 = very desperate
        self.desperation_threshold: float = 0.05  # win_delta below this triggers desperation
        self.epsilon_base: float = 1.0           # Base exploration bonus
        self.epsilon_desperation_scale: float = 2.0  # Multiplier when desperate
        
        # Default feature extractor for chess boards
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = self._default_board_features

    def _sample_feature_mask(self) -> Optional[List[int]]:
        if not self.feature_mask_indices:
            return None
        min_size, max_size = self.feature_mask_size_range
        max_size = max(min_size, min(max_size, len(self.feature_mask_indices)))
        size = random.randint(min_size, max_size)
        size = min(size, len(self.feature_mask_indices))
        return sorted(random.sample(self.feature_mask_indices, size))
    
    @staticmethod
    def _default_board_features(board) -> List[float]:
        """Default feature extractor for chess boards.
        
        Creates a unified representation for KPK and KRK endgames:
        - Piece counts and positions (12 features)
        - Pawn ranks (16 features)
        - King positions (4 features)
        - King distances and opposition (8 features)
        - Turn indicator (1 feature)
        - KRK-specific: Rook position, distances, cuts, box area (11 features)
        - Signed deltas: WR-BK and WK-BK file/rank (4 features)
        - Betweenness: BK between WK-WR on file/rank (2 features)
        - DELAYED HINT: is_mate_in_1 (disabled by default, set M5_ENABLE_MATE_HINT=1)
        
        Total: ~59 features (varies based on piece availability)
        
        OPTION C ARCHITECTURE: Mate hint is disabled by default to enable pure
        discovery via M5.1 emergence mechanics (co-occurrence, recursive spawning).
        Enable only when stuck to help the network break through plateaus.
        """
        if not HAS_CHESS or not hasattr(board, 'piece_map'):
            return []
        
        import chess
        features = []
        
        # Piece counts (12 features: 6 piece types x 2 colors)
        for color in [chess.WHITE, chess.BLACK]:
            for ptype in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]:
                count = len(board.pieces(ptype, color))
                features.append(float(count))
        
        # Pawn ranks (16 features: up to 8 pawns x 2 colors, padded)
        for color in [chess.WHITE, chess.BLACK]:
            pawn_ranks = []
            for sq in board.pieces(chess.PAWN, color):
                rank = chess.square_rank(sq) if color == chess.WHITE else 7 - chess.square_rank(sq)
                pawn_ranks.append(float(rank) / 7.0)  # Normalize to 0-1
            # Pad to 8 pawns
            pawn_ranks.extend([0.0] * (8 - len(pawn_ranks)))
            features.extend(pawn_ranks[:8])
        
        # King positions (4 features: file, rank for each king)
        wk_file, wk_rank, bk_file, bk_rank = 0.5, 0.5, 0.5, 0.5  # Defaults
        for color in [chess.WHITE, chess.BLACK]:
            king_sqs = list(board.pieces(chess.KING, color))
            if king_sqs:
                sq = king_sqs[0]
                f = float(chess.square_file(sq)) / 7.0
                r = float(chess.square_rank(sq)) / 7.0
                features.append(f)
                features.append(r)
                if color == chess.WHITE:
                    wk_file, wk_rank = f, r
                else:
                    bk_file, bk_rank = f, r
            else:
                features.extend([0.0, 0.0])
        
        # King distance to pawn (1 feature)
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        white_king = list(board.pieces(chess.KING, chess.WHITE))
        pawn_file, pawn_rank = 0.5, 0.5
        if white_pawns and white_king:
            pawn_sq = white_pawns[0]
            king_sq = white_king[0]
            pawn_file = float(chess.square_file(pawn_sq)) / 7.0
            pawn_rank = float(chess.square_rank(pawn_sq)) / 7.0
            dist = max(abs(chess.square_file(pawn_sq) - chess.square_file(king_sq)),
                      abs(chess.square_rank(pawn_sq) - chess.square_rank(king_sq)))
            features.append(float(dist) / 7.0)
        else:
            features.append(0.0)
        
        # ============ NEW FEATURES FOR OPPOSITION DETECTION ============
        
        # King-to-king file distance (1 feature) - 0 = same file = possible opposition
        king_file_dist = abs(wk_file - bk_file)
        features.append(king_file_dist)
        
        # King-to-king rank distance (1 feature)
        king_rank_dist = abs(wk_rank - bk_rank)
        features.append(king_rank_dist)
        
        # King-to-king Chebyshev distance (1 feature) - overall king proximity
        king_chebyshev_dist = max(king_file_dist, king_rank_dist)
        features.append(king_chebyshev_dist)
        
        # Same file indicator (1 feature) - 1.0 if kings on same file
        same_file = 1.0 if abs(wk_file - bk_file) < 0.01 else 0.0
        features.append(same_file)
        
        # Turn indicator (1 feature) - 1.0 = white to move, 0.0 = black to move
        turn = 1.0 if board.turn == chess.WHITE else 0.0
        features.append(turn)
        
        # Pawn-to-promotion distance (1 feature) - how close is pawn to 8th rank
        if white_pawns:
            promo_dist = 1.0 - pawn_rank  # Lower = closer to promotion
        else:
            promo_dist = 1.0
        features.append(promo_dist)
        
        # Enemy king to pawn distance (1 feature) - defender approach
        black_king = list(board.pieces(chess.KING, chess.BLACK))
        if white_pawns and black_king:
            pawn_sq = white_pawns[0]
            bk_sq = black_king[0]
            enemy_dist = max(abs(chess.square_file(pawn_sq) - chess.square_file(bk_sq)),
                           abs(chess.square_rank(pawn_sq) - chess.square_rank(bk_sq)))
            features.append(float(enemy_dist) / 7.0)
        else:
            features.append(1.0)
        
        # Opposition indicator (1 feature) - 1.0 if perfect opposition
        # Opposition = same file AND odd rank distance (1, 3, 5)
        rank_dist_squares = int(abs(wk_rank * 7 - bk_rank * 7) + 0.5)
        has_opposition = same_file > 0.5 and rank_dist_squares % 2 == 1
        features.append(1.0 if has_opposition else 0.0)
        
        # ============ KRK-SPECIFIC FEATURES (Rook Endgame) ============
        # These features are critical for learning the "box method" in KRK endgames
        
        white_rooks = list(board.pieces(chess.ROOK, chess.WHITE))
        wk_sq = white_king[0] if white_king else None
        bk_sq = black_king[0] if black_king else None
        
        if white_rooks:
            rook_sq = white_rooks[0]
            rook_file = float(chess.square_file(rook_sq)) / 7.0
            rook_rank = float(chess.square_rank(rook_sq)) / 7.0
            
            # Rook position (2 features)
            features.append(rook_file)
            features.append(rook_rank)
            
            # King-rook distance (1 feature) - how close is our king to our rook
            if wk_sq:
                kr_dist = max(abs(chess.square_file(rook_sq) - chess.square_file(wk_sq)),
                             abs(chess.square_rank(rook_sq) - chess.square_rank(wk_sq)))
                features.append(float(kr_dist) / 7.0)
            else:
                features.append(0.5)
            
            # Rook-enemy-king distance (1 feature) - how close is rook to enemy king
            if bk_sq:
                re_dist = max(abs(chess.square_file(rook_sq) - chess.square_file(bk_sq)),
                             abs(chess.square_rank(rook_sq) - chess.square_rank(bk_sq)))
                features.append(float(re_dist) / 7.0)
            else:
                features.append(0.5)
            
            # King protects rook (1 feature) - 1.0 if king adjacent to rook
            if wk_sq:
                king_adj_rook = (abs(chess.square_file(rook_sq) - chess.square_file(wk_sq)) <= 1 and
                               abs(chess.square_rank(rook_sq) - chess.square_rank(wk_sq)) <= 1)
                features.append(1.0 if king_adj_rook else 0.0)
            else:
                features.append(0.0)
            
            # Adjacent row indicator (1 feature) - 1.0 if king and rook on adjacent rows
            if wk_sq:
                adj_rows = abs(chess.square_rank(rook_sq) - chess.square_rank(wk_sq)) == 1
                features.append(1.0 if adj_rows else 0.0)
            else:
                features.append(0.0)
            
            # Rook cuts rank (1 feature) - 1.0 if rook is between our king and enemy king (rank)
            if wk_sq and bk_sq:
                rook_r = chess.square_rank(rook_sq)
                wk_r = chess.square_rank(wk_sq)
                bk_r = chess.square_rank(bk_sq)
                rook_cuts_rank = (min(wk_r, bk_r) < rook_r < max(wk_r, bk_r))
                features.append(1.0 if rook_cuts_rank else 0.0)
            else:
                features.append(0.0)
            
            # Rook cuts file (1 feature) - 1.0 if rook is between our king and enemy king (file)
            if wk_sq and bk_sq:
                rook_f = chess.square_file(rook_sq)
                wk_f = chess.square_file(wk_sq)
                bk_f = chess.square_file(bk_sq)
                rook_cuts_file = (min(wk_f, bk_f) < rook_f < max(wk_f, bk_f))
                features.append(1.0 if rook_cuts_file else 0.0)
            else:
                features.append(0.0)
            
            # Box area (1 feature) - size of area enemy king is confined to
            # Smaller box = closer to mate
            if bk_sq:
                bk_file = chess.square_file(bk_sq)
                bk_rank = chess.square_rank(bk_sq)
                # Box is bounded by edges and rook position
                box_width = min(bk_file, 7 - bk_file)
                box_height = min(bk_rank, 7 - bk_rank)
                box_area = (box_width + 1) * (box_height + 1) / 64.0  # Normalized
                features.append(box_area)
            else:
                features.append(1.0)
            
            # Enemy king on edge (1 feature) - 1.0 if enemy king on board edge
            if bk_sq:
                bk_file = chess.square_file(bk_sq)
                bk_rank = chess.square_rank(bk_sq)
                on_edge = bk_file in (0, 7) or bk_rank in (0, 7)
                features.append(1.0 if on_edge else 0.0)
            else:
                features.append(0.0)
            
            # ============ SIGNED DELTAS (4 features) ============
            # Signed deltas enable symmetry/transfer learning across board
            
            # Signed delta file WR-BK (1 feature) - positive = rook right of enemy king
            if bk_sq:
                wr_file = chess.square_file(rook_sq)
                bk_f = chess.square_file(bk_sq)
                signed_delta_file_wr_bk = (wr_file - bk_f) / 7.0  # Normalized -1 to 1
                features.append(signed_delta_file_wr_bk)
            else:
                features.append(0.0)
            
            # Signed delta rank WR-BK (1 feature) - positive = rook above enemy king
            if bk_sq:
                wr_rank = chess.square_rank(rook_sq)
                bk_r = chess.square_rank(bk_sq)
                signed_delta_rank_wr_bk = (wr_rank - bk_r) / 7.0
                features.append(signed_delta_rank_wr_bk)
            else:
                features.append(0.0)
            
            # Signed delta file WK-BK (1 feature) - positive = our king right of enemy
            if wk_sq and bk_sq:
                wk_f = chess.square_file(wk_sq)
                bk_f = chess.square_file(bk_sq)
                signed_delta_file_wk_bk = (wk_f - bk_f) / 7.0
                features.append(signed_delta_file_wk_bk)
            else:
                features.append(0.0)
            
            # Signed delta rank WK-BK (1 feature) - positive = our king above enemy
            if wk_sq and bk_sq:
                wk_r = chess.square_rank(wk_sq)
                bk_r = chess.square_rank(bk_sq)
                signed_delta_rank_wk_bk = (wk_r - bk_r) / 7.0
                features.append(signed_delta_rank_wk_bk)
            else:
                features.append(0.0)
            
            # ============ BETWEENNESS (2 features) ============
            # Captures tempo dynamics - is enemy king interposed between our pieces?
            
            # BK between WK-WR on file (signed) - 1.0 if BK between and "ahead" of WR
            if wk_sq and bk_sq:
                wk_f = chess.square_file(wk_sq)
                bk_f = chess.square_file(bk_sq)
                wr_f = chess.square_file(rook_sq)
                
                # Check if BK is between WK and WR on file
                min_f, max_f = min(wk_f, wr_f), max(wk_f, wr_f)
                if min_f < bk_f < max_f:
                    # BK is between - sign indicates direction from WR perspective
                    between_file = 1.0 if bk_f > wr_f else -1.0
                else:
                    between_file = 0.0
                features.append(between_file)
            else:
                features.append(0.0)
            
            # BK between WK-WR on rank (signed) - 1.0 if BK between and "above" WR
            if wk_sq and bk_sq:
                wk_r = chess.square_rank(wk_sq)
                bk_r = chess.square_rank(bk_sq)
                wr_r = chess.square_rank(rook_sq)
                
                # Check if BK is between WK and WR on rank
                min_r, max_r = min(wk_r, wr_r), max(wk_r, wr_r)
                if min_r < bk_r < max_r:
                    # BK is between - sign indicates direction from WR perspective
                    between_rank = 1.0 if bk_r > wr_r else -1.0
                else:
                    between_rank = 0.0
                features.append(between_rank)
            else:
                features.append(0.0)
                
        else:
            # No rook - pad with zeros (17 features now: 11 original + 4 signed + 2 between)
            features.extend([0.0] * 17)
        
        # ============ MATE-IN-1 DETECTION (1 feature) - DELAYED HINT ============
        # OPTION C: Disabled by default for pure discovery mode
        # Set M5_ENABLE_MATE_HINT=1 to enable when stuck (after trying pure emergence)
        # This gives M5.1 a chance to discover mate patterns from lower-level sensors
        import os
        enable_mate_hint = os.environ.get("M5_ENABLE_MATE_HINT", "0") == "1"
        
        if enable_mate_hint:
            is_mate_in_1 = 0.0
            try:
                for move in board.legal_moves:
                    board.push(move)
                    if board.is_checkmate():
                        is_mate_in_1 = 1.0
                        board.pop()
                        break
                    board.pop()
            except Exception:
                pass
            features.append(is_mate_in_1)
        else:
            # Pure discovery mode - append 0 to keep feature vector length consistent
            features.append(0.0)
        
        return features
    
    def spawn_cell(self, config: Optional[StemCellConfig] = None) -> Optional[StemCellTerminal]:
        """Spawn a new stem cell if under limit."""
        active = [c for c in self.cells.values() if c.state in (
            StemCellState.DORMANT, StemCellState.EXPLORING, StemCellState.CANDIDATE
        )]
        
        if len(active) >= self.max_cells:
            return None
        
        cell_id = f"stem_{self._next_id:04d}"
        self._next_id += 1
        
        cell = StemCellTerminal(
            cell_id=cell_id,
            config=config or self.default_config,
            feature_extractor=self.feature_extractor,  # Pass feature extractor!
        )
        cell.activate()
        mask = self._sample_feature_mask()
        if mask:
            cell.metadata["feature_mask"] = mask
        if self.pattern_mode:
            cell.metadata["pattern_mode"] = self.pattern_mode
        if self.stage_goal_features:
            cell.metadata["goal_features"] = list(self.stage_goal_features)
        if self.stage_goal_weights:
            cell.metadata["goal_weights"] = list(self.stage_goal_weights)
        if self.stage_match_mode:
            cell.metadata["match_mode"] = self.stage_match_mode
        if self.stage_goal_match_mode:
            cell.metadata["goal_match_mode"] = self.stage_goal_match_mode
        if self.stage_match_threshold is not None:
            cell.metadata["threshold"] = self.stage_match_threshold
        self.cells[cell_id] = cell
        return cell
    
    def tick(
        self,
        board: Any,
        reward: float,
        tick: int = 0,
    ) -> List[str]:
        """
        Process one tick for all active stem cells.
        
        Returns:
            List of cell IDs that stored samples
        """
        stored = []
        
        # Maybe spawn new cell
        if random.random() < self.spawn_rate:
            self.spawn_cell()
        
        # Update all active cells
        for cell_id, cell in list(self.cells.items()):
            if cell.state in (StemCellState.EXPLORING, StemCellState.CANDIDATE):
                if cell.observe(board, reward, tick):
                    stored.append(cell_id)
        
        return stored
    
    def update_trial_xp(self, affordance_delta: float) -> Dict[str, Tuple[int, str]]:
        """
        Update XP for all TRIAL cells based on affordance delta.
        
        Called after each move to provide move-level feedback.
        
        Args:
            affordance_delta: Change in affordance (positive = good move)
            
        Returns:
            Dict mapping cell_id to (xp_change, result)
        """
        results = {}
        for cell_id, cell in self.cells.items():
            if cell.state == StemCellState.TRIAL:
                xp_change, result = cell.update_xp(affordance_delta)
                results[cell_id] = (xp_change, result)
        return results

    def update_trial_xp_for_cells(
        self,
        cell_ids: Iterable[str],
        affordance_delta: float,
    ) -> Dict[str, Tuple[int, str]]:
        """
        Update XP for a specific subset of TRIAL cells.

        Args:
            cell_ids: Iterable of stem cell IDs to update
            affordance_delta: Change in affordance (positive = good move)

        Returns:
            Dict mapping cell_id to (xp_change, result)
        """
        results = {}
        for cell_id in cell_ids:
            cell = self.cells.get(cell_id)
            if cell and cell.state == StemCellState.TRIAL:
                xp_change, result = cell.update_xp(affordance_delta)
                results[cell_id] = (xp_change, result)
        return results

    def update_actuator_xp_for_cells(
        self,
        cell_ids: Iterable[str],
        affordance_delta: float,
    ) -> Dict[str, Tuple[int, str]]:
        """
        Update actuator XP for a subset of TRIAL cells.
        """
        results = {}
        for cell_id in cell_ids:
            cell = self.cells.get(cell_id)
            if cell and cell.state == StemCellState.TRIAL:
                xp_change, result = cell.update_actuator_xp(affordance_delta)
                results[cell_id] = (xp_change, result)
        return results

    def nudge_goal_vector(
        self,
        cell_id: str,
        actual_features: Dict[str, float],
        goal_features: List[str],
        reward_signal: float,
        lr: float = 0.1,
    ) -> bool:
        """
        Move a cell's goal vector toward (or away from) the observed features.
        """
        cell = self.cells.get(cell_id)
        if not cell:
            return False

        if reward_signal == 0.0:
            return False

        direction = 1.0 if reward_signal > 0 else -1.0
        goal_vec = list(cell.metadata.get("goal_vector") or [0.0] * len(goal_features))
        if len(goal_vec) < len(goal_features):
            goal_vec += [0.0] * (len(goal_features) - len(goal_vec))
        elif len(goal_vec) > len(goal_features):
            goal_vec = goal_vec[: len(goal_features)]

        for idx, name in enumerate(goal_features):
            actual = actual_features.get(name, 0.0)
            goal_vec[idx] += direction * lr * (actual - goal_vec[idx])
            goal_vec[idx] = max(-1.0, min(1.0, goal_vec[idx]))

        cell.metadata["goal_vector"] = goal_vec
        cell.metadata["goal_features"] = list(goal_features)
        return True

    def ensure_goal_actuator_for_cell(
        self,
        cell: StemCellTerminal,
        registry: "TopologyRegistry",
        parent_node_id: str,
        tick: int = 0,
    ) -> Optional[str]:
        """
        Ensure a goal actuator node exists for a promoted sensor cell.
        """
        if not cell.trial_node_id:
            return None

        existing = cell.metadata.get("goal_actuator_node_id")
        if existing:
            return existing

        node_id = f"goal_act_{cell.cell_id}"
        if registry.get_node(node_id) is not None:
            cell.metadata["goal_actuator_node_id"] = node_id
            return node_id

        node_spec = {
            "id": node_id,
            "type": "TERMINAL",
            "group": "stem_goal_actuator",
            "factory": "recon_lite_chess.goal_actuators:create_goal_actuator",
            "pattern_signature": cell.pattern_signature or [],
            "meta": {
                "cell_id": cell.cell_id,
                "source_trial_node": cell.trial_node_id,
                "subgraph": "krk",
            },
        }
        feature_mask = cell.metadata.get("feature_mask")
        if feature_mask:
            node_spec["meta"]["feature_mask"] = list(feature_mask)
        match_mode = cell.metadata.get("match_mode")
        if match_mode:
            node_spec["meta"]["match_mode"] = match_mode
        threshold = cell.metadata.get("threshold")
        if threshold is not None:
            node_spec["meta"]["threshold"] = threshold
        goal_features = cell.metadata.get("goal_features")
        if goal_features:
            node_spec["meta"]["goal_features"] = list(goal_features)
        goal_weights = cell.metadata.get("goal_weights")
        if goal_weights:
            node_spec["meta"]["goal_weights"] = list(goal_weights)
        goal_match_mode = cell.metadata.get("goal_match_mode")
        if goal_match_mode:
            node_spec["meta"]["goal_match_mode"] = goal_match_mode
        registry.add_node(node_spec, tick=tick)

        try:
            if registry.get_edge(parent_node_id, node_id, "SUB") is None:
                registry.add_edge(parent_node_id, node_id, "SUB", weight=0.5, tick=tick)
        except Exception:
            pass

        registry.save()
        cell.metadata["goal_actuator_node_id"] = node_id
        cell.metadata.setdefault("actuator_xp", cell.XP_INITIAL)
        return node_id

    def ensure_turn_composites_for_cell(
        self,
        cell: StemCellTerminal,
        registry: "TopologyRegistry",
        parent_node_id: Optional[str] = None,
        tick: int = 0,
    ) -> List[str]:
        """
        Spawn AND(pattern, white_to_play) and AND(pattern, black_to_play) composites.
        """
        if not cell.trial_node_id:
            return []

        composites = cell.metadata.setdefault("turn_composites", {})
        parent_id = parent_node_id or cell.metadata.get("trial_prep", {}).get("parent_id") or "krk_detect"
        created = []

        for turn_id in ("white", "black"):
            node_id = f"and_{cell.cell_id}_{turn_id}_to_play"
            if composites.get(turn_id):
                continue
            if registry.get_node(node_id) is None:
                node_spec = {
                    "id": node_id,
                    "type": "SCRIPT",
                    "group": "stem_turn_composite",
                    "meta": {
                        "aggregation": "and",
                        "source_trial_node": cell.trial_node_id,
                        "source_cell_id": cell.cell_id,
                        "turn": turn_id,
                    },
                }
                registry.add_node(node_spec, tick=tick)

            try:
                if registry.get_edge(parent_id, node_id, "SUB") is None:
                    registry.add_edge(parent_id, node_id, "SUB", weight=0.5, tick=tick)
            except Exception:
                continue

            try:
                if registry.get_edge(node_id, cell.trial_node_id, "SUB") is None:
                    registry.add_edge(node_id, cell.trial_node_id, "SUB", weight=0.5, tick=tick)
            except Exception:
                pass

            turn_node = f"krk_{turn_id}_to_play"
            if registry.get_node(turn_node) is not None:
                try:
                    if registry.get_edge(node_id, turn_node, "SUB") is None:
                        registry.add_edge(node_id, turn_node, "SUB", weight=0.5, tick=tick)
                except Exception:
                    pass

            composites[turn_id] = node_id
            created.append(node_id)

        if created:
            registry.save()

        return created
    
    def get_trial_cells(self) -> List[StemCellTerminal]:
        """Get all cells in TRIAL state."""
        return [c for c in self.cells.values() if c.state == StemCellState.TRIAL]
    
    def can_promote_to_trial(self) -> bool:
        """
        Check if there's room for another TRIAL cell.
        
        SPARSITY SLEDGEHAMMER: Limits TRIAL slots to force competition.
        Only the top-performing sensors get to stay in TRIAL tier.
        
        Returns:
            True if there's room for another TRIAL cell
        """
        current_trial_count = len(self.get_trial_cells())
        return current_trial_count < self.max_trial_slots
    
    def get_trial_capacity(self) -> Dict[str, int]:
        """Get TRIAL tier capacity info."""
        current = len(self.get_trial_cells())
        return {
            "current": current,
            "max": self.max_trial_slots,
            "available": max(0, self.max_trial_slots - current),
        }
    
    def get_trial_stats(self) -> Dict[str, Any]:
        """Get statistics about TRIAL cells."""
        trial_cells = self.get_trial_cells()
        return {
            "count": len(trial_cells),
            "xp_values": {c.cell_id: c.xp for c in trial_cells},
            "total_xp": sum(c.xp for c in trial_cells),
            "avg_xp": sum(c.xp for c in trial_cells) / max(1, len(trial_cells)),
        }
    
    def get_specialization_candidates(self) -> List[StemCellTerminal]:
        """Get cells ready for specialization."""
        return [c for c in self.cells.values() if c.should_specialize()]
    
    def specialize_all_ready(self) -> List[Dict[str, Any]]:
        """Attempt to specialize all ready cells."""
        specs = []
        for cell in self.get_specialization_candidates():
            spec = cell.specialize()
            if spec:
                specs.append(spec)
        return specs
    
    def prune_failed(self) -> int:
        """Remove pruned cells from pool."""
        pruned = [cid for cid, c in self.cells.items() if c.state == StemCellState.PRUNED]
        for cid in pruned:
            del self.cells[cid]
        return len(pruned)
    
    def prune_inert_cells(self, current_cycle: int, max_inactive: int = 20) -> int:
        """
        Prune TRIAL cells that haven't contributed to CONFIRM signals.
        
        This is "Inertia Pruning" - removing cells that fire but don't
        actually influence network behavior. Forces sparsity and survival
        of the fittest.
        
        Args:
            current_cycle: Current training cycle number
            max_inactive: Maximum cycles without CONFIRM before pruning
            
        Returns:
            Number of cells pruned
        """
        inert_cells = [
            cid for cid, cell in self.cells.items()
            if cell.state == StemCellState.TRIAL and cell.is_inert(current_cycle, max_inactive)
        ]
        
        for cid in inert_cells:
            self.cells[cid].state = StemCellState.PRUNED
            self.cells[cid].metadata["pruned_reason"] = "inertia"
            self.cells[cid].metadata["pruned_cycle"] = current_cycle
            self.cells[cid].metadata["cycles_without_confirm"] = (
                current_cycle - (self.cells[cid].last_confirm_cycle or 0)
            )
        
        return len(inert_cells)
    
    def mark_cells_confirmed(self, confirmed_node_ids: List[str], current_cycle: int) -> int:
        """
        Mark cells as confirmed based on their graph node states.
        
        Called after a game/tick to update inertia tracking for cells whose
        corresponding TRIAL nodes reached CONFIRMED state.
        
        Args:
            confirmed_node_ids: List of graph node IDs that reached CONFIRMED
            current_cycle: Current training cycle number
            
        Returns:
            Number of cells marked as confirmed
        """
        count = 0
        for cid, cell in self.cells.items():
            if cell.state == StemCellState.TRIAL and cell.trial_node_id:
                if cell.trial_node_id in confirmed_node_ids:
                    cell.mark_confirmed(current_cycle)
                    count += 1
        return count
    
    # =========== LAG META-TERMINAL METHODS ===========
    # These enable emergent discovery of temporal trends via M5.1
    
    def spawn_lag_terminal(
        self,
        source_id: str,
        feature_index: int,
        parent_lag_id: Optional[str] = None,
    ) -> Optional[LagMetaTerminal]:
        """
        Spawn a new lag terminal for a feature.
        
        Can create recursive lags (lag of a lag) for t-k delays up to MAX_LAG_DEPTH.
        
        Args:
            source_id: ID of the source terminal/cell we're tracking
            feature_index: Which feature in the vector to track
            parent_lag_id: If set, this is a lag of another lag (for t-2, t-3, etc.)
        
        Returns:
            New LagMetaTerminal or None if depth limit reached
        """
        # Calculate depth
        depth = 1
        if parent_lag_id and parent_lag_id in self.lag_terminals:
            parent = self.lag_terminals[parent_lag_id]
            depth = parent.depth + 1
            if depth > MAX_LAG_DEPTH:
                return None  # Depth limit reached
        
        # Check for duplicate
        for lag in self.lag_terminals.values():
            if lag.source_id == source_id and lag.feature_index == feature_index:
                if lag.parent_lag_id == parent_lag_id:
                    return lag  # Already exists
        
        lag_id = f"lag_{self._next_lag_id:04d}"
        self._next_lag_id += 1
        
        lag = LagMetaTerminal(
            lag_id=lag_id,
            source_id=source_id,
            feature_index=feature_index,
            depth=depth,
            parent_lag_id=parent_lag_id,
        )
        
        self.lag_terminals[lag_id] = lag
        return lag
    
    def spawn_comparator(
        self,
        lag_terminal_id: str,
        mode: ComparatorMode = ComparatorMode.LESS,
        tolerance: float = 0.02,
    ) -> Optional[TemporalComparator]:
        """
        Spawn a comparator for a lag terminal.
        
        Args:
            lag_terminal_id: ID of the lag terminal to compare
            mode: Type of comparison (LESS, GREATER, EQUAL, DELTA)
            tolerance: Fuzzy tolerance for continuous signals
        
        Returns:
            New TemporalComparator or None if lag doesn't exist
        """
        if lag_terminal_id not in self.lag_terminals:
            return None
        
        # Check for duplicate
        for comp in self.comparators.values():
            if comp.lag_terminal_id == lag_terminal_id and comp.mode == mode:
                return comp  # Already exists
        
        comp_id = f"comp_{self._next_comparator_id:04d}"
        self._next_comparator_id += 1
        
        comp = TemporalComparator(
            comparator_id=comp_id,
            lag_terminal_id=lag_terminal_id,
            mode=mode,
            tolerance=tolerance,
        )
        
        self.comparators[comp_id] = comp
        return comp
    
    def update_lags(self, features: List[float]) -> Dict[str, float]:
        """
        Update all lag terminals with new feature values.
        
        Also updates feature variance tracking for proactive spawning.
        
        Args:
            features: Current feature vector
            
        Returns:
            Dict of lag_id -> delta for comparator consumption
        """
        deltas = {}
        
        for lag_id, lag in self.lag_terminals.items():
            if lag.feature_index < len(features):
                # Get value - either from parent lag or from feature vector
                if lag.parent_lag_id and lag.parent_lag_id in self.lag_terminals:
                    # This is a lag of a lag - use parent's previous value
                    parent = self.lag_terminals[lag.parent_lag_id]
                    value = parent.previous_value
                else:
                    value = features[lag.feature_index]
                
                delta = lag.update(value)
                deltas[lag_id] = delta
        
        # Update feature variance tracking
        for i, val in enumerate(features):
            if i not in self.feature_variance:
                self.feature_variance[i] = []
            self.feature_variance[i].append(val)
            # Keep only last N values
            if len(self.feature_variance[i]) > self._feature_variance_window:
                self.feature_variance[i].pop(0)
        
        return deltas
    
    def evaluate_comparators(self, deltas: Dict[str, float]) -> Dict[str, bool]:
        """
        Evaluate all comparators with updated deltas.
        
        Args:
            deltas: Dict of lag_id -> delta from update_lags()
            
        Returns:
            Dict of comparator_id -> fired
        """
        results = {}
        for comp_id, comp in self.comparators.items():
            if comp.lag_terminal_id in deltas:
                delta = deltas[comp.lag_terminal_id]
                fired, _ = comp.evaluate(delta)
                results[comp_id] = fired
        return results
    
    def spawn_lags_for_high_variance(
        self,
        min_variance: float = 0.1,
        max_spawns: int = 3,
        allow_recursive: bool = False,
        recursive_chance: float = 0.5,
    ) -> List[str]:
        """
        Proactively spawn lag terminals for high-variance features with UCB1 exploration.
        
        High-variance features are more likely to be relevant for trend detection.
        Uses UCB1 (Upper Confidence Bound) to balance exploitation (high variance/correlation)
        with exploration (under-visited features).
        
        UCB1 score = std_dev + sqrt(2 * ln(total_visits) / (feature_visits + 1))
        
        Args:
            min_variance: Minimum variance to trigger lag spawning (default 0.1)
            max_spawns: Maximum number of lags to spawn per call
            
        Returns:
            List of newly spawned lag IDs
        """
        import math
        
        spawned = []
        
        # Initialize visit tracking if not present
        if not hasattr(self, '_feature_visits'):
            self._feature_visits = {}
        if not hasattr(self, '_total_spawn_visits'):
            self._total_spawn_visits = 1
        
        self._total_spawn_visits += 1
        
        # Build candidate list with UCB1 scores
        candidates = []
        for feature_idx, values in self.feature_variance.items():
            if len(values) < 5:  # Need minimum data
                continue
            
            # Check if we already have a lag for this feature
            feature_lags = [
                lag for lag in self.lag_terminals.values()
                if lag.feature_index == feature_idx
            ]
            has_lag = bool(feature_lags)
            if has_lag and not allow_recursive:
                continue
            
            # Compute variance and std
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance)
            
            if variance < min_variance:
                continue
            
            # UCB1 exploration bonus with desperation scaling
            # When desperate, exploration is boosted via get_epsilon()
            visits = self._feature_visits.get(feature_idx, 0)
            epsilon = getattr(self, 'epsilon_base', 1.0)
            if hasattr(self, 'get_epsilon'):
                epsilon = self.get_epsilon()  # Desperation-scaled epsilon
            ucb_bonus = epsilon * math.sqrt(2 * math.log(self._total_spawn_visits) / (visits + 1))
            
            # Combined score: std_dev + UCB bonus (prioritize high variance + uncertainty)
            ucb_score = std_dev + ucb_bonus
            
            parent_lag_id = None
            if has_lag and allow_recursive and feature_lags:
                parent_lag = max(feature_lags, key=lambda lag: lag.depth)
                if random.random() < recursive_chance:
                    parent_lag_id = parent_lag.lag_id

            candidates.append((ucb_score, feature_idx, variance, std_dev, parent_lag_id))
        
        # Sort by UCB score descending (best = highest std + uncertainty)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Spawn top candidates
        for ucb_score, feature_idx, variance, std_dev, parent_lag_id in candidates[:max_spawns]:
            lag = self.spawn_lag_terminal(
                source_id=f"feature_{feature_idx}",
                feature_index=feature_idx,
                parent_lag_id=parent_lag_id,
            )
            if lag:
                spawned.append(lag.lag_id)
                
                # Update visit count for this feature
                self._feature_visits[feature_idx] = self._feature_visits.get(feature_idx, 0) + 1
                        
                # Also spawn LESS and GREATER comparators
                self.spawn_comparator(lag.lag_id, ComparatorMode.LESS)
                self.spawn_comparator(lag.lag_id, ComparatorMode.GREATER)
        
        return spawned
    
    def update_desperation(self, win_rate: float) -> float:
        """
        Update desperation metric based on win rate change.
        
        Desperation is used to scale UCB exploration bonus:
        - High desperation (stall) → more exploration (VOLATILE cells prioritized)
        - Low desperation (improving) → more exploitation
        
        Args:
            win_rate: Current cycle's win rate (0.0 to 1.0)
            
        Returns:
            Updated desperation value (0.0 to 1.0)
        """
        # Compute win delta
        win_delta = win_rate - self.prev_win_rate
        
        # Update desperation: higher when win_delta is below threshold
        if win_delta < self.desperation_threshold:
            # Ramp desperation based on how far below threshold
            # win_delta=0.05 → desperation=0, win_delta=-0.05 → desperation=1
            self.desperation = min(1.0, max(0.0, 
                (self.desperation_threshold - win_delta) / (2 * self.desperation_threshold)
            ))
        else:
            # Improving: decay desperation
            self.desperation = max(0.0, self.desperation - 0.2)
        
        # Store for next cycle
        self.prev_win_rate = self.current_win_rate
        self.current_win_rate = win_rate
        
        return self.desperation
    
    def get_epsilon(self) -> float:
        """Get desperation-scaled exploration epsilon for UCB."""
        return self.epsilon_base * (1.0 + self.desperation * self.epsilon_desperation_scale)
    
    def tier_stats(self) -> Dict[str, int]:
        """Get count of cells by tier."""
        counts = {tier.name: 0 for tier in CellTier}
        for cell in self.cells.values():
            if hasattr(cell, 'cell_tier'):
                counts[cell.cell_tier.name] += 1
        return counts
    
    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        states = {}
        for cell in self.cells.values():
            state = cell.state.name
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_cells": len(self.cells),
            "by_state": states,
            "next_id": self._next_id,
        }
    
    def save_stem_cells(self, path: Path) -> None:
        """
        Save stem cell states to JSON for persistence between stages.
        
        This enables knowledge transfer across curriculum stages - cells don't
        reset when advancing to harder stages, allowing accumulated learning
        to compound.
        
        Args:
            path: Path to save stem cells JSON
        """
        data = {
            "next_id": self._next_id,
            "max_cells": self.max_cells,
            "max_trial_slots": self.max_trial_slots,
            "cells": {
                cell_id: cell.to_dict() for cell_id, cell in self.cells.items()
            },
            "win_coactivation": {
                f"{k[0]}|{k[1]}": v for k, v in self.win_coactivation.items()
            },
            "win_active_counts": self.win_active_counts,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    
    def load_stem_cells(self, path: Path) -> bool:
        """
        Load stem cell states from JSON.
        
        Args:
            path: Path to stem cells JSON
            
        Returns:
            True if loaded successfully, False if file doesn't exist or failed
        """
        path = Path(path)
        if not path.exists():
            return False
        
        try:
            data = json.loads(path.read_text())
            
            self._next_id = data.get("next_id", 0)
            self.max_cells = data.get("max_cells", self.max_cells)
            self.max_trial_slots = data.get("max_trial_slots", self.max_trial_slots)
            
            # Load cells
            self.cells = {}
            for cell_id, cell_data in data.get("cells", {}).items():
                try:
                    cell = StemCellTerminal.from_dict(cell_data, self.default_config)
                    self.cells[cell_id] = cell
                except Exception:
                    pass  # Skip malformed cells
            
            # Load coactivation tracking
            self.win_coactivation = {}
            for k, v in data.get("win_coactivation", {}).items():
                parts = k.split("|")
                if len(parts) == 2:
                    self.win_coactivation[(parts[0], parts[1])] = v
            
            self.win_active_counts = data.get("win_active_counts", {})
            
            return True
        except Exception:
            return False
    
    # =========================================================================
    # WIN-COACTIVATION TRACKING (M5 Recursive Branching - AND-Gate Discovery)
    # =========================================================================
    
    def track_win_coactivation(self, active_cell_ids: List[str], game_won: bool) -> None:
        """
        Track which cells fire together during wins for AND-gate detection.
        
        CRITICAL: Normalization is per-cell, not per-game. This prevents rare
        but impactful tactical patterns from being washed out by common features.
        
        Args:
            active_cell_ids: List of cell IDs that were active during the game
            game_won: Whether the game was won
        """
        if not game_won or not active_cell_ids:
            return
        
        # Track individual cell activity in wins (for normalization)
        for cell_id in active_cell_ids:
            self.win_active_counts[cell_id] = self.win_active_counts.get(cell_id, 0) + 1
        
        # Track co-activations (pairs that fire together)
        for i, a in enumerate(active_cell_ids):
            for b in active_cell_ids[i+1:]:
                key = tuple(sorted([a, b]))
                self.win_coactivation[key] = self.win_coactivation.get(key, 0) + 1
    
    def find_win_correlated_pairs(
        self, 
        min_coactivations: int = 50, 
        min_ratio: float = 0.85
    ) -> List[Tuple[str, str, float, int]]:
        """
        Find cell pairs that fire together 85%+ of the time during wins.
        
        NORMALIZATION: ratio = co_fires / min(wins_a_active, wins_b_active)
        This prevents rare but impactful patterns from being diluted by
        common positional features (Zero-Variance Trap prevention).
        
        Args:
            min_coactivations: Minimum co-fire count to consider (default 50)
            min_ratio: Minimum correlation ratio (default 0.85 = 85%)
            
        Returns:
            List of (cell_a, cell_b, ratio, co_count) tuples, sorted by ratio desc
        """
        correlated_pairs = []
        for key, co_count in self.win_coactivation.items():
            if co_count < min_coactivations:
                continue
            
            a, b = key
            
            # Normalize by the smaller of the two activity counts
            wins_a = self.win_active_counts.get(a, 0)
            wins_b = self.win_active_counts.get(b, 0)
            denominator = min(wins_a, wins_b) if min(wins_a, wins_b) > 0 else 1
            
            ratio = co_count / denominator
            if ratio >= min_ratio:
                correlated_pairs.append((a, b, ratio, co_count))
        
        return sorted(correlated_pairs, key=lambda x: -x[2])  # Sort by ratio desc
    
    def get_coactivation_stats(self) -> Dict[str, Any]:
        """Get statistics about win-coactivation tracking."""
        return {
            "tracked_pairs": len(self.win_coactivation),
            "tracked_cells": len(self.win_active_counts),
            "total_coactivations": sum(self.win_coactivation.values()),
            "top_pairs": self.find_win_correlated_pairs(min_coactivations=10, min_ratio=0.5)[:5],
        }
    
    # =========================================================================
    # M5.1 PROACTIVE COMPOSITION SPAWNING
    # =========================================================================
    
    # Tracking for already-tried compositions
    _tried_compositions: Set[Tuple[str, str, str]] = set()  # (cell_a, cell_b, mode)
    
    def spawn_composition(
        self,
        cell_ids: List[str],
        mode: str = "and",  # "and" or "or"
        parent_node_id: Optional[str] = None,
        fast_track: bool = True,
    ) -> Optional[str]:
        """
        M5.1 PROACTIVE COMPOSITION: Create an AND/OR gate from sensor cells.
        
        Creates a composition cell that tracks XP at the gate level. Child cells
        delegate their XP to this parent, and follow its fate (solidify/demote together).
        
        Args:
            cell_ids: List of child cell IDs to compose
            mode: "and" (all must fire) or "or" (any can fire)
            parent_node_id: Optional graph node to wire to
            fast_track: If True, use accelerated XP (faster promotion/demotion)
            
        Returns:
            Composition cell ID if created, None if failed (duplicate/limit)
        """
        if len(cell_ids) < 2:
            return None  # Need at least 2 cells to compose
        
        # Check if already tried this composition
        sorted_ids = tuple(sorted(cell_ids))
        composition_key = (*sorted_ids, mode)
        if composition_key in self._tried_compositions:
            return None  # Already tried
        
        # Verify all child cells exist and get their depth
        child_cells = []
        max_child_depth = 0
        for cid in cell_ids:
            cell = self.cells.get(cid)
            if not cell:
                return None  # Child not found
            child_cells.append(cell)
            max_child_depth = max(max_child_depth, cell.depth)
        
        # Check depth limit (MAX_COMPOSITION_DEPTH = 3)
        new_depth = max_child_depth + 1
        if new_depth > 3:
            return None  # Too deep
        
        # Create composition cell
        comp_id = f"{mode}_{sorted_ids[0]}_{sorted_ids[1]}_{self._next_id}"
        self._next_id += 1
        
        comp_cell = StemCellTerminal(
            cell_id=comp_id,
            config=self.default_config,
        )
        
        # Mark as composition
        comp_cell.is_composition = True
        comp_cell.children = list(cell_ids)
        comp_cell.depth = new_depth
        comp_cell.state = StemCellState.TRIAL
        
        # Fast-track XP if requested (accelerated evaluation)
        if fast_track:
            comp_cell.grace_games = 10  # Shorter grace period
            comp_cell.min_exposure_threshold = 20  # Faster decision
            comp_cell.metadata["fast_track"] = True
        
        # Store metadata
        comp_cell.metadata["composition_mode"] = mode
        comp_cell.metadata["child_ids"] = list(cell_ids)
        comp_cell.metadata["parent_node_id"] = parent_node_id
        comp_cell.metadata["spawn_reason"] = "proactive_composition"
        
        # Set up XP delegation for children
        for child_cell in child_cells:
            child_cell.parent_xp_owner = comp_id
            child_cell.metadata["delegated_to"] = comp_id
        
        # Add to cells and mark as tried
        self.cells[comp_id] = comp_cell
        self._tried_compositions.add(composition_key)
        
        return comp_id
    
    def spawn_compositions_from_correlated_pairs(
        self,
        min_coactivations: int = 30,
        min_ratio: float = 0.70,
        max_spawns: int = 3,
    ) -> List[str]:
        """
        M5.1 PROACTIVE: Auto-spawn AND gates from win-correlated pairs.
        
        Finds sensor pairs that fire together during wins and creates
        AND gates to test if the combination is valuable.
        
        Args:
            min_coactivations: Minimum co-fire count
            min_ratio: Minimum correlation ratio
            max_spawns: Maximum compositions to spawn per call
            
        Returns:
            List of new composition cell IDs
        """
        correlated = self.find_win_correlated_pairs(
            min_coactivations=min_coactivations,
            min_ratio=min_ratio,
        )
        
        spawned = []
        for cell_a, cell_b, ratio, co_count in correlated:
            if len(spawned) >= max_spawns:
                break
            
            comp_id = self.spawn_composition(
                cell_ids=[cell_a, cell_b],
                mode="and",
                fast_track=True,
            )
            
            if comp_id:
                spawned.append(comp_id)
        
        return spawned
    
    def spawn_exploratory_cells(
        self,
        count: int = 5,
        target_legs: Optional[List[str]] = None,
    ) -> List[str]:
        """
        M5.1 Emergent Spawning: Create new exploratory stem cells.
        
        Used when the agent is stuck at a "Failure Frontier" and needs
        to explore new hypotheses. Creates fresh EXPLORING cells that
        will start collecting samples immediately.
        
        Args:
            count: Number of cells to spawn
            target_legs: Optional list of leg names to target (e.g., ["kpk_king_leg"])
            
        Returns:
            List of new cell IDs
        """
        new_ids = []
        
        for i in range(count):
            if len(self.cells) >= self.max_cells:
                break
            
            # Generate unique ID
            cell_id = f"emergent_{self._next_id}"
            self._next_id += 1
            
            # Create new exploratory cell
            cell = StemCellTerminal(
                cell_id=cell_id,
                config=self.default_config,
            )
            
            # Mark as emergent spawning
            cell.metadata["spawn_reason"] = "emergent_spawning"
            cell.metadata["spawn_cycle"] = "unknown"  # Will be set by caller
            
            # Target a specific leg if specified
            if target_legs:
                target_leg = random.choice(target_legs)
                cell.metadata["target_leg"] = target_leg
                # Bias the cell toward king-related features if targeting king leg
                if "king" in target_leg.lower():
                    cell.metadata["feature_bias"] = "king_focus"
            
            # Start in EXPLORING state
            cell.state = StemCellState.EXPLORING
            
            self.cells[cell_id] = cell
            new_ids.append(cell_id)
        
        return new_ids
    
    def save(self, path: Path) -> None:
        """Save manager state including win-coactivation tracking."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tuple keys to string for JSON serialization
        win_coact_serializable = {
            f"{a}|{b}": count for (a, b), count in self.win_coactivation.items()
        }
        
        data = {
            "max_cells": self.max_cells,
            "max_trial_slots": self.max_trial_slots,  # SPARSITY: TRIAL tier cap
            "spawn_rate": self.spawn_rate,
            "next_id": self._next_id,
            "cells": {cid: c.to_dict() for cid, c in self.cells.items()},
            # Win-coactivation tracking (M5 Recursive Branching)
            "win_coactivation": win_coact_serializable,
            "win_active_counts": self.win_active_counts,
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "StemCellManager":
        """Load manager state including win-coactivation tracking."""
        with open(path) as f:
            data = json.load(f)
        
        manager = cls(
            max_cells=data.get("max_cells", 20),
            spawn_rate=data.get("spawn_rate", 0.1),
            max_trial_slots=data.get("max_trial_slots", 15),  # SPARSITY: TRIAL tier cap
        )
        manager._next_id = data.get("next_id", 0)
        manager.cells = {
            cid: StemCellTerminal.from_dict(cdata)
            for cid, cdata in data.get("cells", {}).items()
        }
        
        # Restore win-coactivation tracking (M5 Recursive Branching)
        win_coact_raw = data.get("win_coactivation", {})
        manager.win_coactivation = {
            tuple(k.split("|")): v for k, v in win_coact_raw.items()
        }
        manager.win_active_counts = data.get("win_active_counts", {})
        
        return manager

    # =========================================================================
    # VERTICAL M5 GROWTH: Sensor Hoisting
    # =========================================================================
    
    def compute_pattern_similarity(
        self, cell_a: StemCellTerminal, cell_b: StemCellTerminal
    ) -> float:
        """
        Compute pattern similarity between two cells.
        
        Uses cosine similarity of pattern_signature vectors.
        
        Returns:
            Similarity score in [0, 1] where 1 = identical patterns
        """
        sig_a = cell_a.pattern_signature
        sig_b = cell_b.pattern_signature
        
        if sig_a is None or sig_b is None:
            return 0.0
        
        if len(sig_a) != len(sig_b):
            return 0.0
        
        # Cosine similarity
        dot = sum(a * b for a, b in zip(sig_a, sig_b))
        norm_a = sum(a * a for a in sig_a) ** 0.5
        norm_b = sum(b * b for b in sig_b) ** 0.5
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        return (dot / (norm_a * norm_b) + 1) / 2  # Map [-1,1] to [0,1]
    
    # =========================================================================
    # SPATIAL CLUSTERING: Merge Similar Cells (prevent bloat)
    # =========================================================================
    
    def find_similar_cell(
        self,
        signature: List[float],
        min_similarity: float = 0.95,  # 95% = <5% centroid distance
    ) -> Optional[str]:
        """
        Find an existing cell with similar pattern signature.
        
        SPATIAL CLUSTERING: Before spawning a new cell, check if its pattern
        centroid is within 5% of an existing cell. If so, merge instead of spawn.
        
        Args:
            signature: Pattern signature to compare
            min_similarity: Minimum similarity (default 0.95 = 5% distance)
            
        Returns:
            Cell ID of similar existing cell, or None if no match
        """
        for cell_id, cell in self.cells.items():
            if cell.pattern_signature is None:
                continue
            if len(cell.pattern_signature) != len(signature):
                continue
            
            # Compute similarity
            sig_a = signature
            sig_b = cell.pattern_signature
            dot = sum(a * b for a, b in zip(sig_a, sig_b))
            norm_a = sum(a * a for a in sig_a) ** 0.5
            norm_b = sum(b * b for b in sig_b) ** 0.5
            
            if norm_a < 1e-8 or norm_b < 1e-8:
                continue
            
            similarity = (dot / (norm_a * norm_b) + 1) / 2
            
            if similarity >= min_similarity:
                return cell_id
        
        return None
    
    def merge_into_cell(
        self,
        existing_cell_id: str,
        new_samples: List[Any],
    ) -> bool:
        """
        Merge samples into an existing cell instead of spawning a new one.
        
        This prevents combinatorial bloat by consolidating similar patterns.
        
        Args:
            existing_cell_id: Cell ID to merge into
            new_samples: Samples from the would-be new cell
            
        Returns:
            True if merge successful
        """
        cell = self.cells.get(existing_cell_id)
        if cell is None:
            return False
        
        # Add samples to existing cell
        for sample in new_samples:
            cell.samples.append(sample)
        
        # Trim samples if over limit
        max_samples = cell.config.max_samples
        if len(cell.samples) > max_samples:
            cell.samples = cell.samples[-max_samples:]
        
        # Re-analyze pattern with merged samples
        cell.analyze_pattern()
        
        cell.metadata["merge_count"] = cell.metadata.get("merge_count", 0) + 1
        return True
    
    def find_correlated_clusters(
        self, min_similarity: float = 0.85, min_cluster_size: int = 2
    ) -> List[List[str]]:
        """
        Find clusters of TRIAL cells with highly correlated patterns.
        
        These clusters are candidates for hoisting into intermediate nodes.
        
        Args:
            min_similarity: Minimum cosine similarity to cluster (default 0.85)
            min_cluster_size: Minimum cells to form a cluster (default 2)
            
        Returns:
            List of cell_id clusters, each cluster is a list of IDs
        """
        # Get all TRIAL cells with pattern signatures
        trial_cells = [
            (cid, cell) for cid, cell in self.cells.items()
            if cell.state == StemCellState.TRIAL and cell.pattern_signature
        ]
        
        if len(trial_cells) < min_cluster_size:
            return []
        
        # Build similarity matrix
        n = len(trial_cells)
        similarity = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_pattern_similarity(trial_cells[i][1], trial_cells[j][1])
                similarity[i][j] = sim
                similarity[j][i] = sim
        
        # Simple greedy clustering
        clustered = set()
        clusters = []
        
        for i in range(n):
            if i in clustered:
                continue
            
            # Find all cells similar enough to this one
            cluster = [i]
            for j in range(i + 1, n):
                if j in clustered:
                    continue
                if similarity[i][j] >= min_similarity:
                    cluster.append(j)
            
            if len(cluster) >= min_cluster_size:
                cluster_ids = [trial_cells[idx][0] for idx in cluster]
                clusters.append(cluster_ids)
                clustered.update(cluster)
        
        return clusters
    
    def hoist_cluster(
        self,
        cluster_ids: List[str],
        graph: "Graph",  # type: ignore
        parent_node_id: str = "kpk_execute",
        aggregation_mode: str = "and",  # "and" = min(), "avg" = weighted average
    ) -> Optional[str]:
        """
        Hoist a cluster of correlated sensors into a new Intermediate Script Node.
        
        This implements vertical M5 growth by:
        1. Creating a new Script node as parent of the cluster
        2. Moving SUB edges from old parent to new intermediate
        3. The intermediate aggregates activations from the cluster
        
        AGGREGATION MODES:
        - "and": Uses min() - fires ONLY when ALL children are active (TRUE AND gate)
        - "avg": Uses weighted average - fires based on combined child activations
        
        Args:
            cluster_ids: List of cell IDs to hoist
            graph: The Graph to modify
            parent_node_id: Current parent of the cluster cells
            aggregation_mode: "and" for logical AND (min), "avg" for weighted average
            
        Returns:
            ID of the new intermediate node, or None if failed
        """
        from ..graph import Node, NodeType, LinkType
        
        if len(cluster_ids) < 2:
            return None
        
        # Compute merged pattern signature (average of cluster)
        signatures = []
        for cid in cluster_ids:
            cell = self.cells.get(cid)
            if cell and cell.pattern_signature:
                signatures.append(cell.pattern_signature)
        
        if not signatures:
            return None
        
        # Average signature
        merged_sig = [
            sum(s[i] for s in signatures) / len(signatures)
            for i in range(len(signatures[0]))
        ]
        
        # Create intermediate node ID
        intermediate_id = f"cluster_{self._next_id:04d}"
        self._next_id += 1
        
        # Create the intermediate Script node
        intermediate_node = Node(
            nid=intermediate_id,
            ntype=NodeType.SCRIPT,
        )
        intermediate_node.meta["cluster_members"] = cluster_ids
        intermediate_node.meta["pattern_signature"] = merged_sig
        intermediate_node.meta["origin"] = "hoisted"
        # TRUE AND GATE: Set aggregation mode for propagate_activation
        intermediate_node.meta["aggregation"] = aggregation_mode
        # SPECULATIVE: Mark as hypothesis - will be pruned if no improvement after 50 games
        intermediate_node.meta["speculative"] = True
        intermediate_node.meta["birth_game"] = 0  # Will be set by caller
        intermediate_node.meta["prune_after_games"] = 50
        
        # Add to graph
        graph.add_node(intermediate_node)
        
        # Connect intermediate to parent
        graph.add_edge(parent_node_id, intermediate_id, LinkType.SUB)
        
        # Connect cluster members as children of intermediate
        for cid in cluster_ids:
            # Check if the cell has a corresponding node in graph
            if cid in graph.nodes:
                # Remove old SUB edge from parent
                old_edges = [
                    e for e in graph.edges
                    if e.src == parent_node_id and e.dst == cid and e.ltype == LinkType.SUB
                ]
                for edge in old_edges:
                    graph.edges.remove(edge)
                
                # Add new SUB edge from intermediate
                graph.add_edge(intermediate_id, cid, LinkType.SUB)
        
        return intermediate_id
    
    def auto_hoist(
        self,
        graph: "Graph",  # type: ignore
        parent_node_id: str = "kpk_execute",
        min_similarity: float = 0.85,
    ) -> List[str]:
        """
        Automatically find and hoist all correlated clusters.
        
        Args:
            graph: The Graph to modify
            parent_node_id: Parent node for hoisted clusters
            min_similarity: Minimum similarity to cluster
            
        Returns:
            List of created intermediate node IDs
        """
        clusters = self.find_correlated_clusters(min_similarity=min_similarity)
        created = []
        
        for cluster in clusters:
            intermediate_id = self.hoist_cluster(cluster, graph, parent_node_id)
            if intermediate_id:
                created.append(intermediate_id)
        
        return created
    
    # =========================================================================
    # M5 Identity Audit - Representational Parsimony
    # =========================================================================
    
    def identity_audit(
        self,
        candidate_id: str,
        graph: "Graph",  # type: ignore
        min_improvement: float = 0.1,  # SPARSITY CONSTRAINT: 10% better than parent
    ) -> bool:
        """
        Check if a candidate node justifies its complexity over simpler alternatives.
        
        SPARSITY CONSTRAINT: A new branch is only made permanent if its z_sur
        (confirmation strength) is >10% better than existing simpler branches.
        This forces 'Elegant' solutions over 'Brute Force' clusters.
        
        Args:
            candidate_id: Cell ID to audit
            graph: The graph to check existing branches
            min_improvement: Minimum improvement ratio (default 0.2 = 20%)
            
        Returns:
            True if candidate should be kept, False if it should be pruned
        """
        cell = self.cells.get(candidate_id)
        if not cell or cell.state != StemCellState.TRIAL:
            return True  # Only audit TRIAL cells
        
        # Get candidate's confirmation strength
        candidate_sig = cell.pattern_signature
        if not candidate_sig:
            return True  # No signature = nothing to compare
        
        candidate_complexity = len([x for x in candidate_sig if abs(x) > 0.1])
        
        # Find simpler existing cells with similar pattern
        for cid, other in self.cells.items():
            if cid == candidate_id or other.state not in (StemCellState.TRIAL, StemCellState.MATURE):
                continue
            
            other_sig = other.pattern_signature
            if not other_sig:
                continue
            
            other_complexity = len([x for x in other_sig if abs(x) > 0.1])
            
            # Only compare if other is simpler
            if other_complexity >= candidate_complexity:
                continue
            
            # Check similarity
            similarity = self.compute_pattern_similarity(cell, other)
            if similarity < 0.7:  # Only compare functionally similar cells
                continue
            
            # Compare confirmation strength (XP as proxy for z_sur)
            candidate_strength = cell.xp
            other_strength = other.xp
            
            if other_strength <= 0:
                continue
            
            improvement = (candidate_strength - other_strength) / max(1, other_strength)
            
            if improvement < min_improvement:
                # Candidate doesn't justify extra complexity
                cell.metadata["audit_result"] = "prune_no_improvement"
                cell.metadata["simpler_alternative"] = cid
                cell.metadata["improvement_ratio"] = improvement
                return False
        
        cell.metadata["audit_result"] = "keep"
        return True
    
    def sparsity_audit_vs_parent(
        self,
        candidate_id: str,
        min_improvement: float = 0.1,
    ) -> bool:
        """
        SPARSITY CONSTRAINT: Compare child's z_sur specifically to its parent's.
        
        If a new node's confirmation (z_sur) is less than 10% better than its
        parent's, prune it. This forces 'Elegant' solutions over 'Brute Force'.
        
        Args:
            candidate_id: Cell ID to audit
            min_improvement: Minimum improvement ratio (default 0.1 = 10%)
            
        Returns:
            True if candidate should be kept, False if it should be pruned
        """
        cell = self.cells.get(candidate_id)
        if not cell or cell.state != StemCellState.TRIAL:
            return True  # Only audit TRIAL cells
        
        # Get parent's XP (stored during spawn)
        parent_xp = cell.metadata.get("parent_xp")
        parent_cell_id = cell.metadata.get("parent_cell_id")
        
        if parent_xp is None:
            return True  # No parent data = pass (original cells)
        
        # Compare to parent
        candidate_xp = cell.xp
        
        if parent_xp <= 0:
            return True  # Parent had no XP
        
        improvement = (candidate_xp - parent_xp) / max(1, parent_xp)
        
        if improvement < min_improvement:
            # Child doesn't improve >10% over parent → prune
            cell.metadata["sparsity_result"] = "prune_below_parent"
            cell.metadata["parent_cell_id"] = parent_cell_id
            cell.metadata["parent_xp"] = parent_xp
            cell.metadata["improvement_vs_parent"] = improvement
            return False
        
        cell.metadata["sparsity_result"] = "keep_improves_parent"
        cell.metadata["improvement_vs_parent"] = improvement
        return True
    
    # =========================================================================
    # M5 Survival Bond - Orphan Sweep
    # =========================================================================
    
    def orphan_sweep(
        self,
        graph: "Graph",  # type: ignore
        decay_multiplier: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Apply accelerated decay to orphaned cells (parent was pruned).
        
        Implements survival bond: if a parent node dies, its children's
        XP decay is accelerated by 2x, causing faster pruning of
        branches whose foundation has been removed.
        
        Args:
            graph: The graph to check for live nodes
            decay_multiplier: Decay rate multiplier (default 2x)
            
        Returns:
            Stats about orphan processing
        """
        live_nodes = set(graph.nodes.keys())
        
        orphaned = []
        accelerated = []
        demoted = []
        
        for cid, cell in self.cells.items():
            if cell.state != StemCellState.TRIAL:
                continue
            
            if cell.is_orphaned(live_nodes):
                orphaned.append(cid)
                cell.accelerated_decay(decay_multiplier)
                accelerated.append(cid)
                
                # Check if should be demoted
                if cell.xp <= 0:
                    demoted.append(cid)
        
        return {
            "orphaned": len(orphaned),
            "accelerated": accelerated,
            "demoted": demoted,
        }
    
    # =========================================================================
    # KNOWLEDGE TRANSFER - Cross-Endgame Inheritance
    # =========================================================================
    
    @classmethod
    def load_with_transfer(
        cls,
        source_path: Path,
        prefix_map: Optional[Dict[str, str]] = None,
        states_to_transfer: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        new_domain: str = "krk",
    ) -> "StemCellManager":
        """
        Load stem cells from another domain with knowledge transfer.
        
        This enables cross-endgame inheritance: sensors learned in KPK
        can be transferred to KRK, allowing "Universal Sensors" like
        king_distance to work across multiple endgames.
        
        KNOWLEDGE BANK: Cells are renamed with prefix_map to indicate
        their origin and universal applicability.
        
        Args:
            source_path: Path to source stem_cells.json (e.g., from KPK)
            prefix_map: Mapping for cell ID renaming (e.g., {"kpk_": "universal_"})
            states_to_transfer: List of state names to transfer (default: ["TRIAL", "MATURE"])
            top_n: Only transfer top N cells by XP (default: all matching states)
            new_domain: New domain name for metadata (default: "krk")
            
        Returns:
            New StemCellManager with transferred cells
        """
        # Default prefix map
        if prefix_map is None:
            prefix_map = {
                "kpk_sensor_": "universal_sensor_",
                "kpk_": "universal_",
                "stem_": "universal_stem_",
                "TRIAL_": "universal_TRIAL_",
            }
        
        # Default states to transfer
        if states_to_transfer is None:
            states_to_transfer = ["TRIAL", "MATURE"]
        
        # Load source manager
        with open(source_path) as f:
            data = json.load(f)
        
        # Create new manager
        manager = cls(
            max_cells=data.get("max_cells", 50),  # Increase for transfer
            spawn_rate=data.get("spawn_rate", 0.1),
        )
        
        # Filter cells by state
        source_cells = data.get("cells", {})
        transferable = []
        
        for cell_id, cell_data in source_cells.items():
            state = cell_data.get("state", "DORMANT")
            if state in states_to_transfer:
                cell_data["_original_id"] = cell_id
                cell_data["_xp"] = cell_data.get("xp", 0)
                transferable.append(cell_data)
        
        # Sort by XP descending and take top N
        transferable.sort(key=lambda c: c.get("_xp", 0), reverse=True)
        if top_n is not None:
            transferable = transferable[:top_n]
        
        # Transfer cells with renaming
        transferred_count = 0
        for cell_data in transferable:
            old_id = cell_data["_original_id"]
            
            # Apply prefix mapping
            new_id = old_id
            for old_prefix, new_prefix in prefix_map.items():
                if old_id.startswith(old_prefix):
                    new_id = new_prefix + old_id[len(old_prefix):]
                    break
            else:
                # No prefix match - add universal prefix
                new_id = f"universal_{old_id}"
            
            # Update cell_id
            cell_data["cell_id"] = new_id
            
            # Add transfer metadata
            cell_data["metadata"] = cell_data.get("metadata", {})
            cell_data["metadata"]["origin"] = "kpk_transfer"
            cell_data["metadata"]["original_id"] = old_id
            cell_data["metadata"]["transfer_domain"] = new_domain
            cell_data["metadata"]["source_xp"] = cell_data.get("xp", 0)
            
            # Reset XP to 50% of original for new domain
            # This gives transferred cells a head start but requires them to prove utility
            original_xp = cell_data.get("xp", 0)
            cell_data["xp"] = max(25, original_xp // 2)
            
            # Keep pattern signature for universal features
            # The signature captures patterns that may work across domains
            
            # Create cell from dict
            try:
                cell = StemCellTerminal.from_dict(cell_data)
                manager.cells[new_id] = cell
                transferred_count += 1
            except Exception as e:
                print(f"Warning: Failed to transfer cell {old_id}: {e}")
        
        # Copy win-coactivation data with renaming
        win_coact_raw = data.get("win_coactivation", {})
        for key, count in win_coact_raw.items():
            try:
                a, b = key.split("|")
                # Rename if in transferred cells
                new_a = next((c.cell_id for c in manager.cells.values() 
                              if c.metadata.get("original_id") == a), a)
                new_b = next((c.cell_id for c in manager.cells.values() 
                              if c.metadata.get("original_id") == b), b)
                manager.win_coactivation[(new_a, new_b)] = count
            except Exception:
                pass
        
        # Copy win counts with renaming
        for cell_id, count in data.get("win_active_counts", {}).items():
            new_id = next((c.cell_id for c in manager.cells.values() 
                          if c.metadata.get("original_id") == cell_id), cell_id)
            manager.win_active_counts[new_id] = count
        
        manager._next_id = max(
            int(cid.split("_")[-1]) if cid.split("_")[-1].isdigit() else 0
            for cid in manager.cells.keys()
        ) + 1 if manager.cells else 0
        
        print(f"✅ Transferred {transferred_count} cells from {source_path}")
        print(f"   States: {states_to_transfer}")
        print(f"   Top N: {top_n if top_n else 'all'}")
        
        return manager
    
    def get_transferred_cells(self) -> List[StemCellTerminal]:
        """Get all cells that were transferred from another domain."""
        return [
            c for c in self.cells.values()
            if c.metadata.get("origin") == "kpk_transfer"
        ]
    
    def compute_sensor_reuse_ratio(
        self,
        active_cell_ids: List[str],
        game_won: bool,
    ) -> float:
        """
        Compute the ratio of transferred sensors contributing to wins.
        
        BRIDGE METRIC: Tracks how many "KPK-born" sensors are contributing
        to KRK wins. A ratio > 0.5 indicates successful knowledge transfer.
        
        Args:
            active_cell_ids: Cell IDs that were active in this game
            game_won: Whether the game was won
            
        Returns:
            Ratio of transferred cells among active cells (0.0 to 1.0)
        """
        if not active_cell_ids:
            return 0.0
        
        # Count transferred cells among active
        transferred_active = sum(
            1 for cid in active_cell_ids
            if cid in self.cells and 
            self.cells[cid].metadata.get("origin") == "kpk_transfer"
        )
        
        ratio = transferred_active / len(active_cell_ids)
        
        # Track in metadata for reporting
        if game_won:
            self.metadata = getattr(self, 'metadata', {})
            if 'reuse_ratios' not in self.metadata:
                self.metadata['reuse_ratios'] = []
            self.metadata['reuse_ratios'].append(ratio)
            # Keep last 100
            self.metadata['reuse_ratios'] = self.metadata['reuse_ratios'][-100:]
        
        return ratio
    
    def get_reuse_stats(self) -> Dict[str, Any]:
        """Get statistics about sensor reuse from transfers."""
        transferred = self.get_transferred_cells()
        metadata = getattr(self, 'metadata', {})
        ratios = metadata.get('reuse_ratios', [])
        
        return {
            "transferred_count": len(transferred),
            "transferred_by_state": {
                state.name: len([c for c in transferred if c.state == state])
                for state in StemCellState
            },
            "avg_reuse_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
            "recent_reuse_ratios": ratios[-10:] if ratios else [],
            "high_reuse_games": sum(1 for r in ratios if r > 0.5),
        }
    
    def get_active_transferred_cells(
        self,
        current_features: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get IDs of transferred cells that are actively contributing.
        
        For transfer tracking, we consider a transferred cell "active" if:
        1. It's in TRIAL state (hasn't been pruned)
        2. It has samples (captured patterns)
        
        This provides a simpler metric than pattern matching, since
        feature spaces may differ between KPK and KRK domains.
        
        Args:
            current_features: Optional current position features (not used yet)
            
        Returns:
            List of cell IDs that are actively contributing
        """
        active = []
        for cell in self.get_transferred_cells():
            # Cell is "active" if it's still in play (TRIAL state)
            # and has captured patterns
            if cell.state == StemCellState.TRIAL and len(cell.samples) > 0:
                active.append(cell.cell_id)
                # Also track via trial_node_id if present
                if cell.trial_node_id:
                    active.append(cell.trial_node_id)
        return active
    
    def compute_transfer_contribution(self, game_won: bool) -> float:
        """
        Compute how much transferred cells are contributing to outcomes.
        
        This is a simpler metric that counts transferred TRIAL cells
        as "contributing" if they're still active (not pruned).
        
        Args:
            game_won: Whether the current game was won
            
        Returns:
            Ratio of surviving transferred cells to total transferred
        """
        transferred = self.get_transferred_cells()
        if not transferred:
            return 0.0
        
        # Count cells still in TRIAL (surviving)
        surviving = sum(
            1 for c in transferred 
            if c.state == StemCellState.TRIAL
        )
        
        ratio = surviving / len(transferred)
        
        # Track contribution over time
        if game_won:
            metadata = getattr(self, 'metadata', {})
            if 'transfer_contributions' not in metadata:
                metadata['transfer_contributions'] = []
            metadata['transfer_contributions'].append(ratio)
            # Keep last 100
            metadata['transfer_contributions'] = metadata['transfer_contributions'][-100:]
            self.metadata = metadata
        
        return ratio
