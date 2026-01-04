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
    
    # Promote to trial when consistency > 0.35
    if stem.can_enter_trial():
        stem.promote_to_trial(registry, parent_id)
    
    # Update XP based on move outcomes
    stem.update_xp(affordance_delta)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
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



@dataclass
class StemCellConfig:
    """Configuration for stem cell behavior."""
    min_samples: int = 50  # Minimum samples before specialization attempt
    max_samples: int = 500  # Maximum samples to collect
    reward_threshold: float = 0.3  # Only store samples with |reward| > this
    specialization_threshold: float = 0.7  # Min pattern consistency for specialization
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fen": self.fen,
            "features": self.features,
            "reward": self.reward,
            "tick": self.tick,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StemCellSample":
        return cls(
            fen=data["fen"],
            features=data.get("features"),
            reward=data.get("reward", 0.0),
            tick=data.get("tick", 0),
            metadata=data.get("metadata", {}),
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
                if hasattr(features, 'tolist'):
                    features = features.tolist()
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
        if not self.samples or not HAS_NUMPY:
            return 0.0, None
        
        # Get samples with features
        featured = [s for s in self.samples if s.features is not None]
        if len(featured) < 10:
            return 0.0, None
        
        # Stack features
        try:
            features = np.array([s.features for s in featured])
        except Exception:
            return 0.0, None
        
        # Compute centroid
        centroid = np.mean(features, axis=0)
        
        # Compute consistency (inverse of average distance to centroid)
        distances = np.linalg.norm(features - centroid, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        # Consistency: 1.0 if all samples are identical, lower if spread out
        consistency = 1.0 - min(1.0, avg_distance / (max_distance + 1e-6))
        
        # Store centroid for pattern signature and similarity comparison
        self.pattern_signature = centroid.tolist()
        self.pattern_centroid = self.pattern_signature
        return float(consistency), self.pattern_signature
    
    # =========================================================================
    # XP SYSTEM - THREE-TIER LIFECYCLE MANAGEMENT
    # =========================================================================
    
    def can_enter_trial(self, min_samples: int = 100, min_consistency: float = 0.35) -> bool:
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
        parent_id: str,
        current_tick: int = 0,
    ) -> bool:
        """
        Promote cell to TRIAL tier as a transient vertex.
        
        Creates a temporary node in the registry that can be solidified
        or removed based on XP performance.
        
        Args:
            registry: TopologyRegistry for metadata (node created on solidification)
            parent_id: Parent node to wire to
            current_tick: Current tick for metadata
            
        Returns:
            True if promotion successful
        """
        if self.state not in (StemCellState.CANDIDATE, StemCellState.EXPLORING):
            return False
        
        consistency, signature = self.analyze_pattern()
        if consistency < 0.35:
            return False
        
        # Generate trial node ID (will be created on solidification)
        self.trial_node_id = f"TRIAL_{self.cell_id}_{current_tick}"
        
        # Store trial preparation data (don't add to registry yet to avoid graph issues)
        self.trial_parent_id = parent_id
        self.trial_consistency = consistency
        self.trial_signature = signature
        self.trial_tick = current_tick
        
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
        
        if affordance_delta > 0.1:
            # Success: positive affordance delta
            xp_change = self.XP_SUCCESS
            self.xp_successes += 1
            result = "success"
        elif affordance_delta < -0.1:
            # Failure: negative affordance delta
            xp_change = self.XP_FAILURE
            self.xp_failures += 1
            result = "failure"
        else:
            # Neutral: no significant change
            xp_change = 0
            result = "neutral"
        
        self.xp += xp_change
        return xp_change, result
    
    def decay_xp(self) -> int:
        """
        Apply XP decay (cost of living).
        
        Called once per cycle for all TRIAL cells.
        
        Returns:
            XP after decay
        """
        if self.state != StemCellState.TRIAL:
            return self.xp
        
        self.xp += self.XP_DECAY
        return self.xp
    
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
        
        return cell


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
    ):
        self.max_cells = max_cells
        self.spawn_rate = spawn_rate
        self.default_config = config or StemCellConfig()
        self.cells: Dict[str, StemCellTerminal] = {}
        self._next_id = 0
        
        # Default feature extractor for chess boards
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = self._default_board_features
    
    @staticmethod
    def _default_board_features(board) -> List[float]:
        """Default feature extractor for chess boards.
        
        Creates a simple representation including:
        - Piece counts and positions
        - Pawn rank for each pawn
        - King distances
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
        for color in [chess.WHITE, chess.BLACK]:
            king_sqs = list(board.pieces(chess.KING, color))
            if king_sqs:
                sq = king_sqs[0]
                features.append(float(chess.square_file(sq)) / 7.0)
                features.append(float(chess.square_rank(sq)) / 7.0)
            else:
                features.extend([0.0, 0.0])
        
        # King distance to pawn (1 feature)
        white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
        white_king = list(board.pieces(chess.KING, chess.WHITE))
        if white_pawns and white_king:
            pawn_sq = white_pawns[0]
            king_sq = white_king[0]
            dist = max(abs(chess.square_file(pawn_sq) - chess.square_file(king_sq)),
                      abs(chess.square_rank(pawn_sq) - chess.square_rank(king_sq)))
            features.append(float(dist) / 7.0)
        else:
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
    
    def get_trial_cells(self) -> List[StemCellTerminal]:
        """Get all cells in TRIAL state."""
        return [c for c in self.cells.values() if c.state == StemCellState.TRIAL]
    
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
    def save(self, path: Path) -> None:
        """Save manager state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_cells": self.max_cells,
            "spawn_rate": self.spawn_rate,
            "next_id": self._next_id,
            "cells": {cid: c.to_dict() for cid, c in self.cells.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "StemCellManager":
        """Load manager state."""
        with open(path) as f:
            data = json.load(f)
        
        manager = cls(
            max_cells=data.get("max_cells", 20),
            spawn_rate=data.get("spawn_rate", 0.1),
        )
        manager._next_id = data.get("next_id", 0)
        manager.cells = {
            cid: StemCellTerminal.from_dict(cdata)
            for cid, cdata in data.get("cells", {}).items()
        }
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
    ) -> Optional[str]:
        """
        Hoist a cluster of correlated sensors into a new Intermediate Script Node.
        
        This implements vertical M5 growth by:
        1. Creating a new Script node as parent of the cluster
        2. Moving SUB edges from old parent to new intermediate
        3. The intermediate aggregates activations from the cluster
        
        Args:
            cluster_ids: List of cell IDs to hoist
            graph: The Graph to modify
            parent_node_id: Current parent of the cluster cells
            
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

