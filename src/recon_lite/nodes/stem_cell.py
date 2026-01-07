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
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
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
        
        # Inertia Pruning: Track when cell last contributed to CONFIRM signal
        # Used to prune TRIAL cells that haven't been useful for N cycles
        self.last_confirm_cycle: Optional[int] = None
    
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
        parent_id: str = "kpk_detect",
        current_tick: int = 0,
        min_consistency: float = 0.50,  # Balanced threshold (was 0.35, then 0.65)
        wire_to_legs: bool = True,  # NEW: Also wire as child of legs for gating
        leg_node_ids: Optional[List[str]] = None,  # NEW: Which legs to wire to
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
        
        node_spec = {
            "id": self.trial_node_id,
            "type": "TERMINAL",
            "group": "trial",
            "factory": None,
            "meta": {
                "cell_id": self.cell_id,
                "tier": "trial",
                "consistency": consistency,
                "promoted_tick": current_tick,
                "pattern_signature": signature,
                "subgraph": subgraph_name,  # CRITICAL: Required for subgraph execution
            }
        }
        
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
    ) -> Dict[str, Any]:
        """
        Probabilistically spawn template pack, single cell, or variant.
        
        LOTTERY PROBABILITIES (env var configurable):
        - M5_PACK_PROB=0.40: Full Goal Delegation Pack (template)
        - M5_SINGLE_PROB=0.40: Single TERMINAL cell (legacy)
        - M5_VARIANT_PROB=0.20: Mutated variant (signature shuffle)
        
        Args:
            manager: StemCellManager to spawn new cells
            graph: ReCoN graph for pack injection
            current_tick: Current tick for naming
            
        Returns:
            {"type": "pack"|"single"|"variant", "ids": [...]}
        """
        import os
        
        # Load probabilities from env (defaults: 40/40/20)
        pack_prob = float(os.environ.get("M5_PACK_PROB", "0.40"))
        single_prob = float(os.environ.get("M5_SINGLE_PROB", "0.40"))
        # variant_prob = 1 - pack_prob - single_prob (remainder)
        
        roll = random.random()
        
        if roll < pack_prob and graph is not None:
            # Spawn full Goal Delegation Pack
            try:
                from recon_lite.nodes.pack_template import spawn_goal_delegation_pack
                
                # Make condition sensor using parent's pattern
                condition_fn = self._make_pattern_sensor_fn()
                sentinel_fn = lambda env: env.get("reward", 0) > 0.3
                actuator_fn = self._make_exploration_actuator_fn()
                
                depth = self.metadata.get("depth", 0) + 1
                
                pack_ids = spawn_goal_delegation_pack(
                    goal_name=f"explore_{self.cell_id}_{current_tick}",
                    parent_id=self.trial_node_id or "kpk_execute",
                    graph=graph,
                    condition_sensor_fn=condition_fn,
                    sentinel_fn=sentinel_fn,
                    actuator_fn=actuator_fn,
                    depth=depth,
                    is_trial=True,
                    parent_signature=self.pattern_signature,
                    attach_stem_cells=1,  # Create 1 slot for future children
                    mutate_edges=True,
                )
                
                if pack_ids:
                    self.metadata["spawned_pack_root"] = pack_ids.get("root")
                    self.metadata["spawned_pack"] = True
                    return {"type": "pack", "ids": pack_ids}
                    
            except ImportError:
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
        max_trial_slots: int = 15,  # SPARSITY SLEDGEHAMMER: Max TRIAL cells (reduced to force competition)
    ):
        self.max_cells = max_cells
        self.max_trial_slots = max_trial_slots  # Separate cap on TRIAL tier
        self.spawn_rate = spawn_rate
        self.default_config = config or StemCellConfig()
        self.cells: Dict[str, StemCellTerminal] = {}
        self._next_id = 0
        
        # Win-coactivation tracking for AND-gate discovery (M5 Recursive Branching)
        self.win_coactivation: Dict[Tuple[str, str], int] = {}  # (a,b) -> co-fire count
        self.win_active_counts: Dict[str, int] = {}  # cell_id -> wins where active
        
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
