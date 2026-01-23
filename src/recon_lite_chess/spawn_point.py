"""
Stem Cell Spawn Points for KRK_entry Legs

Attaches exploration spawn points to Leg nodes. When an affordance spike
is detected (position looks promising), stem cells explore the feature space
to discover new patterns that lead to checkmate.

Key concepts:
- **Spawn Point**: Attached to each Leg, controls when to spawn new trials
- **Affordance Spike**: When can_deliver_mate=1, spawn exploration
- **Trial Micro-script**: 3-part structure (precond → actuator → postcond)
- **XP-based Pruning**: Keep high-XP patterns, prune low-XP ones
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from recon_lite.graph import Node, NodeType, Graph
from recon_lite_chess.baseline_teacher import KRKTeacher


# Global teacher for feature extraction
_teacher = KRKTeacher()

# Configuration
SPAWN_PROBABILITY = 0.3  # Probability of spawning on affordance spike
MIN_SAMPLES_FOR_XP = 10  # Minimum samples before computing XP
XP_PRUNE_THRESHOLD = 0.2  # Prune if XP below this
XP_PROMOTE_THRESHOLD = 0.7  # Promote to permanent if XP above this


@dataclass
class SpawnPointConfig:
    """Configuration for a spawn point"""
    spawn_probability: float = SPAWN_PROBABILITY
    max_trials: int = 5  # Max concurrent trials per spawn point
    trial_lifetime: int = 50  # Prune after this many ticks without improvement


@dataclass
class TrialMicroScript:
    """
    A trial micro-script exploring one feature combination.
    
    Structure: precond sensors → actuator → postcond check
    
    The trial explores whether a specific feature delta pattern
    correlates with achieving checkmate.
    """
    trial_id: str
    spawn_point_id: str
    
    # Feature subset being explored
    feature_indices: List[int]
    
    # Learning stats
    samples: int = 0
    checkmate_hits: int = 0  # Times this pattern led to mate
    non_mate_hits: int = 0   # Times pattern activated but no mate
    xp: float = 0.0
    ticks_alive: int = 0
    last_update_tick: int = 0
    
    # Learned delta pattern
    delta_mean: Optional[np.ndarray] = field(default=None)
    delta_std: Optional[np.ndarray] = field(default=None)
    
    def update_xp(self):
        """
        Compute XP based on mate success rate.
        
        Formula: XP = checkmate_hits / total_hits * consistency
        """
        total = self.checkmate_hits + self.non_mate_hits
        if total < MIN_SAMPLES_FOR_XP:
            self.xp = 0.0
            return
        
        success_rate = self.checkmate_hits / total
        
        # Consistency bonus: low delta variance = higher XP
        consistency = 1.0
        if self.delta_std is not None and len(self.delta_std) > 0:
            mean_std = np.mean(self.delta_std)
            consistency = 1.0 / (1.0 + mean_std)
        
        self.xp = success_rate * consistency
    
    def should_prune(self, current_tick: int) -> bool:
        """Check if trial should be pruned"""
        if self.samples < MIN_SAMPLES_FOR_XP:
            return False  # Give it time to collect data
        
        if self.xp < XP_PRUNE_THRESHOLD:
            return True
        
        if current_tick - self.last_update_tick > 50:
            return True  # Stale, no recent updates
        
        return False
    
    def should_promote(self) -> bool:
        """Check if trial should be promoted to permanent node"""
        return self.samples >= MIN_SAMPLES_FOR_XP and self.xp >= XP_PROMOTE_THRESHOLD


@dataclass
class SpawnPoint:
    """
    A spawn point attached to a Leg node.
    
    Controls exploration by spawning trial micro-scripts when
    affordance spikes are detected.
    """
    spawn_point_id: str
    leg_id: str
    config: SpawnPointConfig = field(default_factory=SpawnPointConfig)
    
    active_trials: Dict[str, TrialMicroScript] = field(default_factory=dict)
    promoted_trials: List[str] = field(default_factory=list)
    pruned_trials: List[str] = field(default_factory=list)
    
    total_spawns: int = 0
    total_promotions: int = 0
    total_prunes: int = 0
    
    def should_spawn(self, features: np.ndarray) -> bool:
        """
        Check if we should spawn a new trial.
        
        Triggers on affordance spike: can_deliver_mate=1
        """
        # Feature 12 is can_deliver_mate
        can_deliver_mate = features[12] > 0.5
        
        if not can_deliver_mate:
            return False
        
        if len(self.active_trials) >= self.config.max_trials:
            return False
        
        return random.random() < self.config.spawn_probability
    
    def spawn_trial(self, tick: int) -> TrialMicroScript:
        """
        Spawn a new trial micro-script.
        
        Randomly selects a subset of features to explore.
        """
        trial_id = f"{self.spawn_point_id}_trial_{self.total_spawns}"
        self.total_spawns += 1
        
        # Randomly select 1-4 features to explore
        num_features = random.randint(1, 4)
        feature_indices = random.sample(range(_teacher.feature_dim), num_features)
        feature_indices.sort()
        
        trial = TrialMicroScript(
            trial_id=trial_id,
            spawn_point_id=self.spawn_point_id,
            feature_indices=feature_indices,
            last_update_tick=tick
        )
        
        self.active_trials[trial_id] = trial
        return trial
    
    def process_transition(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        resulted_in_checkmate: bool,
        tick: int
    ):
        """
        Update all active trials with a new transition observation.
        
        Args:
            v0: Features before move
            v1: Features after move
            resulted_in_checkmate: Whether the move achieved checkmate
            tick: Current tick number
        """
        for trial in list(self.active_trials.values()):
            # Compute delta for trial's feature subset
            delta = v1[trial.feature_indices] - v0[trial.feature_indices]
            
            # Update statistics
            trial.samples += 1
            trial.ticks_alive += 1
            trial.last_update_tick = tick
            
            if resulted_in_checkmate:
                trial.checkmate_hits += 1
            else:
                trial.non_mate_hits += 1
            
            # Update delta mean/std
            if trial.delta_mean is None:
                trial.delta_mean = delta.copy()
                trial.delta_std = np.zeros_like(delta)
            else:
                # Running mean/std update (Welford's algorithm)
                n = trial.samples
                delta_from_mean = delta - trial.delta_mean
                trial.delta_mean += delta_from_mean / n
                trial.delta_std = trial.delta_std * (n-1) / n + delta_from_mean * (delta - trial.delta_mean) / n
            
            # Update XP
            trial.update_xp()
    
    def prune_and_promote(self, tick: int) -> Tuple[List[str], List[str]]:
        """
        Prune low-XP trials and promote high-XP trials.
        
        Returns:
            (promoted_ids, pruned_ids)
        """
        promoted = []
        pruned = []
        
        for trial_id, trial in list(self.active_trials.items()):
            if trial.should_promote():
                promoted.append(trial_id)
                self.promoted_trials.append(trial_id)
                self.total_promotions += 1
                del self.active_trials[trial_id]
            elif trial.should_prune(tick):
                pruned.append(trial_id)
                self.pruned_trials.append(trial_id)
                self.total_prunes += 1
                del self.active_trials[trial_id]
        
        return promoted, pruned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get spawn point statistics"""
        return {
            "spawn_point_id": self.spawn_point_id,
            "leg_id": self.leg_id,
            "active_trials": len(self.active_trials),
            "total_spawns": self.total_spawns,
            "total_promotions": self.total_promotions,
            "total_prunes": self.total_prunes,
            "trial_xps": {tid: t.xp for tid, t in self.active_trials.items()}
        }


class SpawnPointManager:
    """
    Manages spawn points across all Legs in the KRK_entry graph.
    """
    
    def __init__(self, config: Optional[SpawnPointConfig] = None):
        self.config = config or SpawnPointConfig()
        self.spawn_points: Dict[str, SpawnPoint] = {}
        self.tick = 0
    
    def attach_to_legs(self, graph: Graph, leg_prefix: str = "leg_"):
        """
        Attach spawn points to all Leg nodes in the graph.
        """
        for node_id in graph.nodes:
            if node_id.startswith(leg_prefix):
                spawn_point_id = f"spawn_{node_id}"
                self.spawn_points[spawn_point_id] = SpawnPoint(
                    spawn_point_id=spawn_point_id,
                    leg_id=node_id,
                    config=self.config
                )
        
        print(f"Attached {len(self.spawn_points)} spawn points to legs")
    
    def process_position(
        self,
        board: Any,
        selected_move: Any,
        resulted_in_checkmate: bool
    ):
        """
        Process a position through all spawn points.
        
        This is called after each move to update trials.
        """
        self.tick += 1
        
        # Extract features before and after
        v0 = _teacher.features(board)
        
        board_after = board.copy()
        board_after.push(selected_move)
        v1 = _teacher.features(board_after)
        
        # Check for spawning (on affordance spike)
        for sp in self.spawn_points.values():
            if sp.should_spawn(v0):
                trial = sp.spawn_trial(self.tick)
                print(f"  Spawned trial: {trial.trial_id} exploring features {trial.feature_indices}")
            
            # Update all active trials with this transition
            sp.process_transition(v0, v1, resulted_in_checkmate, self.tick)
        
        # Periodic prune/promote
        if self.tick % 10 == 0:
            self._prune_and_promote()
    
    def _prune_and_promote(self):
        """Run prune/promote cycle across all spawn points"""
        total_promoted = 0
        total_pruned = 0
        
        for sp in self.spawn_points.values():
            promoted, pruned = sp.prune_and_promote(self.tick)
            total_promoted += len(promoted)
            total_pruned += len(pruned)
            
            for trial_id in promoted:
                print(f"  ✓ PROMOTED: {trial_id}")
            for trial_id in pruned:
                print(f"  ✗ Pruned: {trial_id}")
        
        if total_promoted > 0 or total_pruned > 0:
            print(f"Prune/Promote cycle: {total_promoted} promoted, {total_pruned} pruned")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        stats = {
            "tick": self.tick,
            "spawn_points": len(self.spawn_points),
            "total_active_trials": sum(
                len(sp.active_trials) for sp in self.spawn_points.values()
            ),
            "total_promotions": sum(
                sp.total_promotions for sp in self.spawn_points.values()
            ),
            "total_prunes": sum(
                sp.total_prunes for sp in self.spawn_points.values()
            ),
        }
        return stats


# Convenience function for testing
def create_spawn_point_manager(graph: Graph) -> SpawnPointManager:
    """Create and attach spawn points to a graph"""
    manager = SpawnPointManager()
    manager.attach_to_legs(graph)
    return manager
