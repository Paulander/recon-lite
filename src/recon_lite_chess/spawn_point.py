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

from recon_lite.graph import Node, NodeType, Graph, LinkType
from recon_lite_chess.baseline_teacher import KRKTeacher
from recon_lite_chess.krk_baseline_nodes import apply_readout


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
    
    # Terminal subset being explored (sensor IDs)
    sensor_ids: List[str]
    
    # Learning stats
    samples: int = 0
    checkmate_hits: int = 0  # Times this pattern led to mate
    non_mate_hits: int = 0   # Times pattern activated but no mate
    xp: float = 0.0
    ticks_alive: int = 0
    last_update_tick: int = 0
    
    # Learned delta pattern in terminal space (Δs)
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
    
    def spawn_trial(
        self,
        tick: int,
        available_sensor_ids: List[str],
        mature_sensor_ids: Optional[List[str]] = None,
        exploratory_sensor_ids: Optional[List[str]] = None,
        mature_ratio: float = 0.7,
    ) -> TrialMicroScript:
        """
        Spawn a new trial micro-script.
        
        Randomly selects a subset of sensor terminals to explore.
        """
        trial_id = f"{self.spawn_point_id}_trial_{self.total_spawns}"
        self.total_spawns += 1

        # Select pool: 70% mature, 30% exploratory (fallbacks if empty)
        pool = available_sensor_ids
        if mature_sensor_ids is None:
            mature_sensor_ids = []
        if exploratory_sensor_ids is None:
            exploratory_sensor_ids = []

        if mature_sensor_ids or exploratory_sensor_ids:
            pick_mature = random.random() < mature_ratio
            if pick_mature and mature_sensor_ids:
                pool = mature_sensor_ids
            elif (not pick_mature) and exploratory_sensor_ids:
                pool = exploratory_sensor_ids
            elif mature_sensor_ids:
                pool = mature_sensor_ids
            elif exploratory_sensor_ids:
                pool = exploratory_sensor_ids

        # Randomly select 1-4 sensors to explore
        num_sensors = random.randint(1, min(4, max(1, len(pool))))
        sensor_ids = random.sample(pool, num_sensors)
        sensor_ids.sort()
        
        trial = TrialMicroScript(
            trial_id=trial_id,
            spawn_point_id=self.spawn_point_id,
            sensor_ids=sensor_ids,
            last_update_tick=tick
        )
        
        self.active_trials[trial_id] = trial
        return trial
    
    def process_transition(
        self,
        s0: Dict[str, float],
        s1: Dict[str, float],
        resulted_in_checkmate: bool,
        tick: int
    ):
        """
        Update all active trials with a new transition observation.
        
        Args:
            s0: Terminal outputs before move
            s1: Terminal outputs after move
            resulted_in_checkmate: Whether the move achieved checkmate
            tick: Current tick number
        """
        for trial in list(self.active_trials.values()):
            # Compute delta for trial's terminal subset
            deltas = []
            for sensor_id in trial.sensor_ids:
                if sensor_id in s0 and sensor_id in s1:
                    deltas.append(s1[sensor_id] - s0[sensor_id])
            if not deltas:
                continue
            delta = np.array(deltas, dtype=np.float32)
            
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
    
    def prune_and_promote(self, tick: int) -> Tuple[List["TrialMicroScript"], List[str]]:
        """
        Prune low-XP trials and promote high-XP trials.
        
        Returns:
            (promoted_trials, pruned_ids)
        """
        promoted: List[TrialMicroScript] = []
        pruned = []
        
        for trial_id, trial in list(self.active_trials.items()):
            if trial.should_promote():
                promoted.append(trial)
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
    
    def __init__(self, config: Optional[SpawnPointConfig] = None, graph: Optional[Graph] = None):
        self.config = config or SpawnPointConfig()
        self.graph = graph
        self.spawn_points: Dict[str, SpawnPoint] = {}
        self.tick = 0
        self.sensor_ids: List[str] = []
        self.mature_sensor_ids: List[str] = []
        self.exploratory_sensor_ids: List[str] = []
    
    def attach_to_legs(self, graph: Graph, leg_prefix: str = "leg_"):
        """
        Attach spawn points to all Leg nodes in the graph.
        """
        self.graph = graph
        self.sensor_ids = sorted(
            [nid for nid in graph.nodes if nid.startswith("sensor_") and "_post" not in nid]
        )
        self.mature_sensor_ids = []
        self.exploratory_sensor_ids = []
        for sid in self.sensor_ids:
            node = graph.nodes.get(sid)
            if not node:
                continue
            xp = node.meta.get("baseline_xp")
            try:
                xp_val = float(xp) if xp is not None else 0.0
            except Exception:
                xp_val = 0.0
            if xp_val >= XP_PROMOTE_THRESHOLD:
                self.mature_sensor_ids.append(sid)
            else:
                self.exploratory_sensor_ids.append(sid)
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
        
        if self.graph is None:
            return
        # Extract features before and after
        v0 = _teacher.features(board)
        board_after = board.copy()
        board_after.push(selected_move)
        v1 = _teacher.features(board_after)
        
        # Compute terminal outputs (s0, s1)
        s0 = self._compute_terminal_outputs(v0)
        s1 = self._compute_terminal_outputs(v1)
        
        # Check for spawning (on affordance spike)
        for sp in self.spawn_points.values():
            if sp.should_spawn(v0):
                if self.sensor_ids:
                    trial = sp.spawn_trial(
                        self.tick,
                        self.sensor_ids,
                        mature_sensor_ids=self.mature_sensor_ids,
                        exploratory_sensor_ids=self.exploratory_sensor_ids,
                        mature_ratio=0.7,
                    )
                    print(f"  Spawned trial: {trial.trial_id} exploring sensors {trial.sensor_ids}")
            
            # Update all active trials with this transition
            sp.process_transition(s0, s1, resulted_in_checkmate, self.tick)
        
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
            
            for trial in promoted:
                print(f"  ✓ PROMOTED: {trial.trial_id}")
                # Attempt to materialize into the graph if available
                self._promote_trial_to_graph(sp, trial)
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

    def _compute_terminal_outputs(self, features: np.ndarray) -> Dict[str, float]:
        """Compute terminal outputs for all baseline sensors from a feature vector."""
        outputs: Dict[str, float] = {}
        if self.graph is None:
            return outputs
        for sensor_id in self.sensor_ids:
            node = self.graph.nodes.get(sensor_id)
            if not node:
                continue
            readout_type = node.meta.get("readout_type", "identity")
            feature_mask_keys = node.meta.get("feature_mask_keys", [])
            readout_params = node.meta.get("readout_params", {})
            feature_mask = np.zeros(len(features), dtype=bool)
            for key in feature_mask_keys:
                if key.startswith("feature_"):
                    idx = int(key.split("_")[1])
                    if idx < len(features):
                        feature_mask[idx] = True
            if not np.any(feature_mask):
                continue
            sub_v = features[feature_mask]
            try:
                outputs[sensor_id] = float(apply_readout(sub_v, readout_type, readout_params))
            except Exception:
                continue
        return outputs

    def _promote_trial_to_graph(self, sp: SpawnPoint, trial: TrialMicroScript) -> None:
        """
        Materialize a promoted trial as a 3-part micro-script in the active graph.
        
        Structure:
            parent_leg (SCRIPT)
              ├─ SUB → trial_leg (SCRIPT)
              │    ├─ SUB → precond (SCRIPT, aggregation="and")
              │    │      ├─ SUB → sensor_i (TERMINAL)
              │    ├─ SUB → actuator (TERMINAL)
              │    └─ SUB → postcond (SCRIPT, aggregation="and")
              │           ├─ SUB → sensor_i_post (TERMINAL)
        """
        if self.graph is None:
            return
        
        try:
            from recon_lite_chess.krk_baseline_nodes import (
                create_leg_script,
                create_and_gate,
                create_act_script,
                create_sensor_terminal,
                create_actuator_terminal,
            )
        except Exception:
            return
        
        # Build node ids
        leg_id = f"{trial.trial_id}_leg"
        precond_id = f"{trial.trial_id}_precond"
        act_script_id = f"{trial.trial_id}_act_script"
        postcond_id = f"{trial.trial_id}_postcond"
        actuator_id = f"{trial.trial_id}_act"
        
        if leg_id in self.graph.nodes:
            return
        
        # Parent leg: attach under the leg that spawned this trial
        parent_leg_id = sp.leg_id if sp.leg_id in self.graph.nodes else "krk_entry"
        
        # Create nodes
        leg = create_leg_script(leg_id)
        leg.meta["factory"] = "recon_lite_chess.krk_baseline_nodes:create_leg_script"
        leg.meta["origin"] = "spawn_point"
        leg.meta["trial_id"] = trial.trial_id
        
        precond = create_and_gate(precond_id)
        precond.meta["aggregation"] = "and"
        precond.meta["factory"] = "recon_lite_chess.krk_baseline_nodes:create_and_gate"
        precond.meta["origin"] = "spawn_point"
        
        act_script = create_act_script(act_script_id)
        act_script.meta["factory"] = "recon_lite_chess.krk_baseline_nodes:create_act_script"
        act_script.meta["origin"] = "spawn_point"
        
        postcond = create_and_gate(postcond_id)
        postcond.meta["aggregation"] = "and"
        postcond.meta["factory"] = "recon_lite_chess.krk_baseline_nodes:create_and_gate"
        postcond.meta["origin"] = "spawn_point"
        
        actuator = create_actuator_terminal(actuator_id)
        actuator.meta["factory"] = "recon_lite_chess.krk_baseline_nodes:create_actuator_terminal"
        actuator.meta["origin"] = "spawn_point"
        actuator.meta["trial_id"] = trial.trial_id
        
        # Add core nodes first so edges can target them
        self.graph.add_node(leg)
        self.graph.add_node(precond)
        self.graph.add_node(act_script)
        self.graph.add_node(postcond)
        self.graph.add_node(actuator)
        
        # Create sensor terminals for each feature index
        sensor_ids = []
        goal_delta = {}
        if trial.delta_mean is None:
            return
        for sensor_id, delta in zip(trial.sensor_ids, list(trial.delta_mean)):
            sensor_ids.append(sensor_id)
            goal_delta[sensor_id] = float(delta)
            
            sensor_node = self.graph.nodes.get(sensor_id)
            if sensor_node is None:
                continue
            
            # Precondition uses existing sensors
            self.graph.add_edge(precond_id, sensor_id, LinkType.SUB)
            
            # Postcondition uses new sensor clones with same spec
            post_id = f"{sensor_id}_post_{trial.trial_id}"
            if post_id not in self.graph.nodes:
                sensor_post = create_sensor_terminal(post_id)
                sensor_post.meta["factory"] = "recon_lite_chess.krk_baseline_nodes:create_sensor_terminal"
                sensor_post.meta["readout_type"] = sensor_node.meta.get("readout_type", "identity")
                sensor_post.meta["feature_mask_keys"] = list(sensor_node.meta.get("feature_mask_keys", []))
                sensor_post.meta["readout_params"] = dict(sensor_node.meta.get("readout_params", {}))
                sensor_post.meta["origin"] = "spawn_point"
                sensor_post.meta["trial_id"] = trial.trial_id
                self.graph.add_node(sensor_post)
            self.graph.add_edge(postcond_id, post_id, LinkType.SUB)
        
        # Configure actuator targets
        actuator.meta["targets"] = list(sensor_ids)
        actuator.meta["goal_delta"] = goal_delta
        
        # Wire leg structure (no POR between legs or terminals)
        self.graph.add_edge(parent_leg_id, leg_id, LinkType.SUB)
        self.graph.add_edge(leg_id, precond_id, LinkType.SUB)
        self.graph.add_edge(leg_id, act_script_id, LinkType.SUB)
        self.graph.add_edge(leg_id, postcond_id, LinkType.SUB)
        self.graph.add_edge(act_script_id, actuator_id, LinkType.SUB)
        
        # POR sequencing between scripts only
        self.graph.add_edge(precond_id, act_script_id, LinkType.POR)
        self.graph.add_edge(act_script_id, postcond_id, LinkType.POR)


# Convenience function for testing
def create_spawn_point_manager(graph: Graph) -> SpawnPointManager:
    """Create and attach spawn points to a graph"""
    manager = SpawnPointManager()
    manager.attach_to_legs(graph)
    return manager
