"""
General Terminal-Actuator Baseline Architecture

This module implements an environment-agnostic learning system where:
- Terminals are nodes with hard role invariants (SENSOR or ACTUATOR)
- Sensors learn sparse readouts over designer features
- Actuators encode goal patterns in sensor-terminal space
- Learning happens purely through terminal deltas (Δs)

Design principles:
1. Actuators ARE terminals (not separate entities)
2. All sensors train on deltas; only mature sensors used for actuator construction
3. Explicit XP metrics (stability + separation)
4. Environment-agnostic readouts (domain logic in teacher features)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np


# ============================================================================
# Core Enums and Constants
# ============================================================================

class TerminalRole(Enum):
    """Hard role discriminator for terminals"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"


# XP and promotion thresholds
XP_PROMOTE_THRESHOLD = 0.7
MIN_CYCLES_FOR_PROMOTION = 5
MIN_ACTIVATIONS_FOR_PROMOTION = 20


# ============================================================================
# Terminal Specifications (Role-specific payloads)
# ============================================================================

@dataclass
class SensorSpec:
    """
    Payload for SENSOR terminals.
    
    Defines how a sensor reads from the environment's feature vector.
    All readout types are environment-agnostic; domain-specific logic
    (e.g., "distance to edge") should be computed in teacher features.
    """
    feature_mask: np.ndarray  # bool array: which dims of v to read
    readout_type: str         # "identity", "sum", "mean", "min", "max", "threshold", "bucketize"
    readout_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate readout type"""
        valid_types = {"identity", "sum", "mean", "min", "max", "threshold", "bucketize"}
        if self.readout_type not in valid_types:
            raise ValueError(f"Invalid readout_type: {self.readout_type}. Must be one of {valid_types}")


@dataclass
class ActuatorSpec:
    """
    Payload for ACTUATOR terminals.
    
    Encodes a desired change pattern in sensor-terminal space.
    Used at runtime to score actions by how well their predicted Δs
    matches this goal pattern.
    """
    sensor_indices: List[int]  # which mature sensors it targets
    goal_delta: np.ndarray     # desired Δs for those sensors
    match_mode: str = "l2"     # "l2", "cosine", "manhattan"
    
    def __post_init__(self):
        """Validate match mode and dimensions"""
        valid_modes = {"l2", "cosine", "manhattan"}
        if self.match_mode not in valid_modes:
            raise ValueError(f"Invalid match_mode: {self.match_mode}. Must be one of {valid_modes}")
        
        if len(self.sensor_indices) != len(self.goal_delta):
            raise ValueError(
                f"Dimension mismatch: {len(self.sensor_indices)} sensor_indices "
                f"but {len(self.goal_delta)} goal_delta values"
            )


# ============================================================================
# Terminal (Base class with role invariant)
# ============================================================================

@dataclass
class Terminal:
    """
    Base terminal node with hard role invariant.
    
    Every terminal MUST be either a SENSOR or ACTUATOR, validated in __post_init__.
    Terminals connect to script nodes via SUB/SUR links (Article.md).
    
    Role-specific behavior:
    - SENSOR: reads from environment (measurement)
    - ACTUATOR: writes to environment (action selection via goal patterns)
    """
    id: int
    stage: int
    role: TerminalRole
    
    # Role-specific payloads (exactly one must be non-None)
    sensor_spec: Optional[SensorSpec] = None
    actuator_spec: Optional[ActuatorSpec] = None
    
    # Learning stats (common to both roles)
    xp: float = 0.0
    activations: int = 0
    good_hits: int = 0
    bad_hits: int = 0
    cycles_alive: int = 0
    is_mature: bool = False
    
    # Runtime cache
    last_output: float = 0.0
    
    def __post_init__(self):
        """
        Validate role invariant: exactly one payload must be present.
        
        Raises:
            ValueError: If role/payload combination is invalid
        """
        if self.role == TerminalRole.SENSOR:
            if self.sensor_spec is None:
                raise ValueError("SENSOR terminal must have sensor_spec")
            if self.actuator_spec is not None:
                raise ValueError("SENSOR terminal cannot have actuator_spec")
        
        elif self.role == TerminalRole.ACTUATOR:
            if self.actuator_spec is None:
                raise ValueError("ACTUATOR terminal must have actuator_spec")
            if self.sensor_spec is not None:
                raise ValueError("ACTUATOR terminal cannot have sensor_spec")
        
        else:
            raise ValueError(f"Invalid role: {self.role}")
    
    def __repr__(self) -> str:
        """Concise representation for debugging"""
        role_str = self.role.value
        mature_str = "✓" if self.is_mature else "○"
        return f"Terminal({self.id}, {role_str}, XP={self.xp:.2f}, {mature_str})"


# ============================================================================
# Goal Memory (Separate from terminals/actuators)
# ============================================================================

@dataclass
class GoalMemory:
    """
    Stores successful starting states in terminal space.
    
    Used for high-level goal attractors (e.g., "states close to mate").
    Computed using mature sensors only.
    Centroids/clustering happen ONLY in terminal space, ONLY for goals.
    """
    id: int
    s0: np.ndarray      # state vector using mature sensors
    label: str          # e.g., "mate_in_1", "king_at_edge"
    count: int = 1      # how many times this pattern was seen
    
    def __repr__(self) -> str:
        return f"GoalMemory({self.id}, {self.label}, count={self.count}, dim={len(self.s0)})"


# ============================================================================
# Sensor Readout Functions (Environment-Agnostic)
# ============================================================================

def apply_sensor(sensor: Terminal, v: np.ndarray) -> float:
    """
    Apply sensor readout to feature vector.
    
    All readout types are generic; domain-specific logic (e.g., "distance to edge")
    should be computed in the teacher's features() method, not here.
    
    Args:
        sensor: Terminal with role=SENSOR
        v: Full feature vector from environment
    
    Returns:
        Scalar output of sensor
    
    Raises:
        ValueError: If sensor is not a SENSOR terminal
    """
    if sensor.role != TerminalRole.SENSOR:
        raise ValueError(f"Cannot apply_sensor to {sensor.role} terminal")
    
    spec = sensor.sensor_spec
    sub_v = v[spec.feature_mask]
    
    if spec.readout_type == "identity":
        # Single feature, return as-is
        if len(sub_v) != 1:
            raise ValueError(f"identity readout requires exactly 1 feature, got {len(sub_v)}")
        return float(sub_v[0])
    
    elif spec.readout_type == "sum":
        return float(np.sum(sub_v))
    
    elif spec.readout_type == "mean":
        return float(np.mean(sub_v))
    
    elif spec.readout_type == "min":
        return float(np.min(sub_v))
    
    elif spec.readout_type == "max":
        return float(np.max(sub_v))
    
    elif spec.readout_type == "threshold":
        threshold = spec.readout_params.get("threshold", 0.5)
        return 1.0 if np.mean(sub_v) > threshold else 0.0
    
    elif spec.readout_type == "bucketize":
        bins = spec.readout_params.get("bins", [0.0, 0.5, 1.0])
        val = np.mean(sub_v)
        return float(np.digitize(val, bins))
    
    else:
        raise ValueError(f"Unknown readout_type: {spec.readout_type}")


# ============================================================================
# Sensor XP Metric (Explicit Formula)
# ============================================================================

def compute_sensor_xp(sensor: Terminal, 
                      delta_pos: List[float], 
                      delta_neg: List[float]) -> float:
    """
    Compute sensor XP based on stability and separation.
    
    Formula:
        stability = 1 / (1 + std(delta_pos))
        separation = |mean(delta_pos) - mean(delta_neg)| / (std_pos + std_neg + eps)
        xp = 0.6 * stability + 0.4 * separation
    
    Intuition:
    - Stability: prefer sensors with consistent Δt on positive transitions
    - Separation: prefer sensors that behave differently on pos vs neg
    
    Args:
        sensor: Terminal (for metadata, not used in computation)
        delta_pos: List of Δt values on positive transitions
        delta_neg: List of Δt values on negative transitions
    
    Returns:
        XP score in [0, 1] range (approximately)
    """
    if len(delta_pos) == 0:
        return 0.0
    
    mean_pos = np.mean(delta_pos)
    std_pos = np.std(delta_pos) if len(delta_pos) > 1 else 0.0
    
    # Stability: prefer low variance on positives
    stability = 1.0 / (1.0 + std_pos)
    
    # Separation: prefer different behavior on pos vs neg
    if len(delta_neg) > 0:
        mean_neg = np.mean(delta_neg)
        std_neg = np.std(delta_neg) if len(delta_neg) > 1 else 0.0
        separation = abs(mean_pos - mean_neg) / (std_pos + std_neg + 1e-6)
    else:
        separation = 0.0
    
    # Weighted combination (60% stability, 40% separation)
    xp = 0.6 * stability + 0.4 * separation
    return float(xp)


def should_promote_sensor(sensor: Terminal) -> bool:
    """
    Check if sensor meets promotion criteria.
    
    Criteria:
    - XP >= 0.7
    - cycles_alive >= 5
    - activations >= 20
    - not already mature
    
    Returns:
        True if sensor should be promoted to mature
    """
    return (
        not sensor.is_mature and
        sensor.xp >= XP_PROMOTE_THRESHOLD and
        sensor.cycles_alive >= MIN_CYCLES_FOR_PROMOTION and
        sensor.activations >= MIN_ACTIVATIONS_FOR_PROMOTION
    )


# ============================================================================
# Actuator Pattern Extraction (Precise Algorithm)
# ============================================================================

@dataclass
class TransitionData:
    """Per-transition data from teacher"""
    v0: np.ndarray
    v1: np.ndarray
    label: int  # 1 for positive, 0 for negative
    action: Any = None  # environment-specific action
    reward: float = 0.0  # optional dense reward signal


def extract_actuator_patterns(positive_transitions: List[TransitionData],
                              mature_sensors: List[Terminal],
                              eps: float = 0.1,
                              top_k: int = 3) -> List[ActuatorSpec]:
    """
    Extract sparse Δs patterns from positive transitions.
    
    Algorithm:
    1. For each transition: compute Δs using mature sensors only
    2. Pick indices where |Δs_i| > eps
    3. Keep top-K by magnitude
    4. Quantize: sign + magnitude bucket
    5. Use quantized pattern as key to merge/update actuators
    6. Return list of ActuatorSpec with running mean goal_delta
    
    Args:
        positive_transitions: List of transitions with label=1
        mature_sensors: List of mature sensor terminals
        eps: Threshold for significant delta (default 0.1)
        top_k: Maximum number of sensors per actuator pattern (default 3)
    
    Returns:
        List of ActuatorSpec extracted from patterns
    """
    if len(mature_sensors) == 0:
        return []
    
    pattern_map = {}  # key: (indices_tuple, quant_bins_tuple) -> list of raw deltas
    
    for trans in positive_transitions:
        # Compute Δs using mature sensors
        s0 = np.array([apply_sensor(s, trans.v0) for s in mature_sensors])
        s1 = np.array([apply_sensor(s, trans.v1) for s in mature_sensors])
        delta_s = s1 - s0
        
        # Pick significant indices
        significant = np.where(np.abs(delta_s) > eps)[0]
        if len(significant) == 0:
            continue
        
        # Keep top-K by magnitude
        if len(significant) > top_k:
            top_indices = np.argsort(np.abs(delta_s[significant]))[-top_k:]
            significant = significant[top_indices]
        
        # Quantize for pattern key (per-sensor type)
        quant_bins = tuple(
            quantize_delta(mature_sensors[i], float(delta_s[i]), eps=eps)
            for i in significant
        )
        
        # Create pattern key
        key = (tuple(sorted(significant)), quant_bins)
        
        # Accumulate raw deltas for this pattern
        if key not in pattern_map:
            pattern_map[key] = []
        pattern_map[key].append(delta_s[significant])
    
    # Merge patterns: compute mean goal_delta for each
    actuator_specs = []
    for (indices, _quant_bins), delta_list in pattern_map.items():
        mean_delta = np.mean(delta_list, axis=0)
        
        # Map indices back to sensor IDs
        sensor_indices = [mature_sensors[i].id for i in indices]
        
        spec = ActuatorSpec(
            sensor_indices=sensor_indices,
            goal_delta=mean_delta,
            match_mode="l2"
        )
        actuator_specs.append(spec)
    
    return actuator_specs


def find_similar_actuator(actuators: List[Terminal], 
                          spec: ActuatorSpec,
                          similarity_threshold: float = 0.9,
                          delta_eps: float = 0.15) -> Optional[Terminal]:
    """
    Find an existing actuator with similar pattern.
    
    Similarity based on:
    - Jaccard similarity of sensor_indices > threshold
    - Cosine similarity of goal_delta > threshold
    
    Args:
        actuators: List of existing actuator terminals
        spec: New actuator spec to compare
        similarity_threshold: Minimum similarity to consider a match
    
    Returns:
        Matching actuator terminal, or None if no match
    """
    for act in actuators:
        if act.role != TerminalRole.ACTUATOR:
            continue
        
        act_spec = act.actuator_spec
        
        # Jaccard similarity of sensor indices
        set_a = set(spec.sensor_indices)
        set_b = set(act_spec.sensor_indices)
        jaccard = len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) > 0 else 0.0
        
        if jaccard < similarity_threshold:
            continue
        
        # If identical sensor sets, compare goal_delta distance
        if set_a == set_b:
            a_map = {sid: val for sid, val in zip(spec.sensor_indices, spec.goal_delta)}
            b_map = {sid: val for sid, val in zip(act_spec.sensor_indices, act_spec.goal_delta)}
            ordered = sorted(set_a)
            diffs = np.array([a_map[sid] - b_map[sid] for sid in ordered], dtype=np.float32)
            if np.linalg.norm(diffs) <= delta_eps:
                return act
            continue
        
        # Overlap but not identical: allow merge on high Jaccard only
        return act
    
    return None


# ============================================================================
# Sensor Spawning
# ============================================================================

def spawn_sensor(id: int, 
                stage: int,
                feature_dim: int,
                allowed_features: List[int],
                mask_size_range: tuple = (1, 4)) -> Terminal:
    """
    Spawn a new candidate sensor terminal with random mask.
    
    Args:
        id: Unique ID for this sensor
        stage: Training stage (e.g., 0 for KRK bootstrap)
        feature_dim: Total number of features in v
        allowed_features: List of feature indices allowed for this stage
        mask_size_range: (min, max) number of features to include in mask
    
    Returns:
        New sensor terminal (not yet mature)
    """
    # Random mask size
    mask_size = np.random.randint(mask_size_range[0], mask_size_range[1] + 1)
    
    # Random feature selection
    chosen_indices = np.random.choice(allowed_features, size=min(mask_size, len(allowed_features)), replace=False)
    
    # Create boolean mask
    feature_mask = np.zeros(feature_dim, dtype=bool)
    feature_mask[chosen_indices] = True
    
    # Random readout type (identity only for single features)
    if mask_size == 1:
        readout_type = "identity"
    else:
        readout_type = np.random.choice(["sum", "mean", "min", "max", "threshold"])
    
    # Readout params (if needed)
    readout_params = {}
    if readout_type == "threshold":
        readout_params["threshold"] = np.random.uniform(0.3, 0.7)
    
    # Create sensor spec
    spec = SensorSpec(
        feature_mask=feature_mask,
        readout_type=readout_type,
        readout_params=readout_params
    )
    
    # Create terminal
    return Terminal(
        id=id,
        stage=stage,
        role=TerminalRole.SENSOR,
        sensor_spec=spec
    )


def quantize_delta(sensor: Terminal, delta: float, eps: float = 0.1) -> int:
    """
    Quantize delta per sensor type to reduce actuator proliferation.
    Returns an integer bin label.
    """
    readout_type = sensor.sensor_spec.readout_type if sensor.sensor_spec else "identity"
    if readout_type in ("threshold", "bucketize"):
        # Discrete outputs: clamp to {-1, 0, +1}
        return int(np.clip(np.round(delta), -1, 1))
    # Continuous outputs: quantize to coarse bins
    bin_width = max(eps, 0.25)
    q = int(np.round(delta / bin_width))
    return int(np.clip(q, -3, 3))


def enforce_actuator_cap(
    actuators: List[Terminal],
    stage: int,
    max_actuators: int,
) -> tuple[list[Terminal], int]:
    """
    Enforce a max actuator count per stage. Returns (kept, pruned_count).
    """
    if max_actuators <= 0:
        return actuators, 0
    stage_acts = [a for a in actuators if a.stage == stage and a.role == TerminalRole.ACTUATOR]
    if len(stage_acts) <= max_actuators:
        return actuators, 0
    # Keep top by XP
    stage_sorted = sorted(stage_acts, key=lambda a: a.xp, reverse=True)
    keep_ids = {a.id for a in stage_sorted[:max_actuators]}
    pruned_count = len(stage_acts) - max_actuators
    kept = [a for a in actuators if not (a.stage == stage and a.id not in keep_ids)]
    return kept, pruned_count


def enforce_actuator_cap_total(
    actuators: List[Terminal],
    max_total: int,
) -> tuple[list[Terminal], int]:
    """
    Enforce a global cap on total actuator count. Returns (kept, pruned_count).
    """
    if max_total <= 0:
        return actuators, 0
    all_acts = [a for a in actuators if a.role == TerminalRole.ACTUATOR]
    if len(all_acts) <= max_total:
        return actuators, 0
    # Keep top by XP (then uses, then id for stability)
    sorted_acts = sorted(
        all_acts,
        key=lambda a: (a.xp, getattr(a, "uses", a.activations), a.id),
        reverse=True,
    )
    keep_ids = {a.id for a in sorted_acts[:max_total]}
    pruned_count = len(all_acts) - max_total
    kept = [a for a in actuators if not (a.role == TerminalRole.ACTUATOR and a.id not in keep_ids)]
    return kept, pruned_count


# ============================================================================
# Baseline Learner (Main class)
# ============================================================================

class BaselineLearner:
    """
    Main learning system for terminal-actuator baseline.
    
    Manages:
    - Sensor terminals (candidate + mature)
    - Actuator terminals
    - Goal memories
    - XP computation and promotion
    """
    
    def __init__(
        self,
        feature_dim: int,
        stage: int = 0,
        goal_eps: float = 0.15,
        max_goals: int = 100,
        normalize_goals: bool = True,
    ):
        """
        Initialize learner.
        
        Args:
            feature_dim: Dimension of feature vector v
            stage: Training stage (e.g., 0 for KRK bootstrap)
        """
        self.feature_dim = feature_dim
        self.stage = stage
        
        self.sensors: List[Terminal] = []
        self.actuators: List[Terminal] = []
        self.goal_memories: List[GoalMemory] = []
        self.goal_eps = goal_eps
        self.max_goals = max_goals
        self.normalize_goals = normalize_goals
        
        self._next_sensor_id = 0
        self._next_actuator_id = 0
        self._next_goal_id = 0
    
    def spawn_sensor(self, allowed_features: Optional[List[int]] = None) -> Terminal:
        """Spawn a new sensor terminal"""
        if allowed_features is None:
            allowed_features = list(range(self.feature_dim))
        
        sensor = spawn_sensor(
            id=self._next_sensor_id,
            stage=self.stage,
            feature_dim=self.feature_dim,
            allowed_features=allowed_features
        )
        self._next_sensor_id += 1
        return sensor
    
    def add_goal_memory(
        self,
        s0: np.ndarray,
        label: str,
        goal_eps: float | None = None,
        max_goals: int | None = None,
        normalize: bool | None = None,
    ) -> GoalMemory | None:
        """
        Add a goal memory for a successful starting state.
        
        Args:
            s0: State vector using mature sensors
            label: Goal label (e.g., "mate_in_1")
        
        Returns:
            Created or updated GoalMemory
        """
        if s0.size == 0:
            return None

        eps = goal_eps if goal_eps is not None else self.goal_eps
        cap = max_goals if max_goals is not None else self.max_goals
        do_norm = normalize if normalize is not None else self.normalize_goals

        s = np.array(s0, dtype=np.float32)
        if do_norm:
            denom = np.linalg.norm(s) + 1e-6
            s = s / denom

        # Find nearest goal with same label and shape
        best = None
        best_dist = None
        for goal in self.goal_memories:
            if goal.label != label or goal.s0.shape != s.shape:
                continue
            g = goal.s0
            if do_norm:
                g = g / (np.linalg.norm(g) + 1e-6)
            dist = np.linalg.norm(s - g)
            if best is None or dist < best_dist:
                best = goal
                best_dist = dist

        # Merge if close enough
        if best is not None and best_dist is not None and best_dist < eps:
            best.s0 = best.s0 + (s0 - best.s0) / (best.count + 1)
            best.count += 1
            return best

        # Create new goal if under cap (or evict weakest)
        if len(self.goal_memories) >= cap and cap > 0:
            # Evict lowest-count goal with matching label/shape if possible
            candidates = [g for g in self.goal_memories if g.label == label and g.s0.shape == s.shape]
            if candidates:
                evict = min(candidates, key=lambda g: g.count)
            else:
                evict = min(self.goal_memories, key=lambda g: g.count)
            self.goal_memories.remove(evict)

        goal = GoalMemory(
            id=self._next_goal_id,
            s0=s0,
            label=label
        )
        self._next_goal_id += 1
        self.goal_memories.append(goal)
        return goal
    
    def get_mature_sensors(self) -> List[Terminal]:
        """Get list of mature sensors"""
        return [s for s in self.sensors if s.is_mature]
    
    def __repr__(self) -> str:
        mature_count = len(self.get_mature_sensors())
        return (f"BaselineLearner(stage={self.stage}, "
                f"sensors={len(self.sensors)} ({mature_count} mature), "
                f"actuators={len(self.actuators)}, "
                f"goals={len(self.goal_memories)})")
