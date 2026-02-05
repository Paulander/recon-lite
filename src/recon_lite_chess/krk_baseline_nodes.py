"""
KRK Baseline Node Factories

Factory functions for creating ReCoN nodes from compiled baseline topology.
These are referenced in krk_entry_topology.json.
"""

import numpy as np
from typing import Dict, Any
import chess

from recon_lite.graph import Node, NodeType
from recon_lite.learning.baseline import apply_sensor
from recon_lite_chess.baseline_teacher import KRKTeacher


# Global teacher instance for feature extraction
_teacher = KRKTeacher()


def create_krk_entry_root(node_id=None):
    """
    Create KRK entry root node with blackboard caching.
    
    Extracts features once per tick and caches in blackboard.
    """
    actual_id = node_id or "krk_entry"
    
    def predicate(node, env):
        # Initialize blackboard if needed
        if "blackboard" not in node.meta:
            node.meta["blackboard"] = {}
        blackboard = env.setdefault("blackboard", node.meta["blackboard"])
        node.meta["blackboard"] = blackboard

        # Extract features ONCE per tick
        if "krk_features" not in blackboard:
            board = env.get("board")
            if board:
                features = _teacher.features(board)
                blackboard["krk_features"] = features

        # Initialize caches
        blackboard.setdefault("sensor_outputs", {})
        blackboard.setdefault("sensor_specs", {})
        # Goal bank: hydrate from node.meta if present, otherwise init empty (online growth)
        if "goal_bank" not in blackboard:
            if node.meta.get("goal_bank"):
                blackboard["goal_bank"] = node.meta.get("goal_bank")
            else:
                blackboard["goal_bank"] = {
                    "label": node.meta.get("goal_label", "mate_in_1"),
                    "goals": [],
                    "sensor_specs": {},
                    "sensor_weights": {},
                    "goal_eps": float(node.meta.get("goal_eps", 0.15)),
                }
        blackboard["goal_label"] = node.meta.get("goal_label", "mate_in_1")
        blackboard["goal_normalize"] = node.meta.get("goal_normalize", True)
        blackboard["goal_weight"] = node.meta.get("goal_weight", 0.7)
        blackboard["goal_lookahead"] = node.meta.get("goal_lookahead", "max")
        blackboard["goal_min_overlap"] = node.meta.get("goal_min_overlap", 8)
        blackboard["goal_handoff_threshold"] = node.meta.get("goal_handoff_threshold", 0.2)
        node.meta["goal_bank"] = blackboard.get("goal_bank")

        # Compute current goal distance (for handoff gating) when goal bank exists
        if blackboard.get("goal_bank") and blackboard.get("krk_features") is not None:
            dist, overlap = _goal_distance_from_features(
                blackboard["krk_features"],
                blackboard.get("goal_bank"),
                normalize=blackboard.get("goal_normalize", True),
                min_overlap=int(blackboard.get("goal_min_overlap", 8)),
            )
            blackboard["goal_distance_now"] = dist
            blackboard["goal_overlap_now"] = overlap
            thresh = float(blackboard.get("goal_handoff_threshold", 0.2))
            blackboard["goal_ready"] = (dist is not None and dist <= thresh)

        # Keep root in WAITING so children can run this tick
        return False, False
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_krk_hub(node_id=None):
    """
    Create Hub node with bandit selection.
    
    Selects which Leg to activate based on bandit scores.
    """
    actual_id = node_id or "krk_hub"
    
    def predicate(node, env):
        # For now, just pass through
        # Bandit logic will be added later
        # Keep hub waiting so children get requested
        return False, False
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_leg_script(node_id=None):
    """Create Leg SCRIPT node (simple pass-through)"""
    actual_id = node_id or "leg_script"
    
    def predicate(node, env):
        # Keep leg waiting so children get requested
        return False, False
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_act_script(node_id=None):
    """Create actuator wrapper SCRIPT node."""
    actual_id = node_id or "act_script"
    
    def predicate(node, env):
        # Keep actuator wrapper waiting so child terminal runs
        return False, False
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_and_gate(node_id=None):
    """
    Create AND-gate SCRIPT node.
    
    All children must confirm for this to confirm.
    """
    actual_id = node_id or "and_gate"
    
    def predicate(node, env):
        # Aggregation handled by engine
        # Keep AND gate waiting so children can confirm
        return False, False
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_sensor_terminal(node_id=None):
    """
    Create sensor TERMINAL node.
    
    Reads cached features from blackboard, applies readout, caches output.
    """
    actual_id = node_id or "sensor_terminal"
    
    def predicate(node, env):
        blackboard = env.get("blackboard")
        if not blackboard:
            return False, False
        
        # Read cached features
        features = blackboard.get("krk_features")
        if features is None:
            return False, False
        
        # Apply sensor readout
        readout_type = node.meta.get("readout_type", "identity")
        feature_mask_keys = node.meta.get("feature_mask_keys", [])
        readout_params = node.meta.get("readout_params", {})
        
        # Convert feature keys to mask
        feature_mask = np.zeros(len(features), dtype=bool)
        for key in feature_mask_keys:
            # Extract index from "feature_X" format
            if key.startswith("feature_"):
                idx = int(key.split("_")[1])
                if idx < len(features):
                    feature_mask[idx] = True
        
        # Handle case where no features selected
        if not np.any(feature_mask):
            return False, False
        
        # Apply readout
        sub_v = features[feature_mask]
        try:
            output = apply_readout(sub_v, readout_type, readout_params)
        except Exception as e:
            print(f"Warning: Sensor {node.nid} readout failed: {e}")
            return False, False
        
        # Cache output + spec for actuators
        blackboard["sensor_outputs"][node.nid] = output
        weight = None
        if "baseline_xp" in node.meta:
            try:
                weight = 1.0 + max(0.0, float(node.meta.get("baseline_xp", 0.0)))
            except Exception:
                weight = None
        blackboard["sensor_specs"][node.nid] = {
            "readout_type": readout_type,
            "feature_mask_keys": feature_mask_keys,
            "readout_params": readout_params,
            "weight": weight,
        }
        
        # Store in node state
        if not hasattr(node, 'state') or node.state is None:
            node.state = {}
        node.meta["output"] = output
        
        return True, True
    
    return Node(
        nid=actual_id,
        ntype=NodeType.TERMINAL,
        predicate=predicate
    )


def create_actuator_terminal(node_id=None):
    """
    Create actuator TERMINAL node.
    
    Scores moves by similarity to goal_delta, selects best move.
    """
    actual_id = node_id or "actuator_terminal"
    
    def predicate(node, env):
        blackboard = env.get("blackboard")
        if not blackboard:
            return False, False
        sensor_outputs = blackboard.get("sensor_outputs", {})
        sensor_specs = blackboard.get("sensor_specs", {})
        
        # Get targets and goal_delta
        targets = node.meta.get("targets", [])
        goal_delta = node.meta.get("goal_delta", {})
        stage = int(node.meta.get("stage", 0))

        # Optional eval/training filter: only allow selected actuator stages.
        # This is useful for isolating Stage-1 behavior in diagnostics.
        stage_filter = blackboard.get("stage_filter")
        if stage_filter is not None:
            try:
                if stage != int(stage_filter):
                    return False, False
            except Exception:
                pass
        
        if not targets or not goal_delta:
            return False, False
        
        # Get current sensor values
        s0 = {}
        for target_id in targets:
            if target_id in sensor_outputs:
                s0[target_id] = sensor_outputs[target_id]
        
        if len(s0) != len(targets):
            return False, False  # Not all sensors available
        
        # Determine spotlight weight (handoff A + C)
        features = blackboard.get("krk_features")
        mate_possible = False
        if features is not None and len(features) > 12:
            mate_possible = features[12] >= 0.5  # can_deliver_mate

        goal_ready = bool(blackboard.get("goal_ready"))
        # Stage bias: spotlight tactical (stage 0) when goal basin reached AND mate is visible
        if goal_ready and mate_possible:
            stage_weight = 3.0 if stage == 0 else 0.2
        elif mate_possible:
            stage_weight = 2.5 if stage == 0 else 0.5
        else:
            stage_weight = 0.7 if stage == 0 else 1.0

        # Optional goal bank for backchaining (pure terminal-space objective)
        goal_bank = blackboard.get("goal_bank")
        goal_label = blackboard.get("goal_label", "mate_in_1")
        goal_normalize = blackboard.get("goal_normalize", True)
        goal_weight = float(blackboard.get("goal_weight", 0.7))
        goal_lookahead = blackboard.get("goal_lookahead", "max")
        min_goal_overlap = float(blackboard.get("goal_min_overlap", 8))
        goal_entries = []
        if goal_bank and stage > 0:
            if isinstance(goal_bank, dict) and goal_bank.get("label") == goal_label:
                goal_entries = goal_bank.get("goals", [])

        # Score each legal move
        board = env.get("board")
        if not board:
            return False, False
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return False, False
        
        scores = {}
        move_meta: Dict[Any, Dict[str, Any]] = {}
        for move in legal_moves:
            # Simulate move
            board_copy = board.copy()
            board_copy.push(move)
            is_mate = board_copy.is_checkmate()
            
            # Get new features
            features_1 = _teacher.features(board_copy)
            
            # Compute Δs for target sensors
            delta_s = []
            for target_id in targets:
                spec = sensor_specs.get(target_id)
                if spec is None:
                    base_id = target_id.split("_post_")[0]
                    spec = sensor_specs.get(base_id)
                if spec is None:
                    continue
                
                s1 = _apply_spec_to_features(spec, features_1)
                if s1 is None:
                    continue
                
                # Compute delta
                delta = s1 - s0[target_id]
                delta_s.append(delta)
            
            if len(delta_s) == 0:
                continue
                
            # Score by similarity to goal_delta
            goal_deltas = [goal_delta[t] for t in targets if t in goal_delta][:len(delta_s)]
            similarity = cosine_similarity(delta_s, goal_deltas)
            score = similarity * stage_weight

            # Goal distance shaping (Stage > 0 only)
            if goal_entries:
                def _goal_distance_for_board(b):
                    f = _teacher.features(b)
                    # Build current sensor map by stable id (graph specs + goal specs)
                    s_goal: Dict[str, float] = {}
                    goal_specs = {}
                    if isinstance(goal_bank, dict):
                        goal_specs = goal_bank.get("sensor_specs", {}) or {}
                    merged_specs = dict(goal_specs)
                    merged_specs.update(sensor_specs)
                    for sid_key, spec in merged_specs.items():
                        val = _apply_spec_to_features(spec, f)
                        if val is not None:
                            s_goal[sid_key] = float(val)
                    if not s_goal:
                        return None

                    def _dist(entry):
                        gvals = entry.get("values", {})
                        keys = set(s_goal.keys()) & set(gvals.keys())
                        if not keys:
                            return None
                        weights = np.array([
                            _goal_weight_for_sensor(k, goal_bank, merged_specs)
                            for k in keys
                        ], dtype=np.float32)
                        weight_sum = float(np.sum(weights))
                        if weight_sum < min_goal_overlap:
                            return None
                        vec_cur = np.array([s_goal[k] for k in keys], dtype=np.float32)
                        vec_goal = np.array([gvals[k] for k in keys], dtype=np.float32)
                        if goal_normalize:
                            vec_cur = vec_cur / (np.sqrt(np.sum(weights * (vec_cur ** 2))) + 1e-6)
                            vec_goal = vec_goal / (np.sqrt(np.sum(weights * (vec_goal ** 2))) + 1e-6)
                        diff = vec_cur - vec_goal
                        return float(np.sqrt(np.sum(weights * (diff ** 2))))

                    best = None
                    for entry in goal_entries:
                        d = _dist(entry)
                        if d is None:
                            continue
                        if best is None or d < best:
                            best = d
                    return best

                # If lookahead enabled, evaluate after one black reply (worst-case by default)
                d1 = None
                if goal_lookahead and goal_lookahead != "none":
                    d1_candidates = []
                    for reply in board_copy.legal_moves:
                        b2 = board_copy.copy()
                        b2.push(reply)
                        d2 = _goal_distance_for_board(b2)
                        if d2 is not None:
                            d1_candidates.append(d2)
                    if d1_candidates:
                        d1 = max(d1_candidates) if goal_lookahead == "max" else min(d1_candidates)
                else:
                    d1 = _goal_distance_for_board(board_copy)

                if d1 is not None:
                    score += goal_weight * (-float(d1))
                move_meta[move] = {"is_mate": is_mate, "goal_dist": d1}
            else:
                move_meta[move] = {"is_mate": is_mate, "goal_dist": None}

            scores[move] = score
        
        if not scores:
            return False, False
        
        # Select best move
        best_move = max(scores, key=scores.get)
        best_score = scores[best_move]
        best_meta = move_meta.get(best_move, {})
        
        # Store suggestions in environment, keep best across actuators
        suggestions = env.setdefault("actuator_suggestions", [])
        suggestions.append({
            "actuator": node.nid,
            "move": best_move,
            "score": best_score,
        })
        
        best = max(suggestions, key=lambda s: s["score"])
        env["suggested_move"] = best["move"].uci()
        env["move_confidence"] = best["score"]
        env["suggested_actuator"] = best["actuator"]

        # Promote goal prototype from pre-move state when best move hits goal basin or checkmate.
        # This is local/online goal discovery: "if this state leads to a goal, it becomes a goal."
        if best["actuator"] == node.nid:
            should_promote = False
            if best_meta.get("is_mate"):
                should_promote = True
            else:
                d1 = best_meta.get("goal_dist")
                thresh = float(blackboard.get("goal_handoff_threshold", 0.2))
                if d1 is not None and d1 <= thresh:
                    should_promote = True
            if should_promote:
                _promote_goal_from_outputs(blackboard)
        
        # Store in node state
        if not hasattr(node, 'state') or node.state is None:
            node.state = {}
        node.meta["move"] = best_move.uci()
        node.meta["confidence"] = best_score
        
        return True, True
    
    return Node(
        nid=actual_id,
        ntype=NodeType.TERMINAL,
        predicate=predicate
    )


# Helper functions

def apply_readout(sub_v: np.ndarray, readout_type: str, params: Dict) -> float:
    """Apply sensor readout function"""
    if len(sub_v) == 0:
        return 0.0
        
    if readout_type == "identity":
        if len(sub_v) != 1:
            # Fallback to mean if multiple features
            return float(np.mean(sub_v))
        return float(sub_v[0])
    
    elif readout_type == "sum":
        return float(np.sum(sub_v))
    
    elif readout_type == "mean":
        return float(np.mean(sub_v))
    
    elif readout_type == "min":
        return float(np.min(sub_v))
    
    elif readout_type == "max":
        return float(np.max(sub_v))
    
    elif readout_type == "threshold":
        threshold = params.get("threshold", 0.5)
        return 1.0 if np.mean(sub_v) > threshold else 0.0
    
    else:
        # Default to mean for unknown types
        return float(np.mean(sub_v))


def _apply_spec_to_features(spec: Dict[str, Any], features: np.ndarray) -> float | None:
    """Apply a sensor spec dict to features, returning a scalar or None."""
    readout_type = spec.get("readout_type", "identity")
    feature_mask_keys = spec.get("feature_mask_keys", [])
    readout_params = spec.get("readout_params", {})

    feature_mask = np.zeros(len(features), dtype=bool)
    for key in feature_mask_keys:
        if key.startswith("feature_"):
            idx = int(key.split("_")[1])
            if idx < len(features):
                feature_mask[idx] = True

    if not np.any(feature_mask):
        return None

    sub_v = features[feature_mask]
    try:
        return apply_readout(sub_v, readout_type, readout_params)
    except Exception:
        return None


def _goal_distance_from_values(
    current: Dict[str, float],
    goal_bank: Dict[str, Any] | None,
    normalize: bool = True,
    min_overlap: float = 8,
) -> tuple[float | None, int | None]:
    """Compute min distance to goal prototypes using a values dict (sensor_id -> value)."""
    if not goal_bank or not isinstance(goal_bank, dict):
        return None, None
    goals = goal_bank.get("goals", [])
    if not goals or not current:
        return None, None
    sensor_specs = goal_bank.get("sensor_specs", {}) or {}

    best = None
    best_idx = None
    for idx, entry in enumerate(goals):
        gvals = entry.get("values", {})
        keys = set(current.keys()) & set(gvals.keys())
        if not keys:
            continue
        weights = np.array([
            _goal_weight_for_sensor(k, goal_bank, sensor_specs)
            for k in keys
        ], dtype=np.float32)
        weight_sum = float(np.sum(weights))
        if weight_sum < min_overlap:
            continue
        vec_cur = np.array([current[k] for k in keys], dtype=np.float32)
        vec_goal = np.array([gvals[k] for k in keys], dtype=np.float32)
        if normalize:
            vec_cur = vec_cur / (np.sqrt(np.sum(weights * (vec_cur ** 2))) + 1e-6)
            vec_goal = vec_goal / (np.sqrt(np.sum(weights * (vec_goal ** 2))) + 1e-6)
        diff = vec_cur - vec_goal
        dist = float(np.sqrt(np.sum(weights * (diff ** 2))))
        if best is None or dist < best:
            best = dist
            best_idx = idx
    return best, best_idx


def _promote_goal_from_outputs(blackboard: Dict[str, Any]) -> None:
    """Promote current sensor outputs into the goal bank (local, online)."""
    goal_bank = blackboard.get("goal_bank")
    if not isinstance(goal_bank, dict):
        return
    sensor_outputs = blackboard.get("sensor_outputs", {}) or {}
    sensor_specs = blackboard.get("sensor_specs", {}) or {}
    if not sensor_outputs:
        return

    # Ensure sensor specs/weights are present in goal bank
    goal_specs = goal_bank.setdefault("sensor_specs", {})
    goal_weights = goal_bank.setdefault("sensor_weights", {})
    for sid, spec in sensor_specs.items():
        if sid not in goal_specs:
            goal_specs[sid] = spec
        if sid not in goal_weights:
            w = spec.get("weight")
            try:
                w_val = float(w) if w is not None else 1.0
            except Exception:
                w_val = 1.0
            goal_weights[sid] = w_val

    # Build stable values dict
    values = {sid: float(val) for sid, val in sensor_outputs.items()}

    # Merge with existing goals if close
    goal_eps = float(goal_bank.get("goal_eps", 0.15))
    normalize = bool(blackboard.get("goal_normalize", True))
    min_overlap = float(blackboard.get("goal_min_overlap", 8))
    dist, idx = _goal_distance_from_values(values, goal_bank, normalize=normalize, min_overlap=min_overlap)
    if dist is not None and dist <= goal_eps and idx is not None:
        entry = goal_bank["goals"][idx]
        entry["count"] = int(entry.get("count", 1)) + 1
        return

    # Add new goal entry
    goal_bank.setdefault("goals", []).append({
        "values": values,
        "count": 1,
    })


def _goal_distance_from_features(
    features: np.ndarray,
    goal_bank: Dict[str, Any] | None,
    normalize: bool = True,
    min_overlap: float = 8,
) -> tuple[float | None, float]:
    """Compute min distance to goal prototypes in stable sensor-id space."""
    if not goal_bank or not isinstance(goal_bank, dict):
        return None, 0
    goals = goal_bank.get("goals", [])
    if not goals:
        return None, 0
    sensor_specs = goal_bank.get("sensor_specs", {}) or {}

    current: Dict[str, float] = {}
    for sid_key, spec in sensor_specs.items():
        val = _apply_spec_to_features(spec, features)
        if val is not None:
            current[sid_key] = float(val)

    if not current:
        return None, 0

    best = None
    best_overlap = 0.0
    for entry in goals:
        gvals = entry.get("values", {})
        keys = set(current.keys()) & set(gvals.keys())
        if not keys:
            continue
        weights = np.array([
            _goal_weight_for_sensor(k, goal_bank, sensor_specs)
            for k in keys
        ], dtype=np.float32)
        weight_sum = float(np.sum(weights))
        if weight_sum < min_overlap:
            continue
        vec_cur = np.array([current[k] for k in keys], dtype=np.float32)
        vec_goal = np.array([gvals[k] for k in keys], dtype=np.float32)
        if normalize:
            vec_cur = vec_cur / (np.sqrt(np.sum(weights * (vec_cur ** 2))) + 1e-6)
            vec_goal = vec_goal / (np.sqrt(np.sum(weights * (vec_goal ** 2))) + 1e-6)
        diff = vec_cur - vec_goal
        dist = float(np.sqrt(np.sum(weights * (diff ** 2))))
        if best is None or dist < best:
            best = dist
            best_overlap = weight_sum

    return best, best_overlap


def _goal_weight_for_sensor(
    sensor_id: str,
    goal_bank: Dict[str, Any] | None,
    sensor_specs: Dict[str, Any],
) -> float:
    """Resolve a stable weight for a sensor id (XP-weighted if available)."""
    if isinstance(goal_bank, dict):
        weight_map = goal_bank.get("sensor_weights", {}) or {}
        if sensor_id in weight_map:
            try:
                return float(weight_map[sensor_id])
            except Exception:
                pass
    spec = sensor_specs.get(sensor_id, {}) or {}
    for key in ("weight", "xp"):
        if key in spec and spec[key] is not None:
            try:
                return 1.0 + max(0.0, float(spec[key]))
            except Exception:
                continue
    return 1.0


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors"""
    if len(a) == 0 or len(b) == 0:
        return 0.0
        
    a = np.array(a)
    b = np.array(b)
    
    # Pad shorter to match length
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = np.pad(a, (0, max_len - len(a)))
        b = np.pad(b, (0, max_len - len(b)))
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))
