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
    
    def predicate(node, graph, env):
        # Initialize blackboard if needed
        if "blackboard" not in node.meta:
            node.meta["blackboard"] = {}
        
        # Extract features ONCE per tick
        if "krk_features" not in node.meta["blackboard"]:
            board = env.get("board")
            if board:
                features = _teacher.features(board)
                node.meta["blackboard"]["krk_features"] = features
        
        # Initialize sensor output cache
        if "sensor_outputs" not in node.meta["blackboard"]:
            node.meta["blackboard"]["sensor_outputs"] = {}
        
        return True, True
    
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
    
    def predicate(node, graph, env):
        # For now, just pass through
        # Bandit logic will be added later
        return True, True
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_leg_script(node_id=None):
    """Create Leg SCRIPT node (simple pass-through)"""
    actual_id = node_id or "leg_script"
    
    def predicate(node, graph, env):
        return True, True
    
    return Node(
        nid=actual_id,
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_act_script(node_id=None):
    """Create actuator wrapper SCRIPT node."""
    actual_id = node_id or "act_script"
    
    def predicate(node, graph, env):
        return True, True
    
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
    
    def predicate(node, graph, env):
        # Aggregation handled by engine
        return True, True
    
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
    
    def predicate(node, graph, env):
        # Get root blackboard
        root = graph.nodes.get("krk_entry")
        if not root or "blackboard" not in root.meta:
            return False, False
        
        blackboard = root.meta["blackboard"]
        
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
        
        # Cache output for actuators
        blackboard["sensor_outputs"][node.nid] = output
        
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
    
    def predicate(node, graph, env):
        # Get root blackboard
        root = graph.nodes.get("krk_entry")
        if not root or "blackboard" not in root.meta:
            return False, False
        
        blackboard = root.meta["blackboard"]
        sensor_outputs = blackboard.get("sensor_outputs", {})
        
        # Get targets and goal_delta
        targets = node.meta.get("targets", [])
        goal_delta = node.meta.get("goal_delta", {})
        
        if not targets or not goal_delta:
            return False, False
        
        # Get current sensor values
        s0 = {}
        for target_id in targets:
            if target_id in sensor_outputs:
                s0[target_id] = sensor_outputs[target_id]
        
        if len(s0) != len(targets):
            return False, False  # Not all sensors available
        
        # Score each legal move
        board = env.get("board")
        if not board:
            return False, False
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return False, False
        
        scores = {}
        for move in legal_moves:
            # Simulate move
            board_copy = board.copy()
            board_copy.push(move)
            
            # Get new features
            features_1 = _teacher.features(board_copy)
            
            # Compute Δs for target sensors
            delta_s = []
            for target_id in targets:
                # Get sensor spec from graph
                sensor_node = graph.nodes.get(target_id)
                if not sensor_node:
                    # Try without _post suffix
                    sensor_node = graph.nodes.get(target_id.split("_post_")[0])
                if not sensor_node:
                    continue
                
                # Apply readout to new features
                readout_type = sensor_node.meta.get("readout_type", "identity")
                feature_mask_keys = sensor_node.meta.get("feature_mask_keys", [])
                readout_params = sensor_node.meta.get("readout_params", {})
                
                feature_mask = np.zeros(len(features_1), dtype=bool)
                for key in feature_mask_keys:
                    if key.startswith("feature_"):
                        idx = int(key.split("_")[1])
                        if idx < len(features_1):
                            feature_mask[idx] = True
                
                if not np.any(feature_mask):
                    continue
                
                sub_v = features_1[feature_mask]
                try:
                    s1 = apply_readout(sub_v, readout_type, readout_params)
                except Exception:
                    continue
                
                # Compute delta
                delta = s1 - s0[target_id]
                delta_s.append(delta)
            
            if len(delta_s) == 0:
                continue
                
            # Score by similarity to goal_delta
            goal_deltas = [goal_delta[t] for t in targets if t in goal_delta][:len(delta_s)]
            similarity = cosine_similarity(delta_s, goal_deltas)
            scores[move] = similarity
        
        if not scores:
            return False, False
        
        # Select best move
        best_move = max(scores, key=scores.get)
        best_score = scores[best_move]
        
        # Store suggestions in environment, keep best across actuators
        suggestions = env.setdefault("actuator_suggestions", [])
        suggestions.append({
            "actuator": node.nid,
            "move": best_move,
            "score": best_score,
        })
        
        best = max(suggestions, key=lambda s: s["score"])
        env["suggested_move"] = best["move"]
        env["move_confidence"] = best["score"]
        env["suggested_actuator"] = best["actuator"]
        
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
