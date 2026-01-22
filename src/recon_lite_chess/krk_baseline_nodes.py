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


def create_krk_entry_root():
    """
    Create KRK entry root node with blackboard caching.
    
    Extracts features once per tick and caches in blackboard.
    """
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
        id="krk_entry",
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_krk_hub():
    """
    Create Hub node with bandit selection.
    
    Selects which Leg to activate based on bandit scores.
    """
    def predicate(node, graph, env):
        # For now, just pass through
        # Bandit logic will be added in Phase 2
        return True, True
    
    return Node(
        id="krk_hub",
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_leg_script():
    """Create Leg SCRIPT node (simple pass-through)"""
    def predicate(node, graph, env):
        return True, True
    
    return Node(
        id="leg_script",
        ntype=NodeType.SCRIPT,
        predicate=predicate
    )


def create_and_gate():
    """
    Create AND-gate SCRIPT node.
    
    All children must confirm for this to confirm.
    """
    def predicate(node, graph, env):
        # Aggregation handled by engine
        return True, True
    
    return Node(
        id="and_gate",
        ntype=NodeType.SCRIPT,
        predicate=predicate,
        aggregation="and"
    )


def create_sensor_terminal():
    """
    Create sensor TERMINAL node.
    
    Reads cached features from blackboard, applies readout, caches output.
    """
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
        
        # Apply readout
        sub_v = features[feature_mask]
        output = apply_readout(sub_v, readout_type, readout_params)
        
        # Cache output for actuators
        blackboard["sensor_outputs"][node.id] = output
        
        # Store in node state
        node.state["output"] = output
        
        return True, True
    
    return Node(
        id="sensor_terminal",
        ntype=NodeType.TERMINAL,
        predicate=predicate
    )


def create_actuator_terminal():
    """
    Create actuator TERMINAL node.
    
    Scores moves by similarity to goal_delta, selects best move.
    """
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
                sensor_node = graph.nodes.get(target_id.replace("_post_" + str(node.meta.get("actuator_id", "")), ""))
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
                
                sub_v = features_1[feature_mask]
                s1 = apply_readout(sub_v, readout_type, readout_params)
                
                # Compute delta
                delta = s1 - s0[target_id]
                delta_s.append(delta)
            
            # Score by similarity to goal_delta
            goal_deltas = [goal_delta[t] for t in targets]
            similarity = cosine_similarity(delta_s, goal_deltas)
            scores[move] = similarity
        
        # Select best move
        best_move = max(scores, key=scores.get)
        best_score = scores[best_move]
        
        # Store in environment
        env["suggested_move"] = best_move
        env["move_confidence"] = best_score
        
        # Store in node state
        node.state["move"] = best_move.uci()
        node.state["confidence"] = best_score
        
        return True, True
    
    return Node(
        id="actuator_terminal",
        ntype=NodeType.TERMINAL,
        predicate=predicate
    )


# Helper functions

def apply_readout(sub_v: np.ndarray, readout_type: str, params: Dict) -> float:
    """Apply sensor readout function"""
    if readout_type == "identity":
        if len(sub_v) != 1:
            raise ValueError(f"identity readout requires exactly 1 feature, got {len(sub_v)}")
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
        raise ValueError(f"Unknown readout_type: {readout_type}")


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))
