"""Goal-directed actuator nodes for KRK/KPK servoing.

Implements:
  - Goal actuators: pattern-matching terminals that propose target deltas.
  - Goal solver: selects a legal move that best matches a target delta vector.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from recon_lite.graph import Node, NodeType

from .predicates import (
    box_area,
    box_area_after,
    gives_safe_check,
    has_opposition,
    has_opposition_after,
    chebyshev,
)

DEFAULT_GOAL_FEATURES = [
    "box_area_delta",
    "king_distance_delta",
    "opposition_gain",
    "safe_check",
]

FEATURE_SCALES = {
    "box_area_delta": 64.0,
    "king_distance_delta": 8.0,
    "opposition_gain": 1.0,
    "safe_check": 1.0,
    "mate_delivered": 1.0,
}


def _king_distance_any(board: "chess.Board") -> int:
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 8
    return chebyshev(wk, bk)


def _compute_goal_features(board: "chess.Board", move: "chess.Move") -> Dict[str, float]:
    before_box = box_area(board)
    after_box = box_area_after(board, move)

    before_dist = _king_distance_any(board)
    trial = board.copy(stack=False)
    trial.push(move)
    after_dist = _king_distance_any(trial)

    before_opp = 1.0 if has_opposition(board) else 0.0
    after_opp = 1.0 if has_opposition_after(board, move) else 0.0

    return {
        "box_area_delta": (after_box - before_box) / FEATURE_SCALES["box_area_delta"],
        "king_distance_delta": (after_dist - before_dist) / FEATURE_SCALES["king_distance_delta"],
        "opposition_gain": max(0.0, after_opp - before_opp),
        "safe_check": 1.0 if gives_safe_check(board, move) else 0.0,
        "mate_delivered": 1.0 if trial.is_checkmate() else 0.0,
    }


def compute_goal_feature_deltas(
    board: "chess.Board",
    move: "chess.Move",
    goal_features: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute the goal feature deltas for a move.
    """
    feats = _compute_goal_features(board, move)
    if goal_features is None:
        return feats
    return {name: feats.get(name, 0.0) for name in goal_features}


def _align_goal_vectors(
    goal_vector: Iterable[float],
    goal_features: List[str],
    goal_weights: Optional[Iterable[float]],
) -> Tuple[List[float], List[float]]:
    goal_vec = list(goal_vector)
    if len(goal_vec) < len(goal_features):
        goal_vec += [0.0] * (len(goal_features) - len(goal_vec))
    elif len(goal_vec) > len(goal_features):
        goal_vec = goal_vec[: len(goal_features)]

    weights = list(goal_weights) if goal_weights is not None else [1.0] * len(goal_features)
    if len(weights) < len(goal_features):
        weights += [1.0] * (len(goal_features) - len(weights))
    elif len(weights) > len(goal_features):
        weights = weights[: len(goal_features)]

    return goal_vec, weights


def select_move_from_goal(
    board: "chess.Board",
    goal_vector: Iterable[float],
    goal_features: Optional[List[str]] = None,
    goal_weights: Optional[Iterable[float]] = None,
    legal_moves: Optional[Iterable["chess.Move"]] = None,
    match_mode: Optional[str] = None,
) -> Tuple[Optional["chess.Move"], Optional[float]]:
    if not HAS_CHESS or board is None:
        return None, None

    features = goal_features or DEFAULT_GOAL_FEATURES
    targets, weights = _align_goal_vectors(goal_vector, features, goal_weights)

    best_move = None
    best_score = None

    move_iter = legal_moves if legal_moves is not None else board.legal_moves
    mode = match_mode or "l2"
    for mv in move_iter:
        actual = _compute_goal_features(board, mv)
        score = 0.0
        for idx, name in enumerate(features):
            actual_val = actual.get(name, 0.0)
            if mode == "dot":
                score -= weights[idx] * (actual_val * targets[idx])
            else:
                diff = actual_val - targets[idx]
                score += weights[idx] * (diff * diff)

        if best_score is None or score < best_score:
            best_score = score
            best_move = mv

    return best_move, best_score


def _apply_mask(values: List[float], mask: Optional[List[int]]) -> List[float]:
    if not mask:
        return list(values)
    if isinstance(mask[0], bool):
        return [v for v, keep in zip(values, mask) if keep]
    return [values[idx] for idx in mask if idx < len(values)]


def _similarity(
    signature: List[float],
    features: List[float],
    mask: Optional[List[int]],
    mode: str,
) -> float:
    sig_vals = _apply_mask(signature, mask)
    feat_vals = _apply_mask(features, mask)
    if not sig_vals or not feat_vals:
        return 0.0

    if mode == "dot":
        return float(sum(a * b for a, b in zip(sig_vals, feat_vals)))

    if HAS_NUMPY:
        sig_arr = np.array(sig_vals)
        feat_arr = np.array(feat_vals)
        sig_norm = sig_arr / (np.linalg.norm(sig_arr) + 1e-8)
        feat_norm = feat_arr / (np.linalg.norm(feat_arr) + 1e-8)
        return float(np.dot(sig_norm, feat_norm))

    dot = sum(a * b for a, b in zip(sig_vals, feat_vals))
    sig_norm = sum(a * a for a in sig_vals) ** 0.5
    feat_norm = sum(b * b for b in feat_vals) ** 0.5
    if sig_norm == 0.0 or feat_norm == 0.0:
        return 0.0
    return dot / (sig_norm * feat_norm)


def create_goal_actuator_hub(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        env.setdefault("krk", {}).setdefault("actuator_proposals", [])
        return True, True

    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_goal_actuator(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        signature = node.meta.get("pattern_signature")
        features = env.get("features")
        stem_manager = env.get("__stem_manager__")
        feature_mask = node.meta.get("feature_mask")
        match_mode = node.meta.get("match_mode", "cosine")

        similarity = 0.0
        matched = False

        if signature is not None and features is not None:
            try:
                similarity = _similarity(list(signature), list(features), feature_mask, match_mode)
                node.meta["last_similarity"] = similarity
                threshold = node.meta.get("threshold", 0.7)
                matched = similarity >= threshold
            except Exception:
                matched = False
        else:
            # No signature/features: allow low-confidence exploration
            matched = True
            similarity = node.meta.get("fallback_activation", 0.1)

        activation = similarity if matched else 0.0
        node.meta["activation"] = activation

        if matched:
            goal_features = node.meta.get("goal_features") or DEFAULT_GOAL_FEATURES
            goal_vector = node.meta.get("goal_vector")
            goal_weights = node.meta.get("goal_weights")
            goal_match_mode = node.meta.get("goal_match_mode")
            cell_id = node.meta.get("cell_id")
            cell = None
            if stem_manager and cell_id in stem_manager.cells:
                cell = stem_manager.cells[cell_id]
                if feature_mask is None:
                    feature_mask = cell.metadata.get("feature_mask")
                if goal_vector is None:
                    goal_vector = cell.metadata.get("goal_vector")
                    if goal_vector is not None:
                        goal_features = cell.metadata.get("goal_features", goal_features)
                        goal_match_mode = cell.metadata.get("goal_match_mode", goal_match_mode)

            if goal_vector is None:
                if goal_features == ["mate_delivered"]:
                    goal_vector = [1.0]
                else:
                    goal_vector = [random.uniform(-1.0, 1.0) for _ in goal_features]
                node.meta["goal_vector"] = goal_vector
                node.meta["goal_features"] = goal_features
                if cell:
                    cell.metadata["goal_vector"] = list(goal_vector)
                    cell.metadata["goal_features"] = list(goal_features)

            proposal = {
                "node_id": node.nid,
                "cell_id": node.meta.get("cell_id"),
                "activation": activation,
                "goal_vector": list(goal_vector),
                "goal_features": list(goal_features),
                "goal_weights": list(goal_weights) if goal_weights is not None else None,
                "goal_match_mode": goal_match_mode,
            }
            env.setdefault("krk", {}).setdefault("actuator_proposals", []).append(proposal)

        return True, True

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_goal_solver(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        proposals = env.get("krk", {}).get("actuator_proposals", [])
        if not board:
            return True, True

        if not proposals:
            waits = env.setdefault("krk", {}).get("goal_solver_waits", 0)
            waits += 1
            env.setdefault("krk", {})["goal_solver_waits"] = waits
            max_waits = env.get("krk", {}).get("goal_solver_max_waits", 2)
            if waits <= max_waits:
                return False, False
            return True, True

        stem_manager = env.get("__stem_manager__")
        best = None
        best_score = None

        for prop in proposals:
            activation = prop.get("activation", 0.0)
            cell_id = prop.get("cell_id")
            xp_weight = 1.0
            consistency = 1.0
            if stem_manager and cell_id in stem_manager.cells:
                cell = stem_manager.cells[cell_id]
                actuator_xp = cell.metadata.get("actuator_xp", cell.xp)
                xp_weight = max(0.0, min(actuator_xp / 100.0, 1.0))
                consistency = max(0.0, min(getattr(cell, "trial_consistency", 1.0), 1.0))

            score = activation * (0.5 + 0.5 * xp_weight) * (0.5 + 0.5 * consistency)
            if best_score is None or score > best_score:
                best_score = score
                best = prop

        if best is None:
            return True, True

        move_filter = env.get("krk", {}).get("goal_move_filter")
        legal_moves = None
        if move_filter == "rook_only":
            try:
                legal_moves = [
                    mv for mv in board.legal_moves
                    if board.piece_type_at(mv.from_square) == chess.ROOK
                ]
            except Exception:
                legal_moves = None

        move, move_score = select_move_from_goal(
            board,
            best.get("goal_vector", []),
            goal_features=best.get("goal_features"),
            goal_weights=best.get("goal_weights"),
            legal_moves=legal_moves,
            match_mode=best.get("goal_match_mode"),
        )
        if move is None:
            return True, True

        policy = env.setdefault("krk_root", {}).setdefault("policy", {})
        policy["suggested_move"] = move.uci()
        env.setdefault("krk", {}).setdefault("policy", {})["suggested_move"] = move.uci()

        krk_state = env.setdefault("krk", {})
        krk_state["goal_policy"] = {
            "suggested_move": move.uci(),
            "move_score": move_score,
            "winner_node_id": best.get("node_id"),
            "winner_cell_id": best.get("cell_id"),
            "activation_score": best_score,
            "source": "goal_solver",
        }
        krk_state["actuator_winner_cell_id"] = best.get("cell_id")
        krk_state["actuator_winner_node_id"] = best.get("node_id")

        node.meta["suggested_move"] = move.uci()
        node.meta["winner_node_id"] = best.get("node_id")
        node.meta["activation"] = best_score if best_score is not None else 0.0

        return True, True

    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)
