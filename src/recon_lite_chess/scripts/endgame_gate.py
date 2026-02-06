"""
Endgame Gating Node for ReCoN.

This module provides the learned gating mechanism for subgraph selection.
The gate node computes activation scores for each endgame subgraph based
on material detection, and weighted edges determine which subgraph activates.

Architecture:
    GameRoot → WinStrategy → endgame_gate → {kpk_root, kqk_root, krk_root}
                                            (weighted SUB edges)

The gate predicate runs material detection and stores gate activations
in env. The engine then uses edge weights to determine which subgraph
to lock into.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import chess

from recon_lite import Node, NodeType

from recon_lite_chess.sensors.structure import summarize_kpk_material
from recon_lite_chess.scripts.kqk import is_kqk_position


def _detect_krk(board: chess.Board) -> Tuple[bool, bool]:
    """
    Detect KRK (King+Rook vs King) position.
    
    Returns:
        (is_krk, attacker_color): Whether position is KRK and which color has the rook
    """
    pieces = list(board.piece_map().values())
    if len(pieces) != 3:
        return False, False
    
    types = [p.piece_type for p in pieces]
    if types.count(chess.KING) != 2 or types.count(chess.ROOK) != 1:
        return False, False
    
    # Find rook color
    for piece in pieces:
        if piece.piece_type == chess.ROOK:
            return True, piece.color
    
    return False, False


def create_endgame_gate(node_id: str = "endgame_gate") -> Node:
    """
    Create the endgame gating node.
    
    This node detects material patterns and computes activation scores
    for each endgame subgraph. These scores are stored in env and used
    by the engine (with edge weights) to determine which subgraph to activate.
    
    Gate activations are in env["endgame_gate"]["activations"]:
        - kpk: 0.0-1.0 (King+Pawn vs King detected)
        - kqk: 0.0-1.0 (King+Queen vs King detected)
        - krk: 0.0-1.0 (King+Rook vs King detected)
    
    Args:
        node_id: Unique node identifier
        
    Returns:
        Node configured with gating predicate
    """
    
    def gate_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Compute gate activations for endgame subgraphs.
        
        Returns (done, success) tuple. Always succeeds if board is present.
        """
        board = env.get("board")
        if not board:
            return True, False
        
        # Compute activations based on material detection
        activations = {
            "kpk": 0.0,
            "kqk": 0.0,
            "krk": 0.0,
        }
        
        # Check whose turn it is - only activate gates for current player's endgames
        turn = board.turn
        
        # KQK detection (highest priority - strongest piece)
        is_kqk, kqk_attacker = is_kqk_position(board)
        if is_kqk and kqk_attacker == turn:
            activations["kqk"] = 1.0
        
        # KRK detection
        is_krk, krk_attacker = _detect_krk(board)
        if is_krk and krk_attacker == turn:
            activations["krk"] = 1.0
        
        # KPK detection (lowest priority)
        kpk_summary = summarize_kpk_material(board)
        if kpk_summary.get("is_kpk") and kpk_summary.get("attacker_color") == turn:
            activations["kpk"] = 1.0
        
        # Store gate activations in env
        env["endgame_gate"] = {
            "activations": activations,
            "active_endgame": None,
        }
        
        # Determine which endgame is active (highest activation)
        # This will be used by the engine to determine subgraph locking
        max_activation = 0.0
        active_endgame = None
        for eg, activation in activations.items():
            if activation > max_activation:
                max_activation = activation
                active_endgame = eg
        
        if active_endgame:
            env["endgame_gate"]["active_endgame"] = active_endgame
        
        return True, max_activation > 0.0
    
    node = Node(node_id, NodeType.SCRIPT, meta={
        "layer": "gate",
        "subgraph": "main",
        "description": "Endgame subgraph routing gate",
    })
    node.predicate = gate_predicate
    
    return node


# =============================================================================
# GATE EDGE WEIGHTS
# =============================================================================
# Default weights for gate → subgraph edges
# These can be trained via consolidation/plasticity

DEFAULT_GATE_WEIGHTS = {
    "kpk": 1.0,  # King+Pawn vs King
    "kqk": 1.0,  # King+Queen vs King  
    "krk": 1.0,  # King+Rook vs King
}


def get_gate_routing_decision(env: Dict[str, Any], weights: Dict[str, float] = None) -> str:
    """
    Determine which subgraph to route to based on gate activations and weights.
    
    This is used by the training loop to decide which subgraph to lock.
    
    Args:
        env: Environment dict containing gate activations
        weights: Edge weights for each subgraph (default: all 1.0)
        
    Returns:
        Subgraph name to route to (e.g., "kpk") or None if no endgame active
    """
    if weights is None:
        weights = DEFAULT_GATE_WEIGHTS
    
    gate_data = env.get("endgame_gate", {})
    activations = gate_data.get("activations", {})
    
    if not activations:
        return None
    
    # Compute weighted scores
    scores = {}
    for subgraph, activation in activations.items():
        weight = weights.get(subgraph, 1.0)
        scores[subgraph] = activation * weight
    
    # Return highest-scoring subgraph (if any activation)
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] > 0:
        return best[0]
    
    return None


__all__ = [
    "create_endgame_gate",
    "get_gate_routing_decision",
    "DEFAULT_GATE_WEIGHTS",
]
