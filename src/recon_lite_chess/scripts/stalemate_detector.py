"""
Stalemate Detector Script - Shared across all endgame networks.

This module provides ReCoN nodes that detect stalemate danger and can
gate aggressive moves when the enemy king is nearly trapped.

Architecture (Option C - Hybrid):
- Atomic sensors in FeatureHub (computed once, shared)
- This script composes them into actionable signals
- Each endgame network can subscribe via POR links

Usage in endgame networks:
    from recon_lite_chess.scripts.stalemate_detector import (
        create_stalemate_danger_sensor,
        create_stalemate_gate,
        StalemateDangerLevel,
    )
    
    # Add to graph
    g.add_node(create_stalemate_danger_sensor("stalemate_sensor"))
    g.add_node(create_stalemate_gate("stalemate_gate"))
    g.add_edge("aggressive_move_node", "stalemate_gate", LinkType.POR)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import chess

from recon_lite.graph import Node, NodeType


class StalemateDangerLevel(Enum):
    """Danger levels for stalemate risk."""
    SAFE = "safe"           # mobility >= 5, free to be aggressive
    LOW = "low"             # mobility 4, slight caution
    MEDIUM = "medium"       # mobility 3, prefer waiting moves
    HIGH = "high"           # mobility 2, avoid non-mate queen moves
    CRITICAL = "critical"   # mobility <= 1, only allow checkmate


@dataclass
class StalemateAnalysis:
    """Result of stalemate analysis."""
    danger_level: StalemateDangerLevel
    enemy_mobility: int  # 0-8 raw count
    enemy_at_edge: bool
    enemy_in_corner: bool
    danger_score: float  # 0.0-1.0
    recommended_action: str  # "aggressive", "cautious", "wait", "mate_only"


def analyze_stalemate_danger(board: chess.Board) -> StalemateAnalysis:
    """
    Analyze stalemate danger for current position.
    
    This is the core logic shared by all stalemate-aware nodes.
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return StalemateAnalysis(
            danger_level=StalemateDangerLevel.SAFE,
            enemy_mobility=8,
            enemy_at_edge=False,
            enemy_in_corner=False,
            danger_score=0.0,
            recommended_action="aggressive",
        )
    
    # Count escape squares
    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)
    escape_count = 0
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            nf, nr = king_file + df, king_rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                sq = chess.square(nf, nr)
                occupant = board.piece_at(sq)
                if occupant and occupant.color == (not board.turn):
                    continue
                if board.is_attacked_by(board.turn, sq):
                    continue
                escape_count += 1
    
    # Check position
    at_edge = king_rank in (0, 7) or king_file in (0, 7)
    in_corner = enemy_king in [chess.A1, chess.A8, chess.H1, chess.H8]
    near_corner = (
        (king_rank <= 1 and king_file <= 1) or
        (king_rank <= 1 and king_file >= 6) or
        (king_rank >= 6 and king_file <= 1) or
        (king_rank >= 6 and king_file >= 6)
    )
    
    # Determine danger level
    if escape_count <= 1:
        level = StalemateDangerLevel.CRITICAL
        action = "mate_only"
        score = 1.0
    elif escape_count == 2:
        level = StalemateDangerLevel.HIGH
        action = "wait"
        score = 0.8
    elif escape_count == 3:
        level = StalemateDangerLevel.MEDIUM
        action = "cautious"
        score = 0.5
    elif escape_count == 4:
        level = StalemateDangerLevel.LOW
        action = "cautious"
        score = 0.2
    else:
        level = StalemateDangerLevel.SAFE
        action = "aggressive"
        score = 0.0
    
    # Adjust for position
    if in_corner and level != StalemateDangerLevel.CRITICAL:
        score = min(1.0, score + 0.2)
    elif near_corner and at_edge:
        score = min(1.0, score + 0.1)
    
    return StalemateAnalysis(
        danger_level=level,
        enemy_mobility=escape_count,
        enemy_at_edge=at_edge,
        enemy_in_corner=in_corner,
        danger_score=score,
        recommended_action=action,
    )


def is_checkmate_move(board: chess.Board, move: chess.Move) -> bool:
    """Check if a move delivers checkmate."""
    board.push(move)
    is_mate = board.is_checkmate()
    board.pop()
    return is_mate


def get_safe_waiting_moves(board: chess.Board, attacker_color: chess.Color) -> list[chess.Move]:
    """
    Get queen/rook moves that maintain pressure without causing stalemate.
    
    "Waiting moves" triangulate or maintain restriction without further
    reducing enemy king mobility.
    """
    waiting_moves = []
    analysis = analyze_stalemate_danger(board)
    current_mobility = analysis.enemy_mobility
    
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if not piece or piece.color != attacker_color:
            continue
        
        # Only consider queen/rook for waiting
        if piece.piece_type not in (chess.QUEEN, chess.ROOK):
            continue
        
        # Test the move
        board.push(move)
        
        # Skip if causes stalemate
        if board.is_stalemate():
            board.pop()
            continue
        
        # Skip if piece is now hanging
        if board.is_attacked_by(not attacker_color, move.to_square):
            if not board.is_attacked_by(attacker_color, move.to_square):
                board.pop()
                continue
        
        # Check new mobility
        new_analysis = analyze_stalemate_danger(board)
        
        # Good waiting move: doesn't reduce mobility further OR is checkmate
        if board.is_checkmate():
            waiting_moves.insert(0, move)  # Checkmate first!
        elif new_analysis.enemy_mobility >= current_mobility - 1:
            waiting_moves.append(move)
        
        board.pop()
    
    return waiting_moves


# ============================================================================
# ReCoN Node Factories
# ============================================================================

def create_stalemate_danger_sensor(nid: str) -> Node:
    """
    Create a terminal sensor node that computes stalemate danger.
    
    Sets env["stalemate_analysis"] with full StalemateAnalysis.
    Sets env["stalemate_danger"] with danger score (0.0-1.0).
    
    Returns:
        Node that fires TRUE when danger > 0.3, always confirms with score
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not isinstance(board, chess.Board):
            return False, False
        
        analysis = analyze_stalemate_danger(board)
        
        # Store in env for other nodes
        env["stalemate_analysis"] = analysis
        env["stalemate_danger"] = analysis.danger_score
        env["stalemate_recommended_action"] = analysis.recommended_action
        
        # Store in node meta for tracing
        node.meta["danger_score"] = analysis.danger_score
        node.meta["enemy_mobility"] = analysis.enemy_mobility
        node.meta["danger_level"] = analysis.danger_level.value
        
        # Fire if any danger detected
        has_danger = analysis.danger_score > 0.2
        return has_danger, True  # Always confirm with result
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_stalemate_gate(nid: str, danger_threshold: float = 0.5) -> Node:
    """
    Create a gate node that blocks aggressive moves when stalemate danger is high.
    
    This node should be wired via POR from aggressive move selectors.
    When danger exceeds threshold, it sets env["prefer_wait"] = True.
    
    Args:
        nid: Node ID
        danger_threshold: Danger score above which to block (default 0.5)
    
    Returns:
        Node that gates based on stalemate danger
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not isinstance(board, chess.Board):
            return False, False
        
        # Get or compute analysis
        analysis = env.get("stalemate_analysis")
        if analysis is None:
            analysis = analyze_stalemate_danger(board)
            env["stalemate_analysis"] = analysis
            env["stalemate_danger"] = analysis.danger_score
        
        danger = analysis.danger_score
        node.meta["danger_score"] = danger
        node.meta["threshold"] = danger_threshold
        
        if danger >= danger_threshold:
            # Block aggressive moves, prefer waiting
            env["prefer_wait"] = True
            env["stalemate_gate_blocked"] = True
            node.meta["action"] = "blocked"
            return True, True  # Gate fired, block propagation
        else:
            env["prefer_wait"] = False
            env["stalemate_gate_blocked"] = False
            node.meta["action"] = "passed"
            return False, False  # Let through
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_wait_move_selector(nid: str) -> Node:
    """
    Create a node that selects safe waiting moves when stalemate danger is high.
    
    This should be wired as an alternative to aggressive move selectors,
    activated when stalemate_gate fires.
    
    Returns:
        Node that suggests waiting moves
    """
    def _predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if not isinstance(board, chess.Board):
            return False, False
        
        # Only activate if gate blocked us
        if not env.get("prefer_wait"):
            return False, False
        
        attacker = board.turn
        waiting_moves = get_safe_waiting_moves(board, attacker)
        
        if waiting_moves:
            # Select best waiting move (checkmate first, then first safe move)
            best_move = waiting_moves[0]
            node.meta["suggested_move"] = best_move.uci()
            node.meta["move_type"] = "wait"
            node.meta["candidates"] = len(waiting_moves)
            
            # Store for actuator
            env.setdefault("waiting_moves", {})["suggested_move"] = best_move.uci()
            
            return True, True
        
        # No waiting moves found - might need to allow aggressive
        node.meta["suggested_move"] = None
        node.meta["move_type"] = "none"
        return False, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


# ============================================================================
# Convenience exports
# ============================================================================

__all__ = [
    "StalemateDangerLevel",
    "StalemateAnalysis",
    "analyze_stalemate_danger",
    "is_checkmate_move",
    "get_safe_waiting_moves",
    "create_stalemate_danger_sensor",
    "create_stalemate_gate",
    "create_wait_move_selector",
]

