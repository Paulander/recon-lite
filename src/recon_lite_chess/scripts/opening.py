"""Opening Script Hierarchy for M6.

Implements simple opening heuristics as ReCoN scripts:
- Development: Get minor pieces off starting squares
- Castling: Secure the king
- Center Control: Occupy or control central squares

Structure:
    OpeningPhase (script)
    ├── sub → DevelopMinorPieces (script)
    │   ├── sub → DevelopmentSensor (terminal, fan-in)
    │   └── sub → CenterControlSensor (terminal, fan-in)
    ├── por/ret → CastleEarly (script)
    │   └── sub → CastlingSensor (terminal)
    └── por/ret → ControlCenter (script)
        └── sub → CenterControlSensor (same terminal, fan-in)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import chess

from recon_lite.graph import Graph, Node, NodeType, LinkType


# === Opening Sensor Terminals ===

def development_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Counts developed minor pieces.
    
    Stores:
    - developed_count: Number of developed pieces (0-4)
    - development_score: Normalized score (0-1)
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    # Starting squares for minor pieces
    if side:  # White
        knight_starts = [chess.B1, chess.G1]
        bishop_starts = [chess.C1, chess.F1]
    else:  # Black
        knight_starts = [chess.B8, chess.G8]
        bishop_starts = [chess.C8, chess.F8]
    
    # Count pieces that have moved from starting squares
    developed = 0
    total = 4  # 2 knights + 2 bishops
    
    for sq in knight_starts:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type != chess.KNIGHT or piece.color != side:
            developed += 1
    
    for sq in bishop_starts:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type != chess.BISHOP or piece.color != side:
            developed += 1
    
    score = developed / total if total > 0 else 0.0
    
    node.activation.meta["developed_count"] = developed
    node.activation.meta["development_score"] = score
    node.activation.value = score
    
    env["development_score"] = score
    env["developed_count"] = developed
    
    # Success if at least half developed
    return True, developed >= 2


def castling_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Checks castling status.
    
    Stores:
    - can_castle: Whether castling is still possible
    - has_castled: Whether already castled
    - castling_score: 0=can't, 0.5=can, 1.0=done
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    # Check if already castled (king moved from starting square)
    king_sq = board.king(side)
    start_sq = chess.E1 if side else chess.E8
    
    # Heuristic: if king is on g1/c1 (or g8/c8), assume castled
    castled_squares = [chess.G1, chess.C1] if side else [chess.G8, chess.C8]
    has_castled = king_sq in castled_squares
    
    # Check if can still castle
    can_kingside = board.has_kingside_castling_rights(side)
    can_queenside = board.has_queenside_castling_rights(side)
    can_castle = can_kingside or can_queenside
    
    if has_castled:
        score = 1.0
    elif can_castle:
        score = 0.5
    else:
        score = 0.0
    
    node.activation.meta["can_castle"] = can_castle
    node.activation.meta["has_castled"] = has_castled
    node.activation.meta["castling_score"] = score
    node.activation.value = score
    
    env["can_castle"] = can_castle
    env["has_castled"] = has_castled
    
    # Success if already castled or can castle
    return True, has_castled or can_castle


def center_control_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluates center control.
    
    Central squares: e4, d4, e5, d5 (inner), c3-f6 (extended)
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    # Inner center squares
    inner_center = [chess.E4, chess.D4, chess.E5, chess.D5]
    
    # Count occupation and attacks
    occupation_score = 0
    attack_score = 0
    
    for sq in inner_center:
        piece = board.piece_at(sq)
        if piece and piece.color == side:
            occupation_score += 1
        
        # Count attacks
        attackers = board.attackers(side, sq)
        attack_score += len(attackers) * 0.25
    
    # Normalize
    total_score = (occupation_score + min(attack_score, 4)) / 8.0
    
    node.activation.meta["occupation_score"] = occupation_score
    node.activation.meta["attack_score"] = attack_score
    node.activation.meta["center_score"] = total_score
    node.activation.value = total_score
    
    env["center_score"] = total_score
    
    # Success if decent center control
    return True, total_score >= 0.3


# === Opening Script Builders ===

def create_development_sensor() -> Node:
    """Create development sensor terminal."""
    return Node(
        nid="DevelopmentSensor",
        ntype=NodeType.TERMINAL,
        predicate=development_sensor_predicate,
        meta={"sensor_type": "development", "fan_in_allowed": True},
    )


def create_castling_sensor() -> Node:
    """Create castling sensor terminal."""
    return Node(
        nid="CastlingSensor",
        ntype=NodeType.TERMINAL,
        predicate=castling_sensor_predicate,
        meta={"sensor_type": "castling", "fan_in_allowed": True},
    )


def create_center_control_sensor() -> Node:
    """Create center control sensor terminal."""
    return Node(
        nid="CenterControlSensor",
        ntype=NodeType.TERMINAL,
        predicate=center_control_sensor_predicate,
        meta={"sensor_type": "center_control", "fan_in_allowed": True},
    )


def build_opening_hierarchy(g: Graph) -> str:
    """
    Build the opening phase hierarchy in the graph.
    
    Returns the root node ID ("OpeningPhase").
    """
    # Create sensors
    dev_sensor = create_development_sensor()
    castle_sensor = create_castling_sensor()
    center_sensor = create_center_control_sensor()
    
    g.add_node(dev_sensor)
    g.add_node(castle_sensor)
    g.add_node(center_sensor)
    
    # Create opening phase root
    opening_root = Node(
        nid="OpeningPhase",
        ntype=NodeType.SCRIPT,
        meta={"layer": "phase", "phase": "opening"},
    )
    g.add_node(opening_root)
    
    # Create plan scripts
    develop = Node(
        nid="DevelopMinorPieces",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "OPENING"},
    )
    castle = Node(
        nid="CastleEarly",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "OPENING", "alt": True},
    )
    center = Node(
        nid="ControlCenter",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "OPENING", "alt": True},
    )
    
    g.add_node(develop)
    g.add_node(castle)
    g.add_node(center)
    
    # Wire hierarchy: OpeningPhase → plans
    g.add_edge("OpeningPhase", "DevelopMinorPieces", LinkType.SUB)
    g.add_edge("OpeningPhase", "CastleEarly", LinkType.SUB)
    g.add_edge("OpeningPhase", "ControlCenter", LinkType.SUB)
    
    # Wire plans → sensors
    g.add_edge("DevelopMinorPieces", "DevelopmentSensor", LinkType.SUB)
    g.add_edge("DevelopMinorPieces", "CenterControlSensor", LinkType.SUB)  # Fan-in
    
    g.add_edge("CastleEarly", "CastlingSensor", LinkType.SUB)
    
    g.add_edge("ControlCenter", "CenterControlSensor", LinkType.SUB)  # Fan-in (shared)
    
    # Set confirmation policy - any plan succeeding is enough
    g.set_confirm_policy("OpeningPhase", policy="or")
    
    return "OpeningPhase"


def get_opening_move_candidates(board: chess.Board) -> List[Tuple[chess.Move, str, float]]:
    """
    Get candidate opening moves with reasons and scores.
    
    Returns list of (move, reason, score) tuples.
    """
    candidates = []
    
    for move in board.legal_moves:
        reason = ""
        score = 0.0
        
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        
        # Development moves
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            # Check if moving from back rank
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            
            if board.turn:  # White
                if from_rank == 0 and to_rank > 0:
                    score += 0.5
                    reason = f"Develops {chess.piece_name(piece.piece_type)}"
            else:  # Black
                if from_rank == 7 and to_rank < 7:
                    score += 0.5
                    reason = f"Develops {chess.piece_name(piece.piece_type)}"
            
            # Bonus for central development
            if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5,
                                  chess.C3, chess.F3, chess.C6, chess.F6]:
                score += 0.2
                reason += " to center"
        
        # Castling
        if board.is_castling(move):
            score += 0.8
            reason = "Castles"
        
        # Central pawn moves
        if piece.piece_type == chess.PAWN:
            if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                score += 0.4
                reason = "Central pawn"
        
        if score > 0:
            candidates.append((move, reason, score))
    
    return sorted(candidates, key=lambda x: x[2], reverse=True)

