"""Middlegame Script Hierarchy for M6.

Implements middlegame strategic plans as ReCoN scripts:
- AttackKing: Launch attack on enemy king position
- ImproveWorstPiece: Find and activate passive pieces
- CreateWeakness: Target structural weaknesses
- Simplify: Trade when materially ahead

Structure:
    MiddlegamePhase (script)
    ├── sub → AttackKing (script)
    │   ├── sub → KingSafetySensor (terminal, fan-in)
    │   └── sub → ForkDetector (terminal, fan-in)
    ├── por/ret → ImproveWorstPiece (script)
    │   └── sub → PieceActivitySensor (terminal)
    ├── por/ret → CreateWeakness (script)
    │   └── sub → StructureSensor (terminal)
    └── por/ret → Simplify (script)
        └── sub → MaterialSensor (fan-in)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import chess

from recon_lite.graph import Graph, Node, NodeType, LinkType


# === Middlegame Sensor Terminals ===

def king_safety_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluates king safety for both sides.
    
    Factors:
    - Pawn shield
    - Attacking pieces near king
    - Open files towards king
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    enemy = not side
    
    enemy_king = board.king(enemy)
    if enemy_king is None:
        return True, False
    
    # Count attackers near enemy king
    king_zone = _get_king_zone(enemy_king)
    attackers_in_zone = 0
    
    for sq in king_zone:
        attackers = board.attackers(side, sq)
        attackers_in_zone += len(attackers)
    
    # Check pawn shield (enemy perspective)
    shield_score = _pawn_shield_score(board, enemy, enemy_king)
    
    # Higher score = enemy king more vulnerable
    vulnerability = min(1.0, attackers_in_zone * 0.1 + (1.0 - shield_score) * 0.5)
    
    node.activation.meta["enemy_king_vulnerability"] = vulnerability
    node.activation.meta["attackers_in_zone"] = attackers_in_zone
    node.activation.meta["enemy_shield_score"] = shield_score
    node.activation.value = vulnerability
    
    env["enemy_king_vulnerability"] = vulnerability
    
    # Success if enemy king is somewhat vulnerable
    return True, vulnerability > 0.3


def _get_king_zone(king_sq: chess.Square) -> List[chess.Square]:
    """Get squares around the king."""
    zone = []
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            f = king_file + df
            r = king_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                zone.append(chess.square(f, r))
    
    return zone


def _pawn_shield_score(board: chess.Board, color: bool, king_sq: chess.Square) -> float:
    """Evaluate pawn shield strength (0-1)."""
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    
    # Expected shield rank
    shield_rank = king_rank + 1 if color else king_rank - 1
    
    if shield_rank < 0 or shield_rank > 7:
        return 0.0
    
    shield_pawns = 0
    files_to_check = [max(0, king_file - 1), king_file, min(7, king_file + 1)]
    
    for f in files_to_check:
        sq = chess.square(f, shield_rank)
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            shield_pawns += 1
    
    return shield_pawns / 3.0


def piece_activity_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluates piece activity and finds the worst piece.
    
    Activity measured by:
    - Number of squares attacked
    - Central vs rim positioning
    - Blocked vs active
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    worst_piece_sq = None
    worst_activity = float('inf')
    total_activity = 0
    piece_count = 0
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != side:
            continue
        if piece.piece_type == chess.KING:
            continue
        
        # Calculate activity
        mobility = len(list(board.attacks(sq)))
        
        # Bonus for central squares
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        centrality = 1.0 - (abs(3.5 - file) + abs(3.5 - rank)) / 7.0
        
        activity = mobility * 0.7 + centrality * 0.3
        total_activity += activity
        piece_count += 1
        
        if activity < worst_activity:
            worst_activity = activity
            worst_piece_sq = sq
    
    avg_activity = total_activity / piece_count if piece_count > 0 else 0
    
    node.activation.meta["worst_piece_sq"] = worst_piece_sq
    node.activation.meta["worst_activity"] = worst_activity
    node.activation.meta["avg_activity"] = avg_activity
    node.activation.value = avg_activity
    
    env["worst_piece_sq"] = worst_piece_sq
    env["avg_activity"] = avg_activity
    
    # Success if there's room for improvement
    return True, worst_activity < avg_activity * 0.7


def structure_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluates pawn structure weaknesses.
    
    Weaknesses:
    - Isolated pawns
    - Doubled pawns
    - Backward pawns
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    enemy = not board.turn
    
    weaknesses = {
        "isolated": 0,
        "doubled": 0,
        "backward": 0,
    }
    
    enemy_pawns = list(board.pieces(chess.PAWN, enemy))
    
    # Count pawns per file
    file_counts = [0] * 8
    for sq in enemy_pawns:
        file_counts[chess.square_file(sq)] += 1
    
    for sq in enemy_pawns:
        file = chess.square_file(sq)
        
        # Isolated: no friendly pawns on adjacent files
        has_neighbor = False
        if file > 0 and file_counts[file - 1] > 0:
            has_neighbor = True
        if file < 7 and file_counts[file + 1] > 0:
            has_neighbor = True
        
        if not has_neighbor:
            weaknesses["isolated"] += 1
        
        # Doubled: multiple pawns on same file
        if file_counts[file] > 1:
            weaknesses["doubled"] += 1
    
    weakness_score = (
        weaknesses["isolated"] * 0.3 +
        weaknesses["doubled"] * 0.2
    )
    weakness_score = min(1.0, weakness_score)
    
    node.activation.meta["weaknesses"] = weaknesses
    node.activation.meta["weakness_score"] = weakness_score
    node.activation.value = weakness_score
    
    env["enemy_weaknesses"] = weaknesses
    env["weakness_score"] = weakness_score
    
    # Success if enemy has significant weaknesses
    return True, weakness_score > 0.2


# === Middlegame Script Builders ===

def create_king_safety_sensor() -> Node:
    """Create king safety sensor terminal."""
    return Node(
        nid="KingSafetySensor",
        ntype=NodeType.TERMINAL,
        predicate=king_safety_sensor_predicate,
        meta={"sensor_type": "king_safety", "fan_in_allowed": True},
    )


def create_piece_activity_sensor() -> Node:
    """Create piece activity sensor terminal."""
    return Node(
        nid="PieceActivitySensor",
        ntype=NodeType.TERMINAL,
        predicate=piece_activity_sensor_predicate,
        meta={"sensor_type": "piece_activity", "fan_in_allowed": True},
    )


def create_structure_sensor() -> Node:
    """Create structure sensor terminal."""
    return Node(
        nid="StructureSensor",
        ntype=NodeType.TERMINAL,
        predicate=structure_sensor_predicate,
        meta={"sensor_type": "structure", "fan_in_allowed": True},
    )


def build_middlegame_hierarchy(g: Graph) -> str:
    """
    Build the middlegame phase hierarchy in the graph.
    
    Returns the root node ID ("MiddlegamePhase").
    """
    # Create sensors
    king_sensor = create_king_safety_sensor()
    activity_sensor = create_piece_activity_sensor()
    structure_sensor = create_structure_sensor()
    
    g.add_node(king_sensor)
    g.add_node(activity_sensor)
    g.add_node(structure_sensor)
    
    # Create middlegame phase root
    middlegame_root = Node(
        nid="MiddlegamePhase",
        ntype=NodeType.SCRIPT,
        meta={"layer": "phase", "phase": "middlegame"},
    )
    g.add_node(middlegame_root)
    
    # Create plan scripts
    attack = Node(
        nid="AttackKingPlan",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "MIDDLEGAME", "inertia": 0.8},
    )
    improve = Node(
        nid="ImproveWorstPiecePlan",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "MIDDLEGAME", "alt": True},
    )
    weakness = Node(
        nid="CreateWeaknessPlan",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "MIDDLEGAME", "alt": True},
    )
    simplify = Node(
        nid="SimplifyPlan",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "MIDDLEGAME", "alt": True},
    )
    
    g.add_node(attack)
    g.add_node(improve)
    g.add_node(weakness)
    g.add_node(simplify)
    
    # Wire hierarchy: MiddlegamePhase → plans
    g.add_edge("MiddlegamePhase", "AttackKingPlan", LinkType.SUB)
    g.add_edge("MiddlegamePhase", "ImproveWorstPiecePlan", LinkType.SUB)
    g.add_edge("MiddlegamePhase", "CreateWeaknessPlan", LinkType.SUB)
    g.add_edge("MiddlegamePhase", "SimplifyPlan", LinkType.SUB)
    
    # Wire plans → sensors
    g.add_edge("AttackKingPlan", "KingSafetySensor", LinkType.SUB)
    
    g.add_edge("ImproveWorstPiecePlan", "PieceActivitySensor", LinkType.SUB)
    
    g.add_edge("CreateWeaknessPlan", "StructureSensor", LinkType.SUB)
    
    # SimplifyPlan needs MaterialSensor - assumed to exist from earlier
    # Create a placeholder if not present
    if "MaterialSensor" not in g.nodes:
        mat_sensor = Node(
            nid="MaterialSensor_MG",
            ntype=NodeType.TERMINAL,
            meta={"sensor_type": "material", "fan_in_allowed": True},
        )
        g.add_node(mat_sensor)
        g.add_edge("SimplifyPlan", "MaterialSensor_MG", LinkType.SUB)
    else:
        g.add_edge("SimplifyPlan", "MaterialSensor", LinkType.SUB)
    
    # Set confirmation policy
    g.set_confirm_policy("MiddlegamePhase", policy="or")
    
    return "MiddlegamePhase"


def get_middlegame_move_candidates(
    board: chess.Board,
    plan: str = "general",
) -> List[Tuple[chess.Move, str, float]]:
    """
    Get candidate middlegame moves based on active plan.
    
    Args:
        board: Current position
        plan: Active plan ("attack", "improve", "weakness", "simplify", "general")
        
    Returns:
        List of (move, reason, score) tuples.
    """
    candidates = []
    
    for move in board.legal_moves:
        reason = ""
        score = 0.0
        
        # Capture scoring
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                score += _piece_value(captured.piece_type) * 0.3
                reason = f"Captures {chess.piece_name(captured.piece_type)}"
        
        # Plan-specific scoring
        if plan == "attack":
            # Moves toward enemy king
            enemy_king = board.king(not board.turn)
            if enemy_king:
                dist_before = chess.square_distance(move.from_square, enemy_king)
                dist_after = chess.square_distance(move.to_square, enemy_king)
                if dist_after < dist_before:
                    score += 0.3
                    reason += " approaches king"
        
        elif plan == "improve":
            # Moves that improve piece activity
            piece = board.piece_at(move.from_square)
            if piece:
                mobility_before = len(list(board.attacks(move.from_square)))
                # Estimate mobility after (simplified)
                if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                    score += 0.4
                    reason = "Centralizes piece"
        
        elif plan == "simplify":
            # Equal or winning trades
            if board.is_capture(move):
                attacker = board.piece_at(move.from_square)
                victim = board.piece_at(move.to_square)
                if attacker and victim:
                    if _piece_value(victim.piece_type) >= _piece_value(attacker.piece_type):
                        score += 0.5
                        reason = "Good trade"
        
        if score > 0:
            candidates.append((move, reason, score))
    
    return sorted(candidates, key=lambda x: x[2], reverse=True)


def _piece_value(pt: chess.PieceType) -> float:
    """Standard piece values."""
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }
    return values.get(pt, 0.0)

