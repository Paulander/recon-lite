"""Opening Script Hierarchy for M6/M8.

Implements simple opening heuristics as ReCoN scripts:
- Development: Get minor pieces off starting squares
- Castling: Secure the king
- Center Control: Occupy or control central squares
- Knights Before Bishops: Prefer developing knights first
- Fianchetto: Detect and evaluate fianchetto setups
- Central Pawn Structure: Evaluate pawn center control

Structure:
    OpeningPhase (script)
    ├── sub → DevelopMinorPieces (script)
    │   ├── sub → DevelopmentSensor (terminal, fan-in)
    │   ├── sub → CenterControlSensor (terminal, fan-in)
    │   └── sub → KnightsBeforeBishopsSensor (terminal)
    ├── por/ret → CastleEarly (script)
    │   └── sub → CastlingSensor (terminal)
    ├── por/ret → ControlCenter (script)
    │   ├── sub → CenterControlSensor (same terminal, fan-in)
    │   └── sub → PawnCenterSensor (terminal)
    └── por/ret → Fianchetto (script)
        └── sub → FianchettoSensor (terminal)
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


def knights_before_bishops_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluates knights before bishops principle.
    
    In the opening, knights should generally be developed before bishops because:
    - Knights have fewer good squares (center preferred)
    - Bishops can wait to see where they're most effective
    
    Returns True if the principle is being followed.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    # Starting squares
    if side:  # White
        knight_starts = [chess.B1, chess.G1]
        bishop_starts = [chess.C1, chess.F1]
    else:  # Black
        knight_starts = [chess.B8, chess.G8]
        bishop_starts = [chess.C8, chess.F8]
    
    # Count undeveloped pieces
    undeveloped_knights = 0
    undeveloped_bishops = 0
    
    for sq in knight_starts:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.KNIGHT and piece.color == side:
            undeveloped_knights += 1
    
    for sq in bishop_starts:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.BISHOP and piece.color == side:
            undeveloped_bishops += 1
    
    # Principle is followed if knights are developed before or same as bishops
    # i.e., undeveloped_knights <= undeveloped_bishops
    following_principle = undeveloped_knights <= undeveloped_bishops
    
    # Score: 1.0 if all knights developed, decreases with undeveloped knights
    knights_developed = 2 - undeveloped_knights
    bishops_developed = 2 - undeveloped_bishops
    
    # Bonus for following principle, penalty for not
    if following_principle:
        score = (knights_developed / 2.0) * 0.5 + 0.5
    else:
        score = (knights_developed / 2.0) * 0.3
    
    node.activation.meta["undeveloped_knights"] = undeveloped_knights
    node.activation.meta["undeveloped_bishops"] = undeveloped_bishops
    node.activation.meta["following_principle"] = following_principle
    node.activation.value = score
    
    env["knights_before_bishops"] = {
        "undeveloped_knights": undeveloped_knights,
        "undeveloped_bishops": undeveloped_bishops,
        "following_principle": following_principle,
    }
    
    return True, following_principle


def fianchetto_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Detects and evaluates fianchetto setups.
    
    A fianchetto is when a bishop is placed on b2/g2 (white) or b7/g7 (black)
    after moving the b or g pawn.
    
    Evaluates:
    - Is fianchetto complete (bishop on diagonal)
    - Is fianchetto in progress (pawn moved, bishop can go)
    - Quality of the long diagonal
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    # Fianchetto squares
    if side:  # White
        fianchetto_squares = {
            "kingside": {"pawn_from": chess.G2, "pawn_to": chess.G3, "bishop_sq": chess.G2},
            "queenside": {"pawn_from": chess.B2, "pawn_to": chess.B3, "bishop_sq": chess.B2},
        }
        bishop_final_squares = [chess.G2, chess.B2]
    else:  # Black
        fianchetto_squares = {
            "kingside": {"pawn_from": chess.G7, "pawn_to": chess.G6, "bishop_sq": chess.G7},
            "queenside": {"pawn_from": chess.B7, "pawn_to": chess.B6, "bishop_sq": chess.B7},
        }
        bishop_final_squares = [chess.G7, chess.B7]
    
    fianchettos = {
        "kingside": {"complete": False, "in_progress": False, "diagonal_clear": False},
        "queenside": {"complete": False, "in_progress": False, "diagonal_clear": False},
    }
    
    for wing, squares in fianchetto_squares.items():
        # Check if bishop is on fianchetto square
        bishop_piece = board.piece_at(squares["bishop_sq"])
        if bishop_piece and bishop_piece.piece_type == chess.BISHOP and bishop_piece.color == side:
            fianchettos[wing]["complete"] = True
            
            # Check diagonal clarity (how many squares the bishop controls)
            attacks = board.attacks(squares["bishop_sq"])
            fianchettos[wing]["diagonal_clear"] = len(attacks) >= 5
        
        # Check if pawn has moved to allow fianchetto
        pawn_at_to = board.piece_at(squares["pawn_to"])
        if pawn_at_to and pawn_at_to.piece_type == chess.PAWN and pawn_at_to.color == side:
            fianchettos[wing]["in_progress"] = True
    
    # Score calculation
    score = 0.0
    complete_count = 0
    
    for wing, status in fianchettos.items():
        if status["complete"]:
            complete_count += 1
            score += 0.4
            if status["diagonal_clear"]:
                score += 0.1
        elif status["in_progress"]:
            score += 0.15
    
    score = min(1.0, score)
    
    node.activation.meta["fianchettos"] = fianchettos
    node.activation.meta["complete_count"] = complete_count
    node.activation.value = score
    
    env["fianchetto"] = fianchettos
    
    # Success if at least one fianchetto is complete or in progress
    has_fianchetto = any(
        f["complete"] or f["in_progress"]
        for f in fianchettos.values()
    )
    
    return True, has_fianchetto


def pawn_center_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Evaluates central pawn structure.
    
    Classic center: pawns on e4/d4 (white) or e5/d5 (black)
    Hypermodern: control center with pieces, pawns on flanks
    
    Detects pawn structure type and evaluates strength.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    # Central pawn squares
    if side:  # White
        ideal_center = [chess.E4, chess.D4]
        good_center = [chess.E3, chess.D3, chess.C4]
    else:  # Black
        ideal_center = [chess.E5, chess.D5]
        good_center = [chess.E6, chess.D6, chess.C5]
    
    # Count pawns in center
    ideal_pawns = 0
    good_pawns = 0
    
    for sq in ideal_center:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == side:
            ideal_pawns += 1
    
    for sq in good_center:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == side:
            good_pawns += 1
    
    # Determine structure type
    if ideal_pawns == 2:
        structure_type = "classical"
        score = 1.0
    elif ideal_pawns == 1:
        structure_type = "semi_classical"
        score = 0.7
    elif good_pawns >= 2:
        structure_type = "flexible"
        score = 0.5
    elif good_pawns >= 1:
        structure_type = "hypermodern"
        score = 0.4
    else:
        structure_type = "undeveloped"
        score = 0.2
    
    # Check for pawn tension (pawns attacking each other)
    has_tension = False
    for sq in ideal_center:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN and piece.color == side:
            # Check if attacked by enemy pawn
            attackers = board.attackers(not side, sq)
            for attacker_sq in attackers:
                attacker = board.piece_at(attacker_sq)
                if attacker and attacker.piece_type == chess.PAWN:
                    has_tension = True
                    break
    
    node.activation.meta["ideal_pawns"] = ideal_pawns
    node.activation.meta["good_pawns"] = good_pawns
    node.activation.meta["structure_type"] = structure_type
    node.activation.meta["has_tension"] = has_tension
    node.activation.value = score
    
    env["pawn_center"] = {
        "ideal_pawns": ideal_pawns,
        "good_pawns": good_pawns,
        "structure_type": structure_type,
        "has_tension": has_tension,
    }
    
    # Success if we have some center presence
    return True, ideal_pawns >= 1 or good_pawns >= 1


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


def create_knights_before_bishops_sensor() -> Node:
    """Create knights before bishops sensor terminal."""
    return Node(
        nid="KnightsBeforeBishopsSensor",
        ntype=NodeType.TERMINAL,
        predicate=knights_before_bishops_predicate,
        meta={"sensor_type": "knights_before_bishops", "fan_in_allowed": True},
    )


def create_fianchetto_sensor() -> Node:
    """Create fianchetto sensor terminal."""
    return Node(
        nid="FianchettoSensor",
        ntype=NodeType.TERMINAL,
        predicate=fianchetto_sensor_predicate,
        meta={"sensor_type": "fianchetto", "fan_in_allowed": True},
    )


def create_pawn_center_sensor() -> Node:
    """Create pawn center sensor terminal."""
    return Node(
        nid="PawnCenterSensor",
        ntype=NodeType.TERMINAL,
        predicate=pawn_center_sensor_predicate,
        meta={"sensor_type": "pawn_center", "fan_in_allowed": True},
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
    knights_sensor = create_knights_before_bishops_sensor()
    fianchetto_sensor = create_fianchetto_sensor()
    pawn_center_sensor = create_pawn_center_sensor()
    
    g.add_node(dev_sensor)
    g.add_node(castle_sensor)
    g.add_node(center_sensor)
    g.add_node(knights_sensor)
    g.add_node(fianchetto_sensor)
    g.add_node(pawn_center_sensor)
    
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
    fianchetto = Node(
        nid="Fianchetto",
        ntype=NodeType.SCRIPT,
        meta={"layer": "strategic", "category": "OPENING", "alt": True},
    )
    
    g.add_node(develop)
    g.add_node(castle)
    g.add_node(center)
    g.add_node(fianchetto)
    
    # Wire hierarchy: OpeningPhase → plans
    g.add_edge("OpeningPhase", "DevelopMinorPieces", LinkType.SUB)
    g.add_edge("OpeningPhase", "CastleEarly", LinkType.SUB)
    g.add_edge("OpeningPhase", "ControlCenter", LinkType.SUB)
    g.add_edge("OpeningPhase", "Fianchetto", LinkType.SUB)
    
    # Wire plans → sensors
    g.add_edge("DevelopMinorPieces", "DevelopmentSensor", LinkType.SUB)
    g.add_edge("DevelopMinorPieces", "CenterControlSensor", LinkType.SUB)  # Fan-in
    g.add_edge("DevelopMinorPieces", "KnightsBeforeBishopsSensor", LinkType.SUB)
    
    g.add_edge("CastleEarly", "CastlingSensor", LinkType.SUB)
    
    g.add_edge("ControlCenter", "CenterControlSensor", LinkType.SUB)  # Fan-in (shared)
    g.add_edge("ControlCenter", "PawnCenterSensor", LinkType.SUB)
    
    g.add_edge("Fianchetto", "FianchettoSensor", LinkType.SUB)
    g.add_edge("Fianchetto", "DevelopmentSensor", LinkType.SUB)  # Fan-in (shared)
    
    # Set confirmation policy - any plan succeeding is enough
    g.set_confirm_policy("OpeningPhase", policy="or")
    
    return "OpeningPhase"


def get_opening_move_candidates(board: chess.Board) -> List[Tuple[chess.Move, str, float]]:
    """
    Get candidate opening moves with reasons and scores.
    
    Returns list of (move, reason, score) tuples.
    
    Uses enhanced opening principles:
    - Knights before bishops
    - Central development
    - Castling priority
    - Fianchetto options
    """
    candidates = []
    side = board.turn
    
    # Count undeveloped pieces for knights-before-bishops logic
    if side:  # White
        knight_starts = [chess.B1, chess.G1]
        bishop_starts = [chess.C1, chess.F1]
        fianchetto_squares = [chess.G2, chess.B2]
        fianchetto_pawn_moves = [(chess.G2, chess.G3), (chess.B2, chess.B3)]
    else:  # Black
        knight_starts = [chess.B8, chess.G8]
        bishop_starts = [chess.C8, chess.F8]
        fianchetto_squares = [chess.G7, chess.B7]
        fianchetto_pawn_moves = [(chess.G7, chess.G6), (chess.B7, chess.B6)]
    
    undeveloped_knights = sum(
        1 for sq in knight_starts
        if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.KNIGHT
    )
    undeveloped_bishops = sum(
        1 for sq in bishop_starts
        if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.BISHOP
    )
    
    for move in board.legal_moves:
        reason = ""
        score = 0.0
        
        piece = board.piece_at(move.from_square)
        if piece is None:
            continue
        
        # Development moves
        if piece.piece_type == chess.KNIGHT:
            # Check if moving from back rank
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            
            is_developing = (
                (side and from_rank == 0 and to_rank > 0) or
                (not side and from_rank == 7 and to_rank < 7)
            )
            
            if is_developing:
                score += 0.6  # Knights get bonus over bishops
                reason = "Develops knight"
                
                # Extra bonus if following knights-before-bishops
                if undeveloped_knights > 0 and undeveloped_bishops > 0:
                    score += 0.15
                    reason += " (knights first!)"
            
            # Bonus for central development
            if move.to_square in [chess.C3, chess.F3, chess.C6, chess.F6,
                                  chess.E4, chess.D4, chess.E5, chess.D5]:
                score += 0.2
                reason += " to center"
        
        elif piece.piece_type == chess.BISHOP:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            
            is_developing = (
                (side and from_rank == 0 and to_rank > 0) or
                (not side and from_rank == 7 and to_rank < 7)
            )
            
            if is_developing:
                score += 0.45
                reason = "Develops bishop"
                
                # Penalty if knights not developed yet
                if undeveloped_knights > 0:
                    score -= 0.1
                    reason += " (knights first?)"
                
                # Check for fianchetto
                if move.to_square in fianchetto_squares:
                    score += 0.25
                    reason = "Fianchetto bishop"
            
            # Bonus for active squares
            if move.to_square in [chess.C4, chess.F4, chess.C5, chess.F5,
                                  chess.B5, chess.G5, chess.B4, chess.G4]:
                score += 0.15
                reason += " to active square"
        
        # Castling - high priority
        if board.is_castling(move):
            score += 0.85
            reason = "Castles"
            # Bonus for kingside (usually safer/faster)
            if move.to_square in [chess.G1, chess.G8]:
                score += 0.05
                reason += " kingside"
        
        # Central pawn moves
        if piece.piece_type == chess.PAWN:
            if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                score += 0.5
                reason = "Central pawn"
            elif move.to_square in [chess.C4, chess.C5]:
                score += 0.35
                reason = "Flank pawn (English/Sicilian)"
            
            # Fianchetto prep
            for pawn_from, pawn_to in fianchetto_pawn_moves:
                if move.from_square == pawn_from and move.to_square == pawn_to:
                    score += 0.3
                    reason = "Fianchetto prep"
        
        if score > 0:
            candidates.append((move, reason, score))
    
    return sorted(candidates, key=lambda x: x[2], reverse=True)

