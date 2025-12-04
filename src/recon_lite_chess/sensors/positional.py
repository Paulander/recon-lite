"""Positional Sensors for M8.

Implements strategic positional evaluation sensors:
- Bad bishop detection (bishop blocked by own pawns)
- Piece coordination (mutual defense, attacking same targets)
- Outpost detection (strong squares protected by pawns)
- Passed pawn detection and evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import chess

from recon_lite.graph import Node, NodeType


# ============================================================================
# Bad Bishop Detection
# ============================================================================

def is_light_square(sq: chess.Square) -> bool:
    """Check if a square is a light square."""
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def count_pawns_on_color(board: chess.Board, color: bool, light_squares: bool) -> int:
    """Count pawns of a color on light or dark squares."""
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        sq_is_light = is_light_square(sq)
        if sq_is_light == light_squares:
            count += 1
    return count


def detect_bad_bishop(board: chess.Board, color: bool) -> Dict[str, Any]:
    """
    Detect if a side has a bad bishop.
    
    A bishop is "bad" if many of our pawns are on the same color squares,
    blocking the bishop's activity.
    
    Returns:
        - has_bad_bishop: bool
        - bishop_sq: square of the bad bishop (if any)
        - pawns_blocking: number of own pawns on same color complex
        - severity: 0.0-1.0 indicating how bad the bishop is
    """
    bishops = list(board.pieces(chess.BISHOP, color))
    
    if not bishops:
        return {"has_bad_bishop": False, "reason": "no_bishop"}
    
    worst_bishop = None
    worst_severity = 0.0
    worst_blocking = 0
    
    for bishop_sq in bishops:
        bishop_is_light = is_light_square(bishop_sq)
        pawns_same_color = count_pawns_on_color(board, color, bishop_is_light)
        
        # Severity: 0 pawns = 0.0, 5+ pawns = 1.0
        severity = min(1.0, pawns_same_color / 5.0)
        
        # Also consider if bishop is on back rank (less active)
        back_rank = 0 if color == chess.WHITE else 7
        if chess.square_rank(bishop_sq) == back_rank:
            severity = min(1.0, severity + 0.1)
        
        # Check if bishop has any moves
        bishop_moves = len(list(board.attacks(bishop_sq)))
        if bishop_moves < 3:
            severity = min(1.0, severity + 0.2)
        
        if severity > worst_severity:
            worst_severity = severity
            worst_bishop = bishop_sq
            worst_blocking = pawns_same_color
    
    is_bad = worst_severity > 0.4
    
    return {
        "has_bad_bishop": is_bad,
        "bishop_sq": chess.square_name(worst_bishop) if worst_bishop else None,
        "pawns_blocking": worst_blocking,
        "severity": worst_severity,
    }


def bad_bishop_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Sensor predicate for detecting bad bishops.
    
    Stores both sides' bad bishop info in env.
    Returns True if enemy has a bad bishop (our advantage).
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    enemy = not side
    
    our_bad = detect_bad_bishop(board, side)
    enemy_bad = detect_bad_bishop(board, enemy)
    
    node.activation.meta["our_bad_bishop"] = our_bad
    node.activation.meta["enemy_bad_bishop"] = enemy_bad
    node.activation.value = enemy_bad.get("severity", 0.0) - our_bad.get("severity", 0.0)
    
    env["bad_bishop"] = {
        "our": our_bad,
        "enemy": enemy_bad,
    }
    
    # Success if enemy has worse bishop than us
    our_severity = our_bad.get("severity", 0.0)
    enemy_severity = enemy_bad.get("severity", 0.0)
    
    return True, enemy_severity > our_severity + 0.2


# ============================================================================
# Piece Coordination
# ============================================================================

def evaluate_piece_coordination(board: chess.Board, color: bool) -> Dict[str, Any]:
    """
    Evaluate how well pieces work together.
    
    Factors:
    - Mutual defense (pieces protecting each other)
    - Attacking same targets
    - Piece connectivity (rooks on same file/rank)
    
    Returns coordination score 0.0-1.0 and details.
    """
    pieces = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            pieces.append(sq)
    
    if len(pieces) < 2:
        return {"score": 0.0, "reason": "too_few_pieces"}
    
    # Count mutual defenses
    mutual_defenses = 0
    for sq in pieces:
        defenders = board.attackers(color, sq)
        # Count how many of our pieces defend this one
        for defender_sq in defenders:
            if defender_sq in pieces:
                mutual_defenses += 1
    
    # Count pieces attacking same enemy squares
    enemy_attacked = {}
    for sq in pieces:
        attacks = board.attacks(sq)
        for attacked_sq in attacks:
            target = board.piece_at(attacked_sq)
            if target and target.color != color:
                enemy_attacked[attacked_sq] = enemy_attacked.get(attacked_sq, 0) + 1
    
    # Pieces attacking same target multiple times
    coordinated_attacks = sum(1 for count in enemy_attacked.values() if count >= 2)
    
    # Rook connectivity
    rooks = list(board.pieces(chess.ROOK, color))
    rooks_connected = False
    if len(rooks) >= 2:
        r1, r2 = rooks[0], rooks[1]
        # Same file or rank
        if chess.square_file(r1) == chess.square_file(r2):
            rooks_connected = True
        elif chess.square_rank(r1) == chess.square_rank(r2):
            # Check if nothing between them
            min_f = min(chess.square_file(r1), chess.square_file(r2))
            max_f = max(chess.square_file(r1), chess.square_file(r2))
            rank = chess.square_rank(r1)
            clear = True
            for f in range(min_f + 1, max_f):
                if board.piece_at(chess.square(f, rank)):
                    clear = False
                    break
            rooks_connected = clear
    
    # Calculate score
    max_defenses = len(pieces) * (len(pieces) - 1)
    defense_score = mutual_defenses / max_defenses if max_defenses > 0 else 0
    
    attack_score = min(1.0, coordinated_attacks / 3.0)
    
    rook_bonus = 0.2 if rooks_connected else 0.0
    
    total_score = (defense_score * 0.4 + attack_score * 0.4 + rook_bonus)
    
    return {
        "score": total_score,
        "mutual_defenses": mutual_defenses,
        "coordinated_attacks": coordinated_attacks,
        "rooks_connected": rooks_connected,
    }


def piece_coordination_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Sensor predicate for piece coordination.
    
    Returns True if our coordination is significantly better.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    enemy = not side
    
    our_coord = evaluate_piece_coordination(board, side)
    enemy_coord = evaluate_piece_coordination(board, enemy)
    
    node.activation.meta["our_coordination"] = our_coord
    node.activation.meta["enemy_coordination"] = enemy_coord
    node.activation.value = our_coord.get("score", 0.0) - enemy_coord.get("score", 0.0)
    
    env["coordination"] = {
        "our": our_coord,
        "enemy": enemy_coord,
    }
    
    return True, our_coord.get("score", 0) > enemy_coord.get("score", 0) + 0.15


# ============================================================================
# Outpost Detection
# ============================================================================

def find_outposts(board: chess.Board, color: bool) -> List[Dict[str, Any]]:
    """
    Find strong outpost squares for a side.
    
    An outpost is a square that:
    - Can be protected by our pawn
    - Cannot be attacked by enemy pawns
    - Is in the enemy half of the board
    
    Returns list of outpost squares with occupancy info.
    """
    outposts = []
    enemy = not color
    
    # Determine which ranks are enemy territory
    enemy_ranks = range(4, 8) if color == chess.WHITE else range(0, 4)
    
    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)
        if rank not in enemy_ranks:
            continue
        
        file = chess.square_file(sq)
        
        # Check if can be attacked by enemy pawns
        can_be_attacked_by_enemy_pawn = False
        
        # Enemy pawns attack from files +/- 1
        for pawn_file in [file - 1, file + 1]:
            if not (0 <= pawn_file <= 7):
                continue
            
            # Check if there's an enemy pawn that could attack this square
            # Pawns advance, so check ranks behind (from enemy perspective)
            if color == chess.WHITE:
                # Enemy is black, black pawns go down (rank decreases)
                # So enemy pawn could be on ranks above this square
                check_ranks = range(rank + 1, 8)
            else:
                # Enemy is white, white pawns go up
                check_ranks = range(0, rank)
            
            for check_rank in check_ranks:
                check_sq = chess.square(pawn_file, check_rank)
                piece = board.piece_at(check_sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == enemy:
                    can_be_attacked_by_enemy_pawn = True
                    break
            
            if can_be_attacked_by_enemy_pawn:
                break
        
        if can_be_attacked_by_enemy_pawn:
            continue
        
        # Check if can be defended by our pawn
        can_be_defended = False
        for pawn_file in [file - 1, file + 1]:
            if not (0 <= pawn_file <= 7):
                continue
            
            if color == chess.WHITE:
                defend_rank = rank - 1
            else:
                defend_rank = rank + 1
            
            if 0 <= defend_rank <= 7:
                defend_sq = chess.square(pawn_file, defend_rank)
                piece = board.piece_at(defend_sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    can_be_defended = True
                    break
        
        if not can_be_defended:
            continue
        
        # This is an outpost!
        occupant = board.piece_at(sq)
        is_occupied = occupant and occupant.color == color
        
        # Knights are best on outposts
        is_knight_outpost = is_occupied and occupant.piece_type == chess.KNIGHT
        
        outposts.append({
            "square": chess.square_name(sq),
            "is_occupied": is_occupied,
            "occupant": occupant.symbol() if is_occupied else None,
            "is_knight_outpost": is_knight_outpost,
        })
    
    return outposts


def outpost_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Sensor predicate for outpost detection.
    
    Returns True if we have occupied outposts.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    
    outposts = find_outposts(board, side)
    
    occupied_outposts = [o for o in outposts if o["is_occupied"]]
    knight_outposts = [o for o in outposts if o["is_knight_outpost"]]
    
    node.activation.meta["outposts"] = outposts
    node.activation.meta["occupied_count"] = len(occupied_outposts)
    node.activation.meta["knight_outposts"] = len(knight_outposts)
    
    # Score: knight outposts worth more
    score = len(occupied_outposts) * 0.3 + len(knight_outposts) * 0.2
    node.activation.value = min(1.0, score)
    
    env["outposts"] = {
        "all": outposts,
        "occupied": occupied_outposts,
        "knight": knight_outposts,
    }
    
    return True, len(occupied_outposts) > 0


# ============================================================================
# Passed Pawn Detection
# ============================================================================

def find_passed_pawns(board: chess.Board, color: bool) -> List[Dict[str, Any]]:
    """
    Find passed pawns for a side.
    
    A passed pawn has no enemy pawns in front on the same or adjacent files.
    
    Returns list of passed pawns with advancement potential.
    """
    passed = []
    enemy = not color
    
    for pawn_sq in board.pieces(chess.PAWN, color):
        file = chess.square_file(pawn_sq)
        rank = chess.square_rank(pawn_sq)
        
        is_passed = True
        
        # Check files that could block (same and adjacent)
        for check_file in [file - 1, file, file + 1]:
            if not (0 <= check_file <= 7):
                continue
            
            # Check ranks ahead
            if color == chess.WHITE:
                ahead_ranks = range(rank + 1, 8)
            else:
                ahead_ranks = range(0, rank)
            
            for check_rank in ahead_ranks:
                check_sq = chess.square(check_file, check_rank)
                piece = board.piece_at(check_sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == enemy:
                    is_passed = False
                    break
            
            if not is_passed:
                break
        
        if not is_passed:
            continue
        
        # Calculate advancement potential
        if color == chess.WHITE:
            distance_to_promote = 7 - rank
        else:
            distance_to_promote = rank
        
        # Check if path is clear
        path_clear = True
        if color == chess.WHITE:
            for r in range(rank + 1, 8):
                if board.piece_at(chess.square(file, r)):
                    path_clear = False
                    break
        else:
            for r in range(0, rank):
                if board.piece_at(chess.square(file, r)):
                    path_clear = False
                    break
        
        # Check king support
        our_king = board.king(color)
        enemy_king = board.king(enemy)
        
        our_king_dist = chess.square_distance(our_king, pawn_sq) if our_king else 8
        enemy_king_dist = chess.square_distance(enemy_king, pawn_sq) if enemy_king else 8
        
        passed.append({
            "square": chess.square_name(pawn_sq),
            "rank": rank,
            "distance_to_promote": distance_to_promote,
            "path_clear": path_clear,
            "our_king_distance": our_king_dist,
            "enemy_king_distance": enemy_king_dist,
            "supported": our_king_dist < enemy_king_dist,
        })
    
    return passed


def passed_pawn_sensor_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Sensor predicate for passed pawn detection.
    
    Returns True if we have passed pawns.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side = board.turn
    enemy = not side
    
    our_passed = find_passed_pawns(board, side)
    enemy_passed = find_passed_pawns(board, enemy)
    
    node.activation.meta["our_passed_pawns"] = our_passed
    node.activation.meta["enemy_passed_pawns"] = enemy_passed
    
    # Score based on how advanced and supported they are
    def score_passed(pawns):
        total = 0.0
        for p in pawns:
            # Closer to promotion = higher score
            dist = p["distance_to_promote"]
            base = (7 - dist) / 7.0
            if p["path_clear"]:
                base *= 1.5
            if p["supported"]:
                base *= 1.3
            total += base
        return total
    
    our_score = score_passed(our_passed)
    enemy_score = score_passed(enemy_passed)
    
    node.activation.value = our_score - enemy_score
    
    env["passed_pawns"] = {
        "our": our_passed,
        "enemy": enemy_passed,
        "our_score": our_score,
        "enemy_score": enemy_score,
    }
    
    return True, len(our_passed) > 0


# ============================================================================
# Node Factories
# ============================================================================

def create_bad_bishop_sensor() -> Node:
    """Create a bad bishop sensor terminal."""
    return Node(
        nid="BadBishopSensor",
        ntype=NodeType.TERMINAL,
        predicate=bad_bishop_sensor_predicate,
        meta={"sensor_type": "bad_bishop", "fan_in_allowed": True},
    )


def create_piece_coordination_sensor() -> Node:
    """Create a piece coordination sensor terminal."""
    return Node(
        nid="PieceCoordinationSensor",
        ntype=NodeType.TERMINAL,
        predicate=piece_coordination_sensor_predicate,
        meta={"sensor_type": "coordination", "fan_in_allowed": True},
    )


def create_outpost_sensor() -> Node:
    """Create an outpost sensor terminal."""
    return Node(
        nid="OutpostSensor",
        ntype=NodeType.TERMINAL,
        predicate=outpost_sensor_predicate,
        meta={"sensor_type": "outpost", "fan_in_allowed": True},
    )


def create_passed_pawn_sensor() -> Node:
    """Create a passed pawn sensor terminal."""
    return Node(
        nid="PassedPawnSensor",
        ntype=NodeType.TERMINAL,
        predicate=passed_pawn_sensor_predicate,
        meta={"sensor_type": "passed_pawn", "fan_in_allowed": True},
    )

