"""
KQK (King+Queen vs King) Endgame Network.

This module provides a complete KQK checkmate network that can be used
for training and gameplay. Queen checkmate follows a simpler pattern than
KRK because the queen controls more squares.

Queen Mate Pattern:
1. Push the enemy king toward an edge (queen restricts, king approaches)
2. Create a mating net (queen controls escape squares)
3. Deliver checkmate (typically in a corner or edge)

Common mate patterns:
- Back rank mate: Queen delivers check while king blocks escape
- Corner mate: Queen + King corner the enemy king
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import chess

from recon_lite import Graph, LinkType, Node, NodeType

# Import shared stalemate detector
from recon_lite_chess.scripts.stalemate_detector import (
    create_stalemate_danger_sensor,
    create_stalemate_gate,
    create_wait_move_selector,
    analyze_stalemate_danger,
    StalemateDangerLevel,
)


# ============================================================================
# Position Generation
# ============================================================================

def create_random_kqk_board(white_to_move: bool = True) -> str:
    """
    Create a random valid KQK (King+Queen vs King) position.
    
    Args:
        white_to_move: If True, White has K+Q, else Black has K+Q
        
    Returns:
        FEN string for a valid KQK position
    """
    while True:
        # Place the defending king
        defender_king_sq = random.randint(0, 63)
        
        # Place the attacking king (must be > 1 square away)
        attacker_king_sq = random.randint(0, 63)
        if attacker_king_sq == defender_king_sq:
            continue
        
        # Check king distance
        dk_file = chess.square_file(defender_king_sq)
        dk_rank = chess.square_rank(defender_king_sq)
        ak_file = chess.square_file(attacker_king_sq)
        ak_rank = chess.square_rank(attacker_king_sq)
        
        if abs(dk_file - ak_file) <= 1 and abs(dk_rank - ak_rank) <= 1:
            continue  # Kings too close
        
        # Place the queen (not on king squares, not immediately capturing defender)
        queen_sq = random.randint(0, 63)
        if queen_sq in (defender_king_sq, attacker_king_sq):
            continue
        
        # Build the board
        board = chess.Board.empty()
        
        if white_to_move:
            board.set_piece_at(attacker_king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(defender_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE
        else:
            board.set_piece_at(attacker_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.BLACK))
            board.set_piece_at(defender_king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.turn = chess.BLACK
        
        # Validate position
        if not board.is_valid():
            continue
        
        # Skip if defender is already in checkmate or stalemate
        if board.is_game_over():
            continue
        
        # Skip if queen is immediately capturable
        if board.is_attacked_by(not board.turn, queen_sq):
            continue
        
        return board.fen()


def is_kqk_position(board: chess.Board) -> Tuple[bool, Optional[chess.Color]]:
    """
    Check if board is a KQK endgame.
    
    Returns:
        (is_kqk, attacker_color) - attacker_color is None if not KQK
    """
    pieces = list(board.piece_map().values())
    
    # Must have exactly 3 pieces
    if len(pieces) != 3:
        return False, None
    
    white_pieces = [p for p in pieces if p.color == chess.WHITE]
    black_pieces = [p for p in pieces if p.color == chess.BLACK]
    
    # Check for KQ vs K
    def has_king_and_queen(pieces):
        types = [p.piece_type for p in pieces]
        return chess.KING in types and chess.QUEEN in types and len(types) == 2
    
    def has_lone_king(pieces):
        return len(pieces) == 1 and pieces[0].piece_type == chess.KING
    
    if has_king_and_queen(white_pieces) and has_lone_king(black_pieces):
        return True, chess.WHITE
    if has_king_and_queen(black_pieces) and has_lone_king(white_pieces):
        return True, chess.BLACK
    
    return False, None


# ============================================================================
# Detection Functions
# ============================================================================

def get_queen_square(board: chess.Board, color: chess.Color) -> Optional[chess.Square]:
    """Get the queen's square for the given color."""
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == chess.QUEEN:
            return sq
    return None


def defender_on_edge(board: chess.Board, defender_color: chess.Color) -> bool:
    """Check if the defender's king is on an edge."""
    king_sq = board.king(defender_color)
    if king_sq is None:
        return False
    
    file = chess.square_file(king_sq)
    rank = chess.square_rank(king_sq)
    
    return file in (0, 7) or rank in (0, 7)


def defender_in_corner(board: chess.Board, defender_color: chess.Color) -> bool:
    """Check if the defender's king is in a corner."""
    king_sq = board.king(defender_color)
    if king_sq is None:
        return False
    
    return king_sq in [chess.A1, chess.A8, chess.H1, chess.H8]


def queen_restricts_king(board: chess.Board, attacker_color: chess.Color) -> float:
    """
    Calculate how well the queen restricts the enemy king.
    Returns a score 0-1 based on how many escape squares are cut off.
    """
    defender_color = not attacker_color
    defender_king = board.king(defender_color)
    queen_sq = get_queen_square(board, attacker_color)
    
    if defender_king is None or queen_sq is None:
        return 0.0
    
    # Count escape squares
    dk_file = chess.square_file(defender_king)
    dk_rank = chess.square_rank(defender_king)
    
    total_adjacent = 0
    attacked_adjacent = 0
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            nf, nr = dk_file + df, dk_rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                sq = chess.square(nf, nr)
                total_adjacent += 1
                if board.is_attacked_by(attacker_color, sq):
                    attacked_adjacent += 1
    
    return attacked_adjacent / max(1, total_adjacent)


def king_distance(board: chess.Board) -> int:
    """Get Chebyshev distance between the two kings."""
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king is None or black_king is None:
        return 8
    
    wf, wr = chess.square_file(white_king), chess.square_rank(white_king)
    bf, br = chess.square_file(black_king), chess.square_rank(black_king)
    
    return max(abs(wf - bf), abs(wr - br))


def can_deliver_queen_mate(board: chess.Board, attacker_color: chess.Color) -> List[chess.Move]:
    """
    Check if queen can deliver checkmate in 1 move.
    Returns list of mating moves.
    """
    mates = []
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.QUEEN:
            board.push(move)
            if board.is_checkmate():
                mates.append(move)
            board.pop()
    return mates


def can_approach_for_mate(board: chess.Board, attacker_color: chess.Color) -> List[chess.Move]:
    """
    Get king moves that help set up mate (approaching to support queen).
    Also includes "waiting" king moves when distance can't be reduced.
    """
    defender_color = not attacker_color
    defender_king = board.king(defender_color)
    attacker_king = board.king(attacker_color)
    
    if defender_king is None or attacker_king is None:
        return []
    
    approach_moves = []
    waiting_moves = []
    current_dist = king_distance(board)
    
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING and piece.color == attacker_color:
            # Check if this move safely approaches
            board.push(move)
            
            # Skip if causes stalemate
            if board.is_stalemate():
                board.pop()
                continue
            
            new_dist = king_distance(board)
            
            # Prefer moves that reduce distance to enemy king
            if new_dist < current_dist:
                approach_moves.append((move, current_dist - new_dist + 10))  # High priority
            elif new_dist == current_dist:
                # Waiting move - maintains distance, useful for triangulation
                waiting_moves.append((move, 1))
            
            board.pop()
    
    # Combine: approach moves first, then waiting moves
    approach_moves.sort(key=lambda x: -x[1])
    all_moves = approach_moves + waiting_moves
    return [m for m, _ in all_moves]


def get_restriction_moves(board: chess.Board, attacker_color: chess.Color) -> List[chess.Move]:
    """
    Get queen moves that restrict the enemy king further.
    Avoids moves that would cause stalemate.
    """
    defender_color = not attacker_color
    defender_king = board.king(defender_color)
    queen_sq = get_queen_square(board, attacker_color)
    
    if defender_king is None or queen_sq is None:
        return []
    
    restriction_moves = []
    current_restriction = queen_restricts_king(board, attacker_color)
    
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.QUEEN:
            board.push(move)
            
            # Skip if queen hangs
            new_queen_sq = move.to_square
            if board.is_attacked_by(not attacker_color, new_queen_sq):
                board.pop()
                continue
            
            # CRITICAL: Skip if this causes stalemate!
            if board.is_stalemate():
                board.pop()
                continue
            
            new_restriction = queen_restricts_king(board, attacker_color)
            
            # Prefer restricting but not TOO much (leave 1-2 escape squares)
            # Full restriction (1.0) often leads to stalemate
            if new_restriction > current_restriction and new_restriction < 1.0:
                # Good restriction without over-restricting
                restriction_moves.append((move, new_restriction + 0.1))
            elif new_restriction > current_restriction:
                # Still better, but less preferred due to stalemate risk
                restriction_moves.append((move, new_restriction))
            
            board.pop()
    
    # Sort by restriction value
    restriction_moves.sort(key=lambda x: -x[1])
    return [m for m, _ in restriction_moves]


# ============================================================================
# Node Factory Functions
# ============================================================================

def create_kqk_material_detector(nid: str) -> Node:
    """Detector: Is this a KQK position?"""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        is_kqk, attacker = is_kqk_position(board)
        
        node.meta["is_kqk"] = is_kqk
        node.meta["attacker"] = attacker
        env.setdefault("kqk", {})["is_kqk"] = is_kqk
        env.setdefault("kqk", {})["attacker"] = attacker
        
        return is_kqk, is_kqk
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_edge_detector(nid: str) -> Node:
    """Detector: Is defender king on edge?"""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None:
            return False, False
        
        defender = not attacker
        on_edge = defender_on_edge(board, defender)
        
        node.meta["on_edge"] = on_edge
        env.setdefault("kqk", {})["defender_on_edge"] = on_edge
        
        return on_edge, on_edge
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_corner_detector(nid: str) -> Node:
    """Detector: Is defender king in corner?"""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None:
            return False, False
        
        defender = not attacker
        in_corner = defender_in_corner(board, defender)
        
        node.meta["in_corner"] = in_corner
        env.setdefault("kqk", {})["defender_in_corner"] = in_corner
        
        return in_corner, in_corner
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_mate_detector(nid: str) -> Node:
    """Detector: Can we deliver mate in 1?"""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None or board.turn != attacker:
            return False, False
        
        mates = can_deliver_queen_mate(board, attacker)
        
        node.meta["mate_moves"] = [m.uci() for m in mates]
        env.setdefault("kqk", {})["mate_moves"] = mates
        
        can_mate = len(mates) > 0
        return can_mate, can_mate
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_restriction_evaluator(nid: str) -> Node:
    """Evaluator: How well does the queen restrict the king?"""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None:
            return False, False
        
        restriction = queen_restricts_king(board, attacker)
        
        node.meta["restriction"] = restriction
        env.setdefault("kqk", {})["restriction"] = restriction
        
        # Good restriction is > 0.5
        ok = restriction > 0.5
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_drive_moves(nid: str) -> Node:
    """Actuator: Generate queen moves that restrict king."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None or board.turn != attacker:
            return True, False
        
        moves = get_restriction_moves(board, attacker)
        
        if moves:
            suggested = moves[0].uci()
            node.meta["suggested_move"] = suggested
            env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
            return True, True
        
        return True, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_approach_moves(nid: str) -> Node:
    """Actuator: Generate king moves to approach for mate."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None or board.turn != attacker:
            return True, False
        
        moves = can_approach_for_mate(board, attacker)
        
        if moves:
            suggested = moves[0].uci()
            node.meta["suggested_move"] = suggested
            env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
            return True, True
        
        return True, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_mate_moves(nid: str) -> Node:
    """Actuator: Generate checkmate moves."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        mates = env.get("kqk", {}).get("mate_moves", [])
        
        if mates:
            suggested = mates[0].uci()
            node.meta["suggested_move"] = suggested
            env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
            return True, True
        
        return True, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def get_waiting_queen_moves(board: chess.Board, attacker_color: chess.Color) -> List[chess.Move]:
    """
    Get queen moves that maintain restriction while being safe.
    These are "waiting" moves that force the opponent to move into a worse position.
    """
    defender_color = not attacker_color
    queen_sq = get_queen_square(board, attacker_color)
    
    if queen_sq is None:
        return []
    
    waiting_moves = []
    current_restriction = queen_restricts_king(board, attacker_color)
    
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.QUEEN:
            board.push(move)
            
            # Must be safe
            new_queen_sq = move.to_square
            if board.is_attacked_by(not attacker_color, new_queen_sq):
                board.pop()
                continue
            
            # Must not cause stalemate
            if board.is_stalemate():
                board.pop()
                continue
            
            new_restriction = queen_restricts_king(board, attacker_color)
            
            # Accept moves that maintain or improve restriction (but not stalemate)
            if new_restriction >= current_restriction * 0.8:  # Allow slight decrease
                waiting_moves.append((move, new_restriction))
            
            board.pop()
    
    waiting_moves.sort(key=lambda x: -x[1])
    return [m for m, _ in waiting_moves]


def create_kqk_move_selector(nid: str) -> Node:
    """Main move selector that combines all strategies with stalemate awareness."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        attacker = env.get("kqk", {}).get("attacker")
        
        if attacker is None or board.turn != attacker:
            return True, False
        
        # Get stalemate analysis from shared sensor (if available)
        stalemate_analysis = env.get("stalemate_analysis")
        if stalemate_analysis is None:
            stalemate_analysis = analyze_stalemate_danger(board)
            env["stalemate_analysis"] = stalemate_analysis
        
        danger = stalemate_analysis.danger_score
        danger_level = stalemate_analysis.danger_level
        # Only honor prefer_wait when danger is truly CRITICAL
        prefer_wait = env.get("prefer_wait", False) and danger_level == StalemateDangerLevel.CRITICAL
        node.meta["stalemate_danger"] = danger
        node.meta["danger_level"] = danger_level.value
        node.meta["prefer_wait"] = prefer_wait
        
        # Priority 1: ALWAYS allow checkmate
        mates = can_deliver_queen_mate(board, attacker)
        if mates:
            suggested = mates[0].uci()
            node.meta["suggested_move"] = suggested
            node.meta["move_type"] = "mate"
            env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
            return True, True
        
        # If gate requested waiting, honor it before any aggressive move
        if prefer_wait:
            waiting_from_gate = env.get("waiting_moves", {}).get("suggested_move")
            if waiting_from_gate:
                node.meta["suggested_move"] = waiting_from_gate
                node.meta["move_type"] = "wait_gate"
                env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = waiting_from_gate
                return True, True
            # If gate fired but no precomputed move, fall through to waiting search
        
        # If stalemate danger is CRITICAL, only mate is allowed - skip to waiting
        if danger_level == StalemateDangerLevel.CRITICAL:
            node.meta["skip_reason"] = "critical_danger"
            # Fall through to waiting moves
        else:
            # Priority 2: Restrict king more (with stalemate protection)
            # Skip if danger is HIGH or if gate asked to wait
            if not prefer_wait and danger_level not in (StalemateDangerLevel.HIGH, StalemateDangerLevel.CRITICAL):
                restriction_moves = get_restriction_moves(board, attacker)
                if restriction_moves:
                    suggested = restriction_moves[0].uci()
                    node.meta["suggested_move"] = suggested
                    node.meta["move_type"] = "restrict"
                    env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
                    return True, True
            
            # Priority 3: Approach with king (safer, always OK except CRITICAL)
            if not prefer_wait and danger_level != StalemateDangerLevel.CRITICAL:
                approach_moves = can_approach_for_mate(board, attacker)
                if approach_moves:
                    suggested = approach_moves[0].uci()
                    node.meta["suggested_move"] = suggested
                    node.meta["move_type"] = "approach"
                    env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
                    return True, True
        
        # Priority 4: Waiting queen move (safest, triangulate)
        waiting_moves = get_waiting_queen_moves(board, attacker)
        if waiting_moves:
            suggested = waiting_moves[0].uci()
            node.meta["suggested_move"] = suggested
            node.meta["move_type"] = "wait"
            env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
            return True, True
        
        # Fallback: any safe move that doesn't cause stalemate
        for move in board.legal_moves:
            board.push(move)
            queen_sq = get_queen_square(board, attacker)
            safe = queen_sq is None or not board.is_attacked_by(not attacker, queen_sq)
            stalemate = board.is_stalemate()
            board.pop()
            
            if safe and not stalemate:
                suggested = move.uci()
                node.meta["suggested_move"] = suggested
                node.meta["move_type"] = "safe"
                env.setdefault("kqk", {}).setdefault("policy", {})["suggested_move"] = suggested
                return True, True
        
        return True, False
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kqk_wait(nid: str) -> Node:
    """Wait node for board change."""
    from recon_lite_chess import create_wait_for_board_change
    return create_wait_for_board_change(nid)


# ============================================================================
# Network Builder
# ============================================================================

def build_kqk_network() -> Graph:
    """
    Build the complete KQK checkmate ReCoN network.
    
    Structure:
    - kqk_root
      - kqk_phase1_drive (push king to edge)
        - kqk_edge_detector
        - kqk_restriction_eval
        - kqk_drive_moves
      - kqk_phase2_corner (push king to corner)
        - kqk_corner_detector
        - kqk_approach_moves
      - kqk_phase3_mate (deliver checkmate)
        - kqk_mate_detector
        - kqk_mate_moves
      - kqk_move_selector (fallback)
      - kqk_wait
    
    Returns:
        Graph: Complete KQK network
    """
    g = Graph()
    
    # === ROOT ===
    kqk_root = Node("kqk_root", NodeType.SCRIPT, meta={
        "layer": "endgame",
        "subgraph": "kqk",
    })
    g.add_node(kqk_root)
    
    # === MATERIAL DETECTOR ===
    g.add_node(create_kqk_material_detector("kqk_material_check"))
    
    # === PHASE 1: Drive king to edge ===
    phase1 = Node("kqk_phase1_drive", NodeType.SCRIPT, meta={
        "layer": "endgame_phase",
        "subgraph": "kqk",
    })
    g.add_node(phase1)
    g.add_node(create_kqk_edge_detector("kqk_edge_detector"))
    g.add_node(create_kqk_restriction_evaluator("kqk_restriction_eval"))
    g.add_node(create_kqk_drive_moves("kqk_drive_moves"))
    
    # === PHASE 2: Push to corner ===
    phase2 = Node("kqk_phase2_corner", NodeType.SCRIPT, meta={
        "layer": "endgame_phase",
        "subgraph": "kqk",
    })
    g.add_node(phase2)
    g.add_node(create_kqk_corner_detector("kqk_corner_detector"))
    g.add_node(create_kqk_approach_moves("kqk_approach_moves"))
    
    # === PHASE 3: Deliver mate ===
    phase3 = Node("kqk_phase3_mate", NodeType.SCRIPT, meta={
        "layer": "endgame_phase",
        "subgraph": "kqk",
    })
    g.add_node(phase3)
    g.add_node(create_kqk_mate_detector("kqk_mate_detector"))
    g.add_node(create_kqk_mate_moves("kqk_mate_moves"))
    
    # === STALEMATE DETECTOR (shared sensor) ===
    g.add_node(create_stalemate_danger_sensor("kqk_stalemate_sensor"))
    # Gate should only block at CRITICAL stalemate danger (mobility <= 1)
    g.add_node(create_stalemate_gate("kqk_stalemate_gate", danger_threshold=0.99))
    
    # === MOVE SELECTOR (fallback) ===
    g.add_node(create_kqk_move_selector("kqk_move_selector"))
    
    # === WAIT MOVE SELECTOR (when stalemate danger high) ===
    g.add_node(create_wait_move_selector("kqk_safe_wait_selector"))
    
    # === WAIT (script wrapper + terminal) ===
    kqk_wait_script = Node("kqk_wait", NodeType.SCRIPT, meta={
        "layer": "endgame_phase",
        "subgraph": "kqk",
    })
    g.add_node(kqk_wait_script)
    g.add_node(create_kqk_wait("kqk_wait_for_change"))  # The actual terminal
    
    # === WIRING ===
    # Root hierarchy
    g.add_edge("kqk_root", "kqk_material_check", LinkType.SUB)
    g.add_edge("kqk_root", "kqk_stalemate_sensor", LinkType.SUB)  # Runs first to set env
    g.add_edge("kqk_root", "kqk_phase1_drive", LinkType.SUB)
    g.add_edge("kqk_root", "kqk_phase2_corner", LinkType.SUB)
    g.add_edge("kqk_root", "kqk_phase3_mate", LinkType.SUB)
    g.add_edge("kqk_root", "kqk_stalemate_gate", LinkType.SUB)  # Gates aggressive moves
    g.add_edge("kqk_root", "kqk_move_selector", LinkType.SUB)
    g.add_edge("kqk_root", "kqk_safe_wait_selector", LinkType.SUB)  # When gate fires
    g.add_edge("kqk_root", "kqk_wait", LinkType.SUB)
    
    # Wait script contains the wait terminal
    g.add_edge("kqk_wait", "kqk_wait_for_change", LinkType.SUB)
    
    # Phase 1 internals
    g.add_edge("kqk_phase1_drive", "kqk_edge_detector", LinkType.SUB)
    g.add_edge("kqk_phase1_drive", "kqk_restriction_eval", LinkType.SUB)
    g.add_edge("kqk_phase1_drive", "kqk_drive_moves", LinkType.SUB)
    
    # Phase 2 internals
    g.add_edge("kqk_phase2_corner", "kqk_corner_detector", LinkType.SUB)
    g.add_edge("kqk_phase2_corner", "kqk_approach_moves", LinkType.SUB)
    
    # Phase 3 internals
    g.add_edge("kqk_phase3_mate", "kqk_mate_detector", LinkType.SUB)
    g.add_edge("kqk_phase3_mate", "kqk_mate_moves", LinkType.SUB)
    
    # Phase sequencing (POR)
    g.add_edge("kqk_phase1_drive", "kqk_phase2_corner", LinkType.POR)
    g.add_edge("kqk_phase2_corner", "kqk_phase3_mate", LinkType.POR)
    g.add_edge("kqk_phase3_mate", "kqk_wait", LinkType.POR)
    
    return g


__all__ = [
    "build_kqk_network",
    "create_random_kqk_board",
    "is_kqk_position",
    "create_kqk_material_detector",
    "create_kqk_edge_detector",
    "create_kqk_corner_detector",
    "create_kqk_mate_detector",
    "create_kqk_restriction_evaluator",
    "create_kqk_drive_moves",
    "create_kqk_approach_moves",
    "create_kqk_mate_moves",
    "create_kqk_move_selector",
    # Re-export shared stalemate detector for convenience
    "analyze_stalemate_danger",
    "StalemateDangerLevel",
]

