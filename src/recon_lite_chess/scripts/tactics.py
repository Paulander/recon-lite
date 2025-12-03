"""M5.4: Tactical pattern detection and exploitation subgraph.

Handles:
- Fork detection and exploitation
- Pin detection and exploitation
- Hanging piece detection (capture/protect)
- Skewer detection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess

from recon_lite import Graph, LinkType, Node, NodeType, NodeState


# Configuration cache
_TACTICS_CFG = {
    "loaded": False,
    "weights": {
        "fork_priority": 0.9,
        "pin_priority": 0.8,
        "hanging_priority": 0.95,
        "skewer_priority": 0.7,
    }
}


def _load_tactics_cfg() -> Dict[str, float]:
    """Load tactics weights from SWP, fallback to defaults."""
    if _TACTICS_CFG["loaded"]:
        return _TACTICS_CFG["weights"]
    try:
        path = Path("weights/subgraphs/tactics_weight_pack.swp")
        if path.exists():
            data = json.loads(path.read_text())
            _TACTICS_CFG["weights"].update(data.get("priorities", {}))
    except Exception:
        pass
    _TACTICS_CFG["loaded"] = True
    return _TACTICS_CFG["weights"]


# ============================================================================
# Tactical Detection Functions
# ============================================================================

def detect_forks(board: chess.Board) -> List[Dict[str, Any]]:
    """
    Detect potential fork opportunities.
    
    Returns list of fork opportunities with attacking piece, targets, and best move.
    """
    forks = []
    turn = board.turn
    
    # Check each of our pieces that could create a fork
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != turn:
            continue
        
        # Get legal moves for this piece
        for move in board.legal_moves:
            if move.from_square != sq:
                continue
            
            # Simulate the move
            board.push(move)
            
            # Check if the piece now attacks multiple valuable targets
            new_sq = move.to_square
            attacks = board.attacks(new_sq)
            valuable_targets = []
            
            for target_sq in attacks:
                target = board.piece_at(target_sq)
                if target and target.color != turn:
                    if target.piece_type in (chess.QUEEN, chess.ROOK, chess.KING):
                        valuable_targets.append({
                            "square": chess.square_name(target_sq),
                            "piece": target.symbol(),
                        })
            
            board.pop()
            
            if len(valuable_targets) >= 2:
                forks.append({
                    "move": move.uci(),
                    "attacker": piece.symbol(),
                    "from": chess.square_name(sq),
                    "to": chess.square_name(move.to_square),
                    "targets": valuable_targets,
                })
    
    return forks


def detect_pins(board: chess.Board) -> List[Dict[str, Any]]:
    """
    Detect pinned pieces.
    
    Returns list of pins with pinned piece, pinner, and pinned-to piece.
    """
    pins = []
    turn = board.turn
    enemy = not turn
    
    # Check enemy pieces that are pinned
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != enemy:
            continue
        if piece.piece_type == chess.KING:
            continue
        
        if board.is_pinned(enemy, sq):
            # Find what it's pinned to (should be king)
            pin_mask = board.pin(enemy, sq)
            pins.append({
                "pinned_square": chess.square_name(sq),
                "pinned_piece": piece.symbol(),
                "pin_line": bin(pin_mask),
            })
    
    return pins


def detect_hanging_pieces(board: chess.Board) -> Dict[str, List[str]]:
    """
    Detect hanging (undefended but attacked) pieces.
    
    Returns dict with our_hanging and enemy_hanging lists.
    """
    turn = board.turn
    our_hanging = []
    enemy_hanging = []
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.piece_type == chess.KING:
            continue
        
        color = piece.color
        defenders = board.attackers(color, sq)
        attackers = board.attackers(not color, sq)
        
        is_attacked = len(attackers) > 0
        is_defended = len(defenders) > 0
        
        if is_attacked and not is_defended:
            sq_name = chess.square_name(sq)
            if color == turn:
                our_hanging.append(sq_name)
            else:
                enemy_hanging.append(sq_name)
    
    return {
        "our_hanging": our_hanging,
        "enemy_hanging": enemy_hanging,
    }


def detect_skewers(board: chess.Board) -> List[Dict[str, Any]]:
    """
    Detect potential skewer opportunities (attack through a piece to another).
    
    Returns list of skewer opportunities.
    """
    skewers = []
    turn = board.turn
    
    # Check our sliding pieces (rooks, bishops, queens)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != turn:
            continue
        if piece.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            continue
        
        # Get attack rays
        attacks = board.attacks(sq)
        
        for target_sq in attacks:
            target = board.piece_at(target_sq)
            if not target or target.color == turn:
                continue
            
            # Check if there's a more valuable piece behind
            # (simplified - just check if king is behind)
            direction = target_sq - sq
            if direction == 0:
                continue
            
            # This is a simplified skewer detection
            # A full implementation would trace the ray
            if target.piece_type in (chess.QUEEN, chess.ROOK):
                skewers.append({
                    "attacker": chess.square_name(sq),
                    "front_piece": chess.square_name(target_sq),
                })
    
    return skewers


# ============================================================================
# Tactical Move Generators
# ============================================================================

def get_fork_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that create forks."""
    forks = detect_forks(board)
    return [chess.Move.from_uci(f["move"]) for f in forks]


def get_capture_hanging_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that capture hanging pieces."""
    hanging = detect_hanging_pieces(board)
    moves = []
    
    for sq_name in hanging["enemy_hanging"]:
        sq = chess.parse_square(sq_name)
        # Find moves that capture this square
        for move in board.legal_moves:
            if move.to_square == sq:
                moves.append(move)
    
    return moves


def get_protect_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that protect our hanging pieces."""
    hanging = detect_hanging_pieces(board)
    moves = []
    
    for sq_name in hanging["our_hanging"]:
        sq = chess.parse_square(sq_name)
        
        # Find moves that defend this square
        for move in board.legal_moves:
            # Check if this move adds a defender
            board.push(move)
            defenders = board.attackers(board.turn, sq)  # Note: turn flipped
            board.pop()
            
            # If we now have defenders, this is a protecting move
            if len(defenders) > 0:
                moves.append(move)
                break  # One protection move per hanging piece
    
    return moves


# ============================================================================
# Node Factories
# ============================================================================

def create_fork_detector(nid: str) -> Node:
    """Create a sensor node that detects fork opportunities."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        forks = detect_forks(board)
        node.meta["forks"] = forks
        env.setdefault("tactics", {})["forks"] = forks
        
        ok = len(forks) > 0
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_pin_detector(nid: str) -> Node:
    """Create a sensor node that detects pins."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        pins = detect_pins(board)
        node.meta["pins"] = pins
        env.setdefault("tactics", {})["pins"] = pins
        
        ok = len(pins) > 0
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_hanging_detector(nid: str) -> Node:
    """Create a sensor node that detects hanging pieces."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        hanging = detect_hanging_pieces(board)
        node.meta["hanging"] = hanging
        env.setdefault("tactics", {})["hanging"] = hanging
        
        # Alert if enemy has hanging pieces (opportunity) or we have (danger)
        has_opportunity = len(hanging["enemy_hanging"]) > 0
        has_danger = len(hanging["our_hanging"]) > 0
        
        ok = has_opportunity or has_danger
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_exploit_fork(nid: str) -> Node:
    """Create an actuator node that proposes fork moves."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_fork_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "fork",
                "rank": cfg["fork_priority"],
            } for m in moves[:3]]  # Top 3 forks
            
            env.setdefault("tactics_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_capture_hanging(nid: str) -> Node:
    """Create an actuator node that captures hanging pieces."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_capture_hanging_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "capture_hanging",
                "rank": cfg["hanging_priority"],
            } for m in moves[:3]]
            
            env.setdefault("tactics_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_protect_hanging(nid: str) -> Node:
    """Create an actuator node that protects our hanging pieces."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_protect_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "protect_hanging",
                "rank": cfg["hanging_priority"] * 0.9,  # Slightly lower than capture
            } for m in moves[:3]]
            
            env.setdefault("tactics_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


# ============================================================================
# Graph Builder
# ============================================================================

def build_tactics_network() -> Graph:
    """
    Build the tactical pattern detection and exploitation subgraph.
    
    Structure:
        tactics_root
        ├── detect_fork → exploit_fork
        ├── detect_pin → (exploit_pin - future)
        ├── detect_hanging → capture_hanging
        │                  → protect_hanging
        └── detect_skewer → (exploit_skewer - future)
    """
    g = Graph()
    
    # Root node
    root = Node("tactics_root", NodeType.SCRIPT)
    g.add_node(root)
    
    # Fork detection and exploitation - both are SCRIPT nodes for POR
    fork_sensor = Node("detect_fork", NodeType.SCRIPT)
    fork_exploit = Node("exploit_fork", NodeType.SCRIPT)
    g.add_node(fork_sensor)
    g.add_node(fork_exploit)
    g.add_edge("tactics_root", "detect_fork", LinkType.SUB)
    g.add_edge("detect_fork", "exploit_fork", LinkType.POR)
    
    # Pin detection
    pin_sensor = Node("detect_pin", NodeType.SCRIPT)
    g.add_node(pin_sensor)
    g.add_edge("tactics_root", "detect_pin", LinkType.SUB)
    
    # Hanging piece detection and handling - all SCRIPT nodes for POR
    hanging_sensor = Node("detect_hanging", NodeType.SCRIPT)
    capture_hanging_node = Node("capture_hanging", NodeType.SCRIPT)
    protect_hanging_node = Node("protect_hanging", NodeType.SCRIPT)
    g.add_node(hanging_sensor)
    g.add_node(capture_hanging_node)
    g.add_node(protect_hanging_node)
    g.add_edge("tactics_root", "detect_hanging", LinkType.SUB)
    g.add_edge("detect_hanging", "capture_hanging", LinkType.POR)
    g.add_edge("detect_hanging", "protect_hanging", LinkType.POR)
    
    return g


# ============================================================================
# Weight Pack
# ============================================================================

def create_default_tactics_weight_pack() -> Dict[str, Any]:
    """Create the default weight pack for tactics subgraph."""
    return {
        "version": "1.0",
        "subgraph": "tactics",
        "priorities": {
            "fork_priority": 0.9,
            "pin_priority": 0.8,
            "hanging_priority": 0.95,
            "skewer_priority": 0.7,
        },
        "edges": {
            "tactics_root->detect_fork:SUB": 1.0,
            "detect_fork->exploit_fork:POR": 1.0,
            "tactics_root->detect_pin:SUB": 1.0,
            "tactics_root->detect_hanging:SUB": 1.0,
            "detect_hanging->capture_hanging:POR": 1.0,
            "detect_hanging->protect_hanging:POR": 0.8,
        },
    }

