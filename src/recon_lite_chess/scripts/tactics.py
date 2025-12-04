"""M5.4: Tactical pattern detection and exploitation subgraph.

Handles:
- Fork detection and exploitation
- Pin detection and exploitation
- Hanging piece detection (capture/protect)
- Skewer detection and exploitation
- Back rank weakness detection and exploitation
- Discovered attack detection and exploitation
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
        "back_rank_priority": 0.98,  # Back rank mates are very high priority
        "discovered_attack_priority": 0.85,
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
    
    Returns list of skewer opportunities with moves that create them.
    """
    skewers = []
    turn = board.turn
    enemy = not turn
    
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
            
            # Check if there's a piece behind on the same ray
            # Trace the ray from attacker through target
            file_diff = chess.square_file(target_sq) - chess.square_file(sq)
            rank_diff = chess.square_rank(target_sq) - chess.square_rank(sq)
            
            # Normalize to direction
            if file_diff != 0:
                file_diff = file_diff // abs(file_diff)
            if rank_diff != 0:
                rank_diff = rank_diff // abs(rank_diff)
            
            # Trace ray beyond target
            behind_sq = target_sq
            behind_piece = None
            for _ in range(7):
                new_file = chess.square_file(behind_sq) + file_diff
                new_rank = chess.square_rank(behind_sq) + rank_diff
                if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                    break
                behind_sq = chess.square(new_file, new_rank)
                behind_piece = board.piece_at(behind_sq)
                if behind_piece:
                    break
            
            if behind_piece and behind_piece.color == enemy:
                # Valid skewer: target is in front of behind_piece
                # More valuable if front piece is more valuable than behind
                front_value = _piece_value_tactical(target.piece_type)
                behind_value = _piece_value_tactical(behind_piece.piece_type)
                
                if front_value >= behind_value:
                    skewers.append({
                        "attacker_sq": chess.square_name(sq),
                        "attacker": piece.symbol(),
                        "front_sq": chess.square_name(target_sq),
                        "front_piece": target.symbol(),
                        "behind_sq": chess.square_name(behind_sq),
                        "behind_piece": behind_piece.symbol(),
                    })
    
    return skewers


def detect_back_rank_weakness(board: chess.Board) -> Dict[str, Any]:
    """
    Detect if enemy has back rank weakness (king trapped with no escape).
    
    Returns dict with:
    - has_weakness: bool
    - king_sq: enemy king square
    - escape_squares: list of potential escape squares (should be empty for weakness)
    - attacking_moves: legal moves that exploit the weakness
    """
    turn = board.turn
    enemy = not turn
    
    enemy_king = board.king(enemy)
    if enemy_king is None:
        return {"has_weakness": False}
    
    # Back rank depends on color
    back_rank = 7 if enemy == chess.WHITE else 0
    king_rank = chess.square_rank(enemy_king)
    
    if king_rank != back_rank:
        return {"has_weakness": False, "reason": "king_not_on_back_rank"}
    
    # Check if king has escape squares
    king_file = chess.square_file(enemy_king)
    escape_squares = []
    
    # Check squares king could move to (one rank forward)
    escape_rank = back_rank - 1 if enemy == chess.WHITE else back_rank + 1
    
    for df in [-1, 0, 1]:
        escape_file = king_file + df
        if not (0 <= escape_file <= 7):
            continue
        if not (0 <= escape_rank <= 7):
            continue
        
        escape_sq = chess.square(escape_file, escape_rank)
        
        # Check if square is blocked by own piece
        blocker = board.piece_at(escape_sq)
        if blocker and blocker.color == enemy:
            continue
        
        # Check if square is attacked by us
        if board.is_attacked_by(turn, escape_sq):
            continue
        
        escape_squares.append(chess.square_name(escape_sq))
    
    has_weakness = len(escape_squares) == 0
    
    # Find attacking moves if weakness exists
    attacking_moves = []
    if has_weakness:
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if not piece:
                continue
            
            # Rook or Queen moving to back rank
            if piece.piece_type in (chess.ROOK, chess.QUEEN):
                to_rank = chess.square_rank(move.to_square)
                if to_rank == back_rank:
                    # Check if it would give check
                    board.push(move)
                    gives_check = board.is_check()
                    is_mate = board.is_checkmate()
                    board.pop()
                    
                    if gives_check:
                        attacking_moves.append({
                            "move": move.uci(),
                            "is_mate": is_mate,
                        })
    
    return {
        "has_weakness": has_weakness,
        "king_sq": chess.square_name(enemy_king),
        "escape_squares": escape_squares,
        "attacking_moves": attacking_moves,
    }


def detect_discovered_attacks(board: chess.Board) -> List[Dict[str, Any]]:
    """
    Find pieces that can move to reveal an attack from a piece behind.
    
    A discovered attack occurs when moving one piece reveals an attack
    from another piece on a different target.
    
    Returns list of discovered attack opportunities with:
    - blocking_piece: the piece that moves
    - revealing_piece: the piece that gets revealed
    - target: what the revealed piece attacks
    - move: the triggering move
    """
    discovered = []
    turn = board.turn
    enemy = not turn
    
    # Find our sliding pieces that might be blocked
    for slider_sq in chess.SQUARES:
        slider = board.piece_at(slider_sq)
        if not slider or slider.color != turn:
            continue
        if slider.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            continue
        
        # Get the rays this piece could attack along
        slider_attacks = _get_slider_rays(slider_sq, slider.piece_type)
        
        for ray in slider_attacks:
            if len(ray) < 2:
                continue
            
            # Check if first piece on ray is ours (potential blocker)
            blocker_sq = None
            blocker = None
            for sq in ray:
                piece = board.piece_at(sq)
                if piece:
                    if piece.color == turn:
                        blocker_sq = sq
                        blocker = piece
                    break
            
            if blocker_sq is None:
                continue
            
            # Check if there's an enemy piece further on the ray
            target_sq = None
            target = None
            found_blocker = False
            for sq in ray:
                if sq == blocker_sq:
                    found_blocker = True
                    continue
                if not found_blocker:
                    continue
                
                piece = board.piece_at(sq)
                if piece:
                    if piece.color == enemy:
                        target_sq = sq
                        target = piece
                    break
            
            if target_sq is None:
                continue
            
            # Found a potential discovered attack setup
            # Now find moves for the blocker that reveal the attack
            for move in board.legal_moves:
                if move.from_square != blocker_sq:
                    continue
                
                # Check if moving the blocker reveals the attack
                board.push(move)
                
                # After move, can slider attack target?
                can_attack = target_sq in board.attacks(slider_sq)
                gives_check = board.is_check()
                
                board.pop()
                
                if can_attack:
                    discovered.append({
                        "move": move.uci(),
                        "blocker_sq": chess.square_name(blocker_sq),
                        "blocker": blocker.symbol(),
                        "slider_sq": chess.square_name(slider_sq),
                        "slider": slider.symbol(),
                        "target_sq": chess.square_name(target_sq),
                        "target": target.symbol(),
                        "target_value": _piece_value_tactical(target.piece_type),
                        "is_discovered_check": gives_check,
                    })
    
    return discovered


def _get_slider_rays(sq: chess.Square, piece_type: chess.PieceType) -> List[List[chess.Square]]:
    """Get all rays from a square for a sliding piece."""
    rays = []
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    
    # Rook directions
    if piece_type in (chess.ROOK, chess.QUEEN):
        for df, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ray = []
            for i in range(1, 8):
                new_f, new_r = file + df * i, rank + dr * i
                if 0 <= new_f <= 7 and 0 <= new_r <= 7:
                    ray.append(chess.square(new_f, new_r))
                else:
                    break
            if ray:
                rays.append(ray)
    
    # Bishop directions
    if piece_type in (chess.BISHOP, chess.QUEEN):
        for df, dr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ray = []
            for i in range(1, 8):
                new_f, new_r = file + df * i, rank + dr * i
                if 0 <= new_f <= 7 and 0 <= new_r <= 7:
                    ray.append(chess.square(new_f, new_r))
                else:
                    break
            if ray:
                rays.append(ray)
    
    return rays


def _piece_value_tactical(piece_type: chess.PieceType) -> int:
    """Get piece value for tactical calculations."""
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100,  # High value for targeting king
    }
    return values.get(piece_type, 0)


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


def get_skewer_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that create or exploit skewers."""
    skewers = detect_skewers(board)
    moves = []
    
    # Direct captures of the front piece (the skewer target)
    for skewer in skewers:
        front_sq = chess.parse_square(skewer["front_sq"])
        for move in board.legal_moves:
            if move.to_square == front_sq:
                moves.append(move)
    
    return moves


def get_pin_exploit_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that exploit existing pins."""
    pins = detect_pins(board)
    moves = []
    
    for pin in pins:
        pinned_sq = chess.parse_square(pin["pinned_square"])
        
        # Moves that attack the pinned piece (pile on)
        for move in board.legal_moves:
            board.push(move)
            # Check if we now attack the pinned piece more
            attackers = board.attackers(not board.turn, pinned_sq)
            board.pop()
            
            # Capture the pinned piece if possible
            if move.to_square == pinned_sq:
                moves.append(move)
                continue
            
            # Add attackers to pinned piece
            pre_attackers = board.attackers(board.turn, pinned_sq)
            if len(attackers) > len(pre_attackers):
                moves.append(move)
    
    return moves


def get_back_rank_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that exploit back rank weakness."""
    weakness = detect_back_rank_weakness(board)
    
    if not weakness.get("has_weakness"):
        return []
    
    moves = []
    for attack in weakness.get("attacking_moves", []):
        try:
            move = chess.Move.from_uci(attack["move"])
            # Prioritize mates
            if attack.get("is_mate"):
                moves.insert(0, move)
            else:
                moves.append(move)
        except Exception:
            pass
    
    return moves


def get_discovered_attack_moves(board: chess.Board) -> List[chess.Move]:
    """Get moves that create discovered attacks."""
    discovered = detect_discovered_attacks(board)
    moves = []
    
    # Sort by target value (higher value targets first)
    discovered.sort(key=lambda x: x.get("target_value", 0), reverse=True)
    
    for disc in discovered:
        try:
            move = chess.Move.from_uci(disc["move"])
            # Prioritize discovered checks
            if disc.get("is_discovered_check"):
                moves.insert(0, move)
            else:
                moves.append(move)
        except Exception:
            pass
    
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


def create_skewer_detector(nid: str) -> Node:
    """Create a sensor node that detects skewer opportunities."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        skewers = detect_skewers(board)
        node.meta["skewers"] = skewers
        env.setdefault("tactics", {})["skewers"] = skewers
        
        ok = len(skewers) > 0
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_skewer_exploiter(nid: str) -> Node:
    """Create an actuator node that exploits skewer opportunities."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_skewer_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "skewer",
                "rank": cfg["skewer_priority"],
            } for m in moves[:3]]
            
            env.setdefault("tactics_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_pin_exploiter(nid: str) -> Node:
    """Create an actuator node that exploits pins."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_pin_exploit_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "exploit_pin",
                "rank": cfg["pin_priority"],
            } for m in moves[:3]]
            
            env.setdefault("tactics_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_back_rank_detector(nid: str) -> Node:
    """Create a sensor node that detects back rank weaknesses."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        weakness = detect_back_rank_weakness(board)
        node.meta["back_rank"] = weakness
        env.setdefault("tactics", {})["back_rank"] = weakness
        
        ok = weakness.get("has_weakness", False)
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_back_rank_exploiter(nid: str) -> Node:
    """Create an actuator node that exploits back rank weaknesses."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_back_rank_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "back_rank",
                "rank": cfg["back_rank_priority"],
            } for m in moves[:3]]
            
            env.setdefault("tactics_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        
        return False, False
    
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_discovered_attack_detector(nid: str) -> Node:
    """Create a sensor node that detects discovered attack opportunities."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        discovered = detect_discovered_attacks(board)
        node.meta["discovered_attacks"] = discovered
        env.setdefault("tactics", {})["discovered_attacks"] = discovered
        
        ok = len(discovered) > 0
        return ok, ok
    
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_discovered_attack_exploiter(nid: str) -> Node:
    """Create an actuator node that exploits discovered attacks."""
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board:
            return False, False
        
        moves = get_discovered_attack_moves(board)
        if moves:
            cfg = _load_tactics_cfg()
            proposals = [{
                "move": m.uci(),
                "phase": "tactical",
                "reason": "discovered_attack",
                "rank": cfg["discovered_attack_priority"],
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
        ├── detect_pin → exploit_pin
        ├── detect_hanging → capture_hanging
        │                  → protect_hanging
        ├── detect_skewer → exploit_skewer
        ├── detect_back_rank → exploit_back_rank
        └── detect_discovered → exploit_discovered
    """
    g = Graph()
    
    # Root node
    root = Node("tactics_root", NodeType.SCRIPT)
    g.add_node(root)
    
    # Fork detection and exploitation
    fork_sensor = Node("detect_fork", NodeType.SCRIPT)
    fork_exploit = Node("exploit_fork", NodeType.SCRIPT)
    g.add_node(fork_sensor)
    g.add_node(fork_exploit)
    g.add_edge("tactics_root", "detect_fork", LinkType.SUB)
    g.add_edge("detect_fork", "exploit_fork", LinkType.POR)
    
    # Pin detection and exploitation
    pin_sensor = Node("detect_pin", NodeType.SCRIPT)
    pin_exploit = Node("exploit_pin", NodeType.SCRIPT)
    g.add_node(pin_sensor)
    g.add_node(pin_exploit)
    g.add_edge("tactics_root", "detect_pin", LinkType.SUB)
    g.add_edge("detect_pin", "exploit_pin", LinkType.POR)
    
    # Hanging piece detection and handling
    hanging_sensor = Node("detect_hanging", NodeType.SCRIPT)
    capture_hanging_node = Node("capture_hanging", NodeType.SCRIPT)
    protect_hanging_node = Node("protect_hanging", NodeType.SCRIPT)
    g.add_node(hanging_sensor)
    g.add_node(capture_hanging_node)
    g.add_node(protect_hanging_node)
    g.add_edge("tactics_root", "detect_hanging", LinkType.SUB)
    g.add_edge("detect_hanging", "capture_hanging", LinkType.POR)
    g.add_edge("detect_hanging", "protect_hanging", LinkType.POR)
    
    # Skewer detection and exploitation
    skewer_sensor = Node("detect_skewer", NodeType.SCRIPT)
    skewer_exploit = Node("exploit_skewer", NodeType.SCRIPT)
    g.add_node(skewer_sensor)
    g.add_node(skewer_exploit)
    g.add_edge("tactics_root", "detect_skewer", LinkType.SUB)
    g.add_edge("detect_skewer", "exploit_skewer", LinkType.POR)
    
    # Back rank detection and exploitation
    back_rank_sensor = Node("detect_back_rank", NodeType.SCRIPT)
    back_rank_exploit = Node("exploit_back_rank", NodeType.SCRIPT)
    g.add_node(back_rank_sensor)
    g.add_node(back_rank_exploit)
    g.add_edge("tactics_root", "detect_back_rank", LinkType.SUB)
    g.add_edge("detect_back_rank", "exploit_back_rank", LinkType.POR)
    
    # Discovered attack detection and exploitation
    discovered_sensor = Node("detect_discovered", NodeType.SCRIPT)
    discovered_exploit = Node("exploit_discovered", NodeType.SCRIPT)
    g.add_node(discovered_sensor)
    g.add_node(discovered_exploit)
    g.add_edge("tactics_root", "detect_discovered", LinkType.SUB)
    g.add_edge("detect_discovered", "exploit_discovered", LinkType.POR)
    
    return g


# ============================================================================
# Weight Pack
# ============================================================================

def create_default_tactics_weight_pack() -> Dict[str, Any]:
    """Create the default weight pack for tactics subgraph."""
    return {
        "version": "2.0",
        "subgraph": "tactics",
        "priorities": {
            "fork_priority": 0.9,
            "pin_priority": 0.8,
            "hanging_priority": 0.95,
            "skewer_priority": 0.7,
            "back_rank_priority": 0.98,
            "discovered_attack_priority": 0.85,
        },
        "edges": {
            # Fork
            "tactics_root->detect_fork:SUB": 1.0,
            "detect_fork->exploit_fork:POR": 1.0,
            # Pin
            "tactics_root->detect_pin:SUB": 1.0,
            "detect_pin->exploit_pin:POR": 1.0,
            # Hanging
            "tactics_root->detect_hanging:SUB": 1.0,
            "detect_hanging->capture_hanging:POR": 1.0,
            "detect_hanging->protect_hanging:POR": 0.8,
            # Skewer
            "tactics_root->detect_skewer:SUB": 1.0,
            "detect_skewer->exploit_skewer:POR": 1.0,
            # Back rank
            "tactics_root->detect_back_rank:SUB": 1.0,
            "detect_back_rank->exploit_back_rank:POR": 1.0,
            # Discovered attack
            "tactics_root->detect_discovered:SUB": 1.0,
            "detect_discovered->exploit_discovered:POR": 1.0,
        },
    }

