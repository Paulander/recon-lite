"""M5.4: Rook endgame pattern detection and technique subgraph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

from recon_lite import Graph, LinkType, Node, NodeType


_ROOK_CFG = {"loaded": False, "weights": {
    "lucena_priority": 0.95, "philidor_priority": 0.9,
    "rook_behind_priority": 0.8, "cutoff_priority": 0.85,
}}


def _load_rook_cfg() -> Dict[str, float]:
    if _ROOK_CFG["loaded"]:
        return _ROOK_CFG["weights"]
    try:
        path = Path("weights/subgraphs/rook_weight_pack.swp")
        if path.exists():
            data = json.loads(path.read_text())
            _ROOK_CFG["weights"].update(data.get("priorities", {}))
    except Exception:
        pass
    _ROOK_CFG["loaded"] = True
    return _ROOK_CFG["weights"]


def is_rook_endgame(board: chess.Board) -> Tuple[bool, Optional[bool]]:
    """Check if position is a rook endgame."""
    white_pieces, black_pieces = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            (white_pieces if piece.color == chess.WHITE else black_pieces).append(piece.piece_type)
    
    if white_pieces.count(chess.ROOK) != 1 or black_pieces.count(chess.ROOK) != 1:
        return False, None
    for pt in [chess.QUEEN, chess.BISHOP, chess.KNIGHT]:
        if pt in white_pieces or pt in black_pieces:
            return False, None
    
    wp, bp = white_pieces.count(chess.PAWN), black_pieces.count(chess.PAWN)
    if wp > 0 and bp == 0: return True, chess.WHITE
    if bp > 0 and wp == 0: return True, chess.BLACK
    if wp == 0 and bp == 0: return True, None
    return False, None


def detect_lucena_position(board: chess.Board, attacker_color: bool) -> bool:
    """Detect Lucena position (pawn on 7th, king in front)."""
    if attacker_color is None: return False
    pawn_sq = next(iter(board.pieces(chess.PAWN, attacker_color)), None)
    if pawn_sq is None: return False
    target_rank = 6 if attacker_color == chess.WHITE else 1
    if chess.square_rank(pawn_sq) != target_rank: return False
    king_sq = board.king(attacker_color)
    if king_sq is None: return False
    promotion_rank = 7 if attacker_color == chess.WHITE else 0
    return chess.square_rank(king_sq) == promotion_rank


def detect_philidor_position(board: chess.Board, defender_color: bool) -> bool:
    """Detect Philidor defensive setup (rook on 6th rank)."""
    if defender_color is None: return False
    rook_sq = next(iter(board.pieces(chess.ROOK, defender_color)), None)
    if rook_sq is None: return False
    target_rank = 5 if defender_color == chess.WHITE else 2
    return chess.square_rank(rook_sq) == target_rank


def get_cutoff_moves(board: chess.Board, color: bool) -> List[chess.Move]:
    """Get moves that cut off the enemy king."""
    moves = []
    rook_sq = next(iter(board.pieces(chess.ROOK, color)), None)
    if rook_sq is None: return moves
    enemy_king_sq = board.king(not color)
    if enemy_king_sq is None: return moves
    king_file = chess.square_file(enemy_king_sq)
    for move in board.legal_moves:
        if move.from_square == rook_sq:
            if abs(chess.square_file(move.to_square) - king_file) == 1:
                moves.append(move)
    return moves


def get_lucena_bridge_moves(board: chess.Board, attacker_color: bool) -> List[chess.Move]:
    """Get moves that build the bridge in Lucena position."""
    moves = []
    rook_sq = next(iter(board.pieces(chess.ROOK, attacker_color)), None)
    if rook_sq is None: return moves
    target_rank = 3 if attacker_color == chess.WHITE else 4
    for move in board.legal_moves:
        if move.from_square == rook_sq and chess.square_rank(move.to_square) == target_rank:
            moves.append(move)
    return moves


def create_rook_endgame_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board: return False, False
        is_rook, attacker = is_rook_endgame(board)
        node.meta["is_rook_endgame"] = is_rook
        node.meta["attacker"] = attacker
        env.setdefault("rook_endgame", {}).update({"is_rook_endgame": is_rook, "attacker": attacker})
        return is_rook, is_rook
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_lucena_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        rook_data = env.get("rook_endgame", {})
        attacker = rook_data.get("attacker")
        if not board or attacker is None: return False, False
        is_lucena = detect_lucena_position(board, attacker)
        node.meta["is_lucena"] = is_lucena
        env.setdefault("rook_endgame", {})["is_lucena"] = is_lucena
        return is_lucena, is_lucena
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_philidor_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        rook_data = env.get("rook_endgame", {})
        attacker = rook_data.get("attacker")
        if not board or attacker is None: return False, False
        defender = not attacker
        is_philidor = detect_philidor_position(board, defender)
        node.meta["is_philidor"] = is_philidor
        env.setdefault("rook_endgame", {})["is_philidor"] = is_philidor
        return is_philidor, is_philidor
    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_cutoff_move_selector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if not board: return False, False
        moves = get_cutoff_moves(board, board.turn)
        if moves:
            cfg = _load_rook_cfg()
            proposals = [{"move": m.uci(), "phase": "rook_endgame", "reason": "cutoff_king", "rank": cfg["cutoff_priority"]} for m in moves[:3]]
            env.setdefault("rook_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        return False, False
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def create_lucena_bridge_selector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        rook_data = env.get("rook_endgame", {})
        if not board or not rook_data.get("is_lucena"): return False, False
        attacker = rook_data.get("attacker")
        if board.turn != attacker: return False, False
        moves = get_lucena_bridge_moves(board, attacker)
        if moves:
            cfg = _load_rook_cfg()
            proposals = [{"move": m.uci(), "phase": "rook_endgame", "reason": "lucena_bridge", "rank": cfg["lucena_priority"]} for m in moves[:2]]
            env.setdefault("rook_proposals", []).extend(proposals)
            node.meta["proposals"] = proposals
            return True, True
        return False, False
    return Node(nid=nid, ntype=NodeType.SCRIPT, predicate=_predicate)


def build_rook_endgame_network() -> Graph:
    """Build the rook endgame subgraph."""
    g = Graph()
    root = Node("rook_endgame_root", NodeType.SCRIPT)
    g.add_node(root)
    
    # All nodes must be SCRIPT for POR edges
    rook_detector = Node("detect_rook_endgame", NodeType.SCRIPT)
    g.add_node(rook_detector)
    g.add_edge("rook_endgame_root", "detect_rook_endgame", LinkType.SUB)
    
    lucena_detector = Node("detect_lucena", NodeType.SCRIPT)
    lucena_bridge = Node("lucena_bridge_selector", NodeType.SCRIPT)
    g.add_node(lucena_detector)
    g.add_node(lucena_bridge)
    g.add_edge("detect_rook_endgame", "detect_lucena", LinkType.POR)
    g.add_edge("detect_lucena", "lucena_bridge_selector", LinkType.POR)
    
    philidor_detector = Node("detect_philidor", NodeType.SCRIPT)
    g.add_node(philidor_detector)
    g.add_edge("detect_rook_endgame", "detect_philidor", LinkType.POR)
    
    cutoff_selector = Node("cutoff_move_selector", NodeType.SCRIPT)
    g.add_node(cutoff_selector)
    g.add_edge("detect_rook_endgame", "cutoff_move_selector", LinkType.POR)
    
    return g


def create_default_rook_weight_pack() -> Dict[str, Any]:
    return {
        "version": "1.0", "subgraph": "rook_endgame",
        "priorities": {"lucena_priority": 0.95, "philidor_priority": 0.9, "rook_behind_priority": 0.8, "cutoff_priority": 0.85},
        "edges": {
            "rook_endgame_root->detect_rook_endgame:SUB": 1.0,
            "detect_rook_endgame->detect_lucena:POR": 1.0,
            "detect_lucena->lucena_bridge_selector:POR": 1.0,
            "detect_rook_endgame->detect_philidor:POR": 1.0,
            "detect_rook_endgame->cutoff_move_selector:POR": 0.9,
        },
    }

