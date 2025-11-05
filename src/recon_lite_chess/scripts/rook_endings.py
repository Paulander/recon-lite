"""Rook ending technique scaffolding (cut-off, bridge, ladder)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import chess

from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite_chess import create_wait_for_board_change
from recon_lite_chess import predicates as preds


def _chebyshev(a: chess.Square, b: chess.Square) -> int:
    return max(abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b)))


def _our_rook_square(board: chess.Board, color: Optional[bool] = None) -> Optional[chess.Square]:
    target_color = board.turn if color is None else color
    for sq, piece in board.piece_map().items():
        if piece.color == target_color and piece.piece_type == chess.ROOK:
            return sq
    return None


def create_rook_cutoff_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        ready = bool(board) and preds.has_stable_cut(board)
        env.setdefault("rook_techniques", {}).setdefault("cutoff", {})["ready"] = ready
        node.meta["ready"] = ready
        return True, ready

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_rook_bridge_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if board is None:
            env.setdefault("rook_techniques", {}).setdefault("bridge", {})["ready"] = False
            return True, False
        ready = preds.enemy_at_edge(board) and preds.rook_safe_now(board)
        rook_sq = _our_rook_square(board)
        king_sq = board.king(board.turn)
        if ready and rook_sq is not None and king_sq is not None:
            ready = _chebyshev(king_sq, rook_sq) <= 2
        env.setdefault("rook_techniques", {}).setdefault("bridge", {})["ready"] = ready
        node.meta["ready"] = ready
        return True, ready

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_rook_ladder_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        if board is None:
            env.setdefault("rook_techniques", {}).setdefault("ladder", {})["ready"] = False
            return True, False
        rook_sq = _our_rook_square(board)
        enemy_king = board.king(not board.turn)
        our_king = board.king(board.turn)
        ready = False
        if rook_sq is not None and enemy_king is not None and our_king is not None:
            same_file = chess.square_file(rook_sq) == chess.square_file(enemy_king)
            same_rank = chess.square_rank(rook_sq) == chess.square_rank(enemy_king)
            support = _chebyshev(our_king, enemy_king) <= 2
            ready = (same_file or same_rank) and support
        env.setdefault("rook_techniques", {}).setdefault("ladder", {})["ready"] = ready
        node.meta["ready"] = ready
        return True, ready

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_rook_summary_probe(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        snapshot = env.get("rook_techniques", {})
        node.meta["snapshot"] = snapshot
        return True, True

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def build_rook_techniques_network() -> Graph:
    g = Graph()

    # Terminals
    g.add_node(create_rook_cutoff_detector("rook_cutoff_ready"))
    g.add_node(create_rook_bridge_detector("rook_bridge_ready"))
    g.add_node(create_rook_ladder_detector("rook_ladder_ready"))
    g.add_node(create_rook_summary_probe("rook_techniques_summary"))
    g.add_node(create_wait_for_board_change("rook_wait_after_cutoff"))
    g.add_node(create_wait_for_board_change("rook_wait_after_bridge"))
    g.add_node(create_wait_for_board_change("rook_wait_after_ladder"))

    # Script structure
    g.add_node(Node(nid="rook_techniques_root", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_cutoff", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_bridge", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_ladder", ntype=NodeType.SCRIPT))

    # Per-technique sequencing scripts (check -> wait)
    g.add_node(Node(nid="rook_cutoff_check", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_cutoff_wait", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_bridge_check", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_bridge_wait", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_ladder_check", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="rook_ladder_wait", ntype=NodeType.SCRIPT))

    # Hierarchy wiring
    g.add_edge("rook_techniques_root", "rook_cutoff", LinkType.SUB)
    g.add_edge("rook_techniques_root", "rook_bridge", LinkType.SUB)
    g.add_edge("rook_techniques_root", "rook_ladder", LinkType.SUB)
    g.add_edge("rook_techniques_root", "rook_techniques_summary", LinkType.SUB)

    g.add_edge("rook_cutoff", "rook_cutoff_check", LinkType.SUB)
    g.add_edge("rook_cutoff", "rook_cutoff_wait", LinkType.SUB)
    g.add_edge("rook_bridge", "rook_bridge_check", LinkType.SUB)
    g.add_edge("rook_bridge", "rook_bridge_wait", LinkType.SUB)
    g.add_edge("rook_ladder", "rook_ladder_check", LinkType.SUB)
    g.add_edge("rook_ladder", "rook_ladder_wait", LinkType.SUB)

    # Attach terminals
    g.add_edge("rook_cutoff_check", "rook_cutoff_ready", LinkType.SUB)
    g.add_edge("rook_cutoff_wait", "rook_wait_after_cutoff", LinkType.SUB)
    g.add_edge("rook_bridge_check", "rook_bridge_ready", LinkType.SUB)
    g.add_edge("rook_bridge_wait", "rook_wait_after_bridge", LinkType.SUB)
    g.add_edge("rook_ladder_check", "rook_ladder_ready", LinkType.SUB)
    g.add_edge("rook_ladder_wait", "rook_wait_after_ladder", LinkType.SUB)

    # Sequencing across techniques and within each pair
    g.add_edge("rook_cutoff_check", "rook_cutoff_wait", LinkType.POR)
    g.add_edge("rook_bridge_check", "rook_bridge_wait", LinkType.POR)
    g.add_edge("rook_ladder_check", "rook_ladder_wait", LinkType.POR)

    g.add_edge("rook_cutoff", "rook_bridge", LinkType.POR)
    g.add_edge("rook_bridge", "rook_ladder", LinkType.POR)

    return g


__all__ = [
    "build_rook_techniques_network",
    "create_rook_cutoff_detector",
    "create_rook_bridge_detector",
    "create_rook_ladder_detector",
    "create_rook_summary_probe",
]
