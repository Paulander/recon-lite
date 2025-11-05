"""Lightweight KPK (King+Pawn vs King) script scaffolding."""

from __future__ import annotations

from typing import Any, Dict

import chess

from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite_chess import create_wait_for_board_change
from recon_lite_chess.sensors import structure as struct_sensors
from recon_lite_chess.sensors import tactics as tactic_sensors


def create_kpk_material_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        node.meta["summary"] = summary
        env.setdefault("kpk", {})["material"] = summary
        ok = bool(summary.get("is_kpk"))
        return ok, ok

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_push_window(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        color = summary.get("attacker_color")
        ok = bool(summary.get("is_kpk")) and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        node.meta["can_push"] = ok
        env.setdefault("kpk", {}).setdefault("tactics", {})["can_push"] = ok
        return ok, ok

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_opposition_probe(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        color = summary.get("attacker_color")
        ok = bool(summary.get("is_kpk")) and tactic_sensors.has_opposition_alignment(board, attacker_color=color)
        node.meta["has_opposition"] = ok
        env.setdefault("kpk", {}).setdefault("tactics", {})["has_opposition"] = ok
        return ok, ok

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_promotion_probe(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        distance = struct_sensors.pawn_distance_to_promotion(board)
        node.meta["distance"] = distance
        env.setdefault("kpk", {}).setdefault("structure", {})["promotion_distance"] = distance
        ok = distance <= 1
        return ok, ok

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_move_selector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        if not summary.get("is_kpk"):
            return False, False
        color = summary.get("attacker_color")
        pawn_sq = summary.get("pawn_square")
        suggestion = None
        if color is None or pawn_sq is None:
            return False, False

        direction = 8 if color else -8
        push_sq = pawn_sq + direction
        push_move = chess.Move(pawn_sq, push_sq)
        if push_move in board.legal_moves and tactic_sensors.can_push_pawn_safely(board, attacker_color=color):
            suggestion = push_move.uci()
        else:
            attacker_king = summary.get("attacker_king")
            defender_king = summary.get("defender_king")
            best = None
            for move in board.legal_moves:
                piece = board.piece_at(move.from_square)
                if piece and piece.color == color and piece.piece_type == chess.KING:
                    trial = board.copy(stack=False)
                    trial.push(move)
                    new_sq = trial.king(color)
                    if new_sq is None or defender_king is None:
                        continue
                    dist = max(abs(chess.square_file(new_sq) - chess.square_file(defender_king)),
                               abs(chess.square_rank(new_sq) - chess.square_rank(defender_king)))
                    score = dist
                    if attacker_king is not None:
                        score -= max(abs(chess.square_file(attacker_king) - chess.square_file(defender_king)),
                                     abs(chess.square_rank(attacker_king) - chess.square_rank(defender_king))) * 0.25
                    if best is None or score < best[0]:
                        best = (score, move)
            if best is not None:
                suggestion = best[1].uci()

        if suggestion is None:
            legal = list(board.legal_moves)
            if legal:
                suggestion = legal[0].uci()

        if suggestion is None:
            return True, False

        env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = suggestion
        node.meta["last_move"] = suggestion
        return True, True

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def build_kpk_network() -> Graph:
    g = Graph()

    # Terminals
    g.add_node(create_kpk_material_detector("kpk_material_check"))
    g.add_node(create_kpk_push_window("kpk_push_window"))
    g.add_node(create_kpk_opposition_probe("kpk_opposition_probe"))
    g.add_node(create_kpk_promotion_probe("kpk_promotion_probe"))
    g.add_node(create_kpk_move_selector("kpk_move_selector"))
    g.add_node(create_wait_for_board_change("kpk_wait_for_change"))

    # Script backbone
    g.add_node(Node(nid="kpk_root", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_detect", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_execute", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_finish", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="kpk_wait", ntype=NodeType.SCRIPT))

    # Hierarchy
    g.add_edge("kpk_root", "kpk_detect", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_execute", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_finish", LinkType.SUB)
    g.add_edge("kpk_root", "kpk_wait", LinkType.SUB)

    g.add_edge("kpk_detect", "kpk_material_check", LinkType.SUB)
    g.add_edge("kpk_detect", "kpk_push_window", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_move_selector", LinkType.SUB)
    g.add_edge("kpk_execute", "kpk_opposition_probe", LinkType.SUB)
    g.add_edge("kpk_finish", "kpk_promotion_probe", LinkType.SUB)
    g.add_edge("kpk_wait", "kpk_wait_for_change", LinkType.SUB)

    # Sequencing
    g.add_edge("kpk_detect", "kpk_execute", LinkType.POR)
    g.add_edge("kpk_execute", "kpk_finish", LinkType.POR)
    g.add_edge("kpk_finish", "kpk_wait", LinkType.POR)

    return g


__all__ = [
    "build_kpk_network",
    "create_kpk_material_detector",
    "create_kpk_push_window",
    "create_kpk_opposition_probe",
    "create_kpk_promotion_probe",
    "create_kpk_move_selector",
]
