"""Lightweight KPK (King+Pawn vs King) script scaffolding."""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import json

import chess

from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from recon_lite_chess import create_wait_for_board_change
from recon_lite_chess.sensors import structure as struct_sensors
from recon_lite_chess.sensors import tactics as tactic_sensors

# Shared config cache for KPK weights
_CFG_CACHE = {"loaded": False, "weights": {"push_bias": 0.6, "king_distance_weight": 0.25, "safety_weight": 0.15}}


def _load_cfg():
    """Load KPK weights once from the SWP, fallback to defaults."""
    if _CFG_CACHE["loaded"]:
        return _CFG_CACHE["weights"]
    try:
        path = Path("weights/subgraphs/kpk_weight_pack.swp")
        data = json.loads(path.read_text())
        ws = data.get("move_selector", {})
        _CFG_CACHE["weights"].update({
            "push_bias": float(ws.get("push_bias", _CFG_CACHE["weights"]["push_bias"])),
            "king_distance_weight": float(ws.get("king_distance_weight", _CFG_CACHE["weights"]["king_distance_weight"])),
            "safety_weight": float(ws.get("safety_weight", _CFG_CACHE["weights"]["safety_weight"]))
        })
    except Exception:
        pass
    _CFG_CACHE["loaded"] = True
    return _CFG_CACHE["weights"]


def create_kpk_material_detector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        node.meta["summary"] = summary
        env.setdefault("kpk", {})["material"] = summary
        ok = bool(summary.get("is_kpk"))
        return ok, ok

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_push_window(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
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
        weights = _load_cfg()
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
        weights = _load_cfg()
        board = env.get("board")
        distance = struct_sensors.pawn_distance_to_promotion(board)
        node.meta["distance"] = distance
        env.setdefault("kpk", {}).setdefault("structure", {})["promotion_distance"] = distance
        ok = distance <= 1
        return ok, ok

    return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)


def create_kpk_move_selector(nid: str) -> Node:
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
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
        safe_push = (push_move in board.legal_moves) and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        push_score = (weights.get("push_bias", 0.6) + (weights.get("safety_weight", 0.15) if safe_push else 0.0)) if (push_move in board.legal_moves) else -1.0

        attacker_king = summary.get("attacker_king")
        defender_king = summary.get("defender_king")
        best_king = None
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.color == color and piece.piece_type == chess.KING:
                trial = board.copy(stack=False)
                trial.push(move)
                new_sq = trial.king(color)
                if new_sq is None or defender_king is None:
                    continue
                cur_d = max(abs(chess.square_file(attacker_king) - chess.square_file(defender_king)),
                            abs(chess.square_rank(attacker_king) - chess.square_rank(defender_king))) if attacker_king is not None else 0
                new_d = max(abs(chess.square_file(new_sq) - chess.square_file(defender_king)),
                            abs(chess.square_rank(new_sq) - chess.square_rank(defender_king)))
                gain = max(0, cur_d - new_d)
                score = gain * weights.get("king_distance_weight", 0.25)
                if best_king is None or score > best_king[0]:
                    best_king = (score, move)

        if best_king is not None and best_king[0] >= push_score:
            suggestion = best_king[1].uci()
        elif push_score >= 0:
            suggestion = push_move.uci()

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


def run_kpk_episode_with_trace(
    board: chess.Board,
    *,
    max_plies: int = 100,
    max_ticks_per_move: int = 200,
    trace_db: Optional[TraceDB] = None,
    episode_id: str = "kpk-episode",
    pack_paths: Optional[list[Path]] = None,
) -> EpisodeRecord:
    """
    Run a single KPK episode using the batch evaluator and emit an EpisodeRecord.
    If `trace_db` is provided, the episode is appended immediately.
    """
    from demos.experiments.batch_eval import run_kpk_episode as _run

    _, _, ep = _run(
        board.fen(),
        max_plies=max_plies,
        max_ticks_per_move=max_ticks_per_move,
        pack_paths=pack_paths or [],
    )
    ep.episode_id = episode_id
    if pack_paths:
        ep.pack_meta = pack_fingerprint(pack_paths)
    if trace_db:
        trace_db.add_episode(ep)
    return ep


__all__ = [
    "build_kpk_network",
    "create_kpk_material_detector",
    "create_kpk_push_window",
    "create_kpk_opposition_probe",
    "create_kpk_promotion_probe",
    "create_kpk_move_selector",
]
