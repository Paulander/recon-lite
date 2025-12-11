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
    """
    Sensor that checks if pawn can be pushed safely.
    
    Always completes (returns True) because this is a detection node,
    not a gate. The result is stored in meta/env for the move selector to use.
    """
    def _predicate(node: Node, env: Dict[str, Any]):
        weights = _load_cfg()
        board = env.get("board")
        summary = struct_sensors.summarize_kpk_material(board)
        color = summary.get("attacker_color")
        is_kpk = bool(summary.get("is_kpk"))
        can_push = is_kpk and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        node.meta["can_push"] = can_push
        node.meta["is_kpk"] = is_kpk
        env.setdefault("kpk", {}).setdefault("tactics", {})["can_push"] = can_push
        # Always complete the sensor check - the move selector handles both push and king moves
        return is_kpk, is_kpk  # Only block if not a KPK position at all

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
    """
    KPK move selector with promotion-focused strategy.
    
    The goal of KPK is to PROMOTE the pawn. This is the success condition.
    Strategy priorities:
    1. Push pawn to promotion when safe (highest priority near 7th/8th rank)
    2. If pawn push blocked, use king to support/clear path
    3. Only use king moves when pawn push isn't viable
    """
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

        pawn_rank = chess.square_rank(pawn_sq)
        pawn_file = chess.square_file(pawn_sq)
        
        # Calculate distance to promotion (ranks remaining)
        if color == chess.WHITE:
            distance_to_promo = 7 - pawn_rank  # White promotes on rank 7 (8th rank)
        else:
            distance_to_promo = pawn_rank  # Black promotes on rank 0 (1st rank)
        
        direction = 8 if color else -8
        push_sq = pawn_sq + direction
        
        # Check if this is a promotion move (pawn on 7th rank)
        is_promotion_rank = (color == chess.WHITE and pawn_rank == 6) or (color == chess.BLACK and pawn_rank == 1)
        
        if is_promotion_rank:
            # Promotion move - must specify promotion piece
            promo_move = chess.Move(pawn_sq, push_sq, promotion=chess.QUEEN)
            push_move = promo_move  # Use promotion move
            push_legal = promo_move in board.legal_moves
        else:
            # Regular pawn push
            push_move = chess.Move(pawn_sq, push_sq)
            push_legal = push_move in board.legal_moves
        
        safe_push = push_legal and tactic_sensors.can_push_pawn_safely(board, attacker_color=color)
        
        # CRITICAL: Promotion move! Always take it if legal
        if is_promotion_rank and push_legal:
            # This push is promotion! Highest priority - ALWAYS promote
            suggestion = push_move.uci()
            env.setdefault("kpk", {}).setdefault("policy", {})["suggested_move"] = suggestion
            node.meta["last_move"] = suggestion
            node.meta["reason"] = "promotion"
            return True, True
        
        # Calculate push score with distance bonus
        # The closer to promotion, the higher the priority
        base_push_score = weights.get("push_bias", 0.6)
        safety_bonus = weights.get("safety_weight", 0.15) if safe_push else 0.0
        # Massive bonus for pawns close to promotion
        promotion_urgency = (5 - distance_to_promo) * 0.3  # +0.3 per rank advanced
        push_score = (base_push_score + safety_bonus + promotion_urgency) if push_legal else -1.0

        attacker_king = summary.get("attacker_king")
        defender_king = summary.get("defender_king")
        
        # Evaluate king moves
        best_king = None
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.color == color and piece.piece_type == chess.KING:
                trial = board.copy(stack=False)
                trial.push(move)
                new_sq = trial.king(color)
                if new_sq is None:
                    continue
                    
                score = 0.0
                new_file = chess.square_file(new_sq)
                new_rank = chess.square_rank(new_sq)
                
                # Bonus for king supporting the pawn (within 1-2 files, ahead of pawn)
                file_dist_to_pawn = abs(new_file - pawn_file)
                if file_dist_to_pawn <= 1:
                    score += 0.2
                    # Extra bonus if king is ahead of pawn (can support promotion)
                    if color == chess.WHITE and new_rank > pawn_rank:
                        score += 0.1
                    elif color == chess.BLACK and new_rank < pawn_rank:
                        score += 0.1
                
                # Bonus for approaching enemy king (to cut off escape)
                if defender_king is not None:
                    cur_d = max(abs(chess.square_file(attacker_king) - chess.square_file(defender_king)),
                                abs(chess.square_rank(attacker_king) - chess.square_rank(defender_king))) if attacker_king is not None else 0
                    new_d = max(abs(new_file - chess.square_file(defender_king)),
                                abs(new_rank - chess.square_rank(defender_king)))
                    gain = max(0, cur_d - new_d)
                    score += gain * weights.get("king_distance_weight", 0.25)
                
                if best_king is None or score > best_king[0]:
                    best_king = (score, move)

        # Decision: Prioritize pawn push unless blocked or king move is significantly better
        if push_score >= 0 and (best_king is None or push_score > best_king[0] + 0.3):
            # Push the pawn!
            suggestion = push_move.uci()
            node.meta["reason"] = "pawn_push"
        elif best_king is not None:
            # King move to support
            suggestion = best_king[1].uci()
            node.meta["reason"] = "king_support"
        elif push_score >= 0:
            # Fallback to pawn push even if not ideal
            suggestion = push_move.uci()
            node.meta["reason"] = "pawn_push_fallback"

        if suggestion is None:
            # Last resort: any legal move
            legal = list(board.legal_moves)
            if legal:
                suggestion = legal[0].uci()
                node.meta["reason"] = "fallback"

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
