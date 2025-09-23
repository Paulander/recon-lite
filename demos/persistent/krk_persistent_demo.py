#!/usr/bin/env python3
"""
Persistent KRK Chess Demo (ReCoN-driven)

Runs a single ReCoN engine instance across the whole game.
- Logs per-tick frames (network states, requests) for visualization
- Applies moves as actuators set env["chosen_move"]
- After applying a move, lets the opponent respond, then re-REQUESTS ROOT
- Outputs visualization JSON to demos/outputs/krk_persistent_visualization.json
"""

import argparse
from collections import deque
import chess
import sys
from pathlib import Path
import random
from typing import Callable, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger
from recon_lite.graph import NodeState, Graph, LinkType
from demos.shared.krk_network import build_krk_network, create_random_krk_board
from recon_lite_chess import (
    create_krk_root,
    create_phase0_establish_cut, create_phase0_choose_moves,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_king_drive_moves, create_box_shrink_moves,
    create_opposition_moves, create_mate_moves,
    # New confinement-aware nodes
    create_confinement_evaluator, create_barrier_ready_evaluator,
    create_confinement_moves, create_barrier_placement_moves
)
from recon_lite_chess.krk_nodes import wire_default_krk
from recon_lite_chess.predicates import (
    dist_to_edge,
    on_rim,
    has_opposition_after,
    chebyshev,
    box_area,
    box_min_side,
    would_cause_threefold,
)

PHASE_SEQUENCE = ["phase0", "phase1", "phase2", "phase3", "phase4"]
PHASE_SCRIPT_IDS = {
    "phase0": "phase0_establish_cut",
    "phase1": "phase1_drive_to_edge",
    "phase2": "phase2_shrink_box",
    "phase3": "phase3_take_opposition",
    "phase4": "phase4_deliver_mate",
}
PHASE_PREFIXES = {
    "phase0": "p0_",
    "phase1": "p1_",
    "phase2": "p2_",
    "phase3": "p3_",
    "phase4": "p4_",
}


def _build_basic_krk_graph() -> Graph:
    """
    Legacy helper no longer used; kept for reference. Prefer build_krk_network().
    """
    return build_krk_network()


# ---- Deterministic arbiter helpers (demo-local, no library changes) ----

def _find_our_rook_sq(board: chess.Board) -> chess.Square | None:
    color = board.turn
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == chess.ROOK:
            return sq
    return None

def _rook_safe_now(board: chess.Board, rook_sq: chess.Square) -> bool:
    # Safe if enemy king cannot capture rook next move, or our king can immediately recapture
    color = board.turn
    enemy = not color
    ek = board.king(enemy)
    ok = board.king(color)
    if ek is None or ok is None or rook_sq is None:
        return False
    if chebyshev(ek, rook_sq) > 1:
        return True
    b = board.copy(stack=False)
    b.turn = enemy
    cap = chess.Move(ek, rook_sq)
    if cap in b.legal_moves:
        return chebyshev(ok, rook_sq) <= 1
    return True

def _cut_established(board: chess.Board) -> bool:
    """
    Conservative 'cut' heuristic: rook aligned with BK (file/rank), at least 2 away,
    rook not droppable now, and our king is reasonably close to rook or BK.
    """
    color = board.turn
    ek = board.king(not color)
    ok = board.king(color)
    rsq = _find_our_rook_sq(board)
    if ek is None or ok is None or rsq is None:
        return False
    if on_rim(ek):
        return True
    same_file = chess.square_file(rsq) == chess.square_file(ek)
    same_rank = chess.square_rank(rsq) == chess.square_rank(ek)
    aligned = same_file or same_rank
    far_enough = chebyshev(rsq, ek) >= 2
    safe = _rook_safe_now(board, rsq)
    support_ok = chebyshev(ok, ek) <= 3 or chebyshev(ok, rsq) <= 3
    return aligned and far_enough and safe and support_ok

def _can_deliver_mate_now(board: chess.Board) -> bool:
    for mv in board.legal_moves:
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False

def _eligible_phase(board: chess.Board) -> str:
    """
    Centralized eligible phase selection from board features only.
    Highest-to-lowest precedence: 4 → 3 → 2 → 1 → 0.
    """
    enemy = not board.turn
    ek = board.king(enemy)
    if ek is None:
        return "phase0"
    if _can_deliver_mate_now(board):
        return "phase4"
    try:
        if box_min_side(board) <= 1:
            return "phase3"
    except Exception:
        pass
    if on_rim(ek):
        for mv in board.legal_moves:
            if has_opposition_after(board, mv):
                return "phase3"
    if dist_to_edge(ek) == 0:
        return "phase2"
    if _cut_established(board):
        return "phase1"
    return "phase0"


# ---- Phase-aware proposal validation ----

def _prime_phase(graph: Graph, target_phase: str, min_index: int = 0) -> None:
    target_phase = target_phase.lower()
    encountered_target = False
    for idx, phase in enumerate(PHASE_SEQUENCE):
        script_id = PHASE_SCRIPT_IDS.get(phase)
        if script_id not in graph.nodes:
            continue
        prefix = PHASE_PREFIXES.get(phase, "")
        script_node = graph.nodes[script_id]

        if idx < min_index:
            script_node.state = NodeState.CONFIRMED
            for nid, node in graph.nodes.items():
                if prefix and nid.startswith(prefix):
                    node.state = NodeState.CONFIRMED
            continue

        if phase == target_phase:
            encountered_target = True
            script_node.state = NodeState.REQUESTED
            for nid, node in graph.nodes.items():
                if prefix and nid.startswith(prefix):
                    node.state = NodeState.REQUESTED if nid.endswith("_move") else NodeState.INACTIVE
            # Ensure downstream phases are reset
        elif not encountered_target:
            script_node.state = NodeState.CONFIRMED
            for nid, node in graph.nodes.items():
                if prefix and nid.startswith(prefix):
                    node.state = NodeState.CONFIRMED
        else:
            script_node.state = NodeState.INACTIVE
            for nid, node in graph.nodes.items():
                if prefix and nid.startswith(prefix):
                    node.state = NodeState.INACTIVE

PHASE_PRIORITY = {"phase0": 0, "phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4}


def _worst_case_metrics(board: chess.Board, move: chess.Move) -> dict:
    b_after = board.copy()
    b_after.push(move)

    initial_area = box_area(board)
    initial_min_side = box_min_side(board)
    new_area = box_area(b_after)
    new_min_side = box_min_side(b_after)

    worst_area = new_area
    worst_min_side = new_min_side
    if b_after.legal_moves:
        for reply in b_after.legal_moves:
            reply_board = b_after.copy()
            reply_board.push(reply)
            area_after_reply = box_area(reply_board)
            min_side_after_reply = box_min_side(reply_board)
            if area_after_reply > worst_area:
                worst_area = area_after_reply
            if min_side_after_reply > worst_min_side:
                worst_min_side = min_side_after_reply

    return {
        "initial_area": initial_area,
        "initial_min_side": initial_min_side,
        "new_area": new_area,
        "new_min_side": new_min_side,
        "worst_area": worst_area,
        "worst_min_side": worst_min_side,
    }


def _validate_phase2_move(board: chess.Board, move_uci: str) -> tuple[bool, dict]:
    move = chess.Move.from_uci(move_uci)
    metrics = _worst_case_metrics(board, move)

    if board.gives_check(move):
        metrics["failure"] = "gives_check"
        return False, metrics

    initial_min_side = metrics["initial_min_side"]
    worst_min_side = metrics["worst_min_side"]
    initial_area = metrics["initial_area"]
    worst_area = metrics["worst_area"]

    if worst_min_side > initial_min_side:
        metrics["failure"] = "min_side_regresses"
        return False, metrics
    if initial_min_side > 1 and worst_min_side >= initial_min_side:
        metrics["failure"] = "min_side_not_reduced"
        return False, metrics
    if initial_area > 1 and worst_area >= initial_area:
        metrics["failure"] = "area_not_reduced"
        return False, metrics

    metrics["failure"] = None
    return True, metrics


def _detect_proposing_phase(engine: ReConEngine, move_uci: str) -> tuple[str | None, str | None]:
    phase_name = None
    reason = None
    for node in engine.g.nodes.values():
        suggested = node.meta.get("suggested_moves")
        if suggested and move_uci in suggested:
            phase_name = node.meta.get("phase")
            reason = node.meta.get("reason")
            break
    return phase_name, reason


def _select_candidate(board: chess.Board, proposals: list[dict], debug_logger: RunLogger | None) -> tuple[dict | None, list[dict]]:
    if not proposals:
        return None, proposals

    ordered = sorted(proposals, key=lambda p: p.get("rank", -1), reverse=True)
    for candidate in ordered:
        phase = candidate.get("phase")
        if phase == "phase2":
            ok, metrics = _validate_phase2_move(board, candidate["move"])
            candidate["validation"] = metrics
            if not ok:
                if debug_logger:
                    debug_logger.snapshot(
                        engine=None,
                        note=f"Rejected phase2 move {candidate['move']} (not shrinking)",
                        env={
                            "failure": metrics.get("failure"),
                            "initial_min_side": metrics.get("initial_min_side"),
                            "worst_min_side": metrics.get("worst_min_side"),
                            "initial_area": metrics.get("initial_area"),
                            "worst_area": metrics.get("worst_area"),
                        },
                        thoughts="Phase2 validation failed",
                        new_requests=[],
                    )
                continue
        return candidate, ordered

    return None, ordered


def _decision_cycle(engine: ReConEngine,
                    board: chess.Board,
                    env: dict,
                    *,
                    tick_watchdog: int,
                    min_decision_ticks: int,
                    viz_logger: RunLogger | None,
                    debug_logger: RunLogger | None,
                    plies: int) -> tuple[dict | None, list[dict], int]:
    env["board"] = board
    env["chosen_move"] = None
    proposals: list[dict] = []
    ticks = 0
    min_index = 3 if env.get("stage", 0) >= 1 else 0

    while ticks < tick_watchdog and not board.is_game_over():
        ticks += 1
        now_req = engine.step(env)

        if viz_logger is not None or debug_logger is not None:
            dbg = {}
            if env.get("debug_phase1"):
                dbg["debug_phase1"] = env["debug_phase1"]
            if env.get("debug_phase2"):
                dbg["debug_phase2"] = env["debug_phase2"]

            view_env = {
                "fen": board.fen(),
                "evaluation_tick": ticks,
                "ply": plies + 1,
                "chosen_move": env.get("chosen_move"),
            }
            if "pressure" in env:
                view_env["pressure"] = env.get("pressure")
            if "require_min_side_shrink" in env:
                view_env["require_min_side_shrink"] = env.get("require_min_side_shrink")

            if viz_logger is not None and ticks == 1 and not viz_logger.events:
                viz_logger.attach_graph([
                    {"src": e.src, "dst": e.dst, "type": e.ltype.name}
                    for e in engine.g.edges
                ])

            if viz_logger is not None:
                viz_logger.snapshot(
                    engine=engine,
                    note=f"Persistent eval tick {ticks} (ply {plies+1})",
                    env=view_env,
                    thoughts="Persistent evaluation...",
                    new_requests=list(now_req.keys()) if now_req else [],
                )

            if debug_logger is not None:
                debug_payload = dict(view_env)
                debug_payload.update(dbg)
                debug_logger.snapshot(
                    engine=engine,
                    note=f"Persistent eval tick {ticks} (ply {plies+1})",
                    env=debug_payload,
                    thoughts="Persistent evaluation...",
                    new_requests=list(now_req.keys()) if now_req else [],
                )

        proposed_move = env.get("chosen_move")
        if proposed_move:
            phase_name, reason = _detect_proposing_phase(engine, proposed_move)
            rank = PHASE_PRIORITY.get(phase_name or "", -1)
            if rank < min_index:
                env["chosen_move"] = None
                continue
            proposals.append({
                "move": proposed_move,
                "phase": phase_name or "unknown",
                "rank": rank,
                "reason": reason or env.get("last_reason"),
            })
            env["chosen_move"] = None

        if ticks >= min_decision_ticks and proposals:
            break
        if ticks >= min_decision_ticks and not proposals and not now_req:
            break

    selected, ordered = _select_candidate(board, proposals, debug_logger)
    if debug_logger is not None and proposals:
        debug_logger.snapshot(
            engine=None,
            note="decision_proposals",
            env={
                "ply": plies + 1,
                "proposals": proposals,
            },
            thoughts="Collected proposals",
            new_requests=[],
        )
    return selected, ordered, ticks


def _update_stage(env: dict, board: chess.Board) -> int:
    stage = env.get("stage", 0)
    try:
        if stage < 1:
            enemy_sq = board.king(not board.turn)
            if enemy_sq is not None and on_rim(enemy_sq):
                stage = 1
            elif box_min_side(board) <= 1:
                stage = 1
    except Exception:
        pass
    env["stage"] = stage
    return stage


def _leg2_choose(board: chess.Board, env: dict) -> tuple[Optional[dict], list[dict]]:
    """Prefer phase4/phase3 moves once the rim is secured and log candidates."""
    from recon_lite_chess.actuators import (
        choose_move_phase4,
        choose_move_phase3,
        choose_move_phase1,
    )

    proposals: list[dict] = []
    selected: Optional[dict] = None
    seen: set[str] = set()

    def run_chooser(phase_name: str, chooser, *, reason: str, alias_phase: Optional[str] = None) -> None:
        nonlocal selected
        try:
            move_uci = chooser(board, env)
        except Exception:
            move_uci = None
        if not move_uci or move_uci in seen:
            return
        seen.add(move_uci)
        phase_tag = alias_phase or phase_name
        rank = PHASE_PRIORITY.get(phase_tag, PHASE_PRIORITY.get(phase_name, 0))
        record = {
            "move": move_uci,
            "phase": phase_tag,
            "rank": rank,
            "reason": reason,
        }
        proposals.append(record)
        if selected is None:
            selected = record

    run_chooser(
        "phase4",
        choose_move_phase4,
        reason="Leg2: mate execution attempt",
    )
    if selected:
        return selected, proposals

    run_chooser(
        "phase3",
        choose_move_phase3,
        reason="Leg2: opposition tightening",
    )
    if selected:
        return selected, proposals

    # Last resort: reuse phase1 heuristics to maintain tempo but keep us in phase3.
    fen_hist = env.get("fen_history") if isinstance(env, dict) else None

    def append_tempo_from_phase1() -> None:
        try:
            tempo_move = choose_move_phase1(board, env)
        except Exception:
            tempo_move = None
        if not tempo_move:
            return
        if env.get("leg2_last_move") == tempo_move:
            return
        if fen_hist and would_cause_threefold(board, chess.Move.from_uci(tempo_move), fen_hist):
            return
        run_chooser(
            "phase1",
            lambda _board, _env: tempo_move,
            reason="Leg2: tempo assist via phase1 heuristics",
            alias_phase="phase3",
        )

    append_tempo_from_phase1()
    if selected and isinstance(env, dict):
        env["leg2_last_move"] = selected["move"]

    return selected, proposals

def play_persistent_game(initial_fen: str | None = None,
                         max_plies: int = 200,
                         tick_watchdog: int = 300,
                         *,
                         split_logs: bool = True,
                         output_basename: str = "krk_persistent",
                         skip_opponent: bool = False,
                         single_phase: Optional[str] = None,
                         seed: Optional[int] = None,
                         step_mode: bool = False,
                         opponent_policy: Optional[Callable[[chess.Board], Optional[chess.Move]]] = None) -> dict:
    if split_logs:
        viz_logger = RunLogger()
        debug_logger = RunLogger()
    else:
        viz_logger = debug_logger = RunLogger()

    if seed is not None:
        random.seed(seed)
    if initial_fen:
        board = chess.Board(initial_fen)
    else:
        # create_random_krk_board returns a FEN string; wrap into a Board
        board = chess.Board(create_random_krk_board(white_to_move=True))

    g = build_krk_network()
    engine = ReConEngine(g)
    root_id = "krk_root"
    g.nodes[root_id].state = NodeState.REQUESTED

    if single_phase:
        phase_key = single_phase.lower()
        if phase_key not in PHASE_SEQUENCE:
            raise ValueError(f"Unknown phase '{single_phase}'")
        _prime_phase(g, phase_key)
        single_phase = phase_key

    # Attach graph edges for visualization
    plies = 0
    rook_lost = False
    # Persistent env across plies to maintain fen history and pressure
    env = {"board": board, "chosen_move": None, "fen_history": deque(maxlen=12), "pressure_steps": 0, "stage": 0}

    while not board.is_game_over() and plies < max_plies:
        stage = _update_stage(env, board)
        min_index = 3 if stage >= 1 else 0
        leg2_mode = (stage >= 1 and not single_phase)
        if leg2_mode:
            selected, ordered = _leg2_choose(board, env)
            phase_tag = selected["phase"] if selected else PHASE_SEQUENCE[min_index]
            if phase_tag not in PHASE_SEQUENCE:
                phase_tag = PHASE_SEQUENCE[min_index]
            _prime_phase(g, phase_tag, min_index=min_index)
            ticks = 1 if ordered else 0
            move_record = selected
            move_uci = selected["move"] if selected else None
            if viz_logger is not None and selected:
                viz_logger.snapshot(
                    engine=engine,
                    note="Leg2 proposal",
                    env={
                        "fen": board.fen(),
                        "ply": plies + 1,
                        "leg2": True,
                    },
                    thoughts="Leg2 direct proposal",
                    new_requests=[],
                )
            if debug_logger is not None and ordered:
                debug_logger.snapshot(
                    engine=None,
                    note="decision_proposals",
                    env={"ply": plies + 1, "proposals": ordered},
                    thoughts="Collected leg2 proposals",
                    new_requests=[],
                )
        else:
            phase_tag = single_phase or _eligible_phase(board)
            target_index = PHASE_PRIORITY.get(phase_tag, 0)
            if target_index < min_index:
                phase_tag = PHASE_SEQUENCE[min_index]
            _prime_phase(g, phase_tag, min_index=min_index)
            local_watchdog = min(tick_watchdog, 60)
            selected, ordered, ticks = _decision_cycle(
                engine,
                board,
                env,
                tick_watchdog=local_watchdog,
                min_decision_ticks=3,
                viz_logger=viz_logger,
                debug_logger=debug_logger,
                plies=plies,
            )
            move_record = selected
            move_uci = move_record["move"] if move_record else None

        if not move_uci:
            from recon_lite_chess.actuators import choose_any_safe_move
            fallback = choose_any_safe_move(board)
            if fallback:
                move_record = {
                    "move": fallback,
                    "phase": "fallback",
                    "rank": -1,
                    "reason": "safety fallback",
                }
                move_uci = fallback
                if viz_logger is not None:
                    viz_logger.snapshot(
                        engine=None,
                        note=f"FALLBACK applied: {fallback}",
                        env={"fen": board.fen(), "ply": plies + 1, "fallback": True},
                        thoughts="No acceptable proposal; applying fallback",
                        new_requests=[],
                    )
                if debug_logger is not None and debug_logger is not viz_logger:
                    debug_logger.snapshot(
                        engine=None,
                        note=f"FALLBACK applied: {fallback}",
                        env={"fen": board.fen(), "ply": plies + 1, "fallback": True},
                        thoughts="No acceptable proposal; applying fallback",
                        new_requests=[],
                    )

        if move_uci:
            try:
                board.push_uci(move_uci)
            except Exception:
                break
            plies += 1

            if move_record and move_record.get("validation") and debug_logger is not None:
                debug_logger.snapshot(
                    engine=None,
                    note=f"Phase2 validation for {move_uci}",
                    env={**move_record["validation"], "ply": plies},
                    thoughts="Recorded shrink metrics",
                    new_requests=[],
                )

            if not any(p.piece_type == chess.ROOK and p.color == chess.WHITE for p in board.piece_map().values()):
                rook_lost = True

            if viz_logger is not None:
                viz_logger.snapshot(
                    engine=None,
                    note=f"Applied move {plies}: {move_uci}",
                    env={"fen": board.fen(), "ply": plies, "recons_move": move_uci},
                    thoughts=f"Applied {move_uci} (persistent)",
                    new_requests=[],
                )
            if debug_logger is not None and debug_logger is not viz_logger:
                debug_logger.snapshot(
                    engine=None,
                    note=f"Applied move {plies}: {move_uci}",
                    env={"fen": board.fen(), "ply": plies, "recons_move": move_uci},
                    thoughts=f"Applied {move_uci} (persistent)",
                    new_requests=[],
                )

            if board.is_game_over() or plies >= max_plies:
                break

            if step_mode:
                break

            if not skip_opponent and not board.is_game_over():
                opp_move_obj = None
                if opponent_policy is not None:
                    candidate = opponent_policy(board.copy())
                    if isinstance(candidate, chess.Move):
                        opp_move_obj = candidate
                    elif isinstance(candidate, str):
                        try:
                            opp_move_obj = chess.Move.from_uci(candidate)
                        except ValueError:
                            opp_move_obj = None
                if opp_move_obj is None:
                    opp_candidates = list(board.legal_moves)
                    if opp_candidates:
                        opp_move_obj = random.choice(opp_candidates)
                if opp_move_obj is not None and opp_move_obj in board.legal_moves:
                    board.push(opp_move_obj)
                    opp_uci = opp_move_obj.uci()
                    if viz_logger is not None:
                        viz_logger.snapshot(
                            engine=None,
                            note=f"Opponent ply {plies}: {opp_uci}",
                            env={"fen": board.fen(), "ply": plies, "opponents_move": opp_uci},
                            thoughts="Opponent move (persistent)",
                            new_requests=[],
                        )
                    if debug_logger is not None and debug_logger is not viz_logger:
                        debug_logger.snapshot(
                            engine=None,
                            note=f"Opponent ply {plies}: {opp_uci}",
                            env={"fen": board.fen(), "ply": plies, "opponents_move": opp_uci},
                            thoughts="Opponent move (persistent)",
                            new_requests=[],
                        )
            if board.is_game_over() or plies >= max_plies:
                break

            if step_mode:
                break

    # Reset only terminals (evaluators/actuators) and preserve confirmed phase states
    # This was overly aggressive earlier and ruined the whole graph: 
    # Per ReCoN engine (_request_child_if_ready): A node (e.g., PHASE1) is only REQUESTED if all POR predecessors (e.g., PHASE0) are TRUE/CONFIRMED (_all_por_predecessors_true
    phase_ids = ["ROOT", "PHASE0", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
    for n in g.nodes.values():
        if n.nid not in phase_ids and n.state == NodeState.CONFIRMED:
            n.state = NodeState.INACTIVE  # Reset confirmed terminals
        elif n.state not in (NodeState.CONFIRMED, NodeState.INACTIVE):
            n.state = NodeState.INACTIVE  # Reset transient states of non-phase nodes

    # Re-arm evaluators (clear meta for re-evaluation)
    for eval_id in ["king_at_edge", "box_can_shrink", "can_take_opposition", "can_deliver_mate"]:
        if eval_id in g.nodes:
            n = g.nodes[eval_id]
            n.meta.clear()  # Reset meta (e.g., caches) for fresh evaluation

    # Re-arm wait gate to detect new FEN (if present)
    if "wait_for_board_change" in g.nodes:
        g.nodes["wait_for_board_change"].meta.pop("last_fen", None)

    # Re-REQUEST root to trigger next cycle with persisted phase states
    g.nodes[root_id].state = NodeState.REQUESTED

    result = {
        "plies": plies,
        "checkmate": board.is_checkmate(),
        "stalemate": board.is_stalemate(),
        "rook_lost": rook_lost,
        "final_fen": board.fen(),
    }

    out_dir = Path("demos/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_logs and debug_logger is not viz_logger:
        viz_path = out_dir / f"{output_basename}_viz.json"
        debug_path = out_dir / f"{output_basename}_debug.json"
        viz_logger.to_json(str(viz_path))
        debug_logger.to_json(str(debug_path))
    else:
        combined_path = out_dir / f"{output_basename}_visualization.json"
        viz_logger.to_json(str(combined_path))

    return result


def preview_decision(board: chess.Board,
                     *,
                     tick_watchdog: int = 60,
                     min_decision_ticks: int = 3,
                     target_phase: str | None = "phase2") -> dict:
    """Run a single decision cycle without applying the move (test helper)."""
    g = build_krk_network()
    engine = ReConEngine(g)
    root_id = "krk_root"
    g.nodes[root_id].state = NodeState.REQUESTED

    env = {"board": board, "chosen_move": None, "fen_history": deque(maxlen=12), "pressure_steps": 0, "stage": 0}
    if target_phase:
        _prime_phase(g, target_phase, min_index=0)
    else:
        stage = _update_stage(env, board)
        min_index = 3 if stage >= 1 else 0
        phase_tag = _eligible_phase(board)
        phase_idx = PHASE_PRIORITY.get(phase_tag, 0)
        if phase_idx < min_index:
            phase_tag = PHASE_SEQUENCE[min_index]
        _prime_phase(g, phase_tag, min_index=min_index)

    decision, proposals, ticks = _decision_cycle(
        engine,
        board,
        env,
        tick_watchdog=tick_watchdog,
        min_decision_ticks=min_decision_ticks,
        viz_logger=None,
        debug_logger=None,
        plies=0,
    )
    return {"decision": decision, "proposals": proposals, "ticks": ticks}


def run_batch(n_games: int = 10, max_plies: int = 200, **play_kwargs) -> dict:
    stats = {
        "games": [],
        "mates": 0,
        "stalls": 0,
        "rook_losses": 0,
        "total_mate_plies": 0,
        "avg_mate_length": None,
    }
    for i in range(n_games):
        res = play_persistent_game(initial_fen=None, max_plies=max_plies, **play_kwargs)
        stats["games"].append(res)
        if res.get("checkmate"):
            stats["mates"] += 1
            stats["total_mate_plies"] += res.get("plies", 0)
        if res.get("rook_lost"):
            stats["rook_losses"] += 1
        # No explicit stall flag in persistent; watchdog fallback is logged only
    if stats["mates"]:
        stats["avg_mate_length"] = stats["total_mate_plies"]/stats["mates"]
    print(stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", type=str, default="", help="Optional FEN to start from")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies")
    parser.add_argument("--batch", type=int, default=0, help="Run N games in batch mode")
    parser.add_argument("--combined-log", action="store_true", help="Write a single combined visualization log")
    parser.add_argument("--output-basename", type=str, default="krk_persistent", help="Base name for output logs")
    parser.add_argument("--skip-opponent", action="store_true", help="Disable opponent replies (useful for debugging)")
    parser.add_argument("--single-phase", choices=PHASE_SEQUENCE, help="Lock the network to a single phase")
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG for reproducible runs")
    parser.add_argument("--step-mode", action="store_true", help="Stop after each ReCoN move without opponent response")
    # Single graph; demo uses the shared KRK network
    args = parser.parse_args()

    if args.batch and args.batch > 0:
        run_batch(
            args.batch,
            max_plies=args.max_plies,
            split_logs=not args.combined_log,
            output_basename=args.output_basename,
            skip_opponent=args.skip_opponent,
            single_phase=args.single_phase,
            seed=args.seed,
            step_mode=args.step_mode,
        )
    else:
        start_fen = args.fen if args.fen else "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
        res = play_persistent_game(
            initial_fen=start_fen,
            max_plies=args.max_plies,
            split_logs=not args.combined_log,
            output_basename=args.output_basename,
            skip_opponent=args.skip_opponent,
            single_phase=args.single_phase,
            seed=args.seed,
            step_mode=args.step_mode,
        )
        print(res)


if __name__ == "__main__":
    main()
