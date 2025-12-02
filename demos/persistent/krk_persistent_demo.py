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
import chess.engine
import sys
from pathlib import Path
import random
from typing import Callable, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger
from recon_lite.graph import NodeState, Graph, LinkType
from recon_lite.core.activations import ActivationState, activation_snapshot
from recon_lite.time.microtick import MicrotickConfig
from recon_lite.binding.manager import BindingInstance, BindingTable
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
from recon_lite_chess.strategy import (
    compute_phase_logits,
    ensure_phase_states,
    neutral_outcome_mode,
    neutral_style_bias,
    phase_latents_from_logits,
)
from recon_lite_chess.actuators_blend import choose_blended_move
from recon_lite_chess.predicates import (
    dist_to_edge,
    on_rim,
    has_opposition_after,
    chebyshev,
    box_area,
    box_min_side,
    enemy_nearest_edge_info,
    would_cause_threefold,
)
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from recon_lite.plasticity import (
    PlasticityConfig,
    init_plasticity_state,
    update_eligibility,
    apply_fast_update,
    reset_episode as reset_plasticity_episode,
    snapshot_plasticity,
    extract_episode_summary,
)
from recon_lite.plasticity.bandit import (
    BanditConfig,
    BanditPriors,
    init_bandit_state,
    init_bandit_state_with_priors,
    choose_child,
    assign_reward,
    reset_bandit_episode,
    snapshot_bandit,
    load_priors as load_bandit_priors,
    save_priors as save_bandit_priors,
    export_priors as export_bandit_priors,
    merge_priors as merge_bandit_priors,
)
from recon_lite.plasticity.modulation import (
    ModulationConfig,
    compute_modulators,
)
from recon_lite.plasticity.consolidate import (
    ConsolidationConfig,
    ConsolidationEngine,
)
from recon_lite_chess.eval.heuristic import (
    eval_position,
    compute_reward_tick,
    eval_position_stockfish,
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

DEFAULT_MICROTICK_STEPS = 5
DEFAULT_MICROTICK_ETA = 0.3
DEFAULT_LATENT_LOG_STRIDE = 1

# M3 plasticity defaults
DEFAULT_PLASTICITY_ETA = 0.05
DEFAULT_PLASTICITY_R_MAX = 2.0
DEFAULT_PLASTICITY_W_MIN = 0.1
DEFAULT_PLASTICITY_W_MAX = 3.0
DEFAULT_PLASTICITY_LAMBDA = 0.8

# M3 bandit defaults
DEFAULT_BANDIT_C_EXPLORE = 1.0

# M4 consolidation defaults
DEFAULT_CONSOLIDATE_ETA = 0.01
DEFAULT_CONSOLIDATE_MIN_EPISODES = 10
DEFAULT_CONSOLIDATE_OUTCOME_WEIGHT = 0.5
DEFAULT_CONSOLIDATE_MAX_BASE_DELTA = 0.5

# KRK edges eligible for fast plasticity (POR edges within phases)
KRK_PLASTICITY_EDGES = [
    # Phase internal POR edges (check -> move -> wait sequences)
    ("p0_check", "p0_move", LinkType.POR),
    ("p0_move", "p0_wait", LinkType.POR),
    ("p1_check", "p1_move", LinkType.POR),
    ("p1_move", "p1_wait", LinkType.POR),
    ("p2_check", "p2_move", LinkType.POR),
    ("p2_move", "p2_wait", LinkType.POR),
    ("p3_check", "p3_move", LinkType.POR),
    ("p3_move", "p3_wait", LinkType.POR),
    ("p4_check", "p4_move", LinkType.POR),
    ("p4_move", "p4_wait", LinkType.POR),
    # Phase sequencing POR edges
    ("phase0_establish_cut", "phase1_drive_to_edge", LinkType.POR),
    ("phase1_drive_to_edge", "phase2_shrink_box", LinkType.POR),
    ("phase2_shrink_box", "phase3_take_opposition", LinkType.POR),
    ("phase3_take_opposition", "phase4_deliver_mate", LinkType.POR),
]

# KRK bandit parents and their children (for UCB selection)
KRK_BANDIT_PARENTS = {
    "p1_move": ["king_drive_moves", "confinement_moves", "barrier_placement_moves"],
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


def _square_token(square: chess.Square | None) -> str | None:
    if square is None:
        return None
    return f"square:{chess.square_name(square)}"


def _line_tokens(axis: str, index: int) -> list[str]:
    tokens: list[str] = []
    if axis == "file":
        for rank in range(8):
            tokens.append(_square_token(chess.square(index, rank)))
    else:
        for file in range(8):
            tokens.append(_square_token(chess.square(file, index)))
    return [tok for tok in tokens if tok is not None]


def _box_corner_tokens(enemy_sq: chess.Square | None) -> list[str]:
    if enemy_sq is None:
        return []
    ef, er = chess.square_file(enemy_sq), chess.square_rank(enemy_sq)
    distance = dist_to_edge(enemy_sq)
    min_file = max(0, ef - distance)
    max_file = min(7, ef + distance)
    min_rank = max(0, er - distance)
    max_rank = min(7, er + distance)
    corners = [
        chess.square(min_file, min_rank),
        chess.square(min_file, max_rank),
        chess.square(max_file, min_rank),
        chess.square(max_file, max_rank),
    ]
    return [token for token in (_square_token(sq) for sq in corners) if token is not None]


def _update_binding_table(table: BindingTable, board: chess.Board) -> dict:
    table.invalidate_on_board_change(board)
    color = board.turn
    our_king = board.king(color)
    enemy_king = board.king(not color)
    rook_sq = _find_our_rook_sq(board)

    with table.begin_tick("krk/core/kings") as session:
        if our_king is not None:
            session.reserve(BindingInstance("our_king", {_square_token(our_king)}))
        if enemy_king is not None:
            session.reserve(BindingInstance("enemy_king", {_square_token(enemy_king)}))

    with table.begin_tick("krk/p1/drive") as session:
        if rook_sq is not None:
            session.reserve(BindingInstance("rook_anchor", {_square_token(rook_sq)}))
        if enemy_king is not None:
            session.reserve(BindingInstance("target_enemy", {_square_token(enemy_king)}))

    with table.begin_tick("krk/p2/shrink") as session:
        if enemy_king is not None:
            try:
                fence = enemy_nearest_edge_info(board, enemy_king)
                line_tokens = _line_tokens(fence["axis"], fence["target_line"])
                if line_tokens:
                    session.reserve(BindingInstance("target_fence", set(line_tokens)))
            except Exception:
                pass
            corner_tokens = _box_corner_tokens(enemy_king)
            if corner_tokens:
                session.reserve(BindingInstance("box_corners", set(corner_tokens)))

    return table.snapshot()


def _make_microtick_config(
    board: chess.Board,
    env: dict,
    phase_states: dict,
    *,
    steps: int,
    eta: float,
) -> MicrotickConfig:
    temperature = float(env.get("phase_temperature", 1.4))

    def _compute_targets(states: dict) -> dict:
        logits = compute_phase_logits(board)
        env["phase_logits"] = logits
        targets = phase_latents_from_logits(logits, temperature=temperature)
        for key in targets.keys():
            if key not in states:
                states[key] = ActivationState()
        env["phase_targets"] = targets
        return targets

    return MicrotickConfig(
        states=phase_states,
        compute_targets=_compute_targets,
        steps=max(0, steps),
        eta=float(eta),
        history=False,
    )


def _force_phase_targets(
    board: chess.Board,
    env: dict,
    phase_states: dict,
    temperature: float,
) -> None:
    logits = compute_phase_logits(board)
    env["phase_logits"] = logits
    targets = phase_latents_from_logits(logits, temperature=temperature)
    env["phase_targets"] = targets
    for key, target in targets.items():
        state = phase_states.setdefault(key, ActivationState())
        state.value = target
        state.target = target

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
                    plies: int,
                    phase_states: dict,
                    binding_table: BindingTable,
                    phase_microticks: int,
                    phase_eta: float,
                    latent_log_stride: int,
                    use_blended_actuator: bool,
                    # M3 plasticity/bandit parameters
                    plasticity_state: Optional[dict] = None,
                    plasticity_config: Optional[PlasticityConfig] = None,
                    bandit_state: Optional[dict] = None,
                    bandit_config: Optional[BanditConfig] = None,
                    modulation_config: Optional[ModulationConfig] = None,
                    last_eval: Optional[float] = None,
                    sf_engine: Optional["chess.engine.SimpleEngine"] = None,
                    eval_mode: str = "heuristic") -> tuple[dict | None, list[dict], int, Optional[float]]:
    env["board"] = board
    env["chosen_move"] = None
    proposals: list[dict] = []
    ticks = 0
    min_index = 3 if env.get("stage", 0) >= 1 else 0
    ensure_phase_states(phase_states)

    # M3: Track eval for reward computation
    current_eval = last_eval

    while ticks < tick_watchdog and not board.is_game_over():
        ticks += 1
        env.pop("blended_candidates", None)
        binding_snapshot = _update_binding_table(binding_table, board)
        env["binding"] = binding_snapshot

        # M3: Compute eval before tick (for reward computation)
        if plasticity_config and plasticity_config.enabled:
            if current_eval is None:
                if eval_mode == "stockfish" and sf_engine is not None:
                    current_eval = eval_position_stockfish(board, sf_engine)
                else:
                    current_eval = eval_position(board)

        if phase_microticks > 0:
            env["microticks"] = _make_microtick_config(
                board,
                env,
                phase_states,
                steps=phase_microticks,
                eta=phase_eta,
            )
        else:
            _force_phase_targets(
                board,
                env,
                phase_states,
                temperature=float(env.get("phase_temperature", 1.4)),
            )
            env.pop("microticks", None)

        now_req = engine.step(env)
        phase_latent_values = activation_snapshot(phase_states)
        env["phase_latents"] = phase_latent_values

        # M3: Compute reward and apply plasticity updates
        reward_tick = None
        if plasticity_config and plasticity_config.enabled and plasticity_state:
            # Compute new eval
            if eval_mode == "stockfish" and sf_engine is not None:
                new_eval = eval_position_stockfish(board, sf_engine)
            else:
                new_eval = eval_position(board)

            if current_eval is not None:
                reward_tick = compute_reward_tick(current_eval, new_eval, plasticity_config.r_max)

                # Build fired_edges from current node states
                fired_edges = []
                for e in engine.g.edges:
                    src_node = engine.g.nodes.get(e.src)
                    dst_node = engine.g.nodes.get(e.dst)
                    if src_node and dst_node:
                        # Edge "fired" if both nodes are active/confirmed
                        if src_node.state in (NodeState.TRUE, NodeState.CONFIRMED, NodeState.WAITING):
                            if dst_node.state in (NodeState.REQUESTED, NodeState.WAITING, NodeState.TRUE, NodeState.CONFIRMED):
                                fired_edges.append({"src": e.src, "dst": e.dst, "ltype": e.ltype.name})

                # Update eligibility traces
                update_eligibility(plasticity_state, fired_edges, plasticity_config.lambda_decay)

                # Compute modulators from goal_vector
                goal_vector = env.get("goal_vector", {})
                if not goal_vector:
                    # Build a simple goal_vector from board
                    goal_vector = {"phase_progress": phase_latent_values.get("phase4", 0.0)}
                modulators = compute_modulators(goal_vector, modulation_config)

                # Apply fast update
                deltas = apply_fast_update(
                    plasticity_state,
                    engine.g,
                    reward_tick,
                    modulators.eta_tick_eff,
                    plasticity_config,
                )

                # Store in env for logging
                env["m3_reward_tick"] = reward_tick
                env["m3_modulators"] = modulators.to_dict()
                if deltas:
                    env["m3_weight_deltas"] = deltas

            current_eval = new_eval

        log_latents = latent_log_stride > 0 and (ticks % latent_log_stride == 0)
        latents_payload = phase_latent_values if log_latents else None
        binding_payload = binding_snapshot if log_latents else None

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
            if binding_payload is not None:
                view_env["binding"] = binding_payload
            if use_blended_actuator and log_latents and env.get("blended_candidates"):
                view_env["blended_candidates"] = env.get("blended_candidates")
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
                latents=latents_payload,
                macro=env.get("macro_frame"),
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
                latents=latents_payload,
                macro=env.get("macro_frame"),
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
    if use_blended_actuator and env.get("phase_latents"):
        blended_move, blended_diag = choose_blended_move(board, env["phase_latents"], env)
        if blended_diag:
            env["blended_candidates"] = blended_diag
        if blended_move:
            selected = next((p for p in proposals if p["move"] == blended_move), selected)
            if selected is None:
                top_phase = blended_diag[0]["phase"] if blended_diag else "blended"
                selected = {
                    "move": blended_move,
                    "phase": top_phase,
                    "rank": PHASE_PRIORITY.get(top_phase, 0),
                    "reason": "blended_selector",
                }
                proposals.append(selected)
            if blended_diag:
                ordered = []
                for entry in blended_diag:
                    base = next((p for p in proposals if p["move"] == entry["move"]), None)
                    record = dict(base) if base else {
                        "move": entry["move"],
                        "phase": entry["phase"],
                        "rank": PHASE_PRIORITY.get(entry["phase"], 0),
                        "reason": "blended_candidate",
                    }
                    record["score"] = entry["score"]
                    record["phase_weight"] = entry["phase_weight"]
                    record["phase_score"] = entry["phase_score"]
                    record["cheap_eval"] = entry["cheap_eval"]
                    ordered.append(record)
    if proposals:
        env_payload = {
            "ply": plies + 1,
            "proposals": proposals,
        }
        if viz_logger is not None:
            viz_logger.snapshot(
                engine=engine,
                note="decision_proposals",
                env=dict(env_payload),
                thoughts="Collected proposals",
                new_requests=[],
                latents=env.get("phase_latents"),
                macro=env.get("macro_frame"),
            )
        if debug_logger is not None and debug_logger is not viz_logger:
            debug_logger.snapshot(
                engine=None,
                note="decision_proposals",
                env=env_payload,
                thoughts="Collected proposals",
                new_requests=[],
                latents=env.get("phase_latents"),
                macro=env.get("macro_frame"),
            )
    return selected, ordered, ticks, current_eval


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

"""
Main function for playing a persistent game. This is where we start and "run" the network. 
"""

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
                         stockfish_path: Optional[str] = None,
                         stockfish_depth: int = 2,
                         opponent_policy: Optional[Callable[[chess.Board], Optional[chess.Move]]] = None,
                         log_full_state: bool = False,
                         disable_leg2: bool = False,
                         phase_microticks: int = DEFAULT_MICROTICK_STEPS,
                         phase_eta: float = DEFAULT_MICROTICK_ETA,
                         phase_temperature: float = 1.4,
                         latent_log_stride: int = DEFAULT_LATENT_LOG_STRIDE,
                         use_blended_actuator: bool = False,
                         trace_db: Optional["TraceDB"] = None,
                         trace_episode_id: Optional[str] = None,
                         pack_paths: Optional[list[Path]] = None,
                         # M3 plasticity parameters
                         plasticity_enabled: bool = False,
                         plasticity_eta: float = DEFAULT_PLASTICITY_ETA,
                         plasticity_r_max: float = DEFAULT_PLASTICITY_R_MAX,
                         plasticity_lambda: float = DEFAULT_PLASTICITY_LAMBDA,
                         # M3 bandit parameters
                         bandit_enabled: bool = False,
                         bandit_c_explore: float = DEFAULT_BANDIT_C_EXPLORE,
                         # M3 eval mode
                         eval_mode: str = "heuristic",
                         # M4 consolidation parameters
                         consolidation_enabled: bool = False,
                         consolidation_pack: Optional[Path] = None,
                         bandit_priors_path: Optional[Path] = None,
                         consolidation_eta: float = DEFAULT_CONSOLIDATE_ETA,
                         consolidation_min_episodes: int = DEFAULT_CONSOLIDATE_MIN_EPISODES,
                         consolidation_engine: Optional[ConsolidationEngine] = None) -> dict:
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

    # Arguments done, construct the graph. Basically use the preset krk_network (moved to separate file)
    # and use it to create an engine. Set the root node to REQUESTED. And we are ready to roll.. 
    g = build_krk_network()
    engine = ReConEngine(g)
    root_id = "krk_root"
    g.nodes[root_id].state = NodeState.REQUESTED

    if viz_logger is not None:
        viz_logger.attach_graph([
            {"src": e.src, "dst": e.dst, "type": e.ltype.name, "weight": float(getattr(e, "w", 1.0) or 1.0)}
            for e in engine.g.edges
        ])

    if single_phase:
        phase_key = single_phase.lower()
        if phase_key not in PHASE_SEQUENCE:
            raise ValueError(f"Unknown phase '{single_phase}'")
        _prime_phase(g, phase_key)
        single_phase = phase_key

    # Attach graph edges for visualization
    our_color = board.turn
    plies = 0
    rook_lost = False
    total_ticks = 0
    # Persistent env across plies to maintain fen history and pressure

    # Note: Obviously it's a "smarter" solution to just keep the whole board inside the nodes; it's trivial
    # However as for the case with humans, and other more complex environments - the agent/ReCoN has limited 
    # information available to it; the internal world model doesn't always model the whole environment/underlying reality.
    env = {
        "board": board,
        "chosen_move": None,
        "fen_history": deque(maxlen=12),
        "pressure_steps": 0,
        "stage": 0,
    }
    phase_states = ensure_phase_states({})
    binding_table = BindingTable()
    pack_meta = pack_fingerprint(pack_paths or [])
    tick_records: list[TickRecord] = []
    sf_engine = None
    if stockfish_path:
        try:
            sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception:
            sf_engine = None

    env.update({
        "phase_states": phase_states,
        "phase_temperature": phase_temperature,
        "latents_log_stride": latent_log_stride,
        "outcome_mode": neutral_outcome_mode().as_dict(),
        "style_bias": neutral_style_bias().as_dict(),
        "use_blended_actuator": use_blended_actuator,
    })

    # M3: Initialize plasticity state
    plasticity_config = PlasticityConfig(
        eta_tick=plasticity_eta,
        r_max=plasticity_r_max,
        lambda_decay=plasticity_lambda,
        w_min=DEFAULT_PLASTICITY_W_MIN,
        w_max=DEFAULT_PLASTICITY_W_MAX,
        enabled=plasticity_enabled,
    )
    plasticity_state = init_plasticity_state(g, KRK_PLASTICITY_EDGES) if plasticity_enabled else {}

    # M3: Initialize bandit state (with optional M4 priors)
    bandit_config = BanditConfig(
        c_explore=bandit_c_explore,
        enabled=bandit_enabled,
    )
    bandit_priors: Optional[BanditPriors] = None
    if bandit_enabled and bandit_priors_path and bandit_priors_path.exists():
        try:
            bandit_priors = load_bandit_priors(bandit_priors_path)
        except Exception:
            bandit_priors = None
    if bandit_enabled:
        if bandit_priors:
            bandit_state = init_bandit_state_with_priors(KRK_BANDIT_PARENTS, bandit_priors, prior_weight=0.5)
        else:
            bandit_state = init_bandit_state(KRK_BANDIT_PARENTS)
    else:
        bandit_state = {}

    # M3: Modulation config
    modulation_config = ModulationConfig(
        eta_base=plasticity_eta,
        c_explore_base=bandit_c_explore,
    )

    # M4: Initialize or use provided consolidation engine
    consolidation_config = ConsolidationConfig(
        eta_consolidate=consolidation_eta,
        min_episodes=consolidation_min_episodes,
        enabled=consolidation_enabled,
    )
    if consolidation_engine is not None:
        consol_engine = consolidation_engine
    elif consolidation_enabled:
        consol_engine = ConsolidationEngine(consolidation_config)
        # Load existing consolidation state if available
        if consolidation_pack and consolidation_pack.exists():
            try:
                consol_engine.load_state(consolidation_pack)
            except Exception:
                pass
        # Initialize from graph
        consol_engine.init_from_graph(g)
        # Apply w_base to graph at game start
        consol_engine.apply_w_base_to_graph(g)
    else:
        consol_engine = None

    # M3: Track last eval for reward computation
    last_eval: Optional[float] = None
    episode_reward_sum = 0.0

    def _log_snapshot(*, note: str, env_payload: dict, thoughts: str = "", new_requests=None, include_engine: bool = False):
        if viz_logger is None:
            return
        viz_logger.snapshot(
            engine=engine if include_engine else None,
            note=note,
            env=env_payload,
            thoughts=thoughts,
            new_requests=new_requests or [],
            latents=env.get("phase_latents"),
            macro=env.get("macro_frame"),
        )

    while not board.is_game_over() and plies < max_plies:
        if board.turn != our_color:
            if skip_opponent:
                board.turn = our_color
                continue
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
                _log_snapshot(
                    note=f"Opponent ply {plies}: {opp_uci}",
                    env_payload={"fen": board.fen(), "ply": plies, "opponents_move": opp_uci},
                    thoughts="Opponent move (persistent)",
                    include_engine=log_full_state,
                )
                if debug_logger is not None and debug_logger is not viz_logger:
                    debug_logger.snapshot(
                        engine=None,
                        note=f"Opponent ply {plies}: {opp_uci}",
                        env={"fen": board.fen(), "ply": plies, "opponents_move": opp_uci},
                        thoughts="Opponent move (persistent)",
                        new_requests=[],
                        latents=env.get("phase_latents"),
                        macro=env.get("macro_frame"),
                    )
            else:
                # No legal opponent move; treat as finished
                break
            if board.is_game_over():
                break
            continue
        stage = _update_stage(env, board)
        min_index = 3 if stage >= 1 else 0
        leg2_mode = (stage >= 1 and not single_phase and not disable_leg2)
        ticks = 0
        if leg2_mode:
            selected, ordered = _leg2_choose(board, env)
            phase_tag = selected["phase"] if selected else PHASE_SEQUENCE[min_index]
            if phase_tag not in PHASE_SEQUENCE:
                phase_tag = PHASE_SEQUENCE[min_index]
            _prime_phase(g, phase_tag, min_index=min_index)
            _force_phase_targets(board, env, phase_states, phase_temperature)
            env["binding"] = _update_binding_table(binding_table, board)
            env["phase_latents"] = activation_snapshot(phase_states)
            ticks = 1 if ordered else 0
            move_record = selected
            move_uci = selected["move"] if selected else None
            if viz_logger is not None and selected:
                _log_snapshot(
                    note="Leg2 proposal",
                    env_payload={"fen": board.fen(), "ply": plies + 1, "leg2": True},
                    thoughts="Leg2 direct proposal",
                    include_engine=log_full_state,
                )
            if debug_logger is not None and ordered:
                debug_logger.snapshot(
                    engine=None,
                    note="decision_proposals",
                    env={"ply": plies + 1, "proposals": ordered},
                    thoughts="Collected leg2 proposals",
                    new_requests=[],
                    latents=env.get("phase_latents"),
                )
        else:
            phase_tag = single_phase or _eligible_phase(board)
            target_index = PHASE_PRIORITY.get(phase_tag, 0)
            if target_index < min_index:
                phase_tag = PHASE_SEQUENCE[min_index]
            _prime_phase(g, phase_tag, min_index=min_index)
            local_watchdog = min(tick_watchdog, 60)
            selected, ordered, ticks, last_eval = _decision_cycle(
                engine,
                board,
                env,
                tick_watchdog=local_watchdog,
                min_decision_ticks=3,
                viz_logger=viz_logger,
                debug_logger=debug_logger,
                plies=plies,
                phase_states=phase_states,
                binding_table=binding_table,
                phase_microticks=phase_microticks,
                phase_eta=phase_eta,
                latent_log_stride=latent_log_stride,
                use_blended_actuator=use_blended_actuator,
                # M3 plasticity/bandit
                plasticity_state=plasticity_state,
                plasticity_config=plasticity_config,
                bandit_state=bandit_state,
                bandit_config=bandit_config,
                modulation_config=modulation_config,
                last_eval=last_eval,
                sf_engine=sf_engine,
                eval_mode=eval_mode,
            )
            move_record = selected
            move_uci = move_record["move"] if move_record else None
        total_ticks += ticks

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
                _log_snapshot(
                    note=f"FALLBACK applied: {fallback}",
                    env_payload={"fen": board.fen(), "ply": plies + 1, "fallback": True},
                    thoughts="No acceptable proposal; applying fallback",
                    include_engine=log_full_state,
                )
                if debug_logger is not None and debug_logger is not viz_logger:
                    debug_logger.snapshot(
                        engine=None,
                        note=f"FALLBACK applied: {fallback}",
                        env={"fen": board.fen(), "ply": plies + 1, "fallback": True},
                        thoughts="No acceptable proposal; applying fallback",
                        new_requests=[],
                        latents=env.get("phase_latents"),
                        macro=env.get("macro_frame"),
                    )

        if move_uci:
            try:
                eval_before = None
                eval_after = None
                if sf_engine is not None:
                    try:
                        info_before = sf_engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))
                        score_before = info_before.get("score") if info_before else None
                        eval_before = float(score_before.white().score(mate_score=10000) or 0.0) if score_before else None
                    except Exception:
                        eval_before = None
                board.push_uci(move_uci)
                if sf_engine is not None:
                    try:
                        info_after = sf_engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))
                        score_after = info_after.get("score") if info_after else None
                        eval_after = float(score_after.white().score(mate_score=10000) or 0.0) if score_after else None
                    except Exception:
                        eval_after = None
            except Exception:
                break
            plies += 1
            tick_records.append(
                TickRecord(
                    tick_id=len(tick_records) + 1,
                    phase_estimate=None,
                    goal_vector=env.get("goal_vector"),
                    board_fen=board.fen(),
                    active_nodes=[nid for nid, node in engine.g.nodes.items() if node.state != NodeState.INACTIVE],
                    fired_edges=[],
                    action=move_uci,
                    eval_before=eval_before,
                    eval_after=eval_after,
                    reward_tick=(round(eval_after - eval_before, 3) if eval_after is not None and eval_before is not None else None),
                    meta={
                        "ply": plies,
                        "stage": env.get("stage"),
                        "reason": move_record["reason"] if isinstance(move_record, dict) else None,
                    },
                )
            )
            _force_phase_targets(board, env, phase_states, phase_temperature)
            env["binding"] = _update_binding_table(binding_table, board)
            env["phase_latents"] = activation_snapshot(phase_states)

            if move_record and move_record.get("validation") and debug_logger is not None:
                debug_logger.snapshot(
                    engine=None,
                    note=f"Phase2 validation for {move_uci}",
                    env={**move_record["validation"], "ply": plies},
                    thoughts="Recorded shrink metrics",
                    new_requests=[],
                    macro=env.get("macro_frame"),
                )

            if not any(p.piece_type == chess.ROOK and p.color == chess.WHITE for p in board.piece_map().values()):
                rook_lost = True

            _log_snapshot(
                note=f"Applied move {plies}: {move_uci}",
                env_payload={"fen": board.fen(), "ply": plies, "recons_move": move_uci},
                thoughts=f"Applied {move_uci} (persistent)",
                include_engine=log_full_state,
            )
            if debug_logger is not None and debug_logger is not viz_logger:
                debug_logger.snapshot(
                    engine=None,
                    note=f"Applied move {plies}: {move_uci}",
                    env={"fen": board.fen(), "ply": plies, "recons_move": move_uci},
                    thoughts=f"Applied {move_uci} (persistent)",
                    new_requests=[],
                    latents=env.get("phase_latents"),
                    macro=env.get("macro_frame"),
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

    # M4: Extract episode summary for consolidation
    game_result = board.result() if board.is_game_over() else None
    episode_summary = None
    if plasticity_enabled or bandit_enabled:
        episode_summary = extract_episode_summary(
            plasticity_state if plasticity_enabled else None,
            bandit_state if bandit_enabled else None,
            tick_records,
            game_result,
        )

    # M4: Accumulate episode for consolidation
    if consol_engine and episode_summary:
        consol_engine.accumulate_episode(episode_summary)
        # Check if we should apply consolidation
        if consol_engine.should_apply():
            applied_deltas = consol_engine.apply_to_graph(g)
            if applied_deltas:
                result["consolidation_applied"] = len(applied_deltas)
            # Save updated state if pack path provided
            if consolidation_pack:
                try:
                    consol_engine.save_state(consolidation_pack)
                except Exception:
                    pass

    # M4: Update bandit priors if enabled
    if bandit_enabled and bandit_state and bandit_priors_path:
        try:
            new_priors = export_bandit_priors(bandit_state)
            if bandit_priors:
                merged = merge_bandit_priors(bandit_priors, new_priors, decay=0.95)
            else:
                merged = new_priors
            save_bandit_priors(merged, bandit_priors_path)
        except Exception:
            pass

    # M3: Reset plasticity and bandit for next episode
    if plasticity_enabled and plasticity_state:
        reset_plasticity_episode(plasticity_state, g)
    if bandit_enabled and bandit_state:
        reset_bandit_episode(bandit_state)

    if trace_db is not None:
        ep_id = trace_episode_id or f"krk-{seed or 0}"
        ep = EpisodeRecord(
            episode_id=ep_id,
            result=game_result,
            ticks=tick_records,
            pack_meta=pack_meta,
            notes={"plies": plies, "ticks": total_ticks},
            summary=episode_summary,
        )
        trace_db.add_episode(ep)
    if sf_engine is not None:
        try:
            sf_engine.quit()
        except Exception:
            pass

    out_dir = Path("demos/outputs/persistent")
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
    phase_states = ensure_phase_states({})
    binding_table = BindingTable()
    env.update({
        "phase_states": phase_states,
        "phase_temperature": 1.4,
    })
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

    decision, proposals, ticks, _ = _decision_cycle(
        engine,
        board,
        env,
        tick_watchdog=tick_watchdog,
        min_decision_ticks=min_decision_ticks,
        viz_logger=None,
        debug_logger=None,
        plies=0,
        phase_states=phase_states,
        binding_table=binding_table,
        phase_microticks=DEFAULT_MICROTICK_STEPS,
        phase_eta=DEFAULT_MICROTICK_ETA,
        latent_log_stride=DEFAULT_LATENT_LOG_STRIDE,
        use_blended_actuator=False,
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
    parser.add_argument("--tick-watchdog", type=int, default=300, help="Maximum ticks per decision cycle")
    parser.add_argument("--batch", type=int, default=0, help="Run N games in batch mode")
    parser.add_argument("--combined-log", action="store_true", help="Write a single combined visualization log")
    parser.add_argument("--output-basename", type=str, default="krk_persistent", help="Base name for output logs")
    parser.add_argument("--skip-opponent", action="store_true", help="Disable opponent replies (useful for debugging)")
    parser.add_argument("--single-phase", choices=PHASE_SEQUENCE, help="Lock the network to a single phase")
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG for reproducible runs")
    parser.add_argument("--step-mode", action="store_true", help="Stop after each ReCoN move without opponent response")
    parser.add_argument("--log-full-state", action="store_true", help="Include node states on every snapshot (slower, best for visualization)")
    parser.add_argument("--disable-leg2", action="store_true", help="Force full decision cycles even in late-game leg2 situations")
    parser.add_argument("--phase-microticks", type=int, default=DEFAULT_MICROTICK_STEPS, help="Number of micro-ticks to run before each engine step")
    parser.add_argument("--phase-eta", type=float, default=DEFAULT_MICROTICK_ETA, help="Smoothing factor for micro-ticks")
    parser.add_argument("--phase-temperature", type=float, default=1.4, help="Softmax temperature for phase latents")
    parser.add_argument("--latent-log-stride", type=int, default=DEFAULT_LATENT_LOG_STRIDE, help="Log latents/bindings every N ticks")
    parser.add_argument("--use-blended-actuator", action="store_true", help="Enable phase-weighted blended move chooser")
    parser.add_argument("--engine", type=str, default=None, help="Path to Stockfish for eval/reward logging (optional)")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth when scoring moves")
    parser.add_argument("--trace-out", type=Path, default=None, help="Optional JSONL trace output (EpisodeRecord/TickRecord).")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="Weight pack path(s) to fingerprint in traces.")
    # M3 plasticity arguments
    parser.add_argument("--plasticity", action="store_true", help="Enable M3 fast plasticity (within-game edge weight adaptation)")
    parser.add_argument("--plasticity-eta", type=float, default=DEFAULT_PLASTICITY_ETA, help="Base learning rate for plasticity updates")
    parser.add_argument("--plasticity-r-max", type=float, default=DEFAULT_PLASTICITY_R_MAX, help="Reward clipping bound for plasticity")
    parser.add_argument("--plasticity-lambda", type=float, default=DEFAULT_PLASTICITY_LAMBDA, help="Eligibility trace decay factor")
    # M3 bandit arguments
    parser.add_argument("--bandit", action="store_true", help="Enable M3 bandit control (UCB selection among sibling scripts)")
    parser.add_argument("--bandit-c-explore", type=float, default=DEFAULT_BANDIT_C_EXPLORE, help="Exploration coefficient for UCB")
    # M3 eval mode
    parser.add_argument("--eval-mode", choices=["heuristic", "stockfish"], default="heuristic", help="Evaluation mode for reward computation")
    # M4 consolidation arguments
    parser.add_argument("--consolidate", action="store_true", help="Enable M4 slow consolidation (cross-game weight updates)")
    parser.add_argument("--consolidate-pack", type=Path, default=None, help="Path to load/save consolidation state")
    parser.add_argument("--bandit-priors", type=Path, default=None, help="Path to load/save bandit priors")
    parser.add_argument("--consolidate-eta", type=float, default=DEFAULT_CONSOLIDATE_ETA, help="Consolidation learning rate")
    parser.add_argument("--consolidate-min-episodes", type=int, default=DEFAULT_CONSOLIDATE_MIN_EPISODES, help="Minimum episodes before consolidation")
    # Single graph; demo uses the shared KRK network
    args = parser.parse_args()

    if args.batch and args.batch > 0:
        # M4: Create shared consolidation engine for batch mode
        consol_engine = None
        if args.consolidate:
            consol_config = ConsolidationConfig(
                eta_consolidate=args.consolidate_eta,
                min_episodes=args.consolidate_min_episodes,
                enabled=True,
            )
            consol_engine = ConsolidationEngine(consol_config)
            if args.consolidate_pack and args.consolidate_pack.exists():
                try:
                    consol_engine.load_state(args.consolidate_pack)
                except Exception:
                    pass

        run_batch(
            args.batch,
            max_plies=args.max_plies,
            tick_watchdog=args.tick_watchdog,
            split_logs=not args.combined_log,
            output_basename=args.output_basename,
            skip_opponent=args.skip_opponent,
            single_phase=args.single_phase,
            seed=args.seed,
            step_mode=args.step_mode,
            log_full_state=args.log_full_state,
            disable_leg2=args.disable_leg2,
            phase_microticks=args.phase_microticks,
            phase_eta=args.phase_eta,
            phase_temperature=args.phase_temperature,
            latent_log_stride=args.latent_log_stride,
            use_blended_actuator=args.use_blended_actuator,
            stockfish_path=args.engine,
            stockfish_depth=args.depth,
            # M3 plasticity/bandit
            plasticity_enabled=args.plasticity,
            plasticity_eta=args.plasticity_eta,
            plasticity_r_max=args.plasticity_r_max,
            plasticity_lambda=args.plasticity_lambda,
            bandit_enabled=args.bandit,
            bandit_c_explore=args.bandit_c_explore,
            eval_mode=args.eval_mode,
            # M4 consolidation
            consolidation_enabled=args.consolidate,
            consolidation_pack=args.consolidate_pack,
            bandit_priors_path=args.bandit_priors,
            consolidation_eta=args.consolidate_eta,
            consolidation_min_episodes=args.consolidate_min_episodes,
            consolidation_engine=consol_engine,
        )

        # M4: Save final consolidation state after batch
        if consol_engine and args.consolidate_pack:
            try:
                consol_engine.save_state(args.consolidate_pack)
            except Exception:
                pass
    else:
        start_fen = args.fen if args.fen else "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
        trace_db = TraceDB(args.trace_out) if args.trace_out else None
        res = play_persistent_game(
            initial_fen=start_fen,
            max_plies=args.max_plies,
            tick_watchdog=args.tick_watchdog,
            split_logs=not args.combined_log,
            output_basename=args.output_basename,
            skip_opponent=args.skip_opponent,
            single_phase=args.single_phase,
            seed=args.seed,
            step_mode=args.step_mode,
            log_full_state=args.log_full_state,
            disable_leg2=args.disable_leg2,
            phase_microticks=args.phase_microticks,
            phase_eta=args.phase_eta,
            phase_temperature=args.phase_temperature,
            latent_log_stride=args.latent_log_stride,
            use_blended_actuator=args.use_blended_actuator,
            trace_db=trace_db,
            trace_episode_id="krk-cli-run",
            pack_paths=args.pack,
            stockfish_path=args.engine,
            stockfish_depth=args.depth,
            # M3 plasticity/bandit
            plasticity_enabled=args.plasticity,
            plasticity_eta=args.plasticity_eta,
            plasticity_r_max=args.plasticity_r_max,
            plasticity_lambda=args.plasticity_lambda,
            bandit_enabled=args.bandit,
            bandit_c_explore=args.bandit_c_explore,
            eval_mode=args.eval_mode,
            # M4 consolidation
            consolidation_enabled=args.consolidate,
            consolidation_pack=args.consolidate_pack,
            bandit_priors_path=args.bandit_priors,
            consolidation_eta=args.consolidate_eta,
            consolidation_min_episodes=args.consolidate_min_episodes,
        )
        if trace_db:
            trace_db.flush()
        print(res)


if __name__ == "__main__":
    main()
