#!/usr/bin/env python3
"""Full Game Training Script with Plasticity, Consolidation, and Stem Cells.

This script trains the full game network using:
- Fast plasticity (per-tick weight updates)
- Slow consolidation (cross-game weight updates)
- Stem cell pattern discovery
- Stockfish evaluation for reward signals

Usage:
    # Basic training (10 games)
    uv run python demos/persistent/full_game_train.py --batch 10

    # Full training with all features
    uv run python demos/persistent/full_game_train.py \
        --batch 100 \
        --plasticity \
        --consolidate \
        --consolidate-pack weights/nightly/fullgame_consol.json \
        --stem-cells \
        --stem-cell-path weights/nightly/stem_cells.json \
        --engine /usr/games/stockfish

    # Quick test
    uv run python demos/persistent/full_game_train.py --batch 5 --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import chess
import chess.engine

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.graph import Graph, Node, NodeType, NodeState, LinkType
from recon_lite.engine import ReConEngine
from recon_lite.logger import RunLogger
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint, EpisodeSummary
from recon_lite.plasticity import (
    PlasticityConfig,
    init_plasticity_state,
    update_eligibility,
    apply_fast_update,
    reset_episode as reset_plasticity_episode,
    extract_episode_summary,
)
from recon_lite.plasticity.consolidate import (
    ConsolidationConfig,
    ConsolidationEngine,
)
from recon_lite.nodes.stem_cell import StemCellManager, StemCellConfig
from recon_lite.dynamics.persistence import (
    PersistenceConfig,
    apply_persistence_to_node,
    get_active_plans,
)
from recon_lite_chess.goals.ultimate import (
    UltimateGoal,
    assess_ultimate_goal,
    create_ultimate_goal_node,
)
from recon_lite_chess.goals.strategic import (
    STRATEGIC_PLANS,
    get_active_plans_for_goal,
)
from recon_lite_chess.sensors.material import (
    assess_material,
    create_material_sensor_node,
)
from recon_lite_chess.sensors.phase import (
    estimate_phase,
    create_phase_sensor_node,
)
from recon_lite_chess.scripts.opening import (
    get_opening_move_candidates,
    development_sensor_predicate,
    center_control_sensor_predicate,
)
from recon_lite_chess.scripts.middlegame import (
    get_middlegame_move_candidates,
    king_safety_sensor_predicate,
    piece_activity_sensor_predicate,
)
from recon_lite_chess.scripts.tactics import (
    detect_forks,
    detect_hanging_pieces,
    detect_back_rank_weakness,
    get_fork_moves,
    get_capture_hanging_moves,
    get_back_rank_moves,
)
from recon_lite_chess.eval.heuristic import eval_position, compute_reward_tick
from recon_lite_chess.scripts.kqk import is_kqk_position
from recon_lite_chess.sensors.structure import summarize_kpk_material
from recon_lite_chess.graph import (
    build_unified_graph,
    load_all_weights,
    get_active_edge_traces,
    reset_edge_traces,
)
from tools.graph_snapshot import export_graph_snapshot

# Import the graph builder from full_game_demo
from demos.persistent.full_game_demo import (
    build_full_game_graph,
    select_move,
    extract_board_features,
    GameState,
)


def _normalize_promotion(board: chess.Board, move: Optional[chess.Move]) -> Optional[chess.Move]:
    """
    Default pawn promotion to queen unless an underpromotion uniquely checkmates.
    Prevents accidental stalemates from arbitrary underpromotions.
    """
    if move is None:
        return None
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return move
    
    dest_rank = chess.square_rank(move.to_square)
    if dest_rank not in (0, 7):
        return move
    
    # Already promoting to queen
    if move.promotion == chess.QUEEN:
        return move
    
    queen_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    
    under_mate = False
    queen_mate = False
    
    if move.promotion:
        try:
            board.push(move)
            under_mate = board.is_checkmate()
            board.pop()
        except Exception:
            under_mate = False
    
    try:
        board.push(queen_move)
        queen_mate = board.is_checkmate()
        board.pop()
    except Exception:
        queen_mate = False
    
    if under_mate and not queen_mate:
        return move
    
    if queen_move in board.legal_moves:
        return queen_move
    return move


def _would_be_insufficient_material(board: chess.Board, move: chess.Move) -> bool:
    """Check if making the move results in insufficient material (and not mate)."""
    try:
        board.push(move)
        bad = board.is_insufficient_material() and not board.is_checkmate()
        board.pop()
        return bad
    except Exception:
        return False


def _prefer_legal_queen_promo(board: chess.Board, move: chess.Move) -> chess.Move:
    """
    Ensure pawn promotions default to queen and avoid moves that immediately lead
    to insufficient material (unless checkmate).
    """
    normalized = _normalize_promotion(board, move)
    if normalized and not _would_be_insufficient_material(board, normalized):
        return normalized
    
    # Try other legal moves that do not lead to insufficient material
    for candidate in board.legal_moves:
        norm = _normalize_promotion(board, candidate)
        if not _would_be_insufficient_material(board, norm):
            return norm
    # Fallback: return the original normalized move
    return normalized or move


def _queen_hangs_after(board: chess.Board, move: chess.Move) -> bool:
    """
    Returns True if after making the move, our queen is hanging (attacked, not defended)
    and the position is not checkmate.
    """
    try:
        mover_color = board.turn
        board.push(move)
        q_sq = None
        for sq, piece in board.piece_map().items():
            if piece.color == mover_color and piece.piece_type == chess.QUEEN:
                q_sq = sq
                break
        if q_sq is None:
            board.pop()
            return False
        opponent_color = board.turn  # after push, turn flips to opponent
        hanging = board.is_attacked_by(opponent_color, q_sq) and not board.is_attacked_by(mover_color, q_sq)
        bad = hanging and not board.is_checkmate()
        board.pop()
        return bad
    except Exception:
        return False


def _prefer_safe_move(board: chess.Board, move: chess.Move) -> chess.Move:
    """
    Apply promotion normalization and avoid moves that hang the queen or drop to insufficient material.
    """
    normalized = _normalize_promotion(board, move)
    if normalized and not _would_be_insufficient_material(board, normalized) and not _queen_hangs_after(board, normalized):
        return normalized
    
    for candidate in board.legal_moves:
        norm = _normalize_promotion(board, candidate)
        if not _would_be_insufficient_material(board, norm) and not _queen_hangs_after(board, norm):
            return norm
    
    return normalized or move


# Default parameters
DEFAULT_PLASTICITY_ETA = 0.05
DEFAULT_PLASTICITY_R_MAX = 2.0
DEFAULT_PLASTICITY_W_MIN = 0.1
DEFAULT_PLASTICITY_W_MAX = 3.0
DEFAULT_PLASTICITY_LAMBDA = 0.8

DEFAULT_CONSOLIDATE_ETA = 0.01
DEFAULT_CONSOLIDATE_MIN_EPISODES = 10


# =============================================================================
# ENDGAME SENTINELS FOR SUBGRAPH LOCKING
# =============================================================================

def kpk_sentinel(env: Dict[str, Any]) -> bool:
    """
    Sentinel for KPK subgraph: returns True while position is still KPK.
    Returns False when pawn promotes (position becomes KQK) or material changes.
    """
    board = env.get("board")
    if not board:
        return False
    summary = summarize_kpk_material(board)
    return bool(summary.get("is_kpk"))


def kqk_sentinel(env: Dict[str, Any]) -> bool:
    """
    Sentinel for KQK subgraph: returns True while position is still KQK.
    Returns False when queen is captured or game ends.
    """
    board = env.get("board")
    if not board:
        return False
    is_kqk, _ = is_kqk_position(board)
    return is_kqk


def krk_sentinel(env: Dict[str, Any]) -> bool:
    """
    Sentinel for KRK subgraph: returns True while position is still KRK.
    """
    board = env.get("board")
    if not board:
        return False
    # Simple KRK check: exactly 3 pieces, one rook
    pieces = list(board.piece_map().values())
    if len(pieces) != 3:
        return False
    types = [p.piece_type for p in pieces]
    return types.count(chess.KING) == 2 and types.count(chess.ROOK) == 1

_SUBGRAPH_NODE_PREFIXES = (
    "krk_",
    "kpk_",
    "kqk_",
)

_TACTICS_NODE_PREFIXES = (
    "tactic_",
    "detect_",
    "exploit_",
    "protect_",
)


def _is_non_fullgame_node_id(node_id: str) -> bool:
    return node_id.startswith(_SUBGRAPH_NODE_PREFIXES) or node_id.startswith(_TACTICS_NODE_PREFIXES)


def _fullgame_edge_whitelist(graph: Graph) -> List[str]:
    """
    Return edge keys that belong to the 'full game' graph and should be owned by
    the full-game consolidation pack.

    Critical: this excludes endgame subgraphs (kpk/krk/kqk) and tactics, because
    those are trained/saved in their own packs and should not be overwritten by
    full-game consolidation.
    """
    keys: List[str] = []
    for e in graph.edges:
        if e.ltype not in (LinkType.POR, LinkType.SUB):
            continue
        
        # FIX: Whitelist edges that connect TO a subgraph root (the gate).
        # These are "main" -> "subgraph_root" edges and MUST be owned by fullgame.
        if e.dst.endswith("_root"):
             keys.append(f"{e.src}->{e.dst}:{e.ltype.name}")
             continue

        if _is_non_fullgame_node_id(e.src) or _is_non_fullgame_node_id(e.dst):
            continue
        keys.append(f"{e.src}->{e.dst}:{e.ltype.name}")
    return keys


def get_trainable_edges(graph: Graph) -> List[Tuple[str, str, LinkType]]:
    """
    Get ALL edges that should be trained with plasticity.
    
    The whole network learns - not just specific edge types.
    Every edge that is marked for consolidation will be trained.
    """
    edges = []
    for e in graph.edges:
        # Train ALL edges marked for consolidation
        meta = getattr(e, 'meta', {})
        if meta.get("consolidate", True):  # Default to True if not marked
            edges.append((e.src, e.dst, e.ltype))
    return edges


def compute_stockfish_eval(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int = 2) -> Optional[float]:
    """Get Stockfish evaluation in centipawns."""
    try:
        info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score:
            cp = score.white().score(mate_score=10000)
            return float(cp) if cp is not None else None
    except Exception:
        pass
    return None


def play_training_game(
    game_id: int,
    *,
    initial_fen: Optional[str] = None,
    max_moves: int = 200,
    timeout_loss: bool = False,
    vs_random: bool = True,
    verbose: bool = False,
    stockfish_engine: Optional[chess.engine.SimpleEngine] = None,
    stockfish_depth: int = 2,
    # Plasticity
    plasticity_enabled: bool = False,
    plasticity_config: Optional[PlasticityConfig] = None,
    # Consolidation (shared across games)
    consolidation_engine: Optional[ConsolidationEngine] = None,
    # Stem cells
    stem_manager: Optional[StemCellManager] = None,
    # Snapshot hook
    snapshot_hook: Optional[callable] = None,
    debug_draws: bool = False,
    weights_dir: Path = Path("weights/latest"),
) -> Dict[str, Any]:
    """
    Play a single training game.
    
    Returns:
        Dict with game results and training stats
    """
    # Build the unified graph (includes ALL subgraphs: strategic, KRK, KPK, tactics)
    g = build_unified_graph(
        include_endgames=True,
        include_tactics=True,
        include_sensors=True,
    )
    # Apply latest weights (per-subgraph packs) before consolidation
    load_all_weights(g, weights_dir=weights_dir)
    engine = ReConEngine(g)
    
    # Reset all edge traces at start of game
    reset_edge_traces(g)
    
    # Initialize plasticity
    plasticity_state = {}
    if plasticity_enabled:
        if plasticity_config is None:
            plasticity_config = PlasticityConfig(
                eta_tick=DEFAULT_PLASTICITY_ETA,
                r_max=DEFAULT_PLASTICITY_R_MAX,
                lambda_decay=DEFAULT_PLASTICITY_LAMBDA,
                w_min=DEFAULT_PLASTICITY_W_MIN,
                w_max=DEFAULT_PLASTICITY_W_MAX,
                enabled=True,
            )
        trainable_edges = get_trainable_edges(g)
        plasticity_state = init_plasticity_state(g, trainable_edges)
    
    # Apply consolidation weights if available
    if consolidation_engine:
        # IMPORTANT: do not let full-game consolidation overwrite trained endgame/tactics packs.
        consolidation_engine.init_from_graph(g, edge_whitelist=_fullgame_edge_whitelist(g))
        consolidation_engine.apply_w_base_to_graph(g)
    
    # Set up feature extractor for stem cells
    if stem_manager:
        for cell in stem_manager.cells.values():
            cell.feature_extractor = extract_board_features
    
    # Initialize game state (optionally from FEN)
    if initial_fen:
        state = GameState(board=chess.Board(initial_fen))
    else:
        state = GameState(board=chess.Board())
    agent_color = state.board.turn
    state.last_eval = eval_position(state.board)
    
    tick_records: List[TickRecord] = []
    total_reward = 0.0
    weight_deltas_sum = {}
    persistence_config = PersistenceConfig()
    
    if verbose:
        print(f"  Game {game_id}: Starting...")
    
    while not state.board.is_game_over() and len(state.move_history) < max_moves:
        # Request root to start evaluation
        g.nodes["GameRoot"].state = NodeState.REQUESTED
        
        # Create environment
        env = {"board": state.board}
        
        # =====================================================================
        # SUBGRAPH GOAL DELEGATION: Lock into endgame subgraph when detected
        # =====================================================================
        # Check if we should lock into an endgame subgraph
        # Priority: KQK (strongest) > KRK > KPK (weakest)
        if not engine.subgraph_lock:
            is_kqk, kqk_attacker = is_kqk_position(state.board)
            if is_kqk and kqk_attacker == state.board.turn:
                # We have K+Q vs K and it's our turn - lock into KQK
                engine.lock_subgraph("kqk_root", kqk_sentinel)
            elif krk_sentinel(env):
                # We have K+R vs K - lock into KRK
                # (check who has the rook)
                pieces = list(state.board.piece_map().items())
                rook_sq = next((sq for sq, p in pieces if p.piece_type == chess.ROOK), None)
                if rook_sq is not None:
                    rook_color = state.board.piece_at(rook_sq).color
                    if rook_color == state.board.turn:
                        engine.lock_subgraph("krk_root", krk_sentinel)
            elif kpk_sentinel(env):
                # We have K+P vs K - lock into KPK
                kpk_summary = summarize_kpk_material(state.board)
                if kpk_summary.get("attacker_color") == state.board.turn:
                    engine.lock_subgraph("kpk_root", kpk_sentinel)
        
        # Get evaluation before move
        eval_before = None
        if stockfish_engine:
            eval_before = compute_stockfish_eval(state.board, stockfish_engine, stockfish_depth)
        if eval_before is None:
            eval_before = eval_position(state.board)
        
        # Run engine step (if subgraph locked, this runs internal ticks)
        fired_edges = []
        now_requested = engine.step(env)
        state.tick += 1
        
        # Debug logging (uncomment for troubleshooting)
        # if verbose and state.tick <= 3:
        #     lock_info = f"lock={engine.subgraph_lock.subgraph_root}" if engine.subgraph_lock else "no lock"
        #     kpk_policy = env.get("kpk", {}).get("policy", {}).get("suggested_move", "none")
        #     kqk_policy = env.get("kqk", {}).get("policy", {}).get("suggested_move", "none")
        #     print(f"    Tick {state.tick}: {lock_info}, kpk={kpk_policy}, kqk={kqk_policy}")
        
        # Check for endgame policy suggestions (KRK/KPK/KQK) before other logic
        suggested_move_uci = None
        for key in ("kqk", "krk", "kpk"):
            pol = env.get(key, {}).get("policy") if isinstance(env.get(key), dict) else None
            if pol and pol.get("suggested_move"):
                suggested_move_uci = pol["suggested_move"]
                break
        
        move_from_policy = None
        if suggested_move_uci:
            try:
                candidate = chess.Move.from_uci(suggested_move_uci)
                if candidate in state.board.legal_moves:
                    move_from_policy = candidate
            except Exception:
                move_from_policy = None
        
        # Collect fired edges and update traces for FULL NETWORK consolidation
        # (use permissive "active endpoints" heuristic; strict matching yields zero updates)
        for e in g.edges:
            if e.ltype not in (LinkType.POR, LinkType.SUB):
                continue
            src_node = g.nodes.get(e.src)
            dst_node = g.nodes.get(e.dst)
            if src_node and dst_node:
                src_ok = src_node.state in (NodeState.TRUE, NodeState.CONFIRMED, getattr(NodeState, "WAITING", NodeState.CONFIRMED))
                dst_ok = dst_node.state in (NodeState.REQUESTED, NodeState.TRUE, NodeState.CONFIRMED, getattr(NodeState, "WAITING", NodeState.CONFIRMED))
                if src_ok and dst_ok:
                    # Record for plasticity
                    fired_edges.append({"src": e.src, "dst": e.dst, "ltype": e.ltype.name})
                    
                    # Accumulate edge trace for consolidation (learning signal)
                    # This is the key change: ALL edges accumulate traces
                    if not hasattr(e, 'trace'):
                        e.trace = 0.0
                    e.trace += 1.0  # Increment trace when edge fires
        
        # Get assessments
        ultimate = assess_ultimate_goal(state.board, state.board.turn)
        phase = estimate_phase(state.board)
        
        # Update persistence for strategic plans
        goal_plans = get_active_plans_for_goal(ultimate.goal.name, phase.as_dict())
        for plan_id, base_weight in goal_plans:
            if plan_id in g.nodes:
                evidence = base_weight / 2.0
                apply_persistence_to_node(g.nodes[plan_id], evidence, config=persistence_config)
        
        # Get active plans
        active_plans = get_active_plans(g.nodes, layer="strategic", config=persistence_config)
        
        # Check for tactical opportunities
        forks = detect_forks(state.board)
        hanging = detect_hanging_pieces(state.board)
        
        # Prioritize tactical moves (but never override an explicit endgame policy move)
        move = move_from_policy

        if move is None and forks:
            fork_moves = get_fork_moves(state.board)
            if fork_moves:
                move = fork_moves[0]
        elif move is None and hanging.get("enemy_hanging"):
            hanging_moves = get_capture_hanging_moves(state.board)
            if hanging_moves:
                move = hanging_moves[0]
        
        # Fall back to strategic move selection
        if move is None:
            move = select_move(
                state.board,
                ultimate.goal,
                phase.as_dict(),
                active_plans,
            )
        
        # Safety net: choose any legal move if selector returned None
        if move is None:
            legal = list(state.board.legal_moves)
            if not legal:
                break
            # Prefer queen promotion if available when falling back
            promo_moves = [m for m in legal if m.promotion]
            queen_promos = [m for m in promo_moves if m.promotion == chess.QUEEN]
            if queen_promos:
                move = queen_promos[0]
            elif promo_moves:
                move = promo_moves[0]
            else:
                move = legal[0]
            # Mark fallback in tick meta later
            fallback_used = True
        else:
            fallback_used = False
        
        # Normalize promotion choice and avoid insufficient material / hanging queen.
        move = _prefer_safe_move(state.board, move)

        # Extra guard: in pure KQK, never hang the queen.
        in_kqk, attacker_color = is_kqk_position(state.board)
        if in_kqk and attacker_color == state.board.turn and _queen_hangs_after(state.board, move):
            move = _prefer_safe_move(state.board, move)
        
        if move is None:
            break
        
        # Make the move
        state.board.push(move)
        state.move_history.append(move.uci())
        
        # Get evaluation after move
        eval_after = None
        if stockfish_engine:
            eval_after = compute_stockfish_eval(state.board, stockfish_engine, stockfish_depth)
        if eval_after is None:
            eval_after = eval_position(state.board)
        
        # Compute reward
        reward_tick = 0.0
        if eval_before is not None and eval_after is not None:
            # Reward based on evaluation change
            reward_tick = (eval_after - eval_before) / 100.0  # Scale to reasonable range
            reward_tick = max(-2.0, min(2.0, reward_tick))
        
        # Bonus for checkmate
        if state.board.is_checkmate():
            reward_tick = 2.0
        
        # =================================================================
        # PROMOTION DETECTION & SUBGRAPH TRANSITION: KPK → KQK
        # =================================================================
        if move.promotion:
            # Promotion happened! Big bonus for achieving KPK goal
            reward_tick += 1.0
            
            # Transition from KPK to KQK subgraph
            if engine.subgraph_lock and engine.subgraph_lock.subgraph_root == "kpk_root":
                engine.unlock_subgraph(goal_achieved=True)
                # Check if we should lock into KQK
                is_kqk, kqk_attacker = is_kqk_position(state.board)
                # Note: after our move, it's opponent's turn, so we check if we're the attacker
                if is_kqk and kqk_attacker != state.board.turn:  # We just promoted, opponent to move
                    # Don't lock yet - wait until it's our turn again
                    pass
        
        total_reward += reward_tick
        
        # Update plasticity
        if plasticity_enabled and fired_edges:
            update_eligibility(plasticity_state, fired_edges, plasticity_config.lambda_decay)
            deltas = apply_fast_update(
                plasticity_state,
                g,
                reward_tick,
                plasticity_config.eta_tick,
                plasticity_config,
            )
            for k, v in deltas.items():
                weight_deltas_sum[k] = weight_deltas_sum.get(k, 0.0) + v
        
        # Feed stem cells
        if stem_manager:
            stem_manager.tick(state.board, reward_tick, state.tick)
        
        # Record tick
        tick_records.append(TickRecord(
            tick_id=len(tick_records) + 1,
            board_fen=state.board.fen(),
            action=move.uci(),
            active_nodes=[nid for nid, n in g.nodes.items() if n.state != NodeState.INACTIVE],
            eval_before=eval_before,
            eval_after=eval_after,
            reward_tick=round(reward_tick, 4),
            meta={
                "ply": len(state.move_history),
                "fallback": fallback_used,
                "result": state.board.result(claim_draw=True),
            },
        ))
        
        # Opponent's turn
        if not state.board.is_game_over() and vs_random:
            opp_moves = list(state.board.legal_moves)
            if opp_moves:
                opp_move = random.choice(opp_moves)
                state.board.push(opp_move)
                state.move_history.append(opp_move.uci())
    
    # Game over - extract episode summary
    timed_out = (not state.board.is_game_over()) and len(state.move_history) >= max_moves
    if timed_out and timeout_loss:
        # If the agent fails to convert within the move budget, treat as a loss
        game_result = "0-1" if agent_color == chess.WHITE else "1-0"
    else:
        game_result = state.board.result() if state.board.is_game_over() else "*"
    outcome_score = 0.0
    if game_result == "1-0":
        outcome_score = 1.0
    elif game_result == "0-1":
        outcome_score = -1.0
    elif game_result == "1/2-1/2" and debug_draws:
        try:
            draw_claim = state.board.result(claim_draw=True)
        except Exception:
            draw_claim = "n/a"
        print(f"[draw-debug] game {game_id} draw fen={state.board.fen()} claim={draw_claim}")
    elif timed_out and timeout_loss and debug_draws:
        print(f"[timeout-debug] game {game_id} timeout_loss fen={state.board.fen()} plies={len(state.move_history)}")
    
    # Gather ALL active edge traces for consolidation
    # This tracks the entire network, not just specific edges
    active_traces = get_active_edge_traces(g, threshold=0.01)
    
    episode_summary = None
    if plasticity_enabled:
        episode_summary = extract_episode_summary(
            plasticity_state,
            None,  # No bandit state
            tick_records,
            game_result,
        )
    
    # Accumulate for consolidation using ALL edge traces
    if consolidation_engine:
        # Only consolidate edges owned by the full-game pack.
        allowed_edge_keys = set(consolidation_engine.edge_states.keys())

        # Create episode summary with all active edge deltas
        edge_delta_sums = {}
        for edge_key, trace_value in active_traces.items():
            # Scale trace by outcome: winning reinforces active edges, losing penalizes
            if edge_key in allowed_edge_keys:
                edge_delta_sums[edge_key] = trace_value * outcome_score
        
        # If plasticity ran, merge its deltas too
        if episode_summary and hasattr(episode_summary, 'edge_delta_sums'):
            for k, v in episode_summary.edge_delta_sums.items():
                if k in allowed_edge_keys:
                    edge_delta_sums[k] = edge_delta_sums.get(k, 0.0) + v
        
        # Build comprehensive episode summary
        full_summary = EpisodeSummary(
            outcome_score=outcome_score,
            avg_reward_tick=total_reward / max(1, len(tick_records)),
            edge_delta_sums=edge_delta_sums,
        )
        consolidation_engine.accumulate_episode(full_summary)
    
    # Reset plasticity for next game
    if plasticity_enabled and plasticity_state:
        reset_plasticity_episode(plasticity_state, g)
    
    # Get stem cell discoveries
    discoveries = []
    if stem_manager:
        candidates = stem_manager.get_specialization_candidates()
        for cell in candidates:
            if cell.should_specialize():
                result = cell.specialize()
                if result:
                    discoveries.append(result)
    
    result = {
        "game_id": game_id,
        "moves": len(state.move_history),
        "result": game_result,
        "is_checkmate": state.board.is_checkmate(),
        "is_stalemate": state.board.is_stalemate(),
        "total_reward": round(total_reward, 4),
        "avg_reward": round(total_reward / max(1, len(tick_records)), 4),
        "weight_deltas": len(weight_deltas_sum),
        "discoveries": len(discoveries),
        "final_fen": state.board.fen(),
        "tick_records": tick_records,
    }
    
    if verbose:
        win_status = "WIN" if game_result == "1-0" else "LOSS" if game_result == "0-1" else "DRAW"
        print(f"  Game {game_id}: {win_status} in {len(state.move_history)} moves, reward={total_reward:.2f}")
    
    # Optional graph snapshot after game (captures weights/topology as loaded/applied)
    if snapshot_hook:
        try:
            snapshot_hook(g, game_id)
        except Exception:
            pass
    
    return result


def run_batch_training(
    n_games: int,
    *,
    initial_fens: Optional[List[str]] = None,
    max_moves: int = 200,
    timeout_loss: bool = False,
    vs_random: bool = True,
    verbose: bool = True,
    stockfish_path: Optional[str] = None,
    stockfish_depth: int = 2,
    # Plasticity
    plasticity_enabled: bool = False,
    plasticity_eta: float = DEFAULT_PLASTICITY_ETA,
    # Consolidation
    consolidation_enabled: bool = False,
    consolidation_pack: Optional[Path] = None,
    consolidation_eta: float = DEFAULT_CONSOLIDATE_ETA,
    consolidation_min_episodes: int = DEFAULT_CONSOLIDATE_MIN_EPISODES,
    # Stem cells
    stem_cells_enabled: bool = False,
    stem_cell_path: Optional[Path] = None,
    # Trace output
    trace_out: Optional[Path] = None,
    # Snapshots
    snapshot_dir: Optional[Path] = None,
    snapshot_interval: int = 50,
    debug_draws: bool = False,
    # Weights
    weights_dir: Path = Path("weights/latest"),
) -> Dict[str, Any]:
    """
    Run batch training with multiple games.
    
    Returns:
        Aggregate statistics
    """
    # Initialize Stockfish
    sf_engine = None
    if stockfish_path:
        try:
            sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            if verbose:
                print(f"Using Stockfish: {stockfish_path}")
        except Exception as e:
            print(f"Warning: Could not start Stockfish: {e}")
    
    # Initialize plasticity config
    plasticity_config = None
    if plasticity_enabled:
        plasticity_config = PlasticityConfig(
            eta_tick=plasticity_eta,
            r_max=DEFAULT_PLASTICITY_R_MAX,
            lambda_decay=DEFAULT_PLASTICITY_LAMBDA,
            w_min=DEFAULT_PLASTICITY_W_MIN,
            w_max=DEFAULT_PLASTICITY_W_MAX,
            enabled=True,
        )
    
    # Initialize consolidation engine (shared across games)
    consol_engine = None
    if consolidation_enabled:
        consol_config = ConsolidationConfig(
            eta_consolidate=consolidation_eta,
            min_episodes=consolidation_min_episodes,
            enabled=True,
        )
        consol_engine = ConsolidationEngine(consol_config)
        if consolidation_pack and consolidation_pack.exists():
            try:
                consol_engine.load_state(consolidation_pack)
                if verbose:
                    print(f"Loaded consolidation state from {consolidation_pack}")
            except Exception as e:
                print(f"Warning: Could not load consolidation: {e}")
    
    # Initialize stem cell manager (shared across games)
    stem_manager = None
    if stem_cells_enabled:
        stem_config = StemCellConfig(
            min_samples=30,
            max_samples=200,
            reward_threshold=0.2,
            specialization_threshold=0.6,
            exploration_budget=150,
        )
        stem_manager = StemCellManager(
            max_cells=20,
            spawn_rate=0.05,
            config=stem_config,
        )
        if stem_cell_path and stem_cell_path.exists():
            try:
                stem_manager = StemCellManager.load(stem_cell_path)
                if verbose:
                    print(f"Loaded stem cells from {stem_cell_path}")
            except Exception:
                pass
    
    # Initialize trace DB
    trace_db = TraceDB(trace_out) if trace_out else None
    
    # Run games
    stats = {
        "total_games": n_games,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "checkmates": 0,
        "stalemates": 0,
        "total_moves": 0,
        "total_reward": 0.0,
        "discoveries": 0,
        "games": [],
    }
    
    if verbose:
        print(f"\n=== Training {n_games} games ===")
        print(f"Plasticity: {'ON' if plasticity_enabled else 'OFF'}")
        print(f"Consolidation: {'ON' if consolidation_enabled else 'OFF'}")
        print(f"Stem cells: {'ON' if stem_cells_enabled else 'OFF'}")
        print()
    
    for i in range(n_games):
        # Get initial FEN if provided
        game_fen = None
        if initial_fens and i < len(initial_fens):
            game_fen = initial_fens[i]
        
        # Snapshot hook (per game) if enabled
        snapshot_hook = None
        if snapshot_dir and snapshot_interval > 0 and (i % snapshot_interval == 0):
            def _hook(graph, game_id, *, _dir=snapshot_dir):
                export_graph_snapshot(graph, _dir / f"graph_{game_id:04d}.json", meta={"game": game_id})
            snapshot_hook = _hook
        
        result = play_training_game(
            game_id=i + 1,
            initial_fen=game_fen,
            max_moves=max_moves,
            timeout_loss=timeout_loss,
            vs_random=vs_random,
            verbose=verbose,
            stockfish_engine=sf_engine,
            stockfish_depth=stockfish_depth,
            plasticity_enabled=plasticity_enabled,
            plasticity_config=plasticity_config,
            consolidation_engine=consol_engine,
            stem_manager=stem_manager,
            snapshot_hook=snapshot_hook,
            debug_draws=debug_draws,
            weights_dir=weights_dir,
        )
        
        # Save episode to trace if enabled
        if trace_db:
            ep = EpisodeRecord(
                episode_id=f"game-{i+1}",
                result=result.get("result"),
                ticks=result.get("tick_records", []),
                notes={
                    "final_fen": result.get("final_fen"),
                    "moves": result.get("moves"),
                },
            )
            trace_db.add_episode(ep)
        
        # Aggregate stats
        stats["games"].append(result)
        stats["total_moves"] += result["moves"]
        stats["total_reward"] += result["total_reward"]
        stats["discoveries"] += result["discoveries"]
        
        if result["result"] == "1-0":
            stats["wins"] += 1
        elif result["result"] == "0-1":
            stats["losses"] += 1
        else:
            stats["draws"] += 1
        
        if result["is_checkmate"]:
            stats["checkmates"] += 1
        if result["is_stalemate"]:
            stats["stalemates"] += 1
        
        # Apply consolidation periodically
        if consol_engine and consol_engine.should_apply():
            # Build a unified graph to apply consolidation
            g = build_unified_graph(
                include_endgames=True,
                include_tactics=True,
                include_sensors=True,
            )
            applied = consol_engine.apply_to_graph(g)
            if verbose and applied:
                print(f"  [Consolidation] Applied {len(applied)} weight updates to full network")
        
        # Memory management
        gc.collect()
    
    # Final consolidation save
    if consol_engine and consolidation_pack:
        try:
            consolidation_pack.parent.mkdir(parents=True, exist_ok=True)
            consol_engine.save_state(consolidation_pack)
            if verbose:
                print(f"\nSaved consolidation to {consolidation_pack}")
        except Exception as e:
            print(f"Warning: Could not save consolidation: {e}")
    
    # Save stem cells
    if stem_manager and stem_cell_path:
        try:
            stem_cell_path.parent.mkdir(parents=True, exist_ok=True)
            stem_manager.save(stem_cell_path)
            if verbose:
                print(f"Saved stem cells to {stem_cell_path}")
        except Exception as e:
            print(f"Warning: Could not save stem cells: {e}")
    
    # Flush traces
    if trace_db:
        trace_db.flush()
    
    # Cleanup
    if sf_engine:
        sf_engine.quit()
    
    # Compute final stats
    stats["win_rate"] = stats["wins"] / n_games if n_games > 0 else 0
    stats["avg_moves"] = stats["total_moves"] / n_games if n_games > 0 else 0
    stats["avg_reward"] = stats["total_reward"] / n_games if n_games > 0 else 0
    
    if verbose:
        print(f"\n=== Training Complete ===")
        print(f"Win rate: {stats['win_rate']*100:.1f}%")
        print(f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
        print(f"Checkmates: {stats['checkmates']}, Stalemates: {stats['stalemates']}")
        print(f"Avg moves: {stats['avg_moves']:.1f}")
        print(f"Avg reward: {stats['avg_reward']:.2f}")
        print(f"Stem cell discoveries: {stats['discoveries']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Full Game Training with Plasticity and Consolidation")
    
    # Basic options
    parser.add_argument("--batch", type=int, default=10, help="Number of games to train")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    parser.add_argument("--timeout-loss", action="store_true", help="Treat reaching --max-moves without mate as a loss for the agent")
    parser.add_argument("--vs-random", action="store_true", default=True, help="Play against random opponent")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer games, less depth)")
    parser.add_argument("--fen-file", type=Path, help="File with starting FENs (one per line)")
    
    # Stockfish
    parser.add_argument("--engine", type=str, help="Path to Stockfish")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth")
    
    # Plasticity
    parser.add_argument("--plasticity", action="store_true", help="Enable fast plasticity")
    parser.add_argument("--plasticity-eta", type=float, default=DEFAULT_PLASTICITY_ETA)
    
    # Consolidation
    parser.add_argument("--consolidate", action="store_true", help="Enable slow consolidation")
    parser.add_argument("--consolidate-pack", type=Path, help="Path to load/save consolidation state")
    parser.add_argument("--consolidate-eta", type=float, default=DEFAULT_CONSOLIDATE_ETA)
    parser.add_argument("--consolidate-min-episodes", type=int, default=DEFAULT_CONSOLIDATE_MIN_EPISODES)
    
    # Stem cells
    parser.add_argument("--stem-cells", action="store_true", help="Enable stem cell pattern discovery")
    parser.add_argument("--stem-cell-path", type=Path, help="Path to load/save stem cell state")
    
    # Output
    parser.add_argument("--trace-out", type=Path, help="Path for trace output")
    parser.add_argument("--output-json", type=Path, help="Path to save stats JSON")
    
    # Snapshots
    parser.add_argument("--snapshot-dir", type=Path, help="Directory to save graph snapshots")
    parser.add_argument("--snapshot-interval", type=int, default=50, help="Save snapshot every N games (default: 50)")
    parser.add_argument("--debug-draws", action="store_true", help="Print FEN/claim result when a game ends in draw")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights/latest"), help="Directory to load per-subgraph weights (default: weights/latest)")
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.batch = min(args.batch, 5)
        args.max_moves = 100
        args.depth = 1
    
    # Load FENs from file if provided
    initial_fens = None
    if args.fen_file and args.fen_file.exists():
        with open(args.fen_file) as f:
            initial_fens = [line.strip() for line in f if line.strip()]
        # Adjust batch size to match FEN count if FENs are provided
        if initial_fens:
            args.batch = min(args.batch, len(initial_fens))
    
    verbose = not args.quiet
    
    stats = run_batch_training(
        n_games=args.batch,
        initial_fens=initial_fens,
        max_moves=args.max_moves,
        timeout_loss=args.timeout_loss,
        vs_random=args.vs_random,
        verbose=verbose,
        stockfish_path=args.engine,
        stockfish_depth=args.depth,
        plasticity_enabled=args.plasticity,
        plasticity_eta=args.plasticity_eta,
        consolidation_enabled=args.consolidate,
        consolidation_pack=args.consolidate_pack,
        consolidation_eta=args.consolidate_eta,
        consolidation_min_episodes=args.consolidate_min_episodes,
        stem_cells_enabled=args.stem_cells,
        stem_cell_path=args.stem_cell_path,
        trace_out=args.trace_out,
        snapshot_dir=args.snapshot_dir,
        snapshot_interval=args.snapshot_interval,
        debug_draws=args.debug_draws,
        weights_dir=args.weights_dir,
    )
    
    # Save stats if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        # Remove full game list for smaller output
        output_stats = {k: v for k, v in stats.items() if k != "games"}
        output_stats["games_played"] = len(stats["games"])
        with open(args.output_json, "w") as f:
            json.dump(output_stats, f, indent=2)
        if verbose:
            print(f"Stats saved to {args.output_json}")
    
    # Print summary as JSON for parsing
    print(json.dumps({
        "wins": stats["wins"],
        "losses": stats["losses"],
        "draws": stats["draws"],
        "win_rate": round(stats["win_rate"], 4),
        "checkmates": stats["checkmates"],
    }))


if __name__ == "__main__":
    main()

