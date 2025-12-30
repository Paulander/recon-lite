#!/usr/bin/env python3
"""
KQK (King+Queen vs King) Persistent Training Demo.

This demo runs headless training for the KQK endgame subgraph.
Goal: Checkmate the opponent king using King + Queen.

KQK is easier than KRK because the Queen controls more squares.
Expected win rate: 95%+ against random opponent.

Usage:
    # Single game
    python demos/persistent/kqk_persistent_demo.py --fen "8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1"
    
    # Batch training with plasticity
    python demos/persistent/kqk_persistent_demo.py --batch 100 --plasticity --consolidate
    
    # With Stockfish evaluation
    python demos/persistent/kqk_persistent_demo.py --batch 50 --engine /usr/games/stockfish
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite import ReConEngine
from recon_lite.graph import NodeState, LinkType
from recon_lite.logger import RunLogger
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from recon_lite_chess.scripts.kqk import build_kqk_network, create_random_kqk_board
from recon_lite.plasticity import (
    PlasticityConfig,
    init_plasticity_state,
    update_eligibility,
    apply_fast_update,
    reset_episode as reset_plasticity_episode,
    snapshot_plasticity,
    extract_episode_summary,
)
from recon_lite_chess.eval.heuristic import (
    eval_position,
    eval_position_stockfish,
    compute_reward_tick,
    compute_kqk_reward_shaping,
    compute_kqk_efficiency_bonus,
    KQK_STEP_PENALTY,
)
from recon_lite.plasticity.consolidate import (
    ConsolidationConfig,
    ConsolidationEngine,
)
from recon_lite.nodes.stem_cell import (
    StemCellManager,
    StemCellConfig,
)

# Default configuration
DEFAULT_PLASTICITY_ETA = 0.05
DEFAULT_PLASTICITY_R_MAX = 2.0
DEFAULT_PLASTICITY_W_MIN = 0.1
DEFAULT_PLASTICITY_W_MAX = 3.0
DEFAULT_PLASTICITY_LAMBDA = 0.8
DEFAULT_CONSOLIDATE_ETA = 0.01
DEFAULT_CONSOLIDATE_MIN_EPISODES = 10


def _collect_fired_edges(g) -> list[dict]:
    """Collect fired POR/SUB edges based on active endpoint node states."""
    waiting = getattr(NodeState, "WAITING", None)
    src_states = {NodeState.TRUE, NodeState.CONFIRMED}
    dst_states = {NodeState.REQUESTED, NodeState.TRUE, NodeState.CONFIRMED}
    if waiting is not None:
        src_states.add(waiting)
        dst_states.add(waiting)

    fired = []
    for e in g.edges:
        if e.ltype not in (LinkType.POR, LinkType.SUB):
            continue
        src_node = g.nodes.get(e.src)
        dst_node = g.nodes.get(e.dst)
        if not src_node or not dst_node:
            continue
        if src_node.state in src_states and dst_node.state in dst_states:
            fired.append({"src": e.src, "dst": e.dst, "ltype": e.ltype.name})
    return fired


def _edge_keys_in_graph(g) -> set[str]:
    keys: set[str] = set()
    for e in g.edges:
        if e.ltype not in (LinkType.POR, LinkType.SUB):
            continue
        keys.add(f"{e.src}->{e.dst}:{e.ltype.name}")
    return keys


def _prune_consolidation_to_graph(consol_engine: ConsolidationEngine, g) -> None:
    """
    Prevent stale/foreign edges from persisting inside a reused consolidation pack.

    Consolidation packs can get "polluted" (e.g., if a full-game trainer wrote
    into a KQK pack). Because ConsolidationEngine doesn't auto-prune, we do it
    here so `weights/nightly/kqk_consol.json` only contains edges that exist in
    the KQK graph.
    """
    allowed = _edge_keys_in_graph(g)
    consol_engine.edge_states = {k: v for k, v in consol_engine.edge_states.items() if k in allowed}


def play_persistent_game(
    initial_fen: str | None = None,
    *,
    max_plies: int = 150,
    max_ticks_per_move: int = 200,
    split_logs: bool = True,
    output_basename: str = "kqk_persistent",
    stockfish_path: Optional[str] = None,
    stockfish_depth: int = 2,
    trace_db: Optional[TraceDB] = None,
    trace_episode_id: Optional[str] = None,
    pack_paths: Optional[list[Path]] = None,
    # Plasticity parameters
    plasticity_enabled: bool = False,
    plasticity_eta: float = DEFAULT_PLASTICITY_ETA,
    plasticity_r_max: float = DEFAULT_PLASTICITY_R_MAX,
    plasticity_lambda: float = DEFAULT_PLASTICITY_LAMBDA,
    # Consolidation parameters
    consolidation_enabled: bool = False,
    consolidation_pack: Optional[Path] = None,
    consolidation_eta: float = DEFAULT_CONSOLIDATE_ETA,
    consolidation_min_episodes: int = DEFAULT_CONSOLIDATE_MIN_EPISODES,
    consolidation_engine: Optional[ConsolidationEngine] = None,
    # M8: Stem cell parameters
    stem_cell_manager: Optional[StemCellManager] = None,
) -> dict:
    """
    Play a single KQK game to checkmate.
    
    Args:
        initial_fen: Starting position (generates random if None)
        max_plies: Maximum plies before timeout
        max_ticks_per_move: Maximum ticks to find a move
        stockfish_path: Path to Stockfish for position evaluation
        trace_db: TraceDB for recording episodes
        plasticity_enabled: Enable M3 fast plasticity
        consolidation_enabled: Enable M4 consolidation
        
    Returns:
        dict with plies, result, checkmate status, final_fen
    """
    viz_logger = RunLogger()
    debug_logger = RunLogger() if split_logs else viz_logger

    # Use provided FEN or generate random KQK position
    if initial_fen:
        board = chess.Board(initial_fen)
    else:
        board = chess.Board(create_random_kqk_board(white_to_move=True))
    
    g = build_kqk_network()
    eng = ReConEngine(g)
    g.nodes["kqk_root"].state = NodeState.REQUESTED

    viz_logger.attach_graph([
        {"src": e.src, "dst": e.dst, "type": e.ltype.name, "weight": float(getattr(e, "w", 1.0) or 1.0)}
        for e in eng.g.edges
    ])

    tick_records: list[TickRecord] = []
    pack_meta = pack_fingerprint(pack_paths or [])
    plies = 0
    sf_engine = None
    if stockfish_path:
        try:
            sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception:
            sf_engine = None

    # Initialize plasticity
    plasticity_config = PlasticityConfig(
        eta_tick=plasticity_eta,
        r_max=plasticity_r_max,
        lambda_decay=plasticity_lambda,
        w_min=DEFAULT_PLASTICITY_W_MIN,
        w_max=DEFAULT_PLASTICITY_W_MAX,
        enabled=plasticity_enabled,
    )
    kqk_plasticity_edges = []
    if plasticity_enabled:
        for e in g.edges:
            if e.ltype in (LinkType.POR, LinkType.SUB):
                kqk_plasticity_edges.append((e.src, e.dst, e.ltype))
    plasticity_state = init_plasticity_state(g, kqk_plasticity_edges) if plasticity_enabled else {}

    # Initialize consolidation
    consolidation_config = ConsolidationConfig(
        eta_consolidate=consolidation_eta,
        min_episodes=consolidation_min_episodes,
        enabled=consolidation_enabled,
    )
    if consolidation_engine is not None:
        consol_engine = consolidation_engine
        # Ensure the shared engine is initialized for this graph and applied at game start
        _prune_consolidation_to_graph(consol_engine, g)
        consol_engine.init_from_graph(g)
        consol_engine.apply_w_base_to_graph(g)
    elif consolidation_enabled:
        consol_engine = ConsolidationEngine(consolidation_config)
        if consolidation_pack and consolidation_pack.exists():
            try:
                consol_engine.load_state(consolidation_pack)
            except Exception:
                pass
        _prune_consolidation_to_graph(consol_engine, g)
        consol_engine.init_from_graph(g)
        consol_engine.apply_w_base_to_graph(g)
    else:
        consol_engine = None

    # Track last evaluation for reward computation (pawn units)
    last_eval: Optional[float] = None

    # Main game loop - play until checkmate, stalemate, or max_plies
    while not board.is_game_over() and plies < max_plies:
        env = {"board": board}
        chosen = None
        move_ticks = 0
        fired_edges: list[dict] = []
        
        # Get move from network
        while move_ticks < max_ticks_per_move and chosen is None:
            move_ticks += 1
            now_req = eng.step(env)
            fired_edges = _collect_fired_edges(eng.g)
            if plasticity_enabled and plasticity_state:
                update_eligibility(plasticity_state, fired_edges, plasticity_config.lambda_decay)
            chosen = env.get("kqk", {}).get("policy", {}).get("suggested_move") if isinstance(env.get("kqk"), dict) else None
            viz_logger.snapshot(
                engine=eng,
                note=f"tick {move_ticks}",
                env={"fen": board.fen(), "ply": plies + 1},
                new_requests=list(now_req.keys()),
            )
        
        if not chosen:
            break
        
        # Evaluate position and make move
        try:
            eval_before = (
                eval_position_stockfish(board, sf_engine, depth=stockfish_depth)
                if sf_engine is not None
                else eval_position(board)
            )
            board.push_uci(chosen)
            eval_after = (
                eval_position_stockfish(board, sf_engine, depth=stockfish_depth)
                if sf_engine is not None
                else eval_position(board)
            )
        except Exception:
            break
        
        plies += 1
        last_eval = eval_after

        reward_tick = compute_reward_tick(eval_before, eval_after, plasticity_r_max)
        
        # === KQK REWARD SHAPING ===
        # Add progress-based rewards for edge/restriction/distance
        # Note: We're the attacker (white) with K+Q, playing against random
        board_before_copy = board.copy()
        board_before_copy.pop()  # Undo our move to get board_before
        kqk_shaping = compute_kqk_reward_shaping(board_before_copy, board, chess.WHITE)
        reward_tick += kqk_shaping
        
        # Step penalty to encourage efficient play
        reward_tick -= KQK_STEP_PENALTY
        
        # Checkmate bonus with efficiency scaling
        if board.is_checkmate():
            reward_tick = compute_kqk_efficiency_bonus(plies, plasticity_r_max, optimal_plies=20)

        if plasticity_enabled and plasticity_state:
            deltas = apply_fast_update(
                plasticity_state,
                g,
                reward_tick,
                plasticity_config.eta_tick,
                plasticity_config,
            )
            if deltas:
                env["m3_weight_deltas"] = deltas

        tick_records.append(
            TickRecord(
                tick_id=len(tick_records) + 1,
                board_fen=board.fen(),
                action=chosen,
                active_nodes=[nid for nid, node in eng.g.nodes.items() if node.state != NodeState.INACTIVE],
                eval_before=eval_before,
                eval_after=eval_after,
                reward_tick=round(reward_tick, 4),
                meta={"ply": plies},
            )
        )
        
        # M8: Feed stem cells with significant rewards
        if stem_cell_manager is not None:
            if abs(reward_tick) > 0.2:
                stem_cell_manager.tick(board, reward_tick, len(tick_records))
        
        viz_logger.snapshot(
            engine=eng,
            note="applied_move",
            env={"fen": board.fen(), "ply": plies, "move": chosen},
        )
        
        # Check for game over (checkmate/stalemate)
        if board.is_game_over():
            break
        
        # Opponent plays random move
        opp_moves = list(board.legal_moves)
        if not opp_moves:
            break
        opp = random.choice(opp_moves)
        board.push(opp)
        plies += 1
        viz_logger.snapshot(
            engine=eng,
            note="opponent_move",
            env={"fen": board.fen(), "ply": plies, "move": opp.uci()},
        )
        
        # Check for game over after opponent move
        if board.is_game_over():
            break
        
        # Reset network for next move cycle
        for node in g.nodes.values():
            if node.nid != "kqk_root" and node.state == NodeState.CONFIRMED:
                node.state = NodeState.INACTIVE
            elif node.state not in (NodeState.CONFIRMED, NodeState.INACTIVE):
                node.state = NodeState.INACTIVE
        g.nodes["kqk_root"].state = NodeState.REQUESTED

    # Extract results
    game_result = board.result() if board.is_game_over() else None
    is_checkmate = board.is_checkmate()
    
    episode_summary = None
    if plasticity_enabled:
        episode_summary = extract_episode_summary(
            plasticity_state if plasticity_enabled else None,
            None,
            tick_records,
            game_result,
        )

    result = {
        "plies": plies,
        "game_over": board.is_game_over(),
        "result": game_result,
        "final_fen": board.fen(),
        "checkmate": is_checkmate,
    }
    
    # Apply consolidation if enabled
    if consol_engine and episode_summary:
        consol_engine.accumulate_episode(episode_summary)
        if consol_engine.should_apply():
            applied_deltas = consol_engine.apply_to_graph(g)
            if applied_deltas:
                result["consolidation_applied"] = len(applied_deltas)
            if consolidation_pack:
                try:
                    consol_engine.save_state(consolidation_pack)
                except Exception:
                    pass

    # Reset plasticity for next episode
    if plasticity_enabled and plasticity_state:
        reset_plasticity_episode(plasticity_state, g)

    # Save trace
    if trace_db is not None:
        ep = EpisodeRecord(
            episode_id=trace_episode_id or "kqk-persistent",
            result=game_result,
            ticks=tick_records,
            pack_meta=pack_meta,
            notes={"plies": plies},
            summary=episode_summary,
        )
        trace_db.add_episode(ep)
    
    if sf_engine is not None:
        try:
            sf_engine.quit()
        except Exception:
            pass

    # Save visualization logs
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


def run_batch(n_games: int = 10, max_plies: int = 150, stem_cell_enabled: bool = False, **play_kwargs) -> dict:
    """Run N games in batch mode with memory management."""
    stats = {
        "games": [],
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "incomplete": 0,
        "mates": 0,
        "total_mate_plies": 0,
        "avg_mate_length": None,
    }
    
    # M8: Create shared stem cell manager if enabled
    stem_manager = None
    if stem_cell_enabled:
        stem_manager = StemCellManager(
            max_cells=20,
            spawn_rate=0.05,
            config=StemCellConfig(
                min_samples=50,
                reward_threshold=0.2,
                specialization_threshold=0.6,
            ),
        )
    
    for i in range(n_games):
        res = play_persistent_game(initial_fen=None, max_plies=max_plies, stem_cell_manager=stem_manager, **play_kwargs)
        stats["games"].append(res)
        
        # Track results
        game_result = res.get("result")
        if game_result == "1-0":
            stats["wins"] += 1
        elif game_result == "0-1":
            stats["losses"] += 1
        elif game_result == "1/2-1/2":
            stats["draws"] += 1
        else:
            stats["incomplete"] += 1
        
        if res.get("checkmate"):
            stats["mates"] += 1
            stats["total_mate_plies"] += res.get("plies", 0)
        
        # Memory management
        gc.collect()
        
    if stats["mates"]:
        stats["avg_mate_length"] = stats["total_mate_plies"] / stats["mates"]
    
    # Calculate win rate
    completed = stats["wins"] + stats["losses"] + stats["draws"]
    stats["win_rate"] = stats["wins"] / completed if completed > 0 else 0.0
    
    # M8: Report stem cell stats
    if stem_manager:
        stem_stats = {
            "total_cells": len(stem_manager.cells),
            "candidates": len(stem_manager.get_specialization_candidates()),
            "total_samples": sum(len(c.samples) for c in stem_manager.cells.values()),
        }
        stats["stem_cells"] = stem_stats
        print(f"Stem cells: {stem_stats}")
    
    print(stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="KQK Persistent Training Demo")
    parser.add_argument("--fen", type=str, default=None, help="Starting FEN position")
    parser.add_argument("--max-plies", type=int, default=150, help="Maximum plies per game")
    parser.add_argument("--max-ticks", type=int, default=200, help="Maximum ticks per move")
    parser.add_argument("--engine", type=str, default=None, help="Path to Stockfish")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth")
    parser.add_argument("--trace-out", type=Path, default=None, help="Output trace JSONL file")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="Weight pack files")
    parser.add_argument("--output-basename", type=str, default="kqk_persistent", help="Output file basename")
    
    # Batch mode
    parser.add_argument("--batch", type=int, default=0, help="Run N games in batch mode")
    
    # Plasticity
    parser.add_argument("--plasticity", action="store_true", help="Enable M3 fast plasticity")
    parser.add_argument("--plasticity-eta", type=float, default=DEFAULT_PLASTICITY_ETA)
    parser.add_argument("--plasticity-rmax", type=float, default=DEFAULT_PLASTICITY_R_MAX)
    parser.add_argument("--plasticity-lambda", type=float, default=DEFAULT_PLASTICITY_LAMBDA)
    
    # Consolidation
    parser.add_argument("--consolidate", action="store_true", help="Enable M4 consolidation")
    parser.add_argument("--consolidate-pack", type=Path, default=None, help="Consolidation pack file")
    parser.add_argument("--consolidate-eta", type=float, default=DEFAULT_CONSOLIDATE_ETA)
    parser.add_argument("--consolidate-min-episodes", type=int, default=DEFAULT_CONSOLIDATE_MIN_EPISODES)
    
    # M8 Stem cells
    parser.add_argument("--stem-cells", action="store_true", help="Enable M8 stem cell pattern discovery")
    
    args = parser.parse_args()
    
    # Set up trace DB if output requested
    trace_db = None
    if args.trace_out:
        trace_db = TraceDB(args.trace_out)
    
    if args.batch > 0:
        # Batch mode (use a shared consolidation engine so the pack is updated reliably)
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
            max_ticks_per_move=args.max_ticks,
            output_basename=args.output_basename,
            stockfish_path=args.engine,
            stockfish_depth=args.depth,
            trace_db=trace_db,
            pack_paths=args.pack or None,
            plasticity_enabled=args.plasticity,
            plasticity_eta=args.plasticity_eta,
            plasticity_r_max=args.plasticity_rmax,
            plasticity_lambda=args.plasticity_lambda,
            consolidation_enabled=args.consolidate,
            consolidation_pack=args.consolidate_pack,
            consolidation_eta=args.consolidate_eta,
            consolidation_min_episodes=args.consolidate_min_episodes,
            consolidation_engine=consol_engine,
            stem_cell_enabled=args.stem_cells,
        )
        if consol_engine and args.consolidate_pack:
            try:
                consol_engine.save_state(args.consolidate_pack)
            except Exception:
                pass
    else:
        # Single game
        res = play_persistent_game(
            initial_fen=args.fen,
            max_plies=args.max_plies,
            max_ticks_per_move=args.max_ticks,
            output_basename=args.output_basename,
            stockfish_path=args.engine,
            stockfish_depth=args.depth,
            trace_db=trace_db,
            trace_episode_id="kqk-persistent",
            pack_paths=args.pack or None,
            plasticity_enabled=args.plasticity,
            plasticity_eta=args.plasticity_eta,
            plasticity_r_max=args.plasticity_rmax,
            plasticity_lambda=args.plasticity_lambda,
            consolidation_enabled=args.consolidate,
            consolidation_pack=args.consolidate_pack,
            consolidation_eta=args.consolidate_eta,
            consolidation_min_episodes=args.consolidate_min_episodes,
        )
        print(f"Game finished: {res}")
    
    # Flush trace DB
    if trace_db:
        trace_db.flush()


if __name__ == "__main__":
    main()

