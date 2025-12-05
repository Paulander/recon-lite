#!/usr/bin/env python3
"""
Minimal persistent loop for the KPK subgraph with optional tracing.
Reuses the existing KPK network and per-move selector; no opponent policy beyond random.

M8: Added batch mode and consolidation support matching krk_persistent_demo.py.
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


def create_random_kpk_board(white_to_move: bool = True) -> str:
    """
    Create a random KPK position (K+P vs K) for training.
    
    Rules:
    - White has King + Pawn, Black has only King
    - Pawn is on ranks 2-6 (not on promotion rank, not on starting row for more variety)
    - Kings are not adjacent
    - Position is legal (not in check if it's opponent's turn to create position)
    
    Args:
        white_to_move: Whether white should move first
        
    Returns:
        str: FEN string of the random KPK position
    """
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Place white pawn on ranks 2-6 (indices 1-5), files a-h
        pawn_file = random.randint(0, 7)  # a-h
        pawn_rank = random.randint(2, 5)  # ranks 3-6 (more interesting than rank 2)
        pawn_sq = chess.square(pawn_file, pawn_rank)
        
        # Place white king - not on pawn square, preferably supporting the pawn
        available_for_wk = [sq for sq in chess.SQUARES if sq != pawn_sq]
        # Prefer squares near the pawn for more realistic positions
        near_pawn = [sq for sq in available_for_wk 
                     if abs(chess.square_file(sq) - pawn_file) <= 2 
                     and chess.square_rank(sq) >= pawn_rank - 1]
        if near_pawn:
            wk_sq = random.choice(near_pawn)
        else:
            wk_sq = random.choice(available_for_wk)
        
        # Place black king - not on pawn or white king, not adjacent to white king
        def is_adjacent(sq1, sq2):
            return abs(chess.square_file(sq1) - chess.square_file(sq2)) <= 1 and \
                   abs(chess.square_rank(sq1) - chess.square_rank(sq2)) <= 1
        
        available_for_bk = [sq for sq in chess.SQUARES 
                           if sq != pawn_sq and sq != wk_sq and not is_adjacent(sq, wk_sq)]
        
        if not available_for_bk:
            continue
            
        # Prefer black king in front of the pawn (more realistic defense)
        blocking_squares = [sq for sq in available_for_bk 
                          if chess.square_rank(sq) > pawn_rank
                          and abs(chess.square_file(sq) - pawn_file) <= 1]
        if blocking_squares and random.random() < 0.7:
            bk_sq = random.choice(blocking_squares)
        else:
            bk_sq = random.choice(available_for_bk)
        
        # Build the board and validate
        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE if white_to_move else chess.BLACK
        
        # Validate position
        if board.is_valid():
            # Make sure it's not already game over
            if not board.is_game_over():
                return board.fen()
    
    # Fallback: simple known valid position
    return "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite import ReConEngine  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.graph import NodeState, LinkType  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.logger import RunLogger  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess.scripts.kpk import build_kpk_network  # type: ignore  # pylint: disable=wrong-import-position
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
from recon_lite_chess.eval.heuristic import eval_position, compute_reward_tick

# M8: KPK plasticity defaults (matching KRK)
DEFAULT_PLASTICITY_ETA = 0.05
DEFAULT_PLASTICITY_R_MAX = 2.0
DEFAULT_PLASTICITY_W_MIN = 0.1
DEFAULT_PLASTICITY_W_MAX = 3.0
DEFAULT_PLASTICITY_LAMBDA = 0.8

# M8: Consolidation defaults
DEFAULT_CONSOLIDATE_ETA = 0.01
DEFAULT_CONSOLIDATE_MIN_EPISODES = 10

# M8: KPK edges eligible for fast plasticity
KPK_PLASTICITY_EDGES = [
    # Will be populated from the actual KPK network structure
]


def play_persistent_game(
    initial_fen: str | None = None,
    *,
    max_plies: int = 100,
    max_ticks_per_move: int = 200,
    split_logs: bool = True,
    output_basename: str = "kpk_persistent",
    stockfish_path: Optional[str] = None,
    stockfish_depth: int = 2,
    trace_db: Optional[TraceDB] = None,
    trace_episode_id: Optional[str] = None,
    pack_paths: Optional[list[Path]] = None,
    # M8: Plasticity parameters
    plasticity_enabled: bool = False,
    plasticity_eta: float = DEFAULT_PLASTICITY_ETA,
    plasticity_r_max: float = DEFAULT_PLASTICITY_R_MAX,
    plasticity_lambda: float = DEFAULT_PLASTICITY_LAMBDA,
    # M8: Consolidation parameters
    consolidation_enabled: bool = False,
    consolidation_pack: Optional[Path] = None,
    consolidation_eta: float = DEFAULT_CONSOLIDATE_ETA,
    consolidation_min_episodes: int = DEFAULT_CONSOLIDATE_MIN_EPISODES,
    consolidation_engine: Optional[ConsolidationEngine] = None,
) -> dict:
    viz_logger = RunLogger()
    debug_logger = RunLogger() if split_logs else viz_logger

    # Use provided FEN or generate random KPK position
    if initial_fen:
        board = chess.Board(initial_fen)
    else:
        board = chess.Board(create_random_kpk_board(white_to_move=True))
    g = build_kpk_network()
    eng = ReConEngine(g)
    g.nodes["kpk_root"].state = NodeState.REQUESTED

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

    # M8: Initialize plasticity
    plasticity_config = PlasticityConfig(
        eta_tick=plasticity_eta,
        r_max=plasticity_r_max,
        lambda_decay=plasticity_lambda,
        w_min=DEFAULT_PLASTICITY_W_MIN,
        w_max=DEFAULT_PLASTICITY_W_MAX,
        enabled=plasticity_enabled,
    )
    # Build plasticity edges from graph
    kpk_plasticity_edges = []
    if plasticity_enabled:
        for e in g.edges:
            if e.ltype == LinkType.POR:
                kpk_plasticity_edges.append((e.src, e.dst, e.ltype))
    plasticity_state = init_plasticity_state(g, kpk_plasticity_edges) if plasticity_enabled else {}

    # M8: Initialize or use provided consolidation engine
    consolidation_config = ConsolidationConfig(
        eta_consolidate=consolidation_eta,
        min_episodes=consolidation_min_episodes,
        enabled=consolidation_enabled,
    )
    if consolidation_engine is not None:
        consol_engine = consolidation_engine
    elif consolidation_enabled:
        consol_engine = ConsolidationEngine(consolidation_config)
        if consolidation_pack and consolidation_pack.exists():
            try:
                consol_engine.load_state(consolidation_pack)
            except Exception:
                pass
        consol_engine.init_from_graph(g)
        consol_engine.apply_w_base_to_graph(g)
    else:
        consol_engine = None

    last_eval: Optional[float] = None

    while not board.is_game_over() and plies < max_plies:
        env = {"board": board}
        chosen = None
        move_ticks = 0
        while move_ticks < max_ticks_per_move and chosen is None:
            move_ticks += 1
            now_req = eng.step(env)
            chosen = env.get("kpk", {}).get("policy", {}).get("suggested_move") if isinstance(env.get("kpk"), dict) else None
            viz_logger.snapshot(
                engine=eng,
                note=f"tick {move_ticks}",
                env={"fen": board.fen(), "ply": plies + 1},
                new_requests=list(now_req.keys()),
            )
        if not chosen:
            break
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
            board.push_uci(chosen)
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
                board_fen=board.fen(),
                action=chosen,
                active_nodes=[nid for nid, node in eng.g.nodes.items() if node.state != NodeState.INACTIVE],
                eval_before=eval_before,
                eval_after=eval_after,
                reward_tick=(round(eval_after - eval_before, 3) if eval_after is not None and eval_before is not None else None),
                meta={"ply": plies},
            )
        )
        viz_logger.snapshot(
            engine=eng,
            note="applied_move",
            env={"fen": board.fen(), "ply": plies, "move": chosen},
        )
        if board.is_game_over():
            break
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

    # M8: Extract episode summary for consolidation
    game_result = board.result() if board.is_game_over() else None
    episode_summary = None
    if plasticity_enabled:
        episode_summary = extract_episode_summary(
            plasticity_state if plasticity_enabled else None,
            None,  # No bandit state for KPK yet
            tick_records,
            game_result,
        )

    # M8: Accumulate episode for consolidation
    result = {
        "plies": plies,
        "game_over": board.is_game_over(),
        "result": game_result,
        "final_fen": board.fen(),
        "checkmate": board.is_checkmate(),
    }
    
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

    # M8: Reset plasticity for next episode
    if plasticity_enabled and plasticity_state:
        reset_plasticity_episode(plasticity_state, g)

    if trace_db is not None:
        ep = EpisodeRecord(
            episode_id=trace_episode_id or "kpk-persistent",
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


def run_batch(n_games: int = 10, max_plies: int = 100, **play_kwargs) -> dict:
    """Run N games in batch mode with memory management."""
    stats = {
        "games": [],
        "mates": 0,
        "stalls": 0,
        "total_mate_plies": 0,
        "avg_mate_length": None,
        "promotions": 0,
    }
    for i in range(n_games):
        res = play_persistent_game(initial_fen=None, max_plies=max_plies, **play_kwargs)
        stats["games"].append(res)
        if res.get("checkmate"):
            stats["mates"] += 1
            stats["total_mate_plies"] += res.get("plies", 0)
        # Check for pawn promotion (indicated by piece on 8th rank in final FEN)
        final_fen = res.get("final_fen", "")
        if "Q" in final_fen.split()[0]:  # Queen from promotion
            stats["promotions"] += 1
        
        # Memory management: clean up between games
        gc.collect()
        
    if stats["mates"]:
        stats["avg_mate_length"] = stats["total_mate_plies"] / stats["mates"]
    print(stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent KPK demo with optional trace output")
    parser.add_argument("--fen", type=str, default=None)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--engine", type=str, default=None, help="Path to Stockfish (optional)")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth when scoring moves")
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--pack", action="append", type=Path, default=[])
    parser.add_argument("--output-basename", type=str, default="kpk_persistent", help="Base name for output logs")
    # M8: Batch mode
    parser.add_argument("--batch", type=int, default=0, help="Run N games in batch mode")
    # M8: Plasticity arguments
    parser.add_argument("--plasticity", action="store_true", help="Enable fast plasticity")
    parser.add_argument("--plasticity-eta", type=float, default=DEFAULT_PLASTICITY_ETA)
    parser.add_argument("--plasticity-r-max", type=float, default=DEFAULT_PLASTICITY_R_MAX)
    parser.add_argument("--plasticity-lambda", type=float, default=DEFAULT_PLASTICITY_LAMBDA)
    # M8: Consolidation arguments
    parser.add_argument("--consolidate", action="store_true", help="Enable slow consolidation")
    parser.add_argument("--consolidate-pack", type=Path, default=None, help="Path to load/save consolidation state")
    parser.add_argument("--consolidate-eta", type=float, default=DEFAULT_CONSOLIDATE_ETA)
    parser.add_argument("--consolidate-min-episodes", type=int, default=DEFAULT_CONSOLIDATE_MIN_EPISODES)
    args = parser.parse_args()

    if args.batch and args.batch > 0:
        # M8: Create shared consolidation engine for batch mode
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

        trace_db = TraceDB(args.trace_out) if args.trace_out else None
        run_batch(
            args.batch,
            max_plies=args.max_plies,
            max_ticks_per_move=args.max_ticks,
            output_basename=args.output_basename,
            stockfish_path=args.engine,
            stockfish_depth=args.depth,
            trace_db=trace_db,
            pack_paths=args.pack,
            # M8: Plasticity
            plasticity_enabled=args.plasticity,
            plasticity_eta=args.plasticity_eta,
            plasticity_r_max=args.plasticity_r_max,
            plasticity_lambda=args.plasticity_lambda,
            # M8: Consolidation
            consolidation_enabled=args.consolidate,
            consolidation_pack=args.consolidate_pack,
            consolidation_eta=args.consolidate_eta,
            consolidation_min_episodes=args.consolidate_min_episodes,
            consolidation_engine=consol_engine,
        )
        if trace_db:
            trace_db.flush()

        # M8: Save final consolidation state after batch
        if consol_engine and args.consolidate_pack:
            try:
                consol_engine.save_state(args.consolidate_pack)
            except Exception:
                pass
    else:
        trace_db = TraceDB(args.trace_out) if args.trace_out else None
        res = play_persistent_game(
            initial_fen=args.fen,
            max_plies=args.max_plies,
            max_ticks_per_move=args.max_ticks,
            output_basename=args.output_basename,
            trace_db=trace_db,
            pack_paths=args.pack,
            trace_episode_id="kpk-cli",
            stockfish_path=args.engine,
            stockfish_depth=args.depth,
            # M8: Plasticity
            plasticity_enabled=args.plasticity,
            plasticity_eta=args.plasticity_eta,
            plasticity_r_max=args.plasticity_r_max,
            plasticity_lambda=args.plasticity_lambda,
            # M8: Consolidation
            consolidation_enabled=args.consolidate,
            consolidation_pack=args.consolidate_pack,
            consolidation_eta=args.consolidate_eta,
            consolidation_min_episodes=args.consolidate_min_episodes,
        )
        if trace_db:
            trace_db.flush()
        print(res)


if __name__ == "__main__":
    main()
