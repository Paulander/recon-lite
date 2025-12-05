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
from typing import Dict, Any, List, Optional, Tuple

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

# Import the graph builder from full_game_demo
from demos.persistent.full_game_demo import (
    build_full_game_graph,
    select_move,
    extract_board_features,
    GameState,
)


# Default parameters
DEFAULT_PLASTICITY_ETA = 0.05
DEFAULT_PLASTICITY_R_MAX = 2.0
DEFAULT_PLASTICITY_W_MIN = 0.1
DEFAULT_PLASTICITY_W_MAX = 3.0
DEFAULT_PLASTICITY_LAMBDA = 0.8

DEFAULT_CONSOLIDATE_ETA = 0.01
DEFAULT_CONSOLIDATE_MIN_EPISODES = 10


def get_trainable_edges(graph: Graph) -> List[Tuple[str, str, LinkType]]:
    """Get edges that should be trained with plasticity."""
    edges = []
    for e in graph.edges:
        # Train POR edges (sequential/temporal)
        if e.ltype == LinkType.POR:
            edges.append((e.src, e.dst, e.ltype))
        # Train some SUB edges (hierarchy)
        elif e.ltype == LinkType.SUB:
            # Only train SUB edges for strategy selection
            if "Strategy" in e.src or "Plan" in e.src:
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
    max_moves: int = 200,
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
) -> Dict[str, Any]:
    """
    Play a single training game.
    
    Returns:
        Dict with game results and training stats
    """
    # Build the graph
    g = build_full_game_graph()
    engine = ReConEngine(g)
    
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
        consolidation_engine.init_from_graph(g)
        consolidation_engine.apply_w_base_to_graph(g)
    
    # Set up feature extractor for stem cells
    if stem_manager:
        for cell in stem_manager.cells.values():
            cell.feature_extractor = extract_board_features
    
    # Initialize game state
    state = GameState(board=chess.Board())
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
        
        # Get evaluation before move
        eval_before = None
        if stockfish_engine:
            eval_before = compute_stockfish_eval(state.board, stockfish_engine, stockfish_depth)
        if eval_before is None:
            eval_before = eval_position(state.board)
        
        # Run engine step
        fired_edges = []
        now_requested = engine.step(env)
        state.tick += 1
        
        # Collect fired edges for plasticity
        if plasticity_enabled:
            for e in g.edges:
                src_node = g.nodes.get(e.src)
                dst_node = g.nodes.get(e.dst)
                if src_node and dst_node:
                    if src_node.state in (NodeState.TRUE, NodeState.CONFIRMED) and \
                       dst_node.state == NodeState.REQUESTED:
                        fired_edges.append({"src": e.src, "dst": e.dst, "ltype": e.ltype.name})
        
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
        
        # Prioritize tactical moves
        move = None
        if forks:
            fork_moves = get_fork_moves(state.board)
            if fork_moves:
                move = fork_moves[0]
        elif hanging.get("enemy_hanging"):
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
            meta={"ply": len(state.move_history)},
        ))
        
        # Opponent's turn
        if not state.board.is_game_over() and vs_random:
            opp_moves = list(state.board.legal_moves)
            if opp_moves:
                opp_move = random.choice(opp_moves)
                state.board.push(opp_move)
                state.move_history.append(opp_move.uci())
    
    # Game over - extract episode summary
    game_result = state.board.result() if state.board.is_game_over() else "*"
    outcome_score = 0.0
    if game_result == "1-0":
        outcome_score = 1.0
    elif game_result == "0-1":
        outcome_score = -1.0
    
    episode_summary = None
    if plasticity_enabled:
        episode_summary = extract_episode_summary(
            plasticity_state,
            None,  # No bandit state
            tick_records,
            game_result,
        )
    
    # Accumulate for consolidation
    if consolidation_engine and episode_summary:
        consolidation_engine.accumulate_episode(episode_summary)
    
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
    }
    
    if verbose:
        win_status = "WIN" if game_result == "1-0" else "LOSS" if game_result == "0-1" else "DRAW"
        print(f"  Game {game_id}: {win_status} in {len(state.move_history)} moves, reward={total_reward:.2f}")
    
    return result


def run_batch_training(
    n_games: int,
    *,
    max_moves: int = 200,
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
        result = play_training_game(
            game_id=i + 1,
            max_moves=max_moves,
            vs_random=vs_random,
            verbose=verbose,
            stockfish_engine=sf_engine,
            stockfish_depth=stockfish_depth,
            plasticity_enabled=plasticity_enabled,
            plasticity_config=plasticity_config,
            consolidation_engine=consol_engine,
            stem_manager=stem_manager,
        )
        
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
            # Build a dummy graph to apply consolidation
            g = build_full_game_graph()
            applied = consol_engine.apply_to_graph(g)
            if verbose and applied:
                print(f"  [Consolidation] Applied {len(applied)} weight updates")
        
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
    parser.add_argument("--vs-random", action="store_true", default=True, help="Play against random opponent")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer games, less depth)")
    
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
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.batch = min(args.batch, 5)
        args.max_moves = 100
        args.depth = 1
    
    verbose = not args.quiet
    
    stats = run_batch_training(
        n_games=args.batch,
        max_moves=args.max_moves,
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

