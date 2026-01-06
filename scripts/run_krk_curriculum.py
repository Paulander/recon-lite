#!/usr/bin/env python3
"""KRK Curriculum Training Driver.

Implements backward-chained curriculum learning for King + Rook vs King:
- 10 progressive stages from Mate_In_1 to Full_KRK
- Multiple positions per stage to learn relative patterns
- Hard move penalty rewards
- Box escape detection

Usage:
    python scripts/run_krk_curriculum.py \\
        --games-per-cycle 50 \\
        --output-dir snapshots/evolution/krk_curriculum

Quick test:
    python scripts/run_krk_curriculum.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess

from recon_lite.graph import Graph, Node, NodeType, NodeState
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, EpisodeSummary
from recon_lite.models.registry import TopologyRegistry
from recon_lite.learning.m5_structure import (
    StructureLearner,
    compute_branching_metrics,
    BACKBONE_NODES,
)

try:
    from recon_lite.nodes.stem_cell import StemCellManager, StemCellConfig, StemCellState
    HAS_STEM_CELL = True
except ImportError:
    HAS_STEM_CELL = False
    StemCellManager = None
    StemCellConfig = None
    StemCellState = None

try:
    from recon_lite_chess.graph.builder import build_graph_from_topology
    HAS_BUILDER = True
except ImportError:
    HAS_BUILDER = False

try:
    from recon_lite_chess.training.krk_curriculum import (
        KRK_STAGES,
        KRKStage,
        KRKCurriculumManager,
        KRKStageStats,
        krk_reward,
        box_min_side,
        did_box_grow,
        generate_krk_curriculum_position,
    )
    HAS_KRK_CURRICULUM = True
except ImportError:
    HAS_KRK_CURRICULUM = False
    KRK_STAGES = []

try:
    from recon_lite.engine import ReConEngine, GatingSchedule
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    ReConEngine = None
    GatingSchedule = None

try:
    from recon_lite.plasticity import PlasticityConfig, init_plasticity_state, apply_plasticity
    HAS_PLASTICITY = True
except ImportError:
    HAS_PLASTICITY = False

try:
    from recon_lite.learning.m5_structure import StructureLearner, compute_branching_metrics
    HAS_M5 = True
except ImportError:
    HAS_M5 = False

try:
    from recon_lite_chess.features.krk_features import extract_krk_features
    HAS_KRK_FEATURES = True
except ImportError:
    HAS_KRK_FEATURES = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class KRKCurriculumConfig:
    """Configuration for KRK curriculum training."""
    topology_path: Path = field(default_factory=lambda: Path("topologies/krk_legs_topology.json"))
    games_per_cycle: int = 50
    max_cycles_per_stage: int = 20
    
    # Directories
    output_dir: Path = field(default_factory=lambda: Path("snapshots/evolution/krk_curriculum"))
    trace_dir: Path = field(default_factory=lambda: Path("traces/krk_curriculum"))
    
    # Game settings
    max_moves_per_game: int = 50  # Reduced for faster games
    max_ticks_per_move: int = 10  # Reduced from 50
    min_internal_ticks: int = 2   # Minimal for speed
    
    # Plasticity settings
    plasticity_eta: float = 0.05
    consolidate_eta: float = 0.01
    
    # Curriculum settings
    min_games_per_stage: int = 30
    win_rate_threshold: float = 0.75  # Win rate to advance stage
    
    # Stem cell settings
    stem_cell_spawn_rate: float = 0.05
    stem_cell_max_cells: int = 20
    max_trial_slots: int = 20
    
    # Gating settings (for hierarchy enforcement)
    enable_gating: bool = True
    gating_initial_strictness: float = 0.3
    gating_final_strictness: float = 1.0
    gating_ramp_games: int = 100
    
    # Mode: "simple" (heuristics only) or "recon" (full engine)
    mode: str = "simple"
    
    # M5 structural learning
    enable_m5: bool = False
    m5_spawn_rate: float = 0.1
    m5_hoist_threshold: float = 0.7


# ============================================================================
# Game Execution
# ============================================================================

def play_krk_game_simple(
    board: chess.Board,
    config: KRKCurriculumConfig,
    stage: KRKStage,
) -> Tuple[str, int, bool]:
    """
    Play a single KRK game using simple heuristics (fast mode).
    
    For curriculum validation, we use simple move selection rather than
    full ReCoN engine to enable rapid iteration.
    
    Args:
        board: Starting position
        config: Training configuration
        stage: Current curriculum stage
    
    Returns:
        (result, move_count, box_escaped)
    """
    move_count = 0
    box_escaped = False
    initial_box_min = box_min_side(board)
    
    while move_count < config.max_moves_per_game:
        if board.is_game_over():
            break
        
        # Our move (White) - simple heuristic
        legal = list(board.legal_moves)
        if not legal:
            break
        
        # Simple KRK heuristic: prefer moves that shrink box or give check
        best_move = None
        best_score = -1000
        
        for move in legal:
            score = 0
            board.push(move)
            
            # Checkmate is best
            if board.is_checkmate():
                score = 1000
            # Check is good
            elif board.is_check():
                score = 50
            # Stalemate is very bad
            elif board.is_stalemate():
                score = -500
            else:
                # Prefer smaller box
                new_box_min = box_min_side(board)
                if new_box_min < initial_box_min:
                    score = 20
                elif new_box_min > initial_box_min:
                    score = -30  # Penalty for growing box
                
                # Prefer king approaching enemy king
                our_king = board.king(chess.WHITE)
                their_king = board.king(chess.BLACK)
                if our_king and their_king:
                    dist = chess.square_distance(our_king, their_king)
                    score += (8 - dist) * 2  # Closer is better
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Make move
        move = best_move if best_move else random.choice(legal)
        board.push(move)
        move_count += 1
        
        # Check for box escape
        current_box_min = box_min_side(board)
        if current_box_min > initial_box_min:
            box_escaped = True
            initial_box_min = current_box_min
        
        # Opponent's move (Black) - random (trying to escape)
        if not board.is_game_over():
            legal = list(board.legal_moves)
            if legal:
                # Opponent tries to maximize box (escape)
                opp_best = None
                opp_best_score = -1000
                for m in legal[:10]:  # Limit search for speed
                    board.push(m)
                    opp_score = box_min_side(board)
                    board.pop()
                    if opp_score > opp_best_score:
                        opp_best_score = opp_score
                        opp_best = m
                board.push(opp_best if opp_best else random.choice(legal))
    
    # Determine result
    if board.is_checkmate():
        result = "win" if board.turn == chess.BLACK else "loss"
    elif board.is_stalemate():
        result = "stalemate"
    elif board.is_insufficient_material():
        result = "loss"
    else:
        result = "draw"
    
    return result, move_count, box_escaped


def play_krk_game_recon(
    board: chess.Board,
    engine: "ReConEngine",
    graph: Graph,
    config: KRKCurriculumConfig,
    stage: KRKStage,
    stem_manager: Optional["StemCellManager"] = None,
) -> Tuple[str, int, bool, List[str]]:
    """
    Play a single KRK game using the ReCoN engine.
    
    Args:
        board: Starting position
        engine: ReConEngine instance
        graph: ReCoN graph
        config: Training configuration
        stage: Current curriculum stage
        stem_manager: Optional stem cell manager for M5 learning
    
    Returns:
        (result, move_count, box_escaped, active_nodes_log)
    """
    move_count = 0
    box_escaped = False
    initial_box_min = box_min_side(board)
    active_nodes_log = []
    game_tick = 0
    
    # Sentinel for KRK positions
    def krk_sentinel(env: Dict[str, Any]) -> bool:
        b = env.get("board")
        if not b:
            return False
        # Check if it's still a KRK position
        pieces = list(b.piece_map().values())
        has_rook = any(p.piece_type == chess.ROOK for p in pieces)
        king_count = sum(1 for p in pieces if p.piece_type == chess.KING)
        return has_rook and king_count == 2 and len(pieces) == 3
    
    # Lock the subgraph
    try:
        engine.lock_subgraph("krk_root", krk_sentinel)
    except ValueError:
        # Subgraph might already be locked or not exist
        pass
    
    while move_count < config.max_moves_per_game:
        if board.is_game_over():
            break
        
        # Build environment
        env = {"board": board}
        
        # Run engine ticks
        suggested_move = None
        ticks_this_move = 0
        
        while ticks_this_move < config.max_ticks_per_move:
            try:
                engine.step(env)
            except Exception as e:
                # Engine step failed, use fallback
                break
            ticks_this_move += 1
            
            # Log active nodes (for M5 analysis)
            active = [
                nid for nid, node in graph.nodes.items()
                if node.state in (NodeState.ACTIVE, NodeState.TRUE, NodeState.CONFIRMED)
            ]
            if active:
                active_nodes_log.extend(active)
            
            # Check for suggested move after min ticks
            if ticks_this_move >= config.min_internal_ticks:
                krk_data = env.get("krk_root", {})
                policy = krk_data.get("policy", {})
                suggested_move = policy.get("suggested_move")
                if suggested_move:
                    break
        
        # Make move
        legal_ucis = [m.uci() for m in board.legal_moves]
        if suggested_move and suggested_move in legal_ucis:
            move = chess.Move.from_uci(suggested_move)
        else:
            # Fallback to simple heuristic if engine didn't suggest
            legal = list(board.legal_moves)
            if not legal:
                break
            
            # Use simple box-shrinking heuristic as fallback
            best_move = None
            best_score = -1000
            for m in legal[:20]:  # Limit for speed
                score = 0
                board.push(m)
                if board.is_checkmate():
                    score = 1000
                elif board.is_check():
                    score = 50
                elif board.is_stalemate():
                    score = -500
                else:
                    new_box = box_min_side(board)
                    if new_box < initial_box_min:
                        score = 20
                board.pop()
                if score > best_score:
                    best_score = score
                    best_move = m
            move = best_move if best_move else random.choice(legal)
        
        board.push(move)
        move_count += 1
        game_tick += 1
        
        # Feed observation to stem cells (M5 learning)
        if stem_manager:
            # Calculate interim reward based on position quality
            current_box = box_min_side(board)
            if board.is_checkmate():
                interim_reward = 1.0
            elif board.is_check():
                interim_reward = 0.6
            elif current_box < initial_box_min:
                interim_reward = 0.4  # Good - box is shrinking
            else:
                interim_reward = 0.1
            
            # Feed to all exploring stem cells
            for cell in stem_manager.cells.values():
                cell.observe(board, interim_reward, tick=game_tick)
        
        # Check for box escape
        current_box_min = box_min_side(board)
        if current_box_min > initial_box_min:
            box_escaped = True
            initial_box_min = current_box_min
        
        # Opponent's move
        if not board.is_game_over():
            legal = list(board.legal_moves)
            if legal:
                # Opponent tries to escape
                opp_best = None
                opp_best_score = -1000
                for m in legal[:10]:
                    board.push(m)
                    opp_score = box_min_side(board)
                    board.pop()
                    if opp_score > opp_best_score:
                        opp_best_score = opp_score
                        opp_best = m
                board.push(opp_best if opp_best else random.choice(legal))
    
    # Determine result
    if board.is_checkmate():
        result = "win" if board.turn == chess.BLACK else "loss"
    elif board.is_stalemate():
        result = "stalemate"
    elif board.is_insufficient_material():
        result = "loss"
    else:
        result = "draw"
    
    # Unlock
    try:
        engine.unlock_subgraph("krk_root")
    except:
        pass
    
    return result, move_count, box_escaped, active_nodes_log


# ============================================================================
# Training Loop
# ============================================================================

def run_krk_curriculum(config: KRKCurriculumConfig) -> Dict[str, Any]:
    """
    Run the full KRK curriculum training.
    
    Returns:
        Summary of training results
    """
    print("=" * 70)
    print("KRK CURRICULUM TRAINING")
    print("=" * 70)
    
    if not HAS_KRK_CURRICULUM:
        print("ERROR: KRK curriculum module not available")
        return {"error": "KRK curriculum not available"}
    
    if not HAS_ENGINE:
        print("ERROR: ReConEngine not available")
        return {"error": "Engine not available"}
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize curriculum manager
    curriculum = KRKCurriculumManager(
        min_games_per_stage=config.min_games_per_stage,
        win_rate_threshold=config.win_rate_threshold,
    )
    
    # Initialize engine and graph based on mode
    engine = None
    graph = None
    stem_manager = None
    
    if config.mode == "recon":
        print(f"\nUsing RECON ENGINE mode with M5={config.enable_m5}")
        
        # Load topology and build graph
        if not config.topology_path.exists():
            print(f"ERROR: Topology not found: {config.topology_path}")
            return {"error": "Topology not found"}
        
        registry = TopologyRegistry(config.topology_path)
        
        if HAS_BUILDER:
            graph = build_graph_from_topology(config.topology_path)
            print(f"  Graph loaded: {len(graph.nodes)} nodes")
        else:
            print("ERROR: Graph builder not available")
            return {"error": "Builder not available"}
        
        if HAS_ENGINE:
            engine = ReConEngine(graph)
            
            # Setup gating
            if config.enable_gating and GatingSchedule:
                gating_schedule = GatingSchedule(
                    initial_strictness=config.gating_initial_strictness,
                    final_strictness=config.gating_final_strictness,
                    ramp_games=config.gating_ramp_games,
                )
                engine.gating_schedule = gating_schedule
                print(f"  Gating enabled: {config.gating_initial_strictness} → {config.gating_final_strictness}")
        else:
            print("ERROR: ReConEngine not available")
            return {"error": "Engine not available"}
        
        # Initialize stem cells for M5
        if config.enable_m5 and HAS_STEM_CELL:
            stem_config = StemCellConfig(min_samples=30, max_samples=500)
            stem_manager = StemCellManager(max_cells=config.max_trial_slots, config=stem_config, max_trial_slots=config.max_trial_slots)
            
            # Seed initial stem cells to start observing
            initial_cells = 10
            for _ in range(initial_cells):
                stem_manager.spawn_cell()
            
            print(f"  M5 enabled: max_trial_slots={config.max_trial_slots}, seeded {len(stem_manager.cells)} cells")
    else:
        print(f"\nUsing SIMPLE heuristic mode for fast validation")
    
    # Training state
    total_games = 0
    stage_history = []
    
    print(f"\nStarting curriculum with {len(KRK_STAGES)} stages")
    print(f"Games per cycle: {config.games_per_cycle}")
    print(f"Win rate threshold: {config.win_rate_threshold}")
    print()
    
    # Main training loop
    stages_completed = set()
    while curriculum.current_stage_id < len(KRK_STAGES):
        # Prevent infinite loop on final stage
        if curriculum.current_stage_id in stages_completed:
            print("\n  All stages processed. Ending training.")
            break
        stages_completed.add(curriculum.current_stage_id)
        stage = curriculum.current_stage
        stage_start_games = total_games
        cycle = 0
        
        print(f"\n{'='*60}")
        print(f"STAGE {stage.stage_id}: {stage.name}")
        print(f"Description: {stage.description}")
        print(f"Key Lesson: {stage.key_lesson}")
        print(f"Target Win Rate: {stage.target_win_rate}")
        print(f"{'='*60}")
        
        # Run cycles for this stage
        while cycle < config.max_cycles_per_stage:
            cycle += 1
            cycle_start = datetime.now()
            
            wins = 0
            losses = 0
            stalemates = 0
            draws = 0
            total_moves = 0
            box_escapes = 0
            total_reward = 0.0
            
            print(f"\n  Cycle {cycle}/{config.max_cycles_per_stage}")
            
            active_node_counts = {}  # Track which nodes fire
            
            for game_idx in range(config.games_per_cycle):
                # Get position for current stage
                board = curriculum.get_position()
                
                # Play game based on mode
                if config.mode == "recon" and engine and graph:
                    result, move_count, box_escaped, active_log = play_krk_game_recon(
                        board=board,
                        engine=engine,
                        graph=graph,
                        config=config,
                        stage=stage,
                        stem_manager=stem_manager if config.enable_m5 else None,
                    )
                    # Count active nodes for M5 analysis
                    for nid in active_log:
                        active_node_counts[nid] = active_node_counts.get(nid, 0) + 1
                else:
                    result, move_count, box_escaped = play_krk_game_simple(
                        board=board,
                        config=config,
                        stage=stage,
                    )
                
                # Progress indicator
                if (game_idx + 1) % 10 == 0:
                    print(f"    Game {game_idx + 1}/{config.games_per_cycle}...")
                
                # Update counters
                total_games += 1
                total_moves += move_count
                
                if result == "win":
                    wins += 1
                elif result == "loss":
                    losses += 1
                elif result == "stalemate":
                    stalemates += 1
                else:
                    draws += 1
                
                if box_escaped:
                    box_escapes += 1
                
                # Compute reward
                optimal_moves = stage.get_optimal_moves(board)
                reward = krk_reward(
                    won=(result == "win"),
                    move_count=move_count,
                    optimal_moves=optimal_moves,
                    box_grew=box_escaped,
                    stalemate=(result == "stalemate"),
                )
                total_reward += reward
                
                # Record in curriculum manager
                advanced = curriculum.record_game(
                    won=(result == "win"),
                    move_count=move_count,
                    stalemate=(result == "stalemate"),
                    box_escaped=box_escaped,
                )
                
                if advanced:
                    print(f"\n  >>> STAGE ADVANCED to {curriculum.current_stage.name} <<<")
                    break
            
            # Cycle summary
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            win_rate = wins / config.games_per_cycle
            avg_moves = total_moves / config.games_per_cycle
            avg_reward = total_reward / config.games_per_cycle
            escape_rate = box_escapes / config.games_per_cycle
            
            print(f"    Win Rate: {win_rate:.1%} ({wins}W/{losses}L/{stalemates}S/{draws}D)")
            print(f"    Avg Moves: {avg_moves:.1f}")
            print(f"    Avg Reward: {avg_reward:.3f}")
            print(f"    Box Escapes: {escape_rate:.1%}")
            print(f"    Duration: {cycle_duration:.1f}s")
            
            # Show top active nodes if in ReCoN mode
            if config.mode == "recon" and active_node_counts:
                top_nodes = sorted(active_node_counts.items(), key=lambda x: -x[1])[:5]
                print(f"    Top nodes: {', '.join(f'{n}({c})' for n, c in top_nodes)}")
            
            # M5 Structural Learning Phase
            if config.mode == "recon" and config.enable_m5 and stem_manager and HAS_M5:
                try:
                    from recon_lite.learning.m5_structure import (
                        StructureLearner,
                        compute_branching_metrics,
                    )
                    
                    # Create structural learner
                    struct_learner = StructureLearner(registry=registry)
                    
                    # Apply structural phase
                    struct_result = struct_learner.apply_structural_phase(
                        stem_manager=stem_manager,
                        episodes=[],  # TODO: Pass actual episodes for POR discovery
                        max_promotions=3,
                        parent_candidates=["krk_detect", "krk_execute", "krk_rook_leg", "krk_king_leg"],
                        current_win_rate=win_rate,
                    )
                    
                    # Compute branching metrics
                    metrics = compute_branching_metrics(graph)
                    
                    promoted = struct_result.get("trial_promotions", 0)
                    hoisted = struct_result.get("hoisted_count", 0)
                    depth = metrics.get("max_depth", 1)
                    
                    if promoted > 0 or hoisted > 0 or depth > 1:
                        print(f"    M5: +{promoted} TRIAL, +{hoisted} HOISTED, depth={depth}")
                    
                    # Spawn new sensors if win rate is low (Stall Recovery)
                    if win_rate < 0.3 and len(stem_manager.cells) < config.max_trial_slots and HAS_KRK_FEATURES:
                        # Spawn new stem cells
                        spawned = 0
                        for _ in range(3):  # Spawn up to 3 new sensors
                            new_cell = stem_manager.spawn_cell()
                            if new_cell:
                                spawned += 1
                        
                        if spawned > 0:
                            print(f"    M5 Spawn: +{spawned} new stem cells (stall recovery)")
                    
                except Exception as e:
                    print(f"    M5 Error: {e}")
            
            # Check if stage advanced during cycle
            if curriculum.current_stage_id > stage.stage_id:
                break
            
            # Force advance if we hit max cycles (allow progression even if win rate not met)
            if cycle >= config.max_cycles_per_stage:
                print(f"\n  Max cycles reached. Force advancing...")
                if not curriculum.force_advance():
                    # Already at final stage - exit
                    print("  (At final stage - training complete)")
                break
            
            # Save cycle snapshot
            snapshot_path = config.output_dir / f"stage{stage.stage_id}" / f"cycle_{cycle:04d}.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            
            snapshot_data = {
                "stage_id": stage.stage_id,
                "stage_name": stage.name,
                "cycle": cycle,
                "games": config.games_per_cycle,
                "win_rate": win_rate,
                "avg_moves": avg_moves,
                "avg_reward": avg_reward,
                "escape_rate": escape_rate,
                "total_games": total_games,
                "timestamp": datetime.now().isoformat(),
            }
            snapshot_path.write_text(json.dumps(snapshot_data, indent=2))
        
        # Stage complete - record history
        stage_games = total_games - stage_start_games
        stage_stats = curriculum.stage_stats[stage.stage_id]
        
        stage_history.append({
            "stage_id": stage.stage_id,
            "stage_name": stage.name,
            "games_played": stage_games,
            "cycles": cycle,
            "final_win_rate": stage_stats.win_rate,
            "avg_moves": stage_stats.avg_moves,
            "escape_rate": stage_stats.escape_rate,
        })
        
        print(f"\n  Stage {stage.stage_id} Complete:")
        print(f"    Games: {stage_games}")
        print(f"    Cycles: {cycle}")
        print(f"    Final Win Rate: {stage_stats.win_rate:.1%}")
    
    # Training complete
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Final summary
    summary = {
        "total_games": total_games,
        "stages_completed": len(stage_history),
        "total_stages": len(KRK_STAGES),
        "stage_history": stage_history,
        "curriculum_summary": curriculum.get_summary(),
        "config": {
            "games_per_cycle": config.games_per_cycle,
            "min_games_per_stage": config.min_games_per_stage,
            "win_rate_threshold": config.win_rate_threshold,
            "enable_gating": config.enable_gating,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save final summary
    summary_path = config.output_dir / "curriculum_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")
    
    # Print stage-by-stage results
    print("\nStage Results:")
    print("-" * 60)
    for stage_result in stage_history:
        print(f"  Stage {stage_result['stage_id']}: {stage_result['stage_name']}")
        print(f"    Win Rate: {stage_result['final_win_rate']:.1%}")
        print(f"    Avg Moves: {stage_result['avg_moves']:.1f}")
        print(f"    Games: {stage_result['games_played']}")
    
    return summary


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="KRK Curriculum Training")
    parser.add_argument("--topology", type=Path, default=Path("topologies/krk_legs_topology.json"),
                        help="Path to KRK topology file")
    parser.add_argument("--output-dir", type=Path, default=Path("snapshots/evolution/krk_curriculum"),
                        help="Output directory for snapshots")
    parser.add_argument("--games-per-cycle", type=int, default=50,
                        help="Games per training cycle")
    parser.add_argument("--max-cycles-per-stage", type=int, default=20,
                        help="Maximum cycles per stage before forced advance")
    parser.add_argument("--win-rate-threshold", type=float, default=0.75,
                        help="Win rate to advance to next stage")
    parser.add_argument("--min-games-per-stage", type=int, default=30,
                        help="Minimum games before stage can advance")
    parser.add_argument("--min-tick-depth", type=int, default=3,
                        help="Minimum internal ticks for sensor propagation")
    parser.add_argument("--enable-gating", action="store_true", default=True,
                        help="Enable hierarchical gating")
    parser.add_argument("--no-gating", action="store_false", dest="enable_gating",
                        help="Disable hierarchical gating")
    parser.add_argument("--mode", choices=["simple", "recon"], default="simple",
                        help="Training mode: simple (heuristics) or recon (full engine)")
    parser.add_argument("--enable-m5", action="store_true",
                        help="Enable M5 structural learning (stem cells)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5 games, 3 cycles)")
    
    args = parser.parse_args()
    
    # Build config
    config = KRKCurriculumConfig(
        topology_path=args.topology,
        output_dir=args.output_dir,
        games_per_cycle=5 if args.quick else args.games_per_cycle,
        max_cycles_per_stage=3 if args.quick else args.max_cycles_per_stage,
        win_rate_threshold=args.win_rate_threshold,
        min_games_per_stage=5 if args.quick else args.min_games_per_stage,
        min_internal_ticks=args.min_tick_depth,
        enable_gating=args.enable_gating,
        mode=args.mode,
        enable_m5=args.enable_m5,
    )
    
    # Run training
    summary = run_krk_curriculum(config)
    
    # Exit code based on success
    if "error" in summary:
        sys.exit(1)
    
    # Success if at least 5 stages completed
    stages_completed = summary.get("stages_completed", 0)
    if stages_completed >= 5:
        print(f"\nSUCCESS: Completed {stages_completed} stages!")
        sys.exit(0)
    else:
        print(f"\nPartial success: Completed {stages_completed}/10 stages")
        sys.exit(0)


if __name__ == "__main__":
    main()

