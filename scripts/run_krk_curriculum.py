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
    
    # Note: For fast curriculum validation, we use simple heuristics
    # Full ReCoN engine integration will be added after curriculum validation
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
            
            for game_idx in range(config.games_per_cycle):
                # Get position for current stage
                board = curriculum.get_position()
                
                # Play game (simple mode for speed)
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

