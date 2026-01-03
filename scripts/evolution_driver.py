#!/usr/bin/env python3
"""M5-Evolution Training Driver.

Manages the alternating Online/Structural training cycle:
1. Online Phase: Play games, apply fast plasticity, collect traces
2. Structural Phase: Analyze traces, promote stem cells, prune edges
3. Snapshot: Save topology and evolution visualization
4. Repeat

Usage:
    python scripts/evolution_driver.py \\
        --topology topologies/kpk_topology.json \\
        --games-per-cycle 100 \\
        --cycles 5 \\
        --output-dir reports/evolution/

For quick testing:
    python scripts/evolution_driver.py --quick
"""

from __future__ import annotations

import argparse
import json
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
from recon_lite.learning.m5_structure import StructureLearner
from recon_lite.viz.evolution_viz import (
    diff_topologies,
    render_evolution_snapshot,
    save_topology_snapshot,
)

try:
    from recon_lite.nodes.stem_cell import StemCellManager, StemCellConfig
    HAS_STEM_CELL = True
except ImportError:
    HAS_STEM_CELL = False
    StemCellManager = None
    StemCellConfig = None

try:
    from recon_lite_chess.graph.builder import build_graph_from_topology
    HAS_BUILDER = True
except ImportError:
    HAS_BUILDER = False

try:
    from recon_lite_chess.training.generators import (
        generate_kpk_position,
        generate_kpk_curriculum_position,
        KPK_STAGES,
        KPKStage,
    )
    HAS_GENERATORS = True
except ImportError:
    HAS_GENERATORS = False
    KPK_STAGES = []
    KPKStage = None

try:
    from recon_lite_chess.features.kpk_features import extract_kpk_features
    HAS_FEATURES = True
except ImportError:
    HAS_FEATURES = False

try:
    from recon_lite_chess.training.rewards import get_reward_provider, MoveReward
    HAS_REWARDS = True
except ImportError:
    HAS_REWARDS = False


@dataclass
class EvolutionConfig:
    """Configuration for evolution training."""
    topology_path: Path = field(default_factory=lambda: Path("topologies/kpk_topology.json"))
    games_per_cycle: int = 100
    max_cycles: int = 10
    max_promotions_per_cycle: int = 2
    prune_threshold_games: int = 100
    
    # Directories
    snapshot_dir: Path = field(default_factory=lambda: Path("snapshots/evolution"))
    trace_dir: Path = field(default_factory=lambda: Path("traces/evolution"))
    signature_dir: Path = field(default_factory=lambda: Path("signatures"))
    output_dir: Path = field(default_factory=lambda: Path("reports/evolution"))
    
    # Plasticity settings
    plasticity_eta: float = 0.05
    consolidate_eta: float = 0.01
    
    # Game settings
    max_moves_per_game: int = 100
    max_ticks_per_move: int = 50
    
    # Stem cell settings
    stem_cell_spawn_rate: float = 0.05
    stem_cell_max_cells: int = 10
    stem_cell_min_samples: int = 30
    
    # Curriculum settings
    use_curriculum: bool = True
    current_stage_idx: int = 0
    stage_promotion_threshold: float = 0.8  # Win rate to advance
    min_games_per_stage: int = 50
    use_stockfish_rewards: bool = True


@dataclass
class CycleResult:
    """Result of a single evolution cycle."""
    cycle: int
    games_played: int
    win_rate: float
    optimal_rate: float
    promotions: List[str]
    pruned_edges: List[str]
    topology_snapshot_path: Optional[Path]
    evolution_png_path: Optional[Path]
    duration_seconds: float


def run_online_phase(
    config: EvolutionConfig,
    registry: TopologyRegistry,
    stem_manager: Optional["StemCellManager"],
    cycle: int,
) -> Tuple[List[EpisodeRecord], Dict[str, Any]]:
    """
    Online Phase (ThinkPad/Teacher):
    - Play N KPK games
    - Apply fast plasticity (M3)
    - Collect stem cell samples
    - Log traces
    
    Returns:
        (episodes, stats_dict)
    """
    from recon_lite.engine import ReConEngine
    from recon_lite.plasticity import (
        PlasticityConfig,
        init_plasticity_state,
        update_eligibility,
        apply_fast_update,
    )
    
    # Build graph from topology
    if HAS_BUILDER:
        graph = build_graph_from_topology(config.topology_path, registry)
    else:
        # Fallback: import KPK network builder
        from recon_lite_chess.scripts.kpk import build_kpk_network
        graph = build_kpk_network()
    
    # Initialize engine
    engine = ReConEngine(graph)
    
    # Plasticity config
    plasticity_cfg = PlasticityConfig(
        eta_tick=config.plasticity_eta,
        r_max=2.0,
        w_min=0.1,
        w_max=3.0,
        lambda_decay=0.8,
    )
    
    episodes: List[EpisodeRecord] = []
    wins = 0
    draws = 0
    losses = 0
    
    for game_idx in range(config.games_per_cycle):
        # Generate starting position - use curriculum if enabled
        if config.use_curriculum and HAS_GENERATORS and KPK_STAGES:
            stage_idx = min(config.current_stage_idx, len(KPK_STAGES) - 1)
            gen_board = generate_kpk_curriculum_position(KPK_STAGES[stage_idx])
            fen = gen_board.fen()
        elif HAS_GENERATORS:
            gen_board = generate_kpk_position()
            fen = gen_board.fen() if hasattr(gen_board, 'fen') else str(gen_board)
        else:
            # Default KPK position
            fen = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"
        
        board = chess.Board(fen)
        episode_id = f"cycle{cycle:04d}_game{game_idx:04d}"
        
        # Play game
        result, ep = _play_single_game(
            board=board,
            engine=engine,
            graph=graph,
            plasticity_cfg=plasticity_cfg,
            stem_manager=stem_manager,
            episode_id=episode_id,
            max_moves=config.max_moves_per_game,
            max_ticks=config.max_ticks_per_move,
        )
        
        episodes.append(ep)
        
        # Track outcome
        if result == "win":
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1
    
    total = wins + draws + losses
    stats = {
        "games_played": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0,
        "draw_rate": draws / total if total > 0 else 0,
    }
    
    return episodes, stats


def _play_single_game(
    board: chess.Board,
    engine: "ReConEngine",
    graph: Graph,
    plasticity_cfg: Any,
    stem_manager: Optional["StemCellManager"],
    episode_id: str,
    max_moves: int,
    max_ticks: int,
) -> Tuple[str, EpisodeRecord]:
    """Play a single KPK game and return (result, episode_record)."""
    from recon_lite.plasticity import init_plasticity_state
    
    # Initialize episode record
    ep = EpisodeRecord(episode_id=episode_id)
    ep.summary = EpisodeSummary()
    
    # Reset all node states for fresh game
    for node in graph.nodes.values():
        node.state = NodeState.INACTIVE
    
    # Initialize plasticity state
    p_state = init_plasticity_state(graph)
    
    move_count = 0
    tick_count = 0
    our_color = board.turn
    
    while not board.is_game_over() and move_count < max_moves:
        # Reset node states for fresh move evaluation
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
        
        # Set up environment
        env = {
            "board": board,
            "our_color": our_color,
            "move_count": move_count,
        }
        
        # Run engine steps until move selected
        ticks_this_move = 0
        suggested_move = None
        
        # Request root to start propagation
        root_node = graph.nodes.get("kpk_root")
        if root_node:
            root_node.state = NodeState.REQUESTED
        
        while ticks_this_move < max_ticks:
            engine.step(env)  # Use step() not tick()
            tick_count += 1
            ticks_this_move += 1
            
            # Check for suggested move in env
            kpk_policy = env.get("kpk", {}).get("policy", {})
            if "suggested_move" in kpk_policy:
                suggested_move = kpk_policy["suggested_move"]
                break
            
            # Also check node meta
            for nid, node in graph.nodes.items():
                if "last_move" in node.meta:
                    suggested_move = node.meta["last_move"]
                    del node.meta["last_move"]  # Clear for next move
                    break
            
            if suggested_move:
                break
        
        # Create tick record
        tick_rec = TickRecord(
            tick_id=tick_count,
            board_fen=board.fen(),
            active_nodes=[n.nid for n in graph.nodes.values() 
                         if n.state in (NodeState.ACTIVE, NodeState.WAITING, NodeState.REQUESTED)],
            action=suggested_move,
        )
        ep.ticks.append(tick_rec)
        
        # Make move
        if suggested_move:
            try:
                move = chess.Move.from_uci(suggested_move)
                if move in board.legal_moves:
                    board.push(move)
                    move_count += 1
                    
                    # Update stem cells if available
                    if stem_manager and HAS_STEM_CELL:
                        # Compute simple reward signal
                        reward = 0.1 if not board.is_game_over() else (1.0 if board.is_checkmate() else 0.0)
                        stem_manager.tick(board, reward, tick_count)
                else:
                    # Illegal move, try any legal
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
                        move_count += 1
            except Exception:
                # Fallback
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(legal_moves[0])
                    move_count += 1
        else:
            # No move suggested, pick random legal
            legal_moves = list(board.legal_moves)
            if legal_moves:
                import random
                board.push(random.choice(legal_moves))
                move_count += 1
        
        # Opponent's turn (for KPK, we simulate opponent moving randomly)
        if not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                import random
                board.push(random.choice(legal_moves))
                move_count += 1
    
    # Determine result
    if board.is_checkmate():
        result = "win" if board.turn != our_color else "loss"
    elif board.is_stalemate() or board.is_insufficient_material():
        result = "draw"
    elif board.can_claim_draw():
        result = "draw"
    else:
        result = "draw"  # Max moves reached
    
    ep.result = {"win": "1-0", "loss": "0-1", "draw": "1/2-1/2"}.get(result, "1/2-1/2")
    
    return result, ep



def run_structural_phase(
    config: EvolutionConfig,
    registry: TopologyRegistry,
    stem_manager: Optional["StemCellManager"],
    episodes: List[EpisodeRecord],
    cycle: int,
) -> Dict[str, Any]:
    """
    Structural Phase (GPU/Dreamer):
    - Run M5 motif extraction on traces
    - Promote 1-2 promising stem cells to nodes
    - Check edges for pruning
    
    Returns:
        Stats dict with promotions and pruning info
    """
    learner = StructureLearner(
        registry=registry,
        cooldown_ticks=config.games_per_cycle * 10,
        min_spike_reward=0.3,
        decay_rate=0.95,
        prune_threshold_games=config.prune_threshold_games,
        signature_dir=config.signature_dir,
    )
    
    if stem_manager is None or not HAS_STEM_CELL:
        return {
            "spikes_found": 0,
            "high_impact_cells": 0,
            "promotions_attempted": 0,
            "promotions_succeeded": 0,
            "promotions": [],
            "pruning_results": [],
        }
    
    # Run structural phase
    stats = learner.apply_structural_phase(
        stem_manager=stem_manager,
        episodes=episodes,
        max_promotions=config.max_promotions_per_cycle,
    )
    
    return stats


def save_cycle_snapshot(
    config: EvolutionConfig,
    registry: TopologyRegistry,
    old_snapshot: Dict[str, Any],
    cycle: int,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save evolution snapshot with diff highlighting.
    
    Returns:
        (topology_json_path, evolution_png_path)
    """
    # Get new snapshot
    new_snapshot = registry.get_snapshot()
    
    # Compute diff
    diff = diff_topologies(old_snapshot, new_snapshot)
    
    # Save JSON snapshot
    json_path = save_topology_snapshot(registry, config.snapshot_dir, cycle)
    
    # Render PNG
    png_path = config.snapshot_dir / f"cycle_{cycle:04d}.png"
    render_evolution_snapshot(
        topology=new_snapshot,
        diff=diff,
        output_path=png_path,
        title=f"KPK Network - Cycle {cycle}",
    )
    
    return json_path, png_path


def run_evolution_training(config: EvolutionConfig) -> List[CycleResult]:
    """
    Main evolution training loop.
    
    Alternates between Online and Structural phases for the configured
    number of cycles.
    """
    import time
    
    print(f"\n{'='*60}")
    print("M5-Evolution Training Driver")
    print(f"{'='*60}")
    print(f"Topology: {config.topology_path}")
    print(f"Games per cycle: {config.games_per_cycle}")
    print(f"Max cycles: {config.max_cycles}")
    print(f"{'='*60}\n")
    
    # Initialize registry
    registry = TopologyRegistry(config.topology_path)
    
    # Initialize stem cell manager
    stem_manager = None
    if HAS_STEM_CELL:
        stem_cfg = StemCellConfig(
            min_samples=config.stem_cell_min_samples,
            max_samples=200,
            reward_threshold=0.2,
            specialization_threshold=0.6,
            exploration_budget=500,
        )
        stem_manager = StemCellManager(
            max_cells=config.stem_cell_max_cells,
            spawn_rate=config.stem_cell_spawn_rate,
            config=stem_cfg,
        )
    
    # Create output directories
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    config.signature_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[CycleResult] = []
    
    for cycle in range(1, config.max_cycles + 1):
        cycle_start = time.time()
        
        print(f"\n--- Cycle {cycle}/{config.max_cycles} ---")
        
        # Get pre-cycle snapshot for diff
        old_snapshot = registry.get_snapshot()
        
        # Online Phase
        print(f"  Online Phase: Playing {config.games_per_cycle} games...")
        episodes, online_stats = run_online_phase(
            config=config,
            registry=registry,
            stem_manager=stem_manager,
            cycle=cycle,
        )
        print(f"    Win rate: {online_stats['win_rate']:.1%}")
        print(f"    Games: {online_stats['wins']}W / {online_stats['draws']}D / {online_stats['losses']}L")
        
        # Save traces
        trace_path = config.trace_dir / f"cycle_{cycle:04d}.jsonl"
        trace_db = TraceDB(trace_path)
        for ep in episodes:
            trace_db.add_episode(ep)
        trace_db.flush()
        
        # Structural Phase
        print("  Structural Phase: Analyzing traces...")
        struct_stats = run_structural_phase(
            config=config,
            registry=registry,
            stem_manager=stem_manager,
            episodes=episodes,
            cycle=cycle,
        )
        print(f"    Spikes found: {struct_stats['spikes_found']}")
        print(f"    High-impact cells: {struct_stats['high_impact_cells']}")
        print(f"    Promotions: {struct_stats['promotions_succeeded']}")
        
        # Snapshot
        print("  Saving snapshot...")
        json_path, png_path = save_cycle_snapshot(
            config=config,
            registry=registry,
            old_snapshot=old_snapshot,
            cycle=cycle,
        )
        
        cycle_duration = time.time() - cycle_start
        
        # Record result
        result = CycleResult(
            cycle=cycle,
            games_played=online_stats["games_played"],
            win_rate=online_stats["win_rate"],
            optimal_rate=online_stats["win_rate"],  # TODO: compute properly
            promotions=struct_stats.get("promotions", []),
            pruned_edges=struct_stats.get("pruning_results", []),
            topology_snapshot_path=json_path,
            evolution_png_path=png_path,
            duration_seconds=cycle_duration,
        )
        results.append(result)
        
        print(f"  Cycle completed in {cycle_duration:.1f}s")
        
        # Report stem cell stats
        if stem_manager:
            stats = stem_manager.stats()
            print(f"  Stem cells: {stats['total_cells']} ({stats.get('by_state', {})})")
    
    # Final report
    print(f"\n{'='*60}")
    print("Evolution Training Complete")
    print(f"{'='*60}")
    
    total_games = sum(r.games_played for r in results)
    avg_win_rate = sum(r.win_rate for r in results) / len(results) if results else 0
    total_promotions = sum(len(r.promotions) for r in results)
    
    print(f"Total games: {total_games}")
    print(f"Average win rate: {avg_win_rate:.1%}")
    print(f"Total promotions: {total_promotions}")
    print(f"Snapshots saved to: {config.snapshot_dir}")
    
    # Save summary
    summary_path = config.output_dir / "evolution_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "total_cycles": len(results),
            "total_games": total_games,
            "avg_win_rate": avg_win_rate,
            "total_promotions": total_promotions,
            "cycles": [
                {
                    "cycle": r.cycle,
                    "games": r.games_played,
                    "win_rate": r.win_rate,
                    "promotions": r.promotions,
                    "duration": r.duration_seconds,
                }
                for r in results
            ],
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="M5-Evolution Training Driver"
    )
    parser.add_argument(
        "--topology", "-t",
        type=Path,
        default=Path("topologies/kpk_topology.json"),
        help="Path to topology.json file"
    )
    parser.add_argument(
        "--games-per-cycle", "-g",
        type=int,
        default=100,
        help="Games to play per cycle"
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=10,
        help="Number of evolution cycles"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("reports/evolution"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (10 games, 2 cycles)"
    )
    
    args = parser.parse_args()
    
    config = EvolutionConfig(
        topology_path=args.topology,
        games_per_cycle=10 if args.quick else args.games_per_cycle,
        max_cycles=2 if args.quick else args.cycles,
        output_dir=args.output_dir,
    )
    
    run_evolution_training(config)


if __name__ == "__main__":
    main()
