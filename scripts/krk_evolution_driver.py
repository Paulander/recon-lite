#!/usr/bin/env python3
"""KRK Evolution Training Driver with Knowledge Transfer.

Specialized driver for KRK (King+Rook vs King) endgame training.
Supports knowledge transfer from KPK to accelerate learning.

Features:
- Load stem cells from KPK with automatic renaming (universal sensors)
- Track sensor_reuse_ratio to measure knowledge transfer success
- Enable KRK-specific Box Method POR discovery
- Full M5 structural learning support

Usage:
    # Fresh KRK training
    python scripts/krk_evolution_driver.py \\
        --topology topologies/krk_legs_topology.json \\
        --games-per-cycle 100 \\
        --cycles 20

    # With knowledge transfer from KPK
    python scripts/krk_evolution_driver.py \\
        --transfer-from snapshots/sweeps/gauntlet/forced_hierarchy/snapshots/stem_cells.json \\
        --topology topologies/krk_legs_topology.json \\
        --games-per-cycle 100 \\
        --cycles 20
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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess

from recon_lite.graph import Graph, Node, NodeType, NodeState
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, EpisodeSummary
from recon_lite.models.registry import TopologyRegistry
from recon_lite.learning.m5_structure import (
    StructureLearner,
    compute_branching_metrics,
    BACKBONE_NODES,
)
from recon_lite.viz.evolution_viz import (
    diff_topologies,
    render_evolution_snapshot,
    save_topology_snapshot,
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
    from recon_lite_chess.training.generators import generate_krk_position
    HAS_GENERATORS = True
except ImportError:
    HAS_GENERATORS = False

try:
    from recon_lite_chess.features.krk_features import extract_krk_features
    HAS_FEATURES = True
except ImportError:
    HAS_FEATURES = False


@dataclass
class KRKEvolutionConfig:
    """Configuration for KRK evolution training."""
    topology_path: Path = Path("topologies/krk_legs_topology.json")
    output_dir: Path = Path("reports/krk_evolution")
    games_per_cycle: int = 50
    max_cycles: int = 30
    max_moves_per_game: int = 100
    max_ticks_per_move: int = 50
    
    # Knowledge transfer
    transfer_from: Optional[Path] = None
    transfer_top_n: int = 20  # Top N cells to transfer
    
    # Stem cell config
    stem_cell_max_cells: int = 50
    stem_cell_spawn_rate: float = 0.1
    stem_cell_min_samples: int = 50
    
    # Plasticity
    plasticity_eta: float = 0.05
    
    # Directories
    snapshot_dir: Path = field(default=None)
    trace_dir: Path = field(default=None)
    signature_dir: Path = field(default=None)
    
    def __post_init__(self):
        if self.snapshot_dir is None:
            self.snapshot_dir = self.output_dir / "snapshots"
        if self.trace_dir is None:
            self.trace_dir = self.output_dir / "traces"
        if self.signature_dir is None:
            self.signature_dir = self.output_dir / "signatures"


@dataclass
class CycleResult:
    """Result of one evolution cycle."""
    cycle: int
    games_played: int
    win_rate: float
    sensor_reuse_ratio: float
    promotions: List[str]
    pruned_edges: List[str]
    duration_seconds: float


def create_random_krk_board() -> chess.Board:
    """Create a random KRK position."""
    if HAS_GENERATORS:
        board = generate_krk_position(ensure_winning=True)
        return board
    else:
        # Fallback: simple KRK position
        return chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")


def run_online_phase(
    config: KRKEvolutionConfig,
    registry: TopologyRegistry,
    stem_manager: Optional["StemCellManager"],
    cycle: int,
) -> Tuple[List[EpisodeRecord], Dict[str, Any]]:
    """
    Online Phase: Play KRK games and collect traces.
    """
    from recon_lite import ReConEngine
    from demos.shared.krk_network import build_krk_network
    
    episodes: List[EpisodeRecord] = []
    wins = 0
    draws = 0
    losses = 0
    
    # Track sensor reuse across games
    reuse_ratios: List[float] = []
    
    for game_idx in range(config.games_per_cycle):
        board = create_random_krk_board()
        episode_id = f"cycle{cycle:04d}_game{game_idx:04d}"
        
        # Build fresh KRK network for each game
        graph = build_krk_network()
        engine = ReConEngine(graph)
        
        # Play game
        result, ep, active_cell_ids = play_single_krk_game(
            board=board,
            engine=engine,
            graph=graph,
            stem_manager=stem_manager,
            episode_id=episode_id,
            max_moves=config.max_moves_per_game,
            max_ticks=config.max_ticks_per_move,
        )
        
        ep.notes = ep.notes or {}
        ep.notes["domain"] = "krk"
        episodes.append(ep)
        
        # Track outcome
        if result == "win":
            wins += 1
            # Track sensor reuse for wins
            if stem_manager and HAS_STEM_CELL:
                # Get active transferred cells directly from stem manager
                transferred_active = stem_manager.get_active_transferred_cells()
                
                # Combine with graph-based active cells for tracking
                all_active = list(set(active_cell_ids + transferred_active))
                
                if all_active:
                    stem_manager.track_win_coactivation(all_active, game_won=True)
                
                # Use transfer contribution ratio (simpler, more reliable)
                ratio = stem_manager.compute_transfer_contribution(game_won=True)
                reuse_ratios.append(ratio)
        elif result == "draw":
            draws += 1
        else:
            losses += 1
    
    total = wins + draws + losses
    avg_reuse = sum(reuse_ratios) / len(reuse_ratios) if reuse_ratios else 0.0
    
    # Also get transfer stats for reporting
    transfer_stats = {}
    if stem_manager and HAS_STEM_CELL:
        transfer_stats = stem_manager.get_reuse_stats()
    
    stats = {
        "games_played": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0,
        "sensor_reuse_ratio": avg_reuse,
        "high_reuse_games": sum(1 for r in reuse_ratios if r > 0.5),
        "transfer_stats": transfer_stats,
    }
    
    return episodes, stats


def play_single_krk_game(
    board: chess.Board,
    engine: "ReConEngine",
    graph: Graph,
    stem_manager: Optional["StemCellManager"],
    episode_id: str,
    max_moves: int,
    max_ticks: int,
) -> Tuple[str, EpisodeRecord, List[str]]:
    """
    Play a single KRK game and return (result, episode_record, active_cell_ids).
    """
    import random
    from recon_lite.graph import NodeState
    
    ep = EpisodeRecord(episode_id=episode_id)
    ep.summary = EpisodeSummary()
    
    move_count = 0
    tick_count = 0
    our_color = board.turn
    
    active_cell_ids: List[str] = []
    
    # Set root as requested
    if "krk_root" in graph.nodes:
        graph.nodes["krk_root"].state = NodeState.REQUESTED
    
    while not board.is_game_over() and move_count < max_moves:
        # Reset node states
        for node in graph.nodes.values():
            if node.nid != "krk_root":
                node.state = NodeState.INACTIVE
        
        env = {
            "board": board,
            "our_color": our_color,
            "move_count": move_count,
        }
        
        # Extract KRK features for stem cells
        if HAS_FEATURES:
            features = extract_krk_features(board)
            env["krk_features"] = features.to_dict()
            env["features"] = features.to_vector()
        
        # Run engine
        ticks_this_move = 0
        suggested_move = None
        
        while ticks_this_move < max_ticks and suggested_move is None:
            engine.step(env)
            tick_count += 1
            ticks_this_move += 1
            
            suggested_move = env.get("chosen_move")
        
        # Track active nodes
        for node in graph.nodes.values():
            if node.state in (NodeState.ACTIVE, NodeState.WAITING, NodeState.REQUESTED):
                if node.nid.startswith(("stem_", "TRIAL_", "universal_")):
                    if node.nid not in active_cell_ids:
                        active_cell_ids.append(node.nid)
        
        # Calculate reward
        tick_reward = 0.0
        checkmate = False
        
        if suggested_move:
            try:
                move = chess.Move.from_uci(suggested_move)
                if move in board.legal_moves:
                    board.push(move)
                    move_count += 1
                    
                    # Check for checkmate
                    if board.is_checkmate():
                        checkmate = True
                        tick_reward = 1.0
                    else:
                        # Progress reward based on box shrinkage
                        if HAS_FEATURES:
                            new_features = extract_krk_features(board)
                            box_progress = (64 - new_features.box_area) / 64.0
                            tick_reward = 0.1 + box_progress * 0.3
                            
                            # DRAW SCENT BOOST: Add partial progress rewards
                            # This enables sample collection in draws where Box Method
                            # progress occurred (rook cuts, king edged, etc.)
                            # Enabled via M5_ENABLE_DRAW_SAMPLING=1
                            try:
                                from recon_lite_chess.features.krk_features import get_draw_scent
                                draw_scent = get_draw_scent(board)
                                tick_reward += draw_scent  # Additive boost
                            except ImportError:
                                pass
                    
                    if stem_manager and HAS_STEM_CELL:
                        stem_manager.tick(board, tick_reward, tick_count)
            except Exception:
                tick_reward = -0.1
        
        # Create tick record
        tick_rec = TickRecord(
            tick_id=tick_count,
            board_fen=board.fen(),
            active_nodes=list(active_cell_ids),
            action=suggested_move,
            reward_tick=tick_reward,
        )
        ep.ticks.append(tick_rec)
        
        if checkmate:
            break
        
        if not suggested_move or suggested_move not in [m.uci() for m in board.legal_moves]:
            # Pick random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(random.choice(legal_moves))
                move_count += 1
            else:
                break
        
        # Opponent's turn (random)
        if not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(random.choice(legal_moves))
                move_count += 1
    
    # Determine result
    if board.is_checkmate():
        result = "win" if board.turn != our_color else "loss"
    elif board.is_stalemate() or board.is_insufficient_material():
        result = "draw"
    else:
        result = "draw"  # Timeout
    
    ep.result = {"win": "1-0", "loss": "0-1", "draw": "1/2-1/2"}.get(result, "1/2-1/2")
    
    return result, ep, active_cell_ids


def run_structural_phase(
    config: KRKEvolutionConfig,
    registry: TopologyRegistry,
    stem_manager: Optional["StemCellManager"],
    episodes: List[EpisodeRecord],
    cycle: int,
    current_win_rate: float,
) -> Dict[str, Any]:
    """Structural Phase: Analyze traces and promote cells."""
    learner = StructureLearner(
        registry=registry,
        cooldown_ticks=0,
        min_spike_reward=0.05,  # LOWERED: KRK rewards are small (0.1-0.3)
        signature_dir=config.signature_dir,
    )
    
    if stem_manager is None or not HAS_STEM_CELL:
        return {"spikes_found": 0}
    
    stats = learner.apply_structural_phase(
        stem_manager=stem_manager,
        episodes=episodes,
        max_promotions=3,
        current_win_rate=current_win_rate,
    )
    
    return stats


def run_krk_evolution(config: KRKEvolutionConfig) -> List[CycleResult]:
    """Main KRK evolution training loop."""
    import time
    
    print(f"\n{'='*60}")
    print("KRK Evolution Training Driver")
    print(f"{'='*60}")
    print(f"Topology: {config.topology_path}")
    print(f"Games per cycle: {config.games_per_cycle}")
    print(f"Max cycles: {config.max_cycles}")
    
    if config.transfer_from:
        print(f"Knowledge Transfer: {config.transfer_from}")
    print(f"{'='*60}\n")
    
    # Initialize registry
    registry = TopologyRegistry(config.topology_path)
    
    # Initialize stem cell manager
    stem_manager = None
    if HAS_STEM_CELL:
        if config.transfer_from and config.transfer_from.exists():
            # Load with knowledge transfer
            print(f"  🔗 Loading stem cells with knowledge transfer...")
            stem_manager = StemCellManager.load_with_transfer(
                source_path=config.transfer_from,
                prefix_map={
                    "kpk_sensor_": "universal_sensor_",
                    "kpk_": "universal_",
                    "stem_": "universal_stem_",
                    "TRIAL_": "universal_TRIAL_",
                },
                states_to_transfer=["TRIAL", "MATURE"],
                top_n=config.transfer_top_n,
                new_domain="krk",
            )
            reuse_stats = stem_manager.get_reuse_stats()
            print(f"  ✅ Transferred {reuse_stats['transferred_count']} cells")
        else:
            # Fresh manager
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
    
    # Create directories
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    config.signature_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[CycleResult] = []
    
    for cycle in range(1, config.max_cycles + 1):
        cycle_start = time.time()
        print(f"\n--- Cycle {cycle}/{config.max_cycles} ---")
        
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
        
        # Show sensor reuse ratio (Bridge Metric)
        reuse_ratio = online_stats.get("sensor_reuse_ratio", 0.0)
        transfer_stats = online_stats.get("transfer_stats", {})
        transferred_count = transfer_stats.get("transferred_count", 0)
        
        if transferred_count > 0:
            by_state = transfer_stats.get("transferred_by_state", {})
            trial_count = by_state.get("TRIAL", 0)
            survival_rate = trial_count / transferred_count if transferred_count > 0 else 0.0
            
            emoji = "🔗" if survival_rate > 0.8 else "🌉"
            print(f"    {emoji} Transfer Survival: {survival_rate:.1%} ({trial_count}/{transferred_count} cells in TRIAL)")
            
            if online_stats["wins"] > 0:
                print(f"    📊 Win contribution: {reuse_ratio:.1%}")
            else:
                print(f"    ⚠️ No wins yet - can't measure contribution")
        
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
            current_win_rate=online_stats["win_rate"],
        )
        
        print(f"    Spikes: {struct_stats.get('spikes_found', 0)}")
        print(f"    TRIAL: {struct_stats.get('trial_promotions', 0)}  SOLID: {struct_stats.get('solidified', 0)}")
        
        # KRK Box Method discovery
        box_method = struct_stats.get("krk_box_method", {})
        if box_method.get("tactical_manager_created"):
            print(f"    📦 BOX METHOD MANAGER created!")
        
        cycle_duration = time.time() - cycle_start
        
        result = CycleResult(
            cycle=cycle,
            games_played=online_stats["games_played"],
            win_rate=online_stats["win_rate"],
            sensor_reuse_ratio=reuse_ratio,
            promotions=struct_stats.get("promotions", []),
            pruned_edges=struct_stats.get("pruning_results", []),
            duration_seconds=cycle_duration,
        )
        results.append(result)
        
        print(f"  Completed in {cycle_duration:.1f}s")
        
        # Stem cell stats
        if stem_manager:
            stats = stem_manager.stats()
            print(f"  Stem cells: {stats['total_cells']}")
    
    # Final report
    print(f"\n{'='*60}")
    print("KRK Evolution Complete")
    print(f"{'='*60}")
    
    total_games = sum(r.games_played for r in results)
    avg_win_rate = sum(r.win_rate for r in results) / len(results) if results else 0
    avg_reuse = sum(r.sensor_reuse_ratio for r in results) / len(results) if results else 0
    
    print(f"Total games: {total_games}")
    print(f"Average win rate: {avg_win_rate:.1%}")
    print(f"Average sensor reuse: {avg_reuse:.1%}")
    
    # Save summary
    summary_path = config.output_dir / "krk_evolution_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "total_cycles": len(results),
            "total_games": total_games,
            "avg_win_rate": avg_win_rate,
            "avg_sensor_reuse": avg_reuse,
            "transfer_from": str(config.transfer_from) if config.transfer_from else None,
            "cycles": [
                {
                    "cycle": r.cycle,
                    "win_rate": r.win_rate,
                    "sensor_reuse_ratio": r.sensor_reuse_ratio,
                    "duration": r.duration_seconds,
                }
                for r in results
            ],
        }, f, indent=2)
    
    print(f"Summary: {summary_path}")
    
    # Save stem cells
    if stem_manager and HAS_STEM_CELL:
        stem_path = config.snapshot_dir / "stem_cells.json"
        stem_manager.save(stem_path)
        print(f"Stem cells: {stem_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="KRK Evolution with Knowledge Transfer")
    
    parser.add_argument(
        "--topology", "-t",
        type=Path,
        default=Path("topologies/krk_legs_topology.json"),
        help="Path to KRK topology"
    )
    parser.add_argument(
        "--transfer-from",
        type=Path,
        default=None,
        help="Path to KPK stem cells for knowledge transfer"
    )
    parser.add_argument(
        "--transfer-top-n",
        type=int,
        default=20,
        help="Number of top cells to transfer"
    )
    parser.add_argument(
        "--games-per-cycle", "-g",
        type=int,
        default=50,
        help="Games per cycle"
    )
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=30,
        help="Number of cycles"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("snapshots/krk_evolution"),
        help="Output directory"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (default: timestamp)"
    )
    
    args = parser.parse_args()
    
    # Generate run name
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = args.output_dir / run_name
    
    config = KRKEvolutionConfig(
        topology_path=args.topology,
        output_dir=output_dir,
        games_per_cycle=args.games_per_cycle,
        max_cycles=args.cycles,
        transfer_from=args.transfer_from,
        transfer_top_n=args.transfer_top_n,
    )
    
    run_krk_evolution(config)


if __name__ == "__main__":
    main()

