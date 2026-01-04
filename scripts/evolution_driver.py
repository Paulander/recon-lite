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

# Per-stage cycle configuration (more cycles for harder stages)
STAGE_CYCLES = {
    0: 5,    # SPRINTER - easy, just push
    1: 20,   # ESCORT - learn king support
    2: 30,   # SQUARE_RULE - learn the square
    3: 30,   # BARRIER - learn barrier concept
    4: 40,   # OPPOSITION - key technique
    5: 40,   # SHOULDER - advanced technique
    6: 50,   # TEMPO - timing concepts
    7: 50,   # DISTANT_OPP - hardest
}

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
    stem_cells_load_path: Optional[Path] = None  # Load from previous stage
    
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
    """
    Play a single KPK game and return (result, episode_record).
    
    Uses proper subgraph locking pattern from full_game_train.py.
    """
    from recon_lite.plasticity import init_plasticity_state
    from recon_lite_chess.sensors.structure import summarize_kpk_material
    
    # Sentinel: stay locked while position is KPK
    def kpk_sentinel(env: Dict[str, Any]) -> bool:
        b = env.get("board")
        if not b:
            return False
        summary = summarize_kpk_material(b)
        return bool(summary.get("is_kpk"))
    
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
    
    # Lock into KPK subgraph (this is the key fix!)
    if "kpk_root" in graph.nodes:
        engine.lock_subgraph("kpk_root", kpk_sentinel)
    
    while not board.is_game_over() and move_count < max_moves:
        # Reset node states for fresh move evaluation (but keep lock)
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
        
        # Set up environment
        env = {
            "board": board,
            "our_color": our_color,
            "move_count": move_count,
        }
        
        # Extract features for stem cells if available
        if HAS_FEATURES:
            env["features"] = extract_kpk_features(board)
            env["kpk_features"] = env["features"]
        
        # Run engine steps until move selected
        ticks_this_move = 0
        suggested_move = None
        
        while ticks_this_move < max_ticks:
            engine.step(env)
            tick_count += 1
            ticks_this_move += 1
            
            # Check for suggested move in env (from kpk_move_selector)
            kpk_policy = env.get("kpk", {}).get("policy", {})
            if "suggested_move" in kpk_policy:
                suggested_move = kpk_policy["suggested_move"]
                break
            
            # Also check kqk if pawn promoted
            kqk_policy = env.get("kqk", {}).get("policy", {})
            if "suggested_move" in kqk_policy:
                suggested_move = kqk_policy["suggested_move"]
                break
        
        # Calculate reward for this tick
        # Use move-based reward shaping: fewer moves to promotion = better
        tick_reward = 0.0
        
        # Make move
        move_made = False
        promoted = False
        if suggested_move:
            try:
                move = chess.Move.from_uci(suggested_move)
                if move in board.legal_moves:
                    board.push(move)
                    move_count += 1
                    move_made = True
                    
                    # KPK WIN CONDITION: Promotion!
                    # For KPK training, promotion = success, game ends
                    if move.promotion:
                        promoted = True
                        # Big reward! Scale by efficiency (fewer moves = better)
                        # Max reward at 2 moves (fastest promotion), min at 50 moves
                        efficiency = max(0.5, 1.0 - (move_count - 2) / 48.0)
                        tick_reward = 1.0 * efficiency
                        
                        # Give stem cells the win reward
                        if stem_manager and HAS_STEM_CELL:
                            stem_manager.tick(board, tick_reward, tick_count)
                    else:
                        # Progress reward - pawn moving up ranks is good
                        # Higher ranks get increasing rewards to trigger spikes
                        pawn_rank = 0
                        for sq in board.pieces(chess.PAWN, our_color):
                            pawn_rank = max(pawn_rank, chess.square_rank(sq) if our_color == chess.WHITE else 7 - chess.square_rank(sq))
                        # Rank 6 (about to promote) = 0.5, rank 5 = 0.4, etc.
                        tick_reward = 0.1 + pawn_rank * 0.07
                        
                        if stem_manager and HAS_STEM_CELL:
                            stem_manager.tick(board, tick_reward, tick_count)
            except Exception:
                tick_reward = -0.1  # Bad move attempted
        
        # Create tick record WITH reward
        tick_rec = TickRecord(
            tick_id=tick_count,
            board_fen=board.fen(),
            active_nodes=[n.nid for n in graph.nodes.values() 
                         if n.state in (NodeState.ACTIVE, NodeState.WAITING, NodeState.REQUESTED)],
            action=suggested_move,
            reward_tick=tick_reward,  # Now stored for structural phase!
        )
        ep.ticks.append(tick_rec)
        
        # Exit on promotion
        if promoted:
            break
        
        if not move_made:
            # No valid suggested move, pick any legal
            legal_moves = list(board.legal_moves)
            if legal_moves:
                import random
                board.push(random.choice(legal_moves))
                move_count += 1
            else:
                break  # No legal moves
        
        # Opponent's turn (random)
        if not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                import random
                board.push(random.choice(legal_moves))
                move_count += 1
    
    # Clean up subgraph lock
    if engine.subgraph_lock:
        engine.unlock_subgraph(goal_achieved=promoted or board.is_checkmate())
    
    # Determine result
    # For KPK: promotion = WIN (the goal is to promote safely)
    if promoted:
        result = "win"  # KPK success!
    elif board.is_checkmate():
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
        cooldown_ticks=0,  # Allow promotions every cycle (tick-based cooldown broken)
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
    stem_manager: Any = None,  # Optional: include TRIAL cells
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save evolution snapshot with diff highlighting.
    
    If stem_manager is provided, TRIAL cells are included in the snapshot
    as visualization-only nodes (group="trial").
    
    Returns:
        (topology_json_path, evolution_png_path)
    """
    # Get new snapshot from registry
    new_snapshot = registry.get_snapshot()
    
    # Add TRIAL cells to snapshot for visualization
    if stem_manager and HAS_STEM_CELL:
        trial_cells = stem_manager.get_trial_cells()
        for cell in trial_cells:
            # Create node entry for TRIAL cell  
            node_id = cell.trial_node_id or f"TRIAL_{cell.cell_id}_{cycle}"
            node_entry = {
                "id": node_id,
                "type": "TERMINAL",
                "group": "trial",  # visualization-only marker
                "factory": "recon_lite.learning.m5_structure:create_pattern_sensor",
                "meta": {
                    "cell_id": cell.cell_id,
                    "xp": cell.xp,
                    "xp_successes": cell.xp_successes,
                    "xp_failures": cell.xp_failures,
                    "tier": "trial",
                    "samples": len(cell.samples),
                    "consistency": getattr(cell, "trial_consistency", 0),
                },
                "transient": True,  # Mark as visualization-only
            }
            # Add to nodes dict (keyed by node_id)
            new_snapshot["nodes"][node_id] = node_entry
            
            # Add edge from parent to TRIAL cell
            parent_id = getattr(cell, "trial_parent_id", None) or "kpk_detect"
            edge_key = f"{parent_id}->{node_id}:SUB"
            edge_entry = {
                "src": parent_id,
                "dst": node_id,
                "type": "SUB",
                "weight": 0.5,  # Trial weight
            }
            new_snapshot["edges"][edge_key] = edge_entry
    
    # Compute diff
    diff = diff_topologies(old_snapshot, new_snapshot)
    
    # Save JSON snapshot
    json_path = save_topology_snapshot(registry, config.snapshot_dir, cycle, new_snapshot)
    
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
        # Load from previous stage if path exists
        if config.stem_cells_load_path and config.stem_cells_load_path.exists():
            try:
                stem_manager = StemCellManager.load(config.stem_cells_load_path)
                print(f"  Loaded {len(stem_manager.cells)} stem cells from previous stage")
            except Exception as e:
                print(f"  Warning: Could not load stem cells: {e}")
                stem_manager = None
        
        # Create fresh manager if no load or load failed
        if stem_manager is None:
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
        
        # Three-tier lifecycle stats
        trial_count = struct_stats.get('trial_promotions', 0)
        solid_count = struct_stats.get('solidified', 0)
        demote_count = struct_stats.get('demoted', 0)
        print(f"    → TRIAL: {trial_count}  SOLID: {solid_count}  DEMOTED: {demote_count}")
        
        # Show trial errors if any
        for err in struct_stats.get('trial_errors', [])[:2]:
            print(f"    ⚠ {err}")
        
        # Snapshot
        print("  Saving snapshot...")
        json_path, png_path = save_cycle_snapshot(
            config=config,
            registry=registry,
            old_snapshot=old_snapshot,
            cycle=cycle,
            stem_manager=stem_manager,  # Include TRIAL cells for visualization
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
    
    # Save stem cells for next stage
    if stem_manager and HAS_STEM_CELL:
        stem_cells_path = config.snapshot_dir / "stem_cells.json"
        stem_manager.save(stem_cells_path)
        print(f"Stem cells saved to: {stem_cells_path}")
    
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
        "--run-name", "-n",
        type=str,
        default=None,
        help="Name for this run (creates unique folder). Default: timestamp"
    )
    parser.add_argument(
        "--stage", "-s",
        type=int,
        default=0,
        help="Starting curriculum stage (0-7)"
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        default=None,
        help="End stage (inclusive). If set, runs all stages from --stage to --end-stage"
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Run all 8 curriculum stages sequentially (inherits weights)"
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.9,
        help="Win rate to advance to next stage (default: 0.9)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (10 games, 2 cycles)"
    )
    
    args = parser.parse_args()
    
    # Generate unique run name if not provided
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine stage range
    start_stage = args.stage
    if args.all_stages:
        end_stage = 7
    elif args.end_stage is not None:
        end_stage = args.end_stage
    else:
        end_stage = args.stage
    
    # Run stages sequentially
    prev_topology_path = args.topology
    prev_stem_cells_path = None  # Track stem cells path between stages
    
    for stage_idx in range(start_stage, end_stage + 1):
        stage_name = f"stage{stage_idx}"
        
        # Create stage-specific directories under run name
        base_snap = Path("snapshots/evolution") / run_name / stage_name
        base_trace = Path("traces/evolution") / run_name / stage_name
        base_output = args.output_dir / run_name / stage_name
        
        # Get cycles for this stage (use per-stage config, or CLI arg as override)
        stage_cycles = STAGE_CYCLES.get(stage_idx, args.cycles) if args.cycles == 20 else args.cycles
        
        config = EvolutionConfig(
            topology_path=prev_topology_path,
            games_per_cycle=10 if args.quick else args.games_per_cycle,
            max_cycles=2 if args.quick else stage_cycles,
            output_dir=base_output,
            snapshot_dir=base_snap,
            trace_dir=base_trace,
            current_stage_idx=stage_idx,
            stage_promotion_threshold=args.win_threshold,
            stem_cells_load_path=prev_stem_cells_path,  # Inherit stem cells
        )
        
        # Ensure directories exist
        config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        config.trace_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print stage info
        print(f"\n{'#'*60}")
        print(f"# STAGE {stage_idx}: {KPK_STAGES[stage_idx].name.upper()}")
        print(f"# {KPK_STAGES[stage_idx].description}")
        print(f"{'#'*60}")
        print(f"Run name: {run_name}/{stage_name}")
        
        results = run_evolution_training(config)
        
        # Get last topology for next stage (weight inheritance!)
        last_cycle_snap = config.snapshot_dir / f"cycle_{config.max_cycles:04d}.json"
        if last_cycle_snap.exists():
            prev_topology_path = last_cycle_snap
            print(f"  → Weights inherited from: {last_cycle_snap}")
        
        # Get stem cells path for next stage (stem cell inheritance!)
        stem_cells_snap = config.snapshot_dir / "stem_cells.json"
        if stem_cells_snap.exists():
            prev_stem_cells_path = stem_cells_snap
        
        # Check if we should stop early (win rate too low)
        if results:
            avg_win = sum(r.win_rate for r in results) / len(results)
            if avg_win < 0.5 and stage_idx < end_stage:
                print(f"\n⚠️  Average win rate {avg_win:.1%} < 50%. Consider more training before advancing.")


if __name__ == "__main__":
    main()

