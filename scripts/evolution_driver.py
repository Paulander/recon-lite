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
    from recon_lite.engine import GatingSchedule
    HAS_GATING_SCHEDULE = True
except ImportError:
    HAS_GATING_SCHEDULE = False
    GatingSchedule = None

# Per-stage cycle configuration (more cycles for harder stages)
# Baby-step stages (1-5) need fewer cycles since positions are fixed/easy
STAGE_CYCLES = {
    0: 5,    # SPRINTER - easy, just push
    1: 3,    # GUARDIAN_E - fixed position, trivial
    2: 3,    # GUARDIAN_D - generalization test
    3: 5,    # STEP_ASIDE - learn unblocking
    4: 8,    # SHOULDERING - interference
    5: 10,   # OPPOSITION_LITE - first real test
    6: 20,   # ESCORT - original Stage 1
    7: 30,   # SQUARE_RULE
    8: 30,   # FRONTAL_BLOCKADE
    9: 40,   # KEY_SQUARES
    10: 40,  # PIVOT
    11: 50,  # CORNER_TRAP
    12: 50,  # ZUGZWANG
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


# =============================================================================
# M5.1 STALL RECOVERY - Configuration
# =============================================================================

# Stall detection thresholds (can be overridden via environment variables)
import os

STALL_THRESHOLD_WIN_RATE = float(os.environ.get("M5_STALL_THRESHOLD_WIN_RATE", "0.10"))
STALL_THRESHOLD_CYCLES = int(os.environ.get("M5_STALL_THRESHOLD_CYCLES", "3"))
STALL_SPAWN_MULTIPLIER = float(os.environ.get("M5_STALL_SPAWN_MULTIPLIER", "2.0"))
ENABLE_SCENT_SHAPING = os.environ.get("M5_ENABLE_SCENT_SHAPING", "1") == "1"
SCENT_REWARD = 0.1  # Reward for draws showing King→Pawn approach

# M5.1 Emergent Spawning - for "Failure Frontier" exploration
ENABLE_EMERGENT_SPAWNING = os.environ.get("M5_ENABLE_EMERGENT_SPAWNING", "0") == "1"
EMERGENT_SPAWN_THRESHOLD_CYCLES = int(os.environ.get("M5_EMERGENT_SPAWN_THRESHOLD_CYCLES", "5"))
EMERGENT_SPAWN_THRESHOLD_WIN_RATE = 0.50  # Below 50% win rate triggers emergent spawning
EMERGENT_SPAWN_COUNT = int(os.environ.get("M5_EMERGENT_SPAWN_COUNT", "10"))

# M5.1 Aggressive Hoisting - lower threshold for speculation
MIN_COACTIVATIONS_FOR_HOIST = int(os.environ.get("M5_MIN_COACTIVATIONS_FOR_HOIST", "50"))

# M5.1 Forced Hierarchy (Gauntlet mode)
ENABLE_FORCED_HOISTING = os.environ.get("M5_ENABLE_FORCED_HOISTING", "0") == "1"
FORCED_HOIST_THRESHOLD_WIN_RATE = float(os.environ.get("M5_FORCED_HOIST_THRESHOLD_WIN_RATE", "0.20"))
FORCED_HOIST_INTERVAL_CYCLES = int(os.environ.get("M5_FORCED_HOIST_INTERVAL_CYCLES", "5"))
LEG_LINK_XP_MULTIPLIER = float(os.environ.get("M5_LEG_LINK_XP_MULTIPLIER", "1.0"))


def compute_king_pawn_distance(board: chess.Board, our_color: chess.Color) -> Optional[int]:
    """
    Compute the Manhattan distance between our King and enemy Pawn's promotion path.
    
    For KPK endgames, this measures how close the King is to supporting
    the Pawn's promotion (or blocking the enemy from doing so).
    
    Returns:
        Distance in squares, or None if position doesn't have K+P vs K
    """
    # Find our king and pawn
    our_king_sq = None
    pawn_sq = None
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        
        if piece.piece_type == chess.KING and piece.color == our_color:
            our_king_sq = sq
        elif piece.piece_type == chess.PAWN and piece.color == our_color:
            pawn_sq = sq
    
    if our_king_sq is None or pawn_sq is None:
        return None
    
    # Compute distance from king to pawn
    king_file = chess.square_file(our_king_sq)
    king_rank = chess.square_rank(our_king_sq)
    pawn_file = chess.square_file(pawn_sq)
    pawn_rank = chess.square_rank(pawn_sq)
    
    # Manhattan distance
    return abs(king_file - pawn_file) + abs(king_rank - pawn_rank)


def compute_scent_reward(
    start_board: chess.Board,
    end_board: chess.Board,
    our_color: chess.Color,
    result: str,
) -> float:
    """
    M5.1 Scent-Based Shaping Reward.
    
    For draws that show King approaching the Pawn's promotion path,
    provide a small positive reward (0.1) to encourage exploration
    in the right direction during stall recovery.
    
    Args:
        start_board: Board state at game start
        end_board: Board state at game end
        our_color: Our color
        result: Game result ("win", "draw", "loss")
        
    Returns:
        Scent reward (0.1 if draw with improved position, 0.0 otherwise)
    """
    # Only apply to draws
    if result != "draw":
        return 0.0
    
    # Compute distance change
    start_dist = compute_king_pawn_distance(start_board, our_color)
    end_dist = compute_king_pawn_distance(end_board, our_color)
    
    if start_dist is None or end_dist is None:
        return 0.0
    
    # Reward if King got closer to Pawn
    if end_dist < start_dist:
        return SCENT_REWARD
    
    return 0.0


def build_graph_for_depth_report(registry: TopologyRegistry) -> Graph:
    """Build graph from registry for depth computation."""
    if HAS_BUILDER:
        return build_graph_from_topology(registry.path, registry)
    return Graph()


def compute_hierarchy_stats(graph: Graph) -> Optional[Dict[str, Any]]:
    """
    Compute hierarchical depth statistics for the graph.
    
    Now uses compute_branching_metrics() from m5_structure for M5 Recursive Branching.
    
    Returns:
        Dict with max_depth, non_backbone_count, busiest_node, busiest_children,
        and new M5 branching metrics (speculative_ands, branching_factor, etc.)
    """
    if not graph.nodes:
        return None
    
    backbone_nodes = {"kpk_root", "kpk_detect", "kpk_execute", "kpk_finish", "kpk_wait"}
    
    # Build parent-children map from SUB edges
    children_map: Dict[str, List[str]] = {}
    parent_map: Dict[str, str] = {}
    
    for src, edges in graph.edges.items():
        for dst, edge_list in edges.items():
            for edge in edge_list:
                if edge.ltype.name == "SUB":
                    if src not in children_map:
                        children_map[src] = []
                    children_map[src].append(dst)
                    parent_map[dst] = src
    
    # Compute max depth (BFS from root)
    max_depth = 0
    root = "kpk_root"
    if root in graph.nodes:
        queue = [(root, 0)]
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for child in children_map.get(node, []):
                queue.append((child, depth + 1))
    
    # Count nodes NOT parented to backbone
    non_backbone_count = 0
    for node_id in graph.nodes:
        parent = parent_map.get(node_id)
        if parent and parent not in backbone_nodes:
            non_backbone_count += 1
    
    # Find busiest node (most children)
    busiest_node = "none"
    busiest_children = 0
    for node_id, children in children_map.items():
        if len(children) > busiest_children:
            busiest_node = node_id
            busiest_children = len(children)
    
    # Get M5 Recursive Branching metrics
    try:
        branching_metrics = compute_branching_metrics(graph, BACKBONE_NODES)
    except Exception:
        branching_metrics = {}
    
    return {
        "max_depth": max_depth,
        "non_backbone_count": non_backbone_count,
        "busiest_node": busiest_node,
        "busiest_children": busiest_children,
        # M5 Recursive Branching metrics
        **branching_metrics,
    }


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
    
    # SPARSITY SLEDGEHAMMER: Max TRIAL slots (reduced from 65 to force competition)
    # With max_trial_slots=15, only the top-performing sensors survive
    max_trial_slots: int = 15
    
    # Curriculum settings
    use_curriculum: bool = True
    current_stage_idx: int = 0
    stage_promotion_threshold: float = 0.8  # Win rate to advance
    min_games_per_stage: int = 50
    use_stockfish_rewards: bool = True
    
    # Engine tick settings (for TRIAL node activation)
    # min_internal_ticks: Mandatory propagation depth before allowing early exit.
    # Set to 3+ during Structural Spurt to ensure TRIAL nodes activate.
    min_internal_ticks: int = 0  # 0 = disabled (legacy behavior)
    
    # Think Harder settings (Bach-Integrated architecture)
    # Pure cognitive mode: no random fallback, just stall and learn
    enable_think_harder: bool = False  # Enable threshold decay escalation
    think_harder_thresholds: tuple = (0.7, 0.5, 0.3, 0.1)  # Confidence thresholds
    enable_curiosity_spawning: bool = False  # Spawn sensors on stall
    curiosity_spawn_count: int = 3  # Sensors to spawn per stall
    pure_cognitive_mode: bool = False  # If True, no random fallback at all
    
    # Progressive gating settings
    enable_progressive_gating: bool = False
    gating_initial_strictness: float = 0.30  # 30% - training wheels
    gating_final_strictness: float = 1.0     # 100% - full Bach
    gating_ramp_games: int = 100             # Full strictness by game 100
    gating_win_based: bool = False           # Increase strictness on wins only
    
    # PHASE 1: Heuristic probability ramp (remove heuristics over stages)
    # Stage 0-2: 0.5 → 0.1, Stage 3+: 0.0
    heuristic_prob_initial: float = 0.5   # Stage 0 heuristic probability
    heuristic_prob_ramp_stage: int = 3    # Stage at which heuristic reaches 0.0
    heuristic_usage_log: bool = True      # Log heuristic usage %
    
    # Performance: skip snapshots for pure metric testing
    skip_snapshots: bool = False          # --no-snapshots flag
    
    def get_heuristic_prob(self) -> float:
        """Get heuristic probability for current stage."""
        if self.current_stage_idx >= self.heuristic_prob_ramp_stage:
            return 0.0
        # Linear ramp from initial to 0 over ramp stages
        progress = self.current_stage_idx / self.heuristic_prob_ramp_stage
        return self.heuristic_prob_initial * (1.0 - progress)


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
    apply_scent_shaping: bool = False,  # M5.1: Stall recovery scent reward
) -> Tuple[List[EpisodeRecord], Dict[str, Any], "Graph"]:
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
    
    # Build graph from registry (includes any TRIAL nodes added during structural phase)
    # CRITICAL FIX: Build from registry, not static topology file
    # This ensures TRIAL nodes persisted to the registry actually appear in the graph
    if HAS_BUILDER:
        # Use registry.path to get the base, then refresh with registry state
        graph = build_graph_from_topology(registry.path, registry)
        
        # Hot-reload any newly added nodes from the registry
        from recon_lite_chess.graph.builder import refresh_graph_from_registry
        changes = refresh_graph_from_registry(graph, registry)
        if changes:
            trial_changes = [k for k in changes.keys() if "TRIAL" in k or "cluster" in k]
            if trial_changes:
                print(f"    [Graph] Hot-loaded {len(trial_changes)} TRIAL/cluster nodes")
    else:
        # Fallback: import KPK network builder
        from recon_lite_chess.scripts.kpk import build_kpk_network
        graph = build_kpk_network()
    
    # Initialize engine with progressive gating schedule (Bach-Integrated)
    gating_schedule = None
    if config.enable_progressive_gating and HAS_GATING_SCHEDULE:
        gating_schedule = GatingSchedule(
            initial_strictness=config.gating_initial_strictness,
            final_strictness=config.gating_final_strictness,
            ramp_games=config.gating_ramp_games,
            win_based=config.gating_win_based,
        )
    engine = ReConEngine(graph, gating_schedule=gating_schedule)
    
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
    
    # PHASE-AWARE: Track per-stage outcomes
    per_stage_wins: Dict[int, int] = {}
    per_stage_games: Dict[int, int] = {}
    
    for game_idx in range(config.games_per_cycle):
        # Generate starting position - use curriculum if enabled
        if config.use_curriculum and HAS_GENERATORS and KPK_STAGES:
            stage_idx = min(config.current_stage_idx, len(KPK_STAGES) - 1)
            gen_board = generate_kpk_curriculum_position(KPK_STAGES[stage_idx])
            fen = gen_board.fen()
        elif HAS_GENERATORS:
            stage_idx = 0  # Default stage
            gen_board = generate_kpk_position()
            fen = gen_board.fen() if hasattr(gen_board, 'fen') else str(gen_board)
        else:
            stage_idx = 0
            # Default KPK position
            fen = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"
        
        board = chess.Board(fen)
        episode_id = f"cycle{cycle:04d}_game{game_idx:04d}"
        
        # Set game number for progressive gating
        engine.set_game_number(game_idx)
        
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
            apply_scent_shaping=apply_scent_shaping,  # M5.1
            min_internal_ticks=config.min_internal_ticks,  # Mandatory tick depth
            # Think Harder settings (Bach-Integrated)
            enable_think_harder=config.enable_think_harder,
            think_harder_thresholds=config.think_harder_thresholds,
            enable_curiosity_spawning=config.enable_curiosity_spawning,
            curiosity_spawn_count=config.curiosity_spawn_count,
            pure_cognitive_mode=config.pure_cognitive_mode,
            # PHASE 1: Heuristic probability ramp
            heuristic_prob=config.get_heuristic_prob(),
        )
        
        # Track wins for win-based gating schedule
        if result == "win":
            engine.record_win()
        
        # Store stage info in episode for later analysis
        ep.notes = ep.notes or {}
        ep.notes["stage_idx"] = stage_idx
        episodes.append(ep)
        
        # Track outcome - global and per-stage
        per_stage_games[stage_idx] = per_stage_games.get(stage_idx, 0) + 1
        if result == "win":
            wins += 1
            per_stage_wins[stage_idx] = per_stage_wins.get(stage_idx, 0) + 1
            
            # Track active cells in winning games for sensor reuse
            if stem_manager and HAS_STEM_CELL:
                active_cells = [
                    tick.active_nodes for tick in ep.ticks
                ]
                # Flatten and dedupe active cells across all ticks
                all_active = set()
                for nodes in active_cells:
                    all_active.update(nodes)
                active_cell_ids = [n for n in all_active if n.startswith(("stem_", "TRIAL_", "universal_"))]
                stem_manager.track_win_coactivation(active_cell_ids, game_won=True)
        elif result == "draw":
            draws += 1
        else:
            losses += 1
    
    total = wins + draws + losses
    
    # Compute per-stage win rates
    per_stage_win_rates: Dict[int, float] = {}
    for stage, games in per_stage_games.items():
        stage_wins = per_stage_wins.get(stage, 0)
        per_stage_win_rates[stage] = stage_wins / games if games > 0 else 0.0
    
    # Compute sensor reuse ratio (for knowledge transfer tracking)
    sensor_reuse_ratio = 0.0
    if stem_manager and HAS_STEM_CELL and hasattr(stem_manager, 'get_reuse_stats'):
        reuse_stats = stem_manager.get_reuse_stats()
        sensor_reuse_ratio = reuse_stats.get('avg_reuse_ratio', 0.0)
    
    stats = {
        "games_played": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0,
        "draw_rate": draws / total if total > 0 else 0,
        # PHASE-AWARE stats
        "per_stage_win_rates": per_stage_win_rates,
        "per_stage_games": per_stage_games,
        # KNOWLEDGE TRANSFER stats
        "sensor_reuse_ratio": sensor_reuse_ratio,
    }
    
    return episodes, stats, graph


def _think_harder(
    engine: "ReConEngine",
    graph: Graph,
    env: Dict[str, Any],
    board: chess.Board,
    stall_count: int,
    thresholds: tuple = (0.7, 0.5, 0.3, 0.1),
    stem_manager: Optional["StemCellManager"] = None,
    enable_curiosity: bool = False,
    curiosity_count: int = 3,
) -> Optional[str]:
    """
    Bach-Integrated "Think Harder" escalation.
    
    When no hypothesis is found, escalate through progressively lower
    confidence thresholds. If still stuck, optionally spawn curiosity sensors.
    
    This is the cognitive alternative to random fallback.
    
    Args:
        engine: ReConEngine for move generation
        graph: ReCoN graph
        env: Environment dict
        board: Current chess board
        stall_count: How many stalls have occurred this game
        thresholds: Confidence thresholds to try (descending)
        stem_manager: Optional stem cell manager for curiosity spawning
        enable_curiosity: Whether to spawn sensors on prolonged stall
        curiosity_count: Number of sensors to spawn
        
    Returns:
        Move UCI string if found, None otherwise (pure stall)
    """
    # Level 1: Threshold decay
    # Try progressively lower confirmation thresholds
    original_threshold = getattr(engine, 'confirmation_threshold', 0.5)
    
    for threshold in thresholds:
        # Note: This would require adding confirmation_threshold to ReConEngine
        # For now, we re-run the engine step and hope activation spreads
        engine.step(env)
        
        # Check for suggested move
        for subgraph_key in ["kpk", "krk", "kqk"]:
            policy = env.get(subgraph_key, {}).get("policy", {})
            if "suggested_move" in policy:
                return policy["suggested_move"]
    
    # Level 2: Curiosity spawning (if enabled and stem manager available)
    if enable_curiosity and stem_manager and HAS_STEM_CELL and stall_count > 3:
        _spawn_curiosity_sensors(stem_manager, board, curiosity_count)
    
    # Pure cognitive stall - no random fallback
    return None


def _spawn_curiosity_sensors(
    stem_manager: "StemCellManager",
    board: chess.Board,
    count: int = 3,
) -> List[str]:
    """
    Spawn new TRIAL sensors targeted at current unexplained position.
    
    This implements "targeted spawning" to Binding Gaps:
    - Extract features from current position
    - Create exploratory sensors focused on this specific pattern
    
    Args:
        stem_manager: Stem cell manager
        board: Current chess board
        count: Number of sensors to spawn
        
    Returns:
        List of spawned cell IDs
    """
    spawned_ids = []
    
    try:
        # Try to extract features for this position
        from recon_lite_chess.features import extract_kpk_features
        features = extract_kpk_features(board)
        
        fen = board.fen()
        fen_hash = hash(fen) % 10000
        
        for i in range(count):
            cell_id = f"curiosity_{fen_hash}_{i}"
            
            # Create cell via manager's create_cell if available
            if hasattr(stem_manager, 'create_cell'):
                # Add some variation to the pattern signature
                import random
                varied_features = [
                    f + random.uniform(-0.1, 0.1) for f in features
                ]
                stem_manager.create_cell(
                    cell_id=cell_id,
                    pattern_signature=varied_features,
                    metadata={
                        "spawned_from": "stall",
                        "target_fen": fen,
                        "curiosity": True,
                    }
                )
                spawned_ids.append(cell_id)
    except Exception:
        pass  # Graceful degradation if feature extraction fails
    
    return spawned_ids


def _play_single_game(
    board: chess.Board,
    engine: "ReConEngine",
    graph: Graph,
    plasticity_cfg: Any,
    stem_manager: Optional["StemCellManager"],
    episode_id: str,
    max_moves: int,
    max_ticks: int,
    apply_scent_shaping: bool = False,  # M5.1: Enable scent reward for draws
    min_internal_ticks: int = 0,  # Mandatory tick depth for TRIAL activation
    # Think Harder settings (Bach-Integrated architecture)
    enable_think_harder: bool = False,
    think_harder_thresholds: tuple = (0.7, 0.5, 0.3, 0.1),
    enable_curiosity_spawning: bool = False,
    curiosity_spawn_count: int = 3,
    pure_cognitive_mode: bool = False,  # If True, no random fallback at all
    # PHASE 1: Heuristic probability (0.0 = no heuristics, 1.0 = always use)
    heuristic_prob: float = 0.5,
) -> Tuple[str, EpisodeRecord]:
    """
    Play a single KPK game and return (result, episode_record).
    
    Uses proper subgraph locking pattern from full_game_train.py.
    
    Args:
        board: Starting chess board
        engine: ReConEngine for move generation
        graph: ReCoN graph
        plasticity_cfg: Plasticity configuration
        stem_manager: Optional stem cell manager
        episode_id: Unique episode identifier
        max_moves: Maximum moves before draw
        max_ticks: Maximum ticks per move
        apply_scent_shaping: M5.1 - If True, apply scent reward for draws
        min_internal_ticks: Mandatory propagation depth before allowing early exit.
                           Set to 3+ during Structural Spurt to ensure TRIAL nodes activate.
        enable_think_harder: Bach - Enable threshold decay escalation
        think_harder_thresholds: Confidence thresholds to try (descending)
        enable_curiosity_spawning: Spawn sensors on prolonged stall
        curiosity_spawn_count: Number of sensors to spawn per stall
        pure_cognitive_mode: If True, no random fallback - pure cognitive stall
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
    
    # M5.1: Save starting position for scent reward calculation
    start_board = board.copy()
    
    # Reset all node states for fresh game
    for node in graph.nodes.values():
        node.state = NodeState.INACTIVE
    
    # Initialize plasticity state
    p_state = init_plasticity_state(graph)
    
    move_count = 0
    tick_count = 0
    our_color = board.turn
    
    # Lock into KPK subgraph (this is the key fix!)
    # min_internal_ticks ensures TRIAL nodes have time to activate before Legs suggest moves
    if "kpk_root" in graph.nodes:
        engine.lock_subgraph(
            "kpk_root", 
            kpk_sentinel,
            min_internal_ticks=min_internal_ticks
        )
    
    while not board.is_game_over() and move_count < max_moves:
        # Reset node states for fresh move evaluation (but keep lock)
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
        
        # Set up environment
        env = {
            "board": board,
            "our_color": our_color,
            "move_count": move_count,
            # HEURISTIC SUPPRESSION: When enabled, disable the "approach" heuristic
            # in king_leg to force reliance on TRIAL sensors for direction.
            # This breaks "vibing" and forces deeper hierarchical reasoning.
            "heuristic_suppression": os.environ.get("KPK_HEURISTIC_SUPPRESSION", "0") == "1",
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
            
            # Explicitly call predicates for nodes that might produce moves
            # This includes strategy actuators, arbiters, and any leg nodes with predicates
            # ReConEngine only calls TERMINAL predicates automatically, so we call others manually
            for nid, node in graph.nodes.items():
                if node.predicate:
                    # Call predicates for strategy/actuator/arbiter nodes
                    if any(x in nid for x in ['strategy', 'arbiter', 'leg', 'selector', 'pawn', 'king']):
                        try:
                            node.predicate(node, env)
                        except Exception:
                            pass
            
            # Check for suggested move in env (from kpk_move_selector or kpk_arbiter)
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
        # FIXED: Include all nodes that participated in this tick:
        # - REQUESTED/WAITING/ACTIVE: Currently processing
        # - TRUE/CONFIRMED: Completed successfully (including TRIAL nodes!)
        # This ensures we capture "Hero Sensors" that unlock gated legs.
        active_states = (
            NodeState.ACTIVE, 
            NodeState.WAITING, 
            NodeState.REQUESTED,
            NodeState.TRUE,       # Completed - critical for TRIAL nodes!
            NodeState.CONFIRMED,  # Confirmed - critical for gating detection!
        )
        
        # Collect active nodes and confirmed nodes separately
        active_node_ids = [n.nid for n in graph.nodes.values() if n.state in active_states]
        confirmed_node_ids = [n.nid for n in graph.nodes.values() if n.state == NodeState.CONFIRMED]
        
        tick_rec = TickRecord(
            tick_id=tick_count,
            board_fen=board.fen(),
            active_nodes=active_node_ids,
            action=suggested_move,
            reward_tick=tick_reward,  # Now stored for structural phase!
        )
        ep.ticks.append(tick_rec)
        
        # INERTIA PRUNING: Track which TRIAL nodes reached CONFIRMED state
        # This is used to prune cells that fire but never contribute to gating
        if stem_manager and HAS_STEM_CELL and confirmed_node_ids:
            stem_manager.mark_cells_confirmed(confirmed_node_ids, current_cycle=move_count // 50)
        
        # Exit on promotion
        if promoted:
            break
        
        if not move_made:
            # No valid suggested move - Think Harder or fallback
            stall_count = getattr(engine, '_stall_count', 0) + 1
            engine._stall_count = stall_count
            
            # Try Think Harder escalation (Bach-Integrated)
            if enable_think_harder:
                think_harder_move = _think_harder(
                    engine=engine,
                    graph=graph,
                    env=env,
                    board=board,
                    stall_count=stall_count,
                    thresholds=think_harder_thresholds,
                    stem_manager=stem_manager,
                    enable_curiosity=enable_curiosity_spawning,
                    curiosity_count=curiosity_spawn_count,
                )
                if think_harder_move:
                    try:
                        move = chess.Move.from_uci(think_harder_move)
                        if move in board.legal_moves:
                            board.push(move)
                            move_count += 1
                            move_made = True
                    except Exception:
                        pass
            
            # Random fallback with probability control (PHASE 1: Heuristic Ramp)
            # heuristic_prob = 0.0 means no random fallback (pure cognitive)
            # heuristic_prob = 1.0 means always use random fallback when stuck
            import random as rnd
            use_heuristic = rnd.random() < heuristic_prob and not pure_cognitive_mode
            
            if not move_made and use_heuristic:
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(rnd.choice(legal_moves))
                    move_count += 1
                    # Track heuristic usage in episode
                    ep.notes = ep.notes or {}
                    ep.notes["heuristic_uses"] = ep.notes.get("heuristic_uses", 0) + 1
                else:
                    break  # No legal moves
            elif not move_made and pure_cognitive_mode:
                # Pure cognitive stall - just skip this move
                # This is the "Cognitive Honesty" mode: no random fallback
                # The game may stall, but that's informative for learning
                pass
            elif not move_made and not use_heuristic:
                # PHASE 1: Heuristic suppressed by probability
                # Track as a "cognitive stall" for later analysis
                ep.notes = ep.notes or {}
                ep.notes["cognitive_stalls"] = ep.notes.get("cognitive_stalls", 0) + 1
        
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
    
    # M3 PLASTICITY: Apply weight updates based on game outcome
    # This enables the arbiter to LEARN which move sources lead to wins
    reward = 1.0 if result == "win" else (-0.5 if result == "loss" else 0.1)
    
    # Update arbiter's learned_weights based on selected source
    arbiter_node = graph.nodes.get("kpk_arbiter")
    if arbiter_node:
        learned_weights = arbiter_node.meta.setdefault("learned_weights", {})
        
        # OPTIMIZATION: Bootstrap promotion bonus for faster learning
        if "promotion_bonus" not in learned_weights:
            learned_weights["promotion_bonus"] = 1.3  # Initial boost
        
        selected_source = arbiter_node.meta.get("selected_source", "")
        
        if selected_source and abs(reward) > 0.1:  # Only update on meaningful results
            # Apply M3 fast weight update (Hebbian-like)
            # OPTIMIZATION: Increased learning rate 0.05 → 0.15
            current = learned_weights.get(selected_source, 1.0)
            delta = 0.15 * reward  # Higher learning rate for faster convergence
            learned_weights[selected_source] = max(0.1, min(3.0, current + delta))
            
            # Also track move-specific learning if available
            last_selected = env.get("last_selected_move", {})
            if last_selected and result == "win":
                move_key = f"{selected_source}:{last_selected.get('move', '')}"
                learned_weights[move_key] = learned_weights.get(move_key, 1.0) + 0.05
                
                # OPTIMIZATION: Stronger promotion boost (+0.08 vs +0.03)
                if last_selected.get("is_promotion"):
                    learned_weights["promotion_bonus"] = learned_weights.get("promotion_bonus", 1.3) + 0.08
            
            ep.notes = ep.notes or {}
            ep.notes["plasticity_delta"] = delta
            ep.notes["learned_source"] = selected_source
    
    # M5.1: Apply scent reward for draws showing King→Pawn approach
    if apply_scent_shaping and result == "draw" and stem_manager and HAS_STEM_CELL:
        scent = compute_scent_reward(start_board, board, our_color, result)
        if scent > 0:
            # Apply scent reward to stem cells
            stem_manager.tick(board, scent, tick_count)
            ep.notes = ep.notes or {}
            ep.notes["scent_reward"] = scent
    
    return result, ep



def run_structural_phase(
    config: EvolutionConfig,
    registry: TopologyRegistry,
    stem_manager: Optional["StemCellManager"],
    episodes: List[EpisodeRecord],
    cycle: int,
    current_win_rate: float = 0.0,  # For Perfect Success bypass
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
    
    # Run structural phase with win rate for Perfect Success bypass
    stats = learner.apply_structural_phase(
        stem_manager=stem_manager,
        episodes=episodes,
        max_promotions=config.max_promotions_per_cycle,
        current_win_rate=current_win_rate,
    )
    
    return stats


def save_cycle_snapshot(
    config: EvolutionConfig,
    registry: TopologyRegistry,
    graph: "Graph",  # NEW: Use graph as source of truth for snapshot
    old_snapshot: Dict[str, Any],
    cycle: int,
    stem_manager: Any = None,  # Optional: include TRIAL cells
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save evolution snapshot with diff highlighting.
    
    If stem_manager is provided, TRIAL cells are included in the snapshot
    as visualization-only nodes (group="trial").
    
    CRITICAL: Uses graph.to_snapshot() to capture ALL nodes including
    dynamically spawned packs (AND/OR gates, etc.)
    
    Returns:
        (topology_json_path, evolution_png_path)
    """
    # Get new snapshot from GRAPH (not registry!) - this captures spawned packs
    new_snapshot = graph.to_snapshot()
    
    # Add/update TRIAL cells in snapshot for visualization
    # NOTE: TRIAL nodes are now added via promote_to_trial() which includes
    # critical metadata like 'subgraph'. This code updates XP stats and adds
    # nodes that might not be in the registry yet.
    if stem_manager and HAS_STEM_CELL:
        trial_cells = stem_manager.get_trial_cells()
        for cell in trial_cells:
            # Create node entry for TRIAL cell  
            node_id = cell.trial_node_id or f"TRIAL_{cell.cell_id}_{cycle}"
            parent_id = getattr(cell, "trial_parent_id", None) or "kpk_detect"
            
            # Extract subgraph from parent_id (critical for subgraph execution)
            subgraph_name = parent_id.split("_")[0] if "_" in parent_id else parent_id
            
            # Preserve existing meta if node already in snapshot
            existing_meta = {}
            if node_id in new_snapshot["nodes"]:
                existing_meta = new_snapshot["nodes"][node_id].get("meta", {})
            
            # Merge: preserve existing meta, update with current XP stats
            merged_meta = existing_meta.copy()
            merged_meta.update({
                "cell_id": cell.cell_id,
                "xp": cell.xp,
                "xp_successes": cell.xp_successes,
                "xp_failures": cell.xp_failures,
                "tier": "trial",
                "samples": len(cell.samples),
                "consistency": getattr(cell, "trial_consistency", 0),
                "subgraph": subgraph_name,  # CRITICAL: Required for subgraph execution
            })
            
            node_entry = {
                "id": node_id,
                "type": "TERMINAL",
                "group": "trial",
                "factory": "recon_lite.learning.m5_structure:create_pattern_sensor",
                "meta": merged_meta,
                "transient": True,
            }
            # Add to nodes dict (keyed by node_id)
            new_snapshot["nodes"][node_id] = node_entry
            
            # Add edge from parent to TRIAL cell if not exists
            edge_key = f"{parent_id}->{node_id}:SUB"
            if edge_key not in new_snapshot["edges"]:
                edge_entry = {
                    "src": parent_id,
                    "dst": node_id,
                    "type": "SUB",
                    "weight": 0.5,
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
                max_trial_slots=config.max_trial_slots,  # SPARSITY: Cap TRIAL tier
            )
    
    # Create output directories
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    config.signature_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[CycleResult] = []
    
    # M5.1 STALL RECOVERY - Track consecutive low win-rate cycles
    stall_counter = 0
    stall_recovery_active = False
    original_spawn_rate = config.stem_cell_spawn_rate
    
    # M5.1 EMERGENT SPAWNING - Track consecutive cycles below 50% (Failure Frontier)
    emergent_spawn_counter = 0
    emergent_spawns_triggered = 0
    
    for cycle in range(1, config.max_cycles + 1):
        cycle_start = time.time()
        
        print(f"\n--- Cycle {cycle}/{config.max_cycles} ---")
        
        # M5.1: Show stall recovery status
        if stall_recovery_active:
            print(f"  🔥 STALL RECOVERY MODE (cycle {stall_counter}/{STALL_THRESHOLD_CYCLES})")
        
        # Get pre-cycle snapshot for diff
        old_snapshot = registry.get_snapshot()
        
        # Online Phase - with scent shaping during stall recovery
        print(f"  Online Phase: Playing {config.games_per_cycle} games...")
        episodes, online_stats, graph = run_online_phase(  # Now returns graph!
            config=config,
            registry=registry,
            stem_manager=stem_manager,
            cycle=cycle,
            apply_scent_shaping=stall_recovery_active and ENABLE_SCENT_SHAPING,  # M5.1
        )
        print(f"    Win rate: {online_stats['win_rate']:.1%}")
        print(f"    Games: {online_stats['wins']}W / {online_stats['draws']}D / {online_stats['losses']}L")
        
        # KNOWLEDGE TRANSFER: Show sensor reuse ratio (Bridge Metric)
        reuse_ratio = online_stats.get('sensor_reuse_ratio', 0.0)
        if reuse_ratio > 0:
            emoji = "🔗" if reuse_ratio > 0.5 else "🌉"
            print(f"    {emoji} Sensor Reuse: {reuse_ratio:.1%}")
            if reuse_ratio > 0.5 and stem_manager and HAS_STEM_CELL:
                # Increase plasticity for general strategies (Bridge success)
                config.plasticity_eta = min(0.15, config.plasticity_eta * 1.2)
        
        # PHASE-AWARE: Show per-stage breakdown
        per_stage = online_stats.get('per_stage_win_rates', {})
        if per_stage:
            stage_parts = [f"S{s}:{r:.0%}" for s, r in sorted(per_stage.items())]
            print(f"    📊 Per-stage: {' | '.join(stage_parts)}")
        
        # =====================================================================
        # M5.1 STALL DETECTION & RECOVERY
        # If win_rate < 10% for 3 consecutive cycles:
        # 1. Double spawn_rate for more exploration
        # 2. Enable scent-based shaping for draws
        # =====================================================================
        current_win_rate = online_stats['win_rate']
        
        if current_win_rate < STALL_THRESHOLD_WIN_RATE:
            stall_counter += 1
            if stall_counter >= STALL_THRESHOLD_CYCLES and not stall_recovery_active:
                # ACTIVATE STALL RECOVERY
                stall_recovery_active = True
                if stem_manager:
                    # Double spawn rate
                    stem_manager.spawn_rate = original_spawn_rate * STALL_SPAWN_MULTIPLIER
                    # Increase plasticity for more aggressive learning
                    config.plasticity_eta = min(0.15, config.plasticity_eta * 1.5)
                print(f"    ⚠️  STALL DETECTED: {stall_counter} cycles < {STALL_THRESHOLD_WIN_RATE:.0%}")
                print(f"    🔥 Activating recovery: spawn_rate={stem_manager.spawn_rate if stem_manager else 'N/A'}, scent_shaping=ON")
        else:
            # Reset stall counter on good performance
            if current_win_rate >= STALL_THRESHOLD_WIN_RATE * 2:  # 20%+ resets
                stall_counter = 0
                if stall_recovery_active:
                    # Deactivate stall recovery, restore original rates
                    stall_recovery_active = False
                    if stem_manager:
                        stem_manager.spawn_rate = original_spawn_rate
                    config.plasticity_eta = 0.05  # Reset to default
                    print(f"    ✅ Stall recovery successful, returning to normal mode")
        
        # =====================================================================
        # M5.1 EMERGENT SPAWNING - "Failure Frontier" Exploration
        # When win_rate stays below 50% for N cycles, spawn new sensors
        # tied to the most active King-leg nodes to explore new hypotheses
        # =====================================================================
        if ENABLE_EMERGENT_SPAWNING:
            if current_win_rate < EMERGENT_SPAWN_THRESHOLD_WIN_RATE:
                emergent_spawn_counter += 1
                
                if emergent_spawn_counter >= EMERGENT_SPAWN_THRESHOLD_CYCLES:
                    # TRIGGER EMERGENT SPAWNING
                    emergent_spawns_triggered += 1
                    print(f"    🌱 EMERGENT SPAWNING triggered (cycle {emergent_spawn_counter}/{EMERGENT_SPAWN_THRESHOLD_CYCLES})")
                    
                    if stem_manager and HAS_STEM_CELL:
                        # Find most active King-leg related sensors
                        king_leg_cells = [
                            c for c in stem_manager.cells.values()
                            if "king" in c.cell_id.lower() or 
                               c.metadata.get("local_root_id", "").startswith("kpk_king")
                        ]
                        
                        # Sort by sample count (most active first)
                        king_leg_cells.sort(key=lambda c: len(c.samples), reverse=True)
                        
                        spawned_count = 0
                        for parent_cell in king_leg_cells[:3]:  # Top 3 most active
                            if parent_cell.state in (StemCellState.TRIAL, StemCellState.MATURE):
                                # Spawn children tied to this King-leg sensor
                                spawn_per_parent = EMERGENT_SPAWN_COUNT // 3
                                new_ids = parent_cell.spawn_neighbors(
                                    stem_manager, 
                                    spawn_count=spawn_per_parent,
                                    target_leg="kpk_king_leg"
                                )
                                spawned_count += len(new_ids)
                        
                        # Also spawn some general exploratory cells
                        general_spawns = stem_manager.spawn_exploratory_cells(
                            count=EMERGENT_SPAWN_COUNT - spawned_count,
                            target_legs=["kpk_king_leg", "kpk_pawn_leg"],
                        )
                        spawned_count += len(general_spawns) if general_spawns else 0
                        
                        print(f"    🌱 Spawned {spawned_count} new sensors for exploration")
                    
                    # Reset counter but don't disable (allow repeated spawning)
                    emergent_spawn_counter = 0
            else:
                # Good performance, reset emergent counter
                if current_win_rate >= EMERGENT_SPAWN_THRESHOLD_WIN_RATE + 0.1:  # 60%+ resets
                    emergent_spawn_counter = 0
        
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
            current_win_rate=online_stats['win_rate'],  # For Perfect Success bypass
        )
        print(f"    Spikes found: {struct_stats['spikes_found']}")
        print(f"    High-impact cells: {struct_stats['high_impact_cells']}")
        
        # Three-tier lifecycle stats
        trial_count = struct_stats.get('trial_promotions', 0)
        solid_count = struct_stats.get('solidified', 0)
        demote_count = struct_stats.get('demoted', 0)
        hoisted_count = len(struct_stats.get('hoisted_clusters', []))
        speculative_count = len(struct_stats.get('speculative_hoists', []))
        forced_count = len(struct_stats.get('forced_hoists', []))
        spawned_count = len(struct_stats.get('spawned_neighbors', []))
        
        # INERTIA PRUNING: Remove TRIAL cells that haven't contributed to CONFIRM signals
        inert_count = 0
        if stem_manager and HAS_STEM_CELL:
            max_inactive = int(os.environ.get("M5_INERTIA_PRUNE_CYCLES", "20"))
            inert_count = stem_manager.prune_inert_cells(cycle, max_inactive=max_inactive)
            if inert_count > 0:
                print(f"    ⚡ Inertia Pruning: {inert_count} idle TRIAL cells removed")
        
        print(f"    → TRIAL: {trial_count}  SOLID: {solid_count}  DEMOTED: {demote_count}  HOISTED: {hoisted_count}  SPAWNED: {spawned_count}")
        
        # M5.1 Forced Hierarchy stats
        if speculative_count > 0 or forced_count > 0:
            print(f"    🔨 Crisis Mode: speculative_hoists={speculative_count}  forced_hoists={forced_count}")
        
        # M5 Recursive Branching stats (new)
        vertical_promo = struct_stats.get('vertical_promotions', 0)
        por_chains = struct_stats.get('discovered_por_chains', 0)
        seq_conf = struct_stats.get('sequence_confidence', 0)
        if vertical_promo > 0 or por_chains > 0:
            print(f"    🌿 Branching: vertical_promos={vertical_promo}  POR_chains={por_chains}  seq_confidence={seq_conf:.2f}")
        
        # Show trial errors if any
        for err in struct_stats.get('trial_errors', [])[:2]:
            print(f"    ⚠ {err}")
        
        # Snapshot (skip if --no-snapshots for faster metric runs)
        json_path, png_path = None, None
        if not config.skip_snapshots:
            print("  Saving snapshot...")
            # Use structural phase graph if available (contains spawned packs)
            # Fallback to online phase graph if structural phase didn't build one
            snapshot_graph = struct_stats.get('graph') or graph
            json_path, png_path = save_cycle_snapshot(
                config=config,
                registry=registry,
                graph=snapshot_graph,  # Prefer structural phase graph with packs
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
        
        # HIERARCHICAL DEPTH REPORT (M5 Recursive Branching Metrics)
        try:
            # Compute graph hierarchy stats
            graph = build_graph_for_depth_report(registry)
            depth_stats = compute_hierarchy_stats(graph)
            if depth_stats:
                # Basic hierarchy stats
                print(f"  📊 Hierarchy: max_depth={depth_stats['max_depth']}  non_backbone={depth_stats['non_backbone_count']}  busiest={depth_stats['busiest_node']}({depth_stats['busiest_children']})")
                
                # M5 Branching metrics (new)
                if depth_stats.get('branching_factor') is not None:
                    spec_ands = depth_stats.get('speculative_ands', 0)
                    branch_factor = depth_stats.get('branching_factor', 0)
                    por_count = depth_stats.get('por_count', 0)
                    por_ratio = depth_stats.get('por_ratio', 0)
                    print(f"  🌿 Branching: spec_ANDs={spec_ands}  branch_factor={branch_factor}  POR_edges={por_count} ({por_ratio:.1%})")
        except Exception:
            pass  # Silent fail on hierarchy computation
    
    # M4 CONSOLIDATION: At stage-end, log and persist learned weights
    try:
        arbiter_spec = registry._nodes.get("kpk_arbiter") if registry else None
        if arbiter_spec and hasattr(arbiter_spec, "meta") and arbiter_spec.meta:
            learned_weights = arbiter_spec.meta.get("learned_weights", {})
            if learned_weights:
                print(f"\n  📊 M4 Consolidation: {len(learned_weights)} learned weight adjustments")
                for k, v in sorted(learned_weights.items())[:5]:  # Show top 5
                    print(f"    {k}: {v:.3f}")
    except Exception:
        pass  # Silent fail on M4 logging
    
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
    parser.add_argument(
        "--min-tick-depth",
        type=int,
        default=0,
        help="Mandatory internal tick depth before allowing move suggestion (0=disabled). "
             "Set to 3+ during Structural Spurt to ensure TRIAL nodes activate."
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Skip saving topology snapshots (faster for pure metric testing)"
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
            min_internal_ticks=args.min_tick_depth,  # Mandatory tick depth for TRIAL activation
            skip_snapshots=args.no_snapshots,  # --no-snapshots for faster metric runs
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

