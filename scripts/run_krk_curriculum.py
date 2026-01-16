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
import cProfile
import io
import json
import os
import pstats
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess

from recon_lite.graph import Graph, Node, NodeType, NodeState, LinkType
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

# ========== PLASTICITY IMPORTS (M3/M4 weight adaptation) ==========
try:
    from recon_lite.plasticity.fast import (
        init_plasticity_state,
        update_eligibility,
        apply_fast_update,
        reset_episode,
        PlasticityConfig,
    )
    HAS_PLASTICITY = True
except ImportError:
    HAS_PLASTICITY = False

# ========== CONSOLIDATION IMPORTS (M4: cross-game weight persistence) ==========
try:
    from recon_lite.plasticity.consolidate import (
        ConsolidationEngine,
        ConsolidationConfig,
        EdgeConsolidationState,
    )
    HAS_CONSOLIDATION = True
except ImportError:
    HAS_CONSOLIDATION = False
    ConsolidationEngine = None
    ConsolidationConfig = None
    PlasticityConfig = None
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
    min_cycles_per_stage: int = 3      # NEW: Force at least 3 cycles even at 100% win
    spawn_cooldown_cycles: int = 3     # NEW: Only spawn every 3 cycles to prevent bloat
    
    # Directories
    output_dir: Path = field(default_factory=lambda: Path("snapshots/evolution/krk_curriculum"))
    trace_dir: Path = field(default_factory=lambda: Path("traces/krk_curriculum"))
    
    # Game settings
    max_moves_per_game: int = 40  # Increased from 30 for Stage 2+ approach
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
    
    # ========== TRACK EDGE ACTIVATIONS FOR CONSOLIDATION ==========
    edge_activations: Dict[str, float] = {}  # Track which edges fired during game
    
    # ========== PLASTICITY INITIALIZATION (M3: per-game weight adaptation) ==========
    plasticity_state = None
    plasticity_config = None
    if HAS_PLASTICITY:
        plasticity_config = PlasticityConfig(
            eta_tick=0.1,  # Learning rate for fast updates
            lambda_decay=0.7,  # Eligibility trace decay
            w_min=0.1,
            w_max=3.0,
            enabled=True,
        )
        plasticity_state = init_plasticity_state(graph)
        # print(f"[PLASTICITY] Initialized {len(plasticity_state)} edge states")
    
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
    
    # Lock the subgraph with enough room for deep POR propagation
    try:
        engine.lock_subgraph("krk_root", krk_sentinel, max_internal_ticks=30, min_internal_ticks=10)
    except ValueError:
        # Subgraph might already be locked or not exist
        pass
    
    # EARLY EXIT: Stage-specific move limits (fail fast if no mate)
    stage_move_limits = {
        "Mate_In_1": 3,
        "Mate_In_2": 40,  # Increased from 5 - needs approach moves
        "Mate_In_3": 40,
        "Edge_Fence_Knight": 40,  # Approach stages need more moves
        "Edge_Fence_Approach": 40,
        "Edge_Fence_Deep": 40,
        "Box_Method": 40,
        "Edge_Confinement": 40,
    }
    max_moves_for_stage = stage_move_limits.get(stage.name, config.max_moves_per_game)
    
    while move_count < min(config.max_moves_per_game, max_moves_for_stage):
        if board.is_game_over():
            break
        
        # Build environment
        env = {"board": board}
        # Pass stem_manager for arbiter access (Phase B deep wiring)
        if stem_manager:
            env["__stem_manager__"] = stem_manager
        
        # Reset engine states before each decision
        engine.reset_states()
        
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
            
            # Track edge activations for consolidation (M4)
            edge_list = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
            for e in edge_list:
                src_node = graph.nodes.get(e.src)
                dst_node = graph.nodes.get(e.dst)
                if src_node and dst_node:
                    src_ok = src_node.state in (NodeState.TRUE, NodeState.CONFIRMED)
                    dst_ok = dst_node.state in (NodeState.REQUESTED, NodeState.TRUE, NodeState.CONFIRMED)
                    if src_ok and dst_ok:
                        edge_key = f"{e.src}->{e.dst}:{e.ltype.name}"
                        edge_activations[edge_key] = edge_activations.get(edge_key, 0.0) + 1.0
            
            # Check for suggested move after min ticks
            # BUG FIX: Also check immediately for checkmates (Stage 0 should win instantly)
            krk_data = env.get("krk_root", {})
            policy = krk_data.get("policy", {})
            found_move = policy.get("suggested_move")
            if found_move:
                suggested_move = found_move  # Capture the move
                # For Stage 0 (mate-in-1), break immediately when we find it
                if ticks_this_move >= config.min_internal_ticks:
                    break
                # Also break if we found a mate (board.is_checkmate() after applying it)
                try:
                    test_move = chess.Move.from_uci(found_move)
                    if test_move in board.legal_moves:
                        test_board = board.copy()
                        test_board.push(test_move)
                        if test_board.is_checkmate():
                            break  # Immediate win - no need to continue ticks
                except Exception:
                    pass
        
        # Make move
        legal_ucis = [m.uci() for m in board.legal_moves]
        if suggested_move and suggested_move in legal_ucis:
            move = chess.Move.from_uci(suggested_move)
        else:
            # ================================================================
            # HEURISTIC FALLBACK (controlled by M5_HEURISTIC_PROB env var)
            # Set M5_HEURISTIC_PROB=0.0 for pure ReCoN mode
            # Set M5_HEURISTIC_PROB=0.5 for 50% heuristic bootstrap
            # ================================================================
            heuristic_prob = float(os.environ.get("M5_HEURISTIC_PROB", "1.0"))
            use_heuristic = random.random() < heuristic_prob
            
            legal = list(board.legal_moves)
            if not legal:
                break
            
            if use_heuristic:
                # Use simple box-shrinking heuristic as fallback
                # PHASE B: Inject stem cell XP-weighted scoring
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
                    
                    # ================================================================
                    # PHASE B: Stem Cell XP-Weighted Move Scoring
                    # Scale scores by avg XP of TRIAL cells that match this state
                    # Bias toward temporal cells (decreasing box_area)
                    # ================================================================
                    if stem_manager and not board.is_checkmate():
                        try:
                            stem_bonus = 0.0
                            cell_count = 0
                            new_box = box_min_side(board)
                            
                            for cell in stem_manager.cells.values():
                                # Only consider TRIAL cells with XP
                                if hasattr(cell, 'state') and cell.state.name == "TRIAL" and cell.xp > 0:
                                    # Use trial_consistency as activation proxy (0-1 range)
                                    consistency = getattr(cell, 'trial_consistency', 0.5)
                                    
                                    # XP-weighted contribution (higher XP = more influence)
                                    weight = min(cell.xp / 10.0, 1.0)  # Normalize: XP 10+ = full weight
                                    
                                    # TEMPORAL BIAS: Boost moves that shrink box
                                    # This teaches the network to prefer constricting moves
                                    temporal_boost = 1.0
                                    if new_box < initial_box_min:
                                        temporal_boost = 1.5  # 50% boost for box shrinking
                                    
                                    stem_bonus += weight * consistency * temporal_boost
                                    cell_count += 1
                            
                            # Apply stem bonus (scaled by cell count for normalization)
                            if cell_count > 0:
                                avg_bonus = stem_bonus / cell_count
                                # LOG: Show when bonus is applied
                                log_bonus = os.environ.get("LOG_STEM_BONUS", "0") == "1"
                                if log_bonus:
                                    print(f"    [STEM-FALLBACK] cells={cell_count}, avg_bonus={avg_bonus:.2f}, score_before={score}")
                                # Threshold: only boost if avg > 0.3 (lower for more influence)
                                if avg_bonus > 0.3:
                                    score += avg_bonus * 25  # Up to +25 from stem cells
                                    if log_bonus:
                                        print(f"    [STEM-FALLBACK] Applied +{avg_bonus * 25:.1f} -> score={score}")
                        except Exception as e:
                            if os.environ.get("LOG_STEM_BONUS", "0") == "1":
                                print(f"    [STEM-FALLBACK] Error: {e}")
                    
                    board.pop()
                    if score > best_score:
                        best_score = score
                        best_move = m
                move = best_move if best_move else random.choice(legal)
            else:
                # PURE MODE: Random move when engine doesn't suggest
                # This forces M5 to learn instead of relying on heuristics
                move = random.choice(legal)
        
        board.push(move)
        move_count += 1
        game_tick += 1
        
        # Feed observation to stem cells (M5 learning)
        if stem_manager:
            # Calculate interim reward based on position quality
            current_box = box_min_side(board)
            box_shrink = initial_box_min - current_box  # Positive if shrunk
            
            if board.is_checkmate():
                interim_reward = 1.0
            elif board.is_check():
                interim_reward = 0.6
            elif box_shrink > 0:
                interim_reward = 0.5  # Good - box is shrinking
            else:
                interim_reward = 0.2  # Neutral position
            
            # MOVE PENALTY: -0.01 per move to encourage tighter mates
            interim_reward -= 0.01 * move_count
            interim_reward = max(interim_reward, 0.0)  # Floor at 0
            
            # PARTIAL XP REWARDS: Update TRIAL cells based on box shrink delta
            # This gives cells XP credit for good moves, not just game outcomes
            if box_shrink > 0:
                xp_delta = 0.1 * abs(box_shrink)  # +0.1 per unit of box shrink
                for cell in stem_manager.cells.values():
                    if hasattr(cell, 'state') and cell.state.name == "TRIAL":
                        cell.xp = int(cell.xp + xp_delta)
            
            # Feed to all exploring stem cells
            for cell in stem_manager.cells.values():
                cell.observe(board, interim_reward, tick=game_tick)
            
            # ========== LAG TERMINAL UPDATES (Temporal Gates) ==========
            # Update lag terminals with feature vector changes for trend detection
            try:
                features = stem_manager.feature_extractor(board)
                if features:
                    # Update all lags and get deltas
                    deltas = stem_manager.update_lags(features)
                    
                    # Evaluate comparators to detect trends
                    trend_signals = stem_manager.evaluate_comparators(deltas)
                    
                    # Record comparator reward correlation for XP
                    for comp_id, fired in trend_signals.items():
                        if comp_id in stem_manager.comparators:
                            stem_manager.comparators[comp_id].record_reward(interim_reward)
            except Exception:
                pass  # Skip lag update on error
        
        # ========== PLASTICITY UPDATE (M3: per-tick edge weight adaptation) ==========
        if HAS_PLASTICITY and plasticity_state and plasticity_config:
            # Get fired edges from active nodes
            fired_edges = []
            for nid, node in graph.nodes.items():
                if node.state in (NodeState.ACTIVE, NodeState.TRUE, NodeState.CONFIRMED):
                    # Find outgoing edges that fired
                    edge_iter = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
                    for edge in edge_iter:
                        if edge.src == nid:
                            fired_edges.append({
                                "src": edge.src,
                                "dst": edge.dst,
                                "ltype": edge.ltype.name,
                            })
            
            # Update eligibility traces
            update_eligibility(
                plasticity_state,
                fired_edges,
                plasticity_config.lambda_decay,
            )
            
            # Apply fast weight updates based on reward
            # BACKWARD CHAINING: Use LEARNED SENSOR PATTERNS, not hardcoded checks
            # When a mastered stage's sensor fires, give bonus reward
            if board.is_checkmate():
                reward_signal = 1.0
            else:
                reward_signal = 0.0
                
                # Check if any mastered sensor pattern matches current position
                # This is the extensible backward chaining mechanism
                if mastered_sensors and stem_manager:
                    current_features = stem_manager.feature_extractor(board)
                    
                    for stage_id, sensor_info in mastered_sensors.items():
                        pattern_sig = sensor_info.get("pattern_signature")
                        if pattern_sig and current_features:
                            # Compute cosine similarity between current features and mastered pattern
                            if len(current_features) == len(pattern_sig):
                                dot_product = sum(f * p for f, p in zip(current_features, pattern_sig))
                                norm_f = sum(f * f for f in current_features) ** 0.5
                                norm_p = sum(p * p for p in pattern_sig) ** 0.5
                                if norm_f > 0 and norm_p > 0:
                                    similarity = dot_product / (norm_f * norm_p)
                                    
                                    # If high similarity to mastered sensor, give bonus reward
                                    # Trust score modulates the reward (higher trust = higher reward)
                                    if similarity > 0.85:
                                        trust = sensor_info.get("trust_score", 0.9)
                                        sensor_reward = 0.7 * trust * similarity
                                        reward_signal = max(reward_signal, sensor_reward)
                
                # Fallback: Basic progress rewards if no sensor match
                if reward_signal == 0.0:
                    if board.is_check():
                        reward_signal = 0.3
                    elif box_min_side(board) < initial_box_min:
                        reward_signal = 0.1
            
            if reward_signal > 0:
                deltas = apply_fast_update(
                    plasticity_state,
                    graph,
                    reward_signal,
                    plasticity_config.eta_tick,
                    plasticity_config,
                )
        
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
    
    # Compute outcome score for consolidation
    outcome_score = 1.0 if result == "win" else (-0.5 if result == "stalemate" else 0.0)
    
    # ========== PLASTICITY FINAL UPDATE & RESET (M3: end-of-game signal) ==========
    if HAS_PLASTICITY and plasticity_state and plasticity_config:
        # Apply final reward signal based on game outcome
        if result == "win":
            final_reward = 1.0  # Strong positive for wins
        elif result == "stalemate":
            final_reward = -0.5  # Penalty for stalemates (missed win)
        else:
            final_reward = 0.0  # No update for losses/draws
        
        if final_reward != 0:
            apply_fast_update(
                plasticity_state,
                graph,
                final_reward,
                plasticity_config.eta_tick * 2.0,  # Double learning rate for final outcome
                plasticity_config,
            )
        
        # DO NOT reset weights - let them persist across games for learning
        # reset_episode(plasticity_state, graph)  # This would undo the learning!
    
    # Unlock
    try:
        engine.unlock_subgraph("krk_root")
    except:
        pass
    
    # Return edge activations for consolidation
    return result, move_count, box_escaped, active_nodes_log, edge_activations


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
    consolidation_engine = None
    
    # ========== INITIALIZE CONSOLIDATION ENGINE (M4: cross-game weight persistence) ==========
    if HAS_CONSOLIDATION and config.mode == "recon":
        consolidation_config = ConsolidationConfig(
            eta_consolidate=config.consolidate_eta,
            min_episodes=10,  # Apply after 10 games
            enabled=True,
        )
        consolidation_engine = ConsolidationEngine(consolidation_config)
        print(f"  Consolidation enabled: eta={config.consolidate_eta}, min_episodes=10")
    
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
            
            # Initialize consolidation engine with graph edges
            if consolidation_engine:
                # Get all KRK edges AND stem cell edges for consolidation
                # Handle both list and dict formats for graph.edges
                edge_list = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
                
                # Include: KRK edges, stem cell edges (stem_, TRIAL_, SC_, pack_, MANAGER_)
                trackable_edges = []
                for e in edge_list:
                    if e.ltype not in (LinkType.POR, LinkType.SUB):
                        continue
                    edge_key = f"{e.src}->{e.dst}:{e.ltype.name}"
                    src_lower = e.src.lower()
                    dst_lower = e.dst.lower()
                    # Track KRK edges
                    if "krk" in src_lower or "krk" in dst_lower:
                        trackable_edges.append(edge_key)
                    # Track stem cell edges (M5 spawned)
                    elif any(prefix in src_lower or prefix in dst_lower 
                             for prefix in ["stem_", "trial_", "sc_", "pack_", "manager_", "and_", "or_"]):
                        trackable_edges.append(edge_key)
                
                consolidation_engine.init_from_graph(graph, edge_whitelist=trackable_edges)
                print(f"  Consolidation tracking {len(trackable_edges)} KRK edges")
            
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
            # Lower reward threshold so more samples are collected
            stem_config = StemCellConfig(
                min_samples=30, 
                max_samples=500,
                reward_threshold=0.1,  # Lowered from 0.3 to capture more positions
            )
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
    cumulative_stall_games = 0  # Track games at low win rate for pack spawning
    
    # Stem cell persistence paths
    stem_cells_path = config.output_dir / "stem_cells.json"
    
    # Load consolidation state if resuming (for transfer learning across stages)
    if consolidation_engine:
        consolidation_state_path = config.output_dir / "final_consolidation.json"
        if consolidation_state_path.exists():
            try:
                consolidation_engine.load_state(consolidation_state_path)
                consolidation_engine.apply_w_base_to_graph(graph)
                print(f"  Loaded consolidation state from previous run (transfer learning enabled)")
            except Exception as e:
                print(f"  Could not load consolidation state: {e}")
    
    # BACKWARD CHAINING: Store mastered stage sensors for reverse linking
    # When a stage hits 90%+, export its condition sensor for next stage's sentinel
    mastered_sensors = {}  # stage_id -> {"sensor_fn": callable, "pattern_signature": [...], "trust_score": float}
    MASTERY_THRESHOLD = 0.70  # Lowered from 0.90 to enable earlier chaining (Stage 2 at 74.5% can export)
    
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
        start_stage_idx = curriculum.current_stage_id  # Track start position for advancement check
        stage = curriculum.current_stage
        stage_start_games = total_games
        cycle = 0
        win_rate_history = []  # Track last 10 cycles for plateau detection
        
        # Abort thresholds (configurable)
        ABORT_GAMES_THRESHOLD = 2000  # Abort if no progress after this many games
        PLATEAU_DELTA_THRESHOLD = 0.05  # 5% improvement required over window
        PLATEAU_WINDOW = 10  # Check last 10 cycles
        
        print(f"\n{'='*60}")
        print(f"STAGE {curriculum.current_stage_id}: {stage.name}")
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
            cycle_episodes = []  # Collect episodes for M5 structural phase
            
            print(f"\n  Cycle {cycle}/{config.max_cycles_per_stage}")
            
            active_node_counts = {}  # Track which nodes fire
            
            for game_idx in range(config.games_per_cycle):
                # Get position for current stage
                board = curriculum.get_position()
                starting_fen = board.fen()  # DEBUG: Capture starting position
                
                # Play game based on mode
                if config.mode == "recon" and engine and graph:
                    result, move_count, box_escaped, active_log, edge_activations = play_krk_game_recon(
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
                    
                    # ========== CONSOLIDATION: Accumulate episode (M4) ==========
                    if consolidation_engine:
                        # Compute outcome score
                        outcome_score = 1.0 if result == "win" else (-0.5 if result == "stalemate" else 0.0)
                        
                        # Compute reward (same as used in plasticity)
                        optimal_moves = stage.get_optimal_moves(board)
                        reward = krk_reward(
                            won=(result == "win"),
                            move_count=move_count,
                            optimal_moves=optimal_moves,
                            box_grew=box_escaped,
                            stalemate=(result == "stalemate"),
                        )
                        
                        # Create episode summary
                        episode_summary = EpisodeSummary(
                            edge_delta_sums={k: v * outcome_score for k, v in edge_activations.items()},
                            outcome_score=outcome_score,
                            avg_reward_tick=reward,
                            total_reward_tick=reward,
                            reward_tick_count=1,
                        )
                        
                        # Accumulate for consolidation
                        consolidation_engine.accumulate_episode(episode_summary)
                        
                        # Apply consolidation if threshold reached
                        if consolidation_engine.should_apply():
                            # DEBUG: Show consolidation stats before apply
                            edges_with_data = sum(1 for s in consolidation_engine.edge_states.values() if s.episode_count > 0)
                            max_delta = max((abs(s.mean_weighted_delta()) for s in consolidation_engine.edge_states.values() if s.episode_count > 0), default=0)
                            
                            deltas = consolidation_engine.apply_to_graph(graph)
                            
                            # If 0 updates, explain why
                            if len(deltas) == 0 and edges_with_data > 0:
                                print(f"    [CONSOLIDATION] Applied {len(deltas)} weight updates (edges_with_data={edges_with_data}, max_delta={max_delta:.6f} - likely stabilized)")
                            else:
                                print(f"    [CONSOLIDATION] Applied {len(deltas)} weight updates")
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
                
                # Collect episode for M5 structural phase
                episode = {
                    "result": result,
                    "move_count": move_count,
                    "reward": reward,
                    "box_escaped": box_escaped,
                    "optimal_moves": optimal_moves,
                    "affordance_delta": reward - 0.5,  # Use reward as proxy for affordance
                    "active_nodes": list(active_node_counts.keys()) if config.mode == "recon" else [],
                }
                cycle_episodes.append(episode)
                
                # ================================================================
                # PSEUDO-XP FOR HEURISTIC OBSERVATIONS
                # Give TRIAL cells small XP for "observing" even when heuristic moves
                # This breaks signal starvation in 0% stages where engine never acts
                # ================================================================
                if config.enable_m5 and stem_manager:
                    OBSERVATION_XP = 0.05  # Small pseudo-reward for being present
                    for cell in stem_manager.cells.values():
                        if hasattr(cell, 'state') and cell.state.name == "TRIAL":
                            # Grant observation XP based on partial success signals
                            if not box_escaped:  # Didn't make things worse
                                cell.xp += OBSERVATION_XP
                            if result == "win":  # Witnessed a win
                                cell.xp += OBSERVATION_XP * 5  # Bigger boost for wins
                
                
                # Record in curriculum manager
                advanced = curriculum.record_game(
                    won=(result == "win"),
                    move_count=move_count,
                    stalemate=(result == "stalemate"),
                    box_escaped=box_escaped,
                )
                
                # Only allow advancement if min_cycles met (force early learning)
                if advanced and cycle < config.min_cycles_per_stage:
                    print(f"\n  🔒 Win rate met but need min {config.min_cycles_per_stage} cycles (at {cycle})")
                    # Reset stats so it doesn't try to advance again
                    advanced = False
                
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
                    
                    # DEBUG: Show first candidate's stats (M5 now handles all promotions)
                    candidates = [c for c in stem_manager.cells.values() 
                                 if c.state == StemCellState.CANDIDATE and len(c.samples) >= 30]
                    if candidates:
                        cell = candidates[0]
                        consistency, _ = cell.analyze_pattern()
                        can_promote = stem_manager.can_promote_to_trial()
                        print(f"    First candidate: {cell.cell_id}, samples={len(cell.samples)}, consistency={consistency:.2f}, can_promote={can_promote}")
                    
                    # M5 STRUCTURAL PHASE HANDLES ALL PROMOTIONS
                    # (Removed manual promotion loop - M5 uses failure bypass, vertical parenting, etc.)
                    
                    # Also run structural phase with collected episodes
                    # Update cumulative stall counter for failure-mode pack spawning
                    if win_rate < 0.10:
                        cumulative_stall_games += config.games_per_cycle
                    else:
                        cumulative_stall_games = max(0, cumulative_stall_games - config.games_per_cycle // 2)  # Decay on success
                    
                    # Inject cumulative stall count into structure learner
                    if hasattr(struct_learner, '_games_at_current_stage'):
                        struct_learner._games_at_current_stage = cumulative_stall_games
                    
                    struct_result = struct_learner.apply_structural_phase(
                        stem_manager=stem_manager,
                        episodes=cycle_episodes,  # FIXED: Pass actual episodes for affordance spikes and POR discovery
                        max_promotions=3,
                        parent_candidates=["krk_detect", "krk_execute", "krk_rook_leg", "krk_king_leg"],
                        current_win_rate=win_rate,
                        mastered_sensors=mastered_sensors,  # BACKWARD CHAINING: previous stage sensors
                        current_stage_id=start_stage_idx,  # Current stage for chain lookup
                    )
                    
                    # ================================================================
                    # FORCED AND-GATE HOISTING
                    # When win rate is 0% but sensors are active (have vocabulary but no grammar)
                    # ================================================================
                    forced_hoists = []
                    if win_rate == 0.0 and active_node_counts and len(active_node_counts) >= 2:
                        # Find the top 2 most active TRIAL sensors
                        trial_activations = [
                            (node_id, count) for node_id, count in active_node_counts.items()
                            if node_id.startswith("TRIAL_") or node_id.startswith("cluster_")
                        ]
                        trial_activations.sort(key=lambda x: -x[1])
                        
                        if len(trial_activations) >= 2:
                            top_2 = trial_activations[:2]
                            sensor_a_id, count_a = top_2[0]
                            sensor_b_id, count_b = top_2[1]
                            
                            # Both must have significant activations (>100)
                            if count_a > 100 and count_b > 100:
                                # Find corresponding cells
                                cell_a = None
                                cell_b = None
                                for cell in stem_manager.cells.values():
                                    if cell.trial_node_id == sensor_a_id:
                                        cell_a = cell
                                    elif cell.trial_node_id == sensor_b_id:
                                        cell_b = cell
                                
                                if cell_a and cell_b:
                                    # Force-hoist into AND-gate (Coordination Manager)
                                    cluster_id = stem_manager.hoist_cluster(
                                        [cell_a.cell_id, cell_b.cell_id],
                                        graph,
                                        parent_node_id="krk_execute",
                                        aggregation_mode="and",
                                    )
                                    if cluster_id:
                                        forced_hoists.append(cluster_id)
                                        # Mark as forced-crisis hoist
                                        if cluster_id in graph.nodes:
                                            graph.nodes[cluster_id].meta["forced_crisis"] = True
                                            graph.nodes[cluster_id].meta["source_activations"] = [count_a, count_b]
                                        
                                        # Persist to registry
                                        node = graph.nodes.get(cluster_id)
                                        if node:
                                            node_spec = {
                                                "id": cluster_id,
                                                "type": node.ntype.name,
                                                "group": "hoisted",
                                                "factory": None,
                                                "meta": node.meta,
                                            }
                                            try:
                                                registry.add_node(node_spec, tick=current_tick)
                                            except Exception:
                                                pass  # Node may already exist
                                        
                                        print(f"    ⚡ FORCED AND-GATE: {cluster_id} from {sensor_a_id}+{sensor_b_id}")
                                        registry.save()
                    
                    # ================================================================
                    # POR CHAIN CONSTRUCTION (Box Method Sequence)
                    # Look for: Rook_Cuts → King_Approaches → Box_Shrinks
                    # ================================================================
                    if active_node_counts and len(active_node_counts) >= 2:
                        # Identify functional node types
                        cut_sensors = [n for n in active_node_counts if "cut" in n.lower() or "rook" in n.lower()]
                        king_sensors = [n for n in active_node_counts if "king" in n.lower() or "distance" in n.lower()]
                        
                        # If we have both cut-related and king-related sensors active
                        if cut_sensors and king_sensors:
                            cut_id = max(cut_sensors, key=lambda x: active_node_counts[x])
                            king_id = max(king_sensors, key=lambda x: active_node_counts[x])
                            
                            # Check if POR link already exists
                            edge_list_por = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
                            existing_por = any(
                                e.src == cut_id and e.dst == king_id 
                                for e in edge_list_por
                            )
                            
                            if not existing_por and cut_id in graph.nodes and king_id in graph.nodes:
                                # Both must be SCRIPT nodes for POR
                                cut_node = graph.nodes[cut_id]
                                king_node = graph.nodes[king_id]
                                
                                if cut_node.ntype == NodeType.SCRIPT and king_node.ntype == NodeType.SCRIPT:
                                    try:
                                        graph.add_edge(cut_id, king_id, LinkType.POR)
                                        print(f"    🔗 POR CHAIN: {cut_id} → {king_id}")
                                    except Exception as e:
                                        pass  # POR may not be valid between these nodes
                    
                    # Compute branching metrics
                    metrics = compute_branching_metrics(graph)
                    
                    promoted = struct_result.get("trial_promotions", 0)
                    hoisted = struct_result.get("hoisted_count", 0) + len(forced_hoists)
                    depth = metrics.get("max_depth", 1)
                    
                    if promoted > 0 or hoisted > 0 or depth > 1:
                        print(f"    M5: +{promoted} TRIAL, +{hoisted} HOISTED, depth={depth}")
                    
                    # Show stem cell stats
                    cell_stats = stem_manager.stats()
                    by_state = cell_stats.get("by_state", {})
                    exploring = by_state.get("EXPLORING", 0)
                    candidate = by_state.get("CANDIDATE", 0) 
                    trial = by_state.get("TRIAL", 0)
                    total_samples = sum(len(c.samples) for c in stem_manager.cells.values())
                    if len(stem_manager.cells) > 0:
                        print(f"    Cells: {exploring}E/{candidate}C/{trial}T, total_samples={total_samples}")
                    
                    # ================================================================
                    # M5.1 PROACTIVE COMPOSITION SPAWNING
                    # Automatically create AND gates from win-correlated sensor pairs
                    # ================================================================
                    if len(stem_manager.win_coactivation) > 0 and cycle > 2:
                        new_compositions = stem_manager.spawn_compositions_from_correlated_pairs(
                            min_coactivations=20,
                            min_ratio=0.60,
                            max_spawns=2,
                        )
                        if new_compositions:
                            print(f"    M5.1: +{len(new_compositions)} AND-gates from co-occurrence")
                            for comp_id in new_compositions:
                                comp_cell = stem_manager.cells.get(comp_id)
                                if comp_cell:
                                    children = comp_cell.children[:2] if len(comp_cell.children) >= 2 else comp_cell.children
                                    print(f"      🔗 {comp_id}: {children[0]} AND {children[1]}")
                    
                    # ================================================================
                    # LAG TERMINAL SPAWNING FOR HIGH-VARIANCE FEATURES
                    # Enables emergent discovery of temporal trends (e.g., shrinking box)
                    # ================================================================
                    if cycle > 2 and len(stem_manager.feature_variance) > 0:
                        new_lags = stem_manager.spawn_lags_for_high_variance(
                            min_variance=0.05,  # Lower threshold for chess features
                            max_spawns=2,
                        )
                        if new_lags:
                            print(f"    ⏱️ Temporal: +{len(new_lags)} lag terminals for high-variance features")
                            print(f"      → {len(stem_manager.comparators)} comparators active")
                    
                    # REFRESH CONSOLIDATION: Re-init to pick up newly spawned M5 edges
                    if consolidation_engine and (promoted > 0 or hoisted > 0 or len(stem_manager.win_coactivation) > 0):
                        edge_list_refresh = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
                        new_edges = []
                        for e in edge_list_refresh:
                            if e.ltype not in (LinkType.POR, LinkType.SUB):
                                continue
                            edge_key = f"{e.src}->{e.dst}:{e.ltype.name}"
                            if edge_key not in consolidation_engine.edge_states:
                                src_lower = e.src.lower()
                                dst_lower = e.dst.lower()
                                if any(prefix in src_lower or prefix in dst_lower 
                                       for prefix in ["stem_", "trial_", "sc_", "pack_", "manager_", "and_", "or_", "krk"]):
                                    new_edges.append(edge_key)
                                    consolidation_engine.edge_states[edge_key] = EdgeConsolidationState(
                                        edge_key=edge_key,
                                        w_base=1.0,
                                        w_init=1.0,
                                    )
                        if new_edges:
                            print(f"      Graph reused: {len(graph.nodes)} nodes (preserving earlier changes)")
                    
                    # Spawn new sensors if win rate is low (Stall Recovery)
                    # Only spawn every spawn_cooldown_cycles to prevent bloat
                    spawn_allowed = (cycle % config.spawn_cooldown_cycles == 0)
                    if spawn_allowed and win_rate < 0.3 and len(stem_manager.cells) < config.max_trial_slots and HAS_KRK_FEATURES:
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
            if curriculum.current_stage_id > start_stage_idx:
                break
            
            # Force advance if we hit max cycles (allow progression even if win rate not met)
            if cycle >= config.max_cycles_per_stage:
                print(f"\n  Max cycles reached. Force advancing...")
                if not curriculum.force_advance():
                    # Already at final stage - exit
                    print("  (At final stage - training complete)")
                break
            
            # Save cycle snapshot
            snapshot_path = config.output_dir / f"stage{start_stage_idx}" / f"cycle_{cycle:04d}.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get consolidation weights for animation (like KPK does)
            w_base = {}
            if consolidation_engine:
                try:
                    w_base = consolidation_engine.get_all_w_base()
                except Exception:
                    pass
            
            snapshot_data = {
                "stage_id": start_stage_idx,
                "stage_name": stage.name,
                "cycle": cycle,
                "games": config.games_per_cycle,
                "win_rate": win_rate,
                "avg_moves": avg_moves,
                "avg_reward": avg_reward,
                "escape_rate": escape_rate,
                "total_games": total_games,
                "timestamp": datetime.now().isoformat(),
                # Consolidation weights for animation (like KPK epoch training)
                "w_base": w_base,
                "consolidation_applied": consolidation_engine.should_apply() if consolidation_engine else False,
            }
            snapshot_path.write_text(json.dumps(snapshot_data, indent=2))
            
            # ================================================================
            # PLATEAU DETECTION AND ABORT LOGIC
            # Prevents wasting 7500 games at 0% - abort if no progress
            # ================================================================
            win_rate_history.append(win_rate)
            stage_games = total_games - stage_start_games
            
            # Check for plateau after PLATEAU_WINDOW cycles
            should_abort = False
            if len(win_rate_history) >= PLATEAU_WINDOW and stage_games >= ABORT_GAMES_THRESHOLD:
                # Calculate improvement over window
                old_rate = sum(win_rate_history[-PLATEAU_WINDOW:-PLATEAU_WINDOW//2]) / (PLATEAU_WINDOW // 2)
                new_rate = sum(win_rate_history[-PLATEAU_WINDOW//2:]) / (PLATEAU_WINDOW // 2)
                delta = new_rate - old_rate
                
                if delta < PLATEAU_DELTA_THRESHOLD and new_rate < 0.05:
                    # Plateaued at low win rate - abort
                    should_abort = True
                    print(f"\n  ⚠️ PLATEAU DETECTED: {delta:.1%} delta over {PLATEAU_WINDOW} cycles")
                    print(f"     Old: {old_rate:.1%} → New: {new_rate:.1%}")
                    print(f"     Aborting stage after {stage_games} games...")
            
            if should_abort:
                print(f"\n  ABORT: Stage {start_stage_idx} plateaued. Force advancing to next stage.")
                break
        
        # Stage complete - record history
        stage_games = total_games - stage_start_games
        stage_stats = curriculum.stage_stats[start_stage_idx]
        
        stage_history.append({
            "stage_id": start_stage_idx,
            "stage_name": stage.name,
            "games_played": stage_games,
            "cycles": cycle,
            "final_win_rate": stage_stats.win_rate,
            "avg_moves": stage_stats.avg_moves,
            "escape_rate": stage_stats.escape_rate,
        })
        
        print(f"\n  Stage {start_stage_idx} Complete:")
        print(f"    Games: {stage_games}")
        print(f"    Cycles: {cycle}")
        print(f"    Final Win Rate: {stage_stats.win_rate:.1%}")
        
        # Persist stem cells for next stage (knowledge transfer)
        if config.enable_m5 and stem_manager:
            stage_stem_path = config.output_dir / f"stem_cells_stage{start_stage_idx}.json"
            stem_manager.save_stem_cells(stage_stem_path)
            print(f"    Stem cells saved: {len(stem_manager.cells)} cells → {stage_stem_path.name}")
        
        # ================================================================
        # BACKWARD CHAINING: Export mastered sensor on 90%+ mastery
        # This sensor becomes the sentinel for the NEXT stage's packs
        # ================================================================
        if stage_stats.win_rate >= MASTERY_THRESHOLD:
            # Create stage entry sensor callable from the stage's generator
            def make_stage_sensor(stage_config):
                """Create a sensor that detects if we're in this stage's configuration."""
                def sensor_fn(board_features):
                    # Use the stage's FEN generator to detect matching patterns
                    # For now, use a simple pattern signature based on stage characteristics
                    return stage_config.get("pattern_match", lambda x: False)(board_features)
                return sensor_fn
            
            # Extract pattern signature from top-performing TRIAL cells in this stage
            pattern_sig = None
            trust_score = stage_stats.win_rate
            if stem_manager:
                trial_cells = [c for c in stem_manager.cells.values() 
                              if c.state.name == "TRIAL" and len(c.samples) > 0]
                if trial_cells:
                    # Use highest-XP cell's signature
                    best_cell = max(trial_cells, key=lambda c: c.xp)
                    pattern_sig = best_cell.pattern_signature
            
            mastered_sensors[start_stage_idx] = {
                "stage_name": stage.name,
                "sensor_fn": make_stage_sensor({"pattern_match": lambda x: True}),  # Placeholder
                "pattern_signature": pattern_sig,
                "trust_score": trust_score,
                "win_rate": stage_stats.win_rate,
            }
            print(f"    🎯 MASTERED: Exported {stage.name} sensor for backward chaining (trust={trust_score:.2f})")
    
    # Training complete
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Save final consolidation state (for transfer learning to next run)
    if consolidation_engine:
        final_consolidation_path = config.output_dir / "final_consolidation.json"
        try:
            consolidation_engine.save_state(final_consolidation_path)
            print(f"\nFinal consolidation state saved to: {final_consolidation_path}")
            
            # Also save final weights snapshot (for animation)
            final_weights = consolidation_engine.get_all_w_base()
            final_weights_path = config.output_dir / "final_weights.json"
            final_weights_data = {
                "timestamp": datetime.now().isoformat(),
                "total_games": total_games,
                "stages_completed": len(stage_history),
                "w_base": final_weights,
            }
            final_weights_path.write_text(json.dumps(final_weights_data, indent=2))
            print(f"Final weights snapshot saved to: {final_weights_path}")
        except Exception as e:
            print(f"Warning: Could not save consolidation state: {e}")
    
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
    parser.add_argument("--profile", action="store_true",
                        help="Enable cProfile to identify bottlenecks")
    
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
    
    # Run training (with optional profiling)
    if args.profile:
        print("\n🔬 PROFILING ENABLED - will output hotspots\n")
        profiler = cProfile.Profile()
        profiler.enable()
        summary = run_krk_curriculum(config)
        profiler.disable()
        
        # Output profile results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 hotspots
        
        profile_output = s.getvalue()
        print("\n" + "="*60)
        print("PROFILE RESULTS (Top 30 by cumulative time):")
        print("="*60)
        print(profile_output)
        
        # Save to file
        profile_path = config.output_dir / "profile_results.txt"
        profile_path.write_text(profile_output)
        print(f"\nProfile saved to: {profile_path}")
    else:
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

