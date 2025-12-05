#!/usr/bin/env python3
"""M6/M8 Full Game Demo - Play a complete chess game from start to finish.

This demo showcases the M6 goal hierarchy with M8 stem cell integration:
- Ultimate goals (WIN/DRAW/SURVIVE) based on position
- Strategic plans (attack, develop, simplify, etc.)
- Phase-aware activation (opening → middlegame → endgame)
- Plan persistence (inertia and decay)
- Fan-in sensor terminals
- M8: Stem cell pattern discovery during play

Usage:
    uv run python demos/persistent/full_game_demo.py
    uv run python demos/persistent/full_game_demo.py --max-moves 100 --output game.json
    uv run python demos/persistent/full_game_demo.py --vs-random  # Play against random moves
    uv run python demos/persistent/full_game_demo.py --stem-cells  # Enable pattern discovery
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import chess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.graph import Graph, Node, NodeType, NodeState, LinkType
from recon_lite.engine import ReConEngine
from recon_lite.dynamics.persistence import (
    PersistenceConfig,
    apply_persistence_to_node,
    get_active_plans,
    compute_plan_competition,
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
from recon_lite_chess.eval.heuristic import eval_position
from recon_lite_chess.scripts.tactics import (
    detect_forks,
    detect_pins,
    detect_hanging_pieces,
    detect_back_rank_weakness,
    detect_skewers,
    detect_discovered_attacks,
    get_fork_moves,
    get_pin_exploit_moves,
    get_skewer_moves,
    get_capture_hanging_moves,
)
from recon_lite.plasticity.consolidate import ConsolidationEngine, ConsolidationConfig
from recon_lite.logger import RunLogger
from recon_lite.nodes.stem_cell import StemCellManager, StemCellConfig, StemCellState

# Import unified graph building
from recon_lite_chess.graph import (
    build_unified_graph,
    compute_subgraph_gates,
    compute_tactics_context_weights,
    TACTIC_TYPES,
)
from recon_lite_chess.graph.unified_builder import load_all_weights, get_subgraph_summary


@dataclass
class GameState:
    """State tracking for the game."""
    board: chess.Board
    move_history: List[str] = field(default_factory=list)
    phase_history: List[Dict[str, float]] = field(default_factory=list)
    goal_history: List[str] = field(default_factory=list)
    active_plans: List[str] = field(default_factory=list)
    tick: int = 0
    last_eval: float = 0.0  # Track position eval for reward computation


# ============================================================================
# M8: Feature Extractor for Stem Cells
# ============================================================================

def extract_board_features(board: chess.Board) -> List[float]:
    """
    Extract a 128-dimensional feature vector from a chess position.
    
    Features:
    - Material counts (12 features: 6 piece types x 2 colors)
    - Pawn structure (16 features: pawns per file x 2 colors)
    - King safety (8 features: pawn shield, attackers near king)
    - Piece activity (16 features: mobility per piece type)
    - Control (32 features: squares controlled)
    - Center control (4 features: inner center occupation)
    - Phase indicators (8 features)
    - Tactical indicators (32 features)
    
    Total: 128 features
    """
    features = []
    
    # === Material (12 features) ===
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(piece_type, color))
            # Normalize by typical max (8 pawns, 2 minors, 2 rooks, 1 queen, 1 king)
            max_counts = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
                         chess.ROOK: 2, chess.QUEEN: 1, chess.KING: 1}
            features.append(count / max_counts[piece_type])
    
    # === Pawn structure (16 features) ===
    for color in [chess.WHITE, chess.BLACK]:
        file_pawns = [0] * 8
        for sq in board.pieces(chess.PAWN, color):
            file_pawns[chess.square_file(sq)] += 1
        features.extend([p / 2.0 for p in file_pawns])  # Max 2 pawns per file
    
    # === King safety (8 features) ===
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        
        # Pawn shield (3 features)
        shield = 0
        shield_rank = king_rank + 1 if color else king_rank - 1
        if 0 <= shield_rank <= 7:
            for df in [-1, 0, 1]:
                f = king_file + df
                if 0 <= f <= 7:
                    sq = chess.square(f, shield_rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shield += 1
        features.append(shield / 3.0)
        
        # Attackers near king
        attackers = 0
        for df in [-2, -1, 0, 1, 2]:
            for dr in [-2, -1, 0, 1, 2]:
                f, r = king_file + df, king_rank + dr
                if 0 <= f <= 7 and 0 <= r <= 7:
                    sq = chess.square(f, r)
                    if board.is_attacked_by(not color, sq):
                        attackers += 1
        features.append(attackers / 25.0)
        
        # King position (centralized vs castled)
        features.append(abs(king_file - 3.5) / 3.5)
        features.append(king_rank / 7.0 if color else (7 - king_rank) / 7.0)
    
    # === Piece activity (16 features) ===
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            total_mobility = 0
            count = 0
            for sq in board.pieces(piece_type, color):
                total_mobility += len(board.attacks(sq))
                count += 1
            avg_mobility = total_mobility / count if count > 0 else 0
            # Normalize by typical max mobility
            max_mobility = {chess.KNIGHT: 8, chess.BISHOP: 13, 
                          chess.ROOK: 14, chess.QUEEN: 27}
            features.append(avg_mobility / max_mobility[piece_type])
            features.append(count / 2.0)  # Piece count
    
    # === Center control (4 features) ===
    center = [chess.E4, chess.D4, chess.E5, chess.D5]
    for color in [chess.WHITE, chess.BLACK]:
        occupation = 0
        control = 0
        for sq in center:
            piece = board.piece_at(sq)
            if piece and piece.color == color:
                occupation += 1
            if board.is_attacked_by(color, sq):
                control += 1
        features.append(occupation / 4.0)
        features.append(control / 4.0)
    
    # === Phase indicators (8 features) ===
    total_material = sum(
        len(board.pieces(pt, c)) * v 
        for c in [chess.WHITE, chess.BLACK]
        for pt, v in [(chess.QUEEN, 9), (chess.ROOK, 5), 
                      (chess.BISHOP, 3), (chess.KNIGHT, 3)]
    )
    features.append(min(1.0, total_material / 62.0))  # Opening indicator
    features.append(max(0.0, 1.0 - total_material / 31.0))  # Endgame indicator
    
    # Development (pieces off back rank)
    for color in [chess.WHITE, chess.BLACK]:
        back_rank = 0 if color else 7
        developed = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, color):
                if chess.square_rank(sq) != back_rank:
                    developed += 1
        features.append(developed / 4.0)
    
    # Castling rights
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    
    # === Tactical indicators (32 features) ===
    # Checks and captures available
    checks = 0
    captures = 0
    for move in board.legal_moves:
        if board.gives_check(move):
            checks += 1
        if board.is_capture(move):
            captures += 1
    features.append(min(1.0, checks / 5.0))
    features.append(min(1.0, captures / 10.0))
    
    # Hanging pieces (undefended)
    for color in [chess.WHITE, chess.BLACK]:
        hanging = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                    hanging += 1
        features.append(hanging / 5.0)
    
    # Passed pawns
    for color in [chess.WHITE, chess.BLACK]:
        passed = 0
        for pawn_sq in board.pieces(chess.PAWN, color):
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            is_passed = True
            
            for check_file in [file - 1, file, file + 1]:
                if not (0 <= check_file <= 7):
                    continue
                ahead_ranks = range(rank + 1, 8) if color else range(0, rank)
                for check_rank in ahead_ranks:
                    check_sq = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color != color:
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                passed += 1
        features.append(passed / 4.0)
    
    # Pad to exactly 128 features
    while len(features) < 128:
        features.append(0.0)
    
    return features[:128]


def compute_reward_signal(board: chess.Board, prev_eval: float) -> float:
    """
    Compute reward signal based on position evaluation change.
    
    Returns a reward in range [-1, 1] based on:
    - Material changes
    - Position improvement
    - Tactical opportunities
    """
    # Get current evaluation
    current_eval = eval_position(board)
    
    # Reward is based on evaluation improvement
    # From the perspective of the side that just moved (which is now not to move)
    eval_delta = current_eval - prev_eval
    
    # Normalize to [-1, 1]
    reward = max(-1.0, min(1.0, eval_delta / 300.0))
    
    # Bonus for checkmate
    if board.is_checkmate():
        reward = 1.0 if board.turn == chess.BLACK else -1.0
    
    # Small bonus for check
    if board.is_check():
        reward += 0.1 if board.turn == chess.BLACK else -0.1
    
    return max(-1.0, min(1.0, reward))


def generate_board_tags(board: chess.Board) -> Dict[str, Any]:
    """
    Generate board overlay tags for visualization.
    
    Returns a dict with:
    - squares: dict mapping square names to lists of tags
    - arrows: list of arrow specs {from, to, color, label}
    """
    tags = {
        "squares": {},
        "arrows": [],
    }
    
    # Add attacked squares
    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq)
        sq_tags = []
        
        # Check if attacked by white
        if board.is_attacked_by(chess.WHITE, sq):
            sq_tags.append("attacked_by_white")
        if board.is_attacked_by(chess.BLACK, sq):
            sq_tags.append("attacked_by_black")
        
        # Center squares
        if sq in [chess.D4, chess.D5, chess.E4, chess.E5]:
            sq_tags.append("center")
        
        # King zones
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq is not None:
                king_file = chess.square_file(king_sq)
                king_rank = chess.square_rank(king_sq)
                sq_file = chess.square_file(sq)
                sq_rank = chess.square_rank(sq)
                
                if abs(sq_file - king_file) <= 1 and abs(sq_rank - king_rank) <= 1:
                    color_name = "white" if color == chess.WHITE else "black"
                    sq_tags.append(f"{color_name}_king_zone")
        
        if sq_tags:
            tags["squares"][sq_name] = sq_tags
    
    # Detect tactical patterns and add arrows
    try:
        # Forks
        forks = detect_forks(board)
        for fork in forks[:3]:  # Limit to top 3
            if fork.get("forking_square") and fork.get("targets"):
                forking_sq = fork["forking_square"]
                for target in fork["targets"][:2]:
                    tags["arrows"].append({
                        "from": forking_sq,
                        "to": target,
                        "color": "red",
                        "label": "fork",
                    })
        
        # Pins
        pins = detect_pins(board)
        for pin in pins[:3]:
            if pin.get("pinned_square") and pin.get("pinning_piece_square"):
                tags["arrows"].append({
                    "from": pin["pinning_piece_square"],
                    "to": pin["pinned_square"],
                    "color": "orange",
                    "label": "pin",
                })
        
        # Hanging pieces
        hanging = detect_hanging_pieces(board)
        for sq in hanging.get("enemy_hanging", [])[:3]:
            if sq in tags["squares"]:
                tags["squares"][sq].append("hanging")
            else:
                tags["squares"][sq] = ["hanging"]
    except Exception:
        pass  # Don't fail on detection errors
    
    return tags


def build_full_game_graph() -> Graph:
    """Build the complete M6 graph for full game play."""
    g = Graph()
    
    # === Sensor Terminals (Fan-in allowed) ===
    material_sensor = create_material_sensor_node()
    phase_sensor = create_phase_sensor_node()
    ultimate_goal_sensor = create_ultimate_goal_node()
    
    g.add_node(material_sensor)
    g.add_node(phase_sensor)
    g.add_node(ultimate_goal_sensor)
    
    # Development and center control sensors
    dev_sensor = Node(
        "DevelopmentSensor", NodeType.TERMINAL,
        predicate=development_sensor_predicate,
        meta={"fan_in_allowed": True},
    )
    center_sensor = Node(
        "CenterControlSensor", NodeType.TERMINAL,
        predicate=center_control_sensor_predicate,
        meta={"fan_in_allowed": True},
    )
    king_safety_sensor = Node(
        "KingSafetySensor", NodeType.TERMINAL,
        predicate=king_safety_sensor_predicate,
        meta={"fan_in_allowed": True},
    )
    activity_sensor = Node(
        "PieceActivitySensor", NodeType.TERMINAL,
        predicate=piece_activity_sensor_predicate,
        meta={"fan_in_allowed": True},
    )
    
    g.add_node(dev_sensor)
    g.add_node(center_sensor)
    g.add_node(king_safety_sensor)
    g.add_node(activity_sensor)
    
    # === Root and Ultimate Layer ===
    game_root = Node("GameRoot", NodeType.SCRIPT, meta={"layer": "root"})
    g.add_node(game_root)
    
    # Strategy branches based on ultimate goal
    win_strategy = Node(
        "WinStrategy", NodeType.SCRIPT,
        meta={"layer": "ultimate", "goal": "WIN", "alt": True},
    )
    draw_strategy = Node(
        "DrawStrategy", NodeType.SCRIPT,
        meta={"layer": "ultimate", "goal": "DRAW", "alt": True},
    )
    survive_strategy = Node(
        "SurviveStrategy", NodeType.SCRIPT,
        meta={"layer": "ultimate", "goal": "SURVIVE", "alt": True},
    )
    
    g.add_node(win_strategy)
    g.add_node(draw_strategy)
    g.add_node(survive_strategy)
    
    # === Strategic Plans ===
    # Create nodes for key strategic plans
    plan_nodes = {}
    for plan_id in ["Develop", "Castle", "CenterControl", "AttackKing", 
                    "Simplify", "ImproveWorstPiece", "CreateWeakness",
                    "ConvertAdvantage", "WinMaterial", "DefendWeakness"]:
        if plan_id in STRATEGIC_PLANS:
            plan = STRATEGIC_PLANS[plan_id]
            node = Node(
                plan_id, NodeType.SCRIPT,
                meta={
                    "layer": "strategic",
                    "category": plan.category.name,
                    "inertia": plan.default_inertia,
                    "decay": plan.default_decay,
                },
            )
            g.add_node(node)
            plan_nodes[plan_id] = node
    
    # Each plan needs a terminal child - use placeholders
    for plan_id, node in plan_nodes.items():
        placeholder = Node(f"{plan_id}_Terminal", NodeType.TERMINAL)
        g.add_node(placeholder)
        g.add_edge(plan_id, f"{plan_id}_Terminal", LinkType.SUB)
    
    # === Wire Hierarchy ===
    # Root → Ultimate goal assessment
    g.add_edge("GameRoot", "UltimateGoal", LinkType.SUB)
    g.add_edge("GameRoot", "PhaseSensor", LinkType.SUB)
    g.add_edge("GameRoot", "MaterialSensor", LinkType.SUB)
    
    # Root → Strategy branches
    g.add_edge("GameRoot", "WinStrategy", LinkType.SUB)
    g.add_edge("GameRoot", "DrawStrategy", LinkType.SUB)
    g.add_edge("GameRoot", "SurviveStrategy", LinkType.SUB)
    
    # Strategy → Plans
    # Win strategy plans
    for plan_id in ["AttackKing", "WinMaterial", "ConvertAdvantage"]:
        if plan_id in plan_nodes:
            g.add_edge("WinStrategy", plan_id, LinkType.SUB)
    
    # Draw strategy plans (own set)
    for plan_id in ["Develop", "Castle", "CenterControl", "ImproveWorstPiece"]:
        if plan_id in plan_nodes:
            g.add_edge("DrawStrategy", plan_id, LinkType.SUB)
    
    # Survive strategy uses Simplify and DefendWeakness
    for plan_id in ["DefendWeakness", "Simplify"]:
        if plan_id in plan_nodes:
            g.add_edge("SurviveStrategy", plan_id, LinkType.SUB)
    
    # Plans → Sensors (fan-in)
    sensor_links = {
        "Develop": ["DevelopmentSensor", "CenterControlSensor"],
        "Castle": ["DevelopmentSensor"],
        "CenterControl": ["CenterControlSensor"],
        "AttackKing": ["KingSafetySensor", "MaterialSensor"],
        "Simplify": ["MaterialSensor", "PhaseSensor"],
        "ImproveWorstPiece": ["PieceActivitySensor"],
        "ConvertAdvantage": ["MaterialSensor", "PhaseSensor"],
        "WinMaterial": ["MaterialSensor"],
        "DefendWeakness": ["KingSafetySensor"],
    }
    
    for plan_id, sensors in sensor_links.items():
        if plan_id in plan_nodes:
            for sensor_id in sensors:
                if sensor_id in g.nodes:
                    g.add_edge(plan_id, sensor_id, LinkType.SUB)
    
    # Set confirmation policies
    g.set_confirm_policy("GameRoot", policy="or")
    g.set_confirm_policy("WinStrategy", policy="or")
    g.set_confirm_policy("DrawStrategy", policy="or")
    g.set_confirm_policy("SurviveStrategy", policy="or")
    
    return g


def run_parallel_tactics(
    board: chess.Board,
    g: Graph,
    context_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Run all tactics detectors in parallel with context-weighted scoring.
    
    Args:
        board: Current board position
        g: Graph with tactic detector nodes
        context_weights: Context multipliers for each tactic type
        
    Returns:
        List of TacticResult dicts with moves and weights
    """
    results = []
    
    # Detection functions for each tactic type
    # Returns (detected: bool, moves: List[str]) - moves as UCI strings
    def _safe_moves_to_uci(moves):
        """Convert moves to UCI strings safely."""
        result = []
        for m in moves:
            if hasattr(m, 'uci'):
                result.append(m.uci())
            elif isinstance(m, str):
                result.append(m)
        return result
    
    tactic_detectors = {
        "fork": lambda b: (len(detect_forks(b)) > 0, _safe_moves_to_uci(get_fork_moves(b))),
        "pin": lambda b: (len(detect_pins(b)) > 0, _safe_moves_to_uci(get_pin_exploit_moves(b))),
        "skewer": lambda b: (len(detect_skewers(b)) > 0, _safe_moves_to_uci(get_skewer_moves(b))),
        "hangingPiece": lambda b: (
            len(detect_hanging_pieces(b).get("enemy_hanging", [])) > 0,
            _safe_moves_to_uci(get_capture_hanging_moves(b))
        ),
        "discoveredAttack": lambda b: (len(detect_discovered_attacks(b)) > 0, []),
        "backRankMate": lambda b: (detect_back_rank_weakness(b), []),
    }
    
    for tactic_type in TACTIC_TYPES:
        detector_node_id = f"detect_{tactic_type}"
        
        # Check if detector exists in graph
        if detector_node_id not in g.nodes:
            continue
        
        # Run detection (if we have a function for it)
        detector_fn = tactic_detectors.get(tactic_type)
        if not detector_fn:
            # For tactics without explicit detectors, skip
            continue
        
        try:
            detected, moves = detector_fn(board)
        except Exception:
            detected, moves = False, []
        
        if not detected:
            continue
        
        # Get learned edge weight from graph
        learned_weight = 1.0
        for edge in g.edges:
            if edge.src == "tactics_root" and edge.dst == detector_node_id:
                learned_weight = float(getattr(edge, "w", 1.0) or 1.0)
                break
        
        # Apply context multiplier
        context_mult = context_weights.get(tactic_type, context_weights.get("base", 1.0))
        final_weight = learned_weight * context_mult
        
        # Update node state for visualization
        g.nodes[detector_node_id].state = NodeState.ACTIVE
        g.nodes[detector_node_id].meta["detected"] = True
        g.nodes[detector_node_id].meta["activation"] = min(1.0, final_weight)
        
        results.append({
            "tactic": tactic_type,
            "moves": moves,
            "learned_weight": learned_weight,
            "context_weight": context_mult,
            "final_weight": final_weight,
        })
    
    return results


def select_move(
    board: chess.Board,
    goal: UltimateGoal,
    phase_weights: Dict[str, float],
    active_plans: List[Tuple[str, float]],
    tactic_results: Optional[List[Dict[str, Any]]] = None,
) -> Optional[chess.Move]:
    """
    Select a move based on active plans, goal, and tactical opportunities.
    
    Integrates:
    - Opening/middlegame heuristics
    - Strategic plan activations
    - Parallel tactics detection results
    
    Args:
        board: Current board position
        goal: Ultimate goal (WIN/DRAW/SURVIVE)
        phase_weights: Game phase weights
        active_plans: List of (plan_id, activation) tuples
        tactic_results: Results from parallel tactics scan
        
    Returns:
        Selected chess.Move or None
    """
    if not board.legal_moves:
        return None
    
    move_scores: Dict[chess.Move, float] = {}
    
    # Get phase for candidate generation
    dominant_phase = max(phase_weights.keys(), key=lambda k: phase_weights[k])
    
    # === Phase 1: Integrate tactical move suggestions ===
    if tactic_results:
        for result in tactic_results:
            moves = result.get("moves", [])
            weight = result.get("final_weight", 1.0)
            
            for move_uci in moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        # Tactics get high priority
                        move_scores[move] = move_scores.get(move, 0) + weight * 2.0
                except Exception:
                    continue
    
    # === Phase 2: Collect candidates from opening heuristics ===
    if dominant_phase == "opening" or phase_weights.get("opening", 0) > 0.3:
        opening_candidates = get_opening_move_candidates(board)
        for move, reason, score in opening_candidates:
            move_scores[move] = move_scores.get(move, 0) + score * phase_weights.get("opening", 0.5)
    
    # === Phase 3: Collect candidates from middlegame heuristics ===
    if dominant_phase == "middlegame" or phase_weights.get("middlegame", 0) > 0.3:
        for plan_id, activation in active_plans[:3]:  # Top 3 plans
            plan_type = "general"
            if "Attack" in plan_id:
                plan_type = "attack"
            elif "Simplify" in plan_id:
                plan_type = "simplify"
            elif "Improve" in plan_id:
                plan_type = "improve"
            
            mg_candidates = get_middlegame_move_candidates(board, plan=plan_type)
            for move, reason, score in mg_candidates:
                move_scores[move] = move_scores.get(move, 0) + score * activation
    
    # === Phase 4: Add base score for captures ===
    for move in board.legal_moves:
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                value = {
                    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9
                }.get(captured.piece_type, 0)
                move_scores[move] = move_scores.get(move, 0) + value * 0.5
    
    # === Fallback: if no candidates, use heuristic eval ===
    if not move_scores:
        best_move = None
        best_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = -eval_position(board)  # Negamax perspective
            board.pop()
            if score > best_eval:
                best_eval = score
                best_move = move
        return best_move
    
    # Select move with highest score (with some randomness for variety)
    sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Pick from top candidates with probability proportional to score
    top_n = min(3, len(sorted_moves))
    candidates = sorted_moves[:top_n]
    
    if len(candidates) == 1:
        return candidates[0][0]
    
    # Weighted random selection
    total_score = sum(s for _, s in candidates)
    if total_score <= 0:
        return random.choice([m for m, _ in candidates])
    
    r = random.random() * total_score
    cumulative = 0
    for move, score in candidates:
        cumulative += score
        if cumulative >= r:
            return move
    
    return candidates[0][0]


def play_game(
    max_moves: int = 200,
    vs_random: bool = False,
    verbose: bool = True,
    weights_path: Optional[Path] = None,
    krk_weights_path: Optional[Path] = None,
    kpk_weights_path: Optional[Path] = None,
    output_viz: bool = False,
    output_basename: str = "full_game",
    enable_stem_cells: bool = False,
    stem_cell_path: Optional[Path] = None,
    use_unified_graph: bool = True,
) -> Tuple[GameState, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Play a complete game using the unified ReCoN architecture.
    
    Args:
        max_moves: Maximum number of moves before declaring draw
        vs_random: If True, opponent plays random moves
        verbose: Print progress
        weights_path: Path to weights directory (default: weights/latest/)
        krk_weights_path: Path to KRK-specific consolidation pack (legacy)
        kpk_weights_path: Path to KPK-specific consolidation pack (legacy)
        output_viz: If True, output visualization JSON
        output_basename: Base name for output files
        enable_stem_cells: If True, use stem cells for pattern discovery
        stem_cell_path: Path to load/save stem cell state
        use_unified_graph: If True, use unified graph with all subgraphs
        
    Returns:
        Final game state, list of frame data for visualization, and discovered patterns
    """
    # Build the unified graph (includes KRK, KPK, and all tactics)
    if use_unified_graph:
        g = build_unified_graph(
            include_endgames=True,
            include_tactics=True,
            include_sensors=True,
        )
        
        # Load all weights from weights directory
        weights_dir = weights_path.parent if weights_path else Path("weights/latest")
        weight_stats = load_all_weights(g, weights_dir)
        
        if verbose:
            subgraph_summary = get_subgraph_summary(g)
            print(f"Built unified graph:")
            for sg_name, sg_info in subgraph_summary.items():
                print(f"  {sg_name}: {sg_info['node_count']} nodes, {sg_info.get('edge_count', 0)} edges")
            print(f"Weights loaded: {weight_stats}")
    else:
        # Legacy: use flat graph
        g = build_full_game_graph()
    
    engine = ReConEngine(g)
    
    # M8: Initialize stem cell manager
    stem_manager = None
    discovered_patterns: List[Dict[str, Any]] = []
    
    if enable_stem_cells:
        stem_config = StemCellConfig(
            min_samples=30,      # Fewer samples needed for chess
            max_samples=200,
            reward_threshold=0.2,  # Capture positions with notable evaluation change
            specialization_threshold=0.6,
            exploration_budget=150,
        )
        stem_manager = StemCellManager(
            max_cells=10,
            spawn_rate=0.08,  # 8% chance to spawn new cell each tick
            config=stem_config,
        )
        
        # Load existing stem cell state if provided
        if stem_cell_path and stem_cell_path.exists():
            try:
                stem_manager = StemCellManager.load(stem_cell_path)
                if verbose:
                    print(f"Loaded stem cell state from {stem_cell_path}")
                    print(f"  Active cells: {stem_manager.stats()}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load stem cells from {stem_cell_path}: {e}")
        
        # Set feature extractor
        for cell in stem_manager.cells.values():
            cell.feature_extractor = extract_board_features
    
    # M8: Load trained weights from consolidation packs
    if weights_path and weights_path.exists():
        try:
            consol_engine = ConsolidationEngine(ConsolidationConfig(enabled=True))
            consol_engine.load_state(weights_path)
            consol_engine.init_from_graph(g)
            consol_engine.apply_w_base_to_graph(g)
            if verbose:
                print(f"Loaded weights from {weights_path}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load weights from {weights_path}: {e}")
    
    # M8: Initialize logger for visualization output
    viz_logger = RunLogger() if output_viz else None
    if viz_logger:
        viz_logger.attach_graph([
            {"src": e.src, "dst": e.dst, "type": e.ltype.name, "weight": float(getattr(e, "w", 1.0) or 1.0)}
            for e in g.edges
        ])
    
    # Initialize game
    state = GameState(board=chess.Board())
    state.last_eval = eval_position(state.board)
    frames: List[Dict[str, Any]] = []
    persistence_config = PersistenceConfig()
    
    if verbose:
        print("Starting full game demo...")
        if enable_stem_cells:
            print("Stem cells ENABLED for pattern discovery")
        print(f"Initial position:\n{state.board}\n")
    
    while not state.board.is_game_over() and len(state.move_history) < max_moves:
        # Request the root to start evaluation
        g.nodes["GameRoot"].state = NodeState.REQUESTED
        
        # Create environment
        env = {"board": state.board}
        
        # Run engine step to evaluate sensors and update states
        engine.step(env)
        state.tick += 1
        
        # Get assessments from sensors
        ultimate = assess_ultimate_goal(state.board, state.board.turn)
        phase = estimate_phase(state.board)
        material = assess_material(state.board)
        
        # Compute subgraph gates for endgames
        subgraph_gates = compute_subgraph_gates(state.board)
        
        # Update subgraph activation in graph metadata
        if "krk_root" in g.nodes:
            g.nodes["krk_root"].meta["gate"] = subgraph_gates.get("krk", 0.0)
            g.nodes["krk_root"].meta["activation"] = subgraph_gates.get("krk", 0.0)
        if "kpk_root" in g.nodes:
            g.nodes["kpk_root"].meta["gate"] = subgraph_gates.get("kpk", 0.0)
            g.nodes["kpk_root"].meta["activation"] = subgraph_gates.get("kpk", 0.0)
        
        # Update persistence for strategic plans
        goal_plans = get_active_plans_for_goal(ultimate.goal.name, phase.as_dict())
        
        # Apply persistence to plan nodes
        for plan_id, base_weight in goal_plans:
            if plan_id in g.nodes:
                evidence = base_weight / 2.0  # Normalize to 0-1
                apply_persistence_to_node(g.nodes[plan_id], evidence, config=persistence_config)
        
        # Get active plans with their competition weights
        active_plans = get_active_plans(g.nodes, layer="strategic", config=persistence_config)
        
        # Run parallel tactics detection with context weighting
        context_weights = compute_tactics_context_weights(
            ultimate_goal=ultimate.goal.name,
            phase=phase.as_dict(),
        )
        tactic_results = run_parallel_tactics(state.board, g, context_weights)
        
        # Select move (now includes tactical results)
        move = select_move(
            state.board,
            ultimate.goal,
            phase.as_dict(),
            active_plans,
            tactic_results=tactic_results,
        )
        
        if move is None:
            break
        
        # Record frame for visualization
        frame = {
            "tick": state.tick,
            "ply": len(state.move_history),
            "fen": state.board.fen(),
            "move": state.board.san(move),
            "env": {
                "ultimate_goal": ultimate.as_dict(),
                "phase": phase.as_dict(),
                "material": material.as_dict(),
                "active_plans": active_plans[:5],
            },
            # Enhanced network state for visualization
            "nodes": {
                node_id: {
                    # Basic state
                    "state": node.state.name if hasattr(node.state, 'name') else str(node.state),
                    "layer": node.meta.get("layer", "unknown") if hasattr(node, 'meta') else "unknown",
                    "subgraph": node.meta.get("subgraph", "main") if hasattr(node, 'meta') else "main",
                    # Activation & persistence values
                    "activation": node.meta.get("activation", 0.0) if hasattr(node, 'meta') else 0.0,
                    "p_value": getattr(node, "p_value", 0.0),
                    "eligibility": getattr(node, "eligibility", 0.0),
                    # Terminal outputs
                    "last_output": node.meta.get("last_output") if hasattr(node, 'meta') else None,
                    "confidence": node.meta.get("confidence", 1.0) if hasattr(node, 'meta') else 1.0,
                    "detected": node.meta.get("detected", False) if hasattr(node, 'meta') else False,
                }
                for node_id, node in g.nodes.items()
            },
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.ltype.name if hasattr(e.ltype, 'name') else str(e.ltype),
                    "weight": float(getattr(e, "w", 1.0) or 1.0),
                    "trace": getattr(e, "trace", 0.0),
                }
                for e in g.edges
            ],
            # Subgraph metadata for visualization grouping
            "subgraphs": {
                "krk": {
                    "gate": subgraph_gates.get("krk", 0.0),
                    "active": subgraph_gates.get("krk", 0.0) > 0.1,
                    "node_count": sum(1 for n in g.nodes.values() if n.meta.get("subgraph") == "krk"),
                },
                "kpk": {
                    "gate": subgraph_gates.get("kpk", 0.0),
                    "active": subgraph_gates.get("kpk", 0.0) > 0.1,
                    "node_count": sum(1 for n in g.nodes.values() if n.meta.get("subgraph") == "kpk"),
                },
                "tactics": {
                    "active": True,
                    "detected": [r["tactic"] for r in tactic_results],
                    "context_weights": context_weights,
                    "node_count": sum(1 for n in g.nodes.values() if "tactics" in str(n.meta.get("subgraph", ""))),
                },
            },
            # Board tags for overlay visualization
            "board_tags": generate_board_tags(state.board),
            # Tactics detection results
            "tactics": tactic_results,
            # Stem cell status
            "stem_cells": [
                {
                    "id": cell_id,
                    "state": cell.state.name if hasattr(cell.state, 'name') else str(cell.state),
                    "samples": len(cell.samples),
                    "exploration_ticks": cell.exploration_ticks,
                }
                for cell_id, cell in (stem_manager.cells.items() if stem_manager else {})
            ] if stem_manager else [],
        }
        frames.append(frame)
        
        # Make the move
        san = state.board.san(move)
        state.board.push(move)
        state.move_history.append(san)
        state.phase_history.append(phase.as_dict())
        state.goal_history.append(ultimate.goal.name)
        state.active_plans = [p[0] for p in active_plans[:3]]
        
        # M8: Compute reward and feed to stem cells
        if stem_manager:
            reward = compute_reward_signal(state.board, state.last_eval)
            stem_manager.tick(state.board, reward, tick=state.tick)
        
        # Update eval for next reward computation
        state.last_eval = eval_position(state.board)
        
        if verbose and len(state.move_history) % 10 == 0:
            print(f"Move {len(state.move_history)}: {san}")
            print(f"  Phase: {phase.dominant_phase()}, Goal: {ultimate.goal.name}")
            print(f"  Active plans: {state.active_plans}")
            if stem_manager:
                stats = stem_manager.stats()
                print(f"  Stem cells: {stats['total_cells']} ({stats['by_state']})")
        
        # Opponent's turn
        if vs_random and not state.board.is_game_over():
            # Random opponent
            opponent_moves = list(state.board.legal_moves)
            if opponent_moves:
                opp_move = random.choice(opponent_moves)
                opp_san = state.board.san(opp_move)
                state.board.push(opp_move)
                state.move_history.append(opp_san)
                state.tick += 1
                
                # Stem cells observe opponent's move result too
                if stem_manager:
                    reward = compute_reward_signal(state.board, state.last_eval)
                    stem_manager.tick(state.board, reward, tick=state.tick)
                    state.last_eval = eval_position(state.board)
        
        # Reset node states for next iteration
        for node in g.nodes.values():
            if node.state != NodeState.INACTIVE:
                node.state = NodeState.INACTIVE
    
    # Game over - process stem cells
    if stem_manager:
        # Attempt to specialize ready cells
        discovered_patterns = stem_manager.specialize_all_ready()
        
        # Prune failed cells
        pruned = stem_manager.prune_failed()
        
        if verbose:
            print(f"\n=== Stem Cell Results ===")
            print(f"Discovered patterns: {len(discovered_patterns)}")
            print(f"Pruned cells: {pruned}")
            print(f"Final stats: {stem_manager.stats()}")
            
            for pattern in discovered_patterns:
                print(f"  - {pattern['sensor_id']}: consistency={pattern['consistency']:.2f}")
        
        # Save stem cell state if path provided
        if stem_cell_path:
            stem_cell_path.parent.mkdir(parents=True, exist_ok=True)
            stem_manager.save(stem_cell_path)
            if verbose:
                print(f"Saved stem cell state to {stem_cell_path}")
    
    if verbose:
        print(f"\nGame ended after {len(state.move_history)} moves")
        print(f"Result: {state.board.result()}")
        print(f"Final position:\n{state.board}")
        
        # PGN
        print("\nPGN:")
        moves_str = " ".join(
            f"{i//2+1}." + (m if i % 2 == 0 else f" {m}")
            for i, m in enumerate(state.move_history)
        )
        print(moves_str)
    
    return state, frames, discovered_patterns


def main():
    parser = argparse.ArgumentParser(description="M6/M8 Full Game Demo with Stem Cells")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves")
    parser.add_argument("--vs-random", action="store_true", help="Play against random")
    parser.add_argument("--output", type=str, help="Output JSON file for visualization")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output (result still printed)")
    parser.add_argument("--weights", type=str, help="Path to consolidation weights (default: weights/latest/fullgame_consol.json)")
    parser.add_argument("--krk-weights", type=str, help="Path to KRK weights (default: weights/latest/krk_consol.json)")
    parser.add_argument("--kpk-weights", type=str, help="Path to KPK weights (default: weights/latest/kpk_consol.json)")
    parser.add_argument("--json-result", action="store_true", help="Output result as JSON (for parsing)")
    parser.add_argument("--viz", action="store_true", help="Output visualization JSON files")
    parser.add_argument("--output-basename", type=str, default="full_game", help="Base name for output files")
    # M8: Stem cell options
    parser.add_argument("--stem-cells", action="store_true", help="Enable stem cell pattern discovery")
    parser.add_argument("--stem-cell-path", type=str, help="Path to stem cells (default: weights/latest/stem_cells.json)")
    args = parser.parse_args()
    
    # Use weights/latest/ by default if available (updated after each nightly run)
    latest_dir = Path("weights/latest")
    
    if args.weights:
        weights_path = Path(args.weights)
    elif (latest_dir / "fullgame_consol.json").exists():
        weights_path = latest_dir / "fullgame_consol.json"
    else:
        weights_path = None
    
    if args.krk_weights:
        krk_weights_path = Path(args.krk_weights)
    elif (latest_dir / "krk_consol.json").exists():
        krk_weights_path = latest_dir / "krk_consol.json"
    else:
        krk_weights_path = None
    
    if args.kpk_weights:
        kpk_weights_path = Path(args.kpk_weights)
    elif (latest_dir / "kpk_consol.json").exists():
        kpk_weights_path = latest_dir / "kpk_consol.json"
    else:
        kpk_weights_path = None
    
    if args.stem_cell_path:
        stem_cell_path = Path(args.stem_cell_path)
    elif (latest_dir / "stem_cells.json").exists():
        stem_cell_path = latest_dir / "stem_cells.json"
    else:
        stem_cell_path = None
    
    # Log which weights are being loaded
    if not args.quiet:
        print("Loading weights:")
        print(f"  Full game: {weights_path or 'None'}")
        print(f"  KRK:       {krk_weights_path or 'None'}")
        print(f"  KPK:       {kpk_weights_path or 'None'}")
        print(f"  Stem cells: {stem_cell_path or 'None'}")
        print()
    
    state, frames, discovered = play_game(
        max_moves=args.max_moves,
        vs_random=args.vs_random,
        verbose=not args.quiet,
        weights_path=weights_path,
        krk_weights_path=krk_weights_path,
        kpk_weights_path=kpk_weights_path,
        output_viz=args.viz,
        output_basename=args.output_basename,
        enable_stem_cells=args.stem_cells,
        stem_cell_path=stem_cell_path,
    )
    
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(frames, f, indent=2)
        if not args.quiet:
            print(f"\nVisualization data saved to {out_path}")
    
    # Save discovered patterns if any
    if discovered and args.output_basename:
        patterns_path = Path(f"weights/discoveries/{args.output_basename}_patterns.json")
        patterns_path.parent.mkdir(parents=True, exist_ok=True)
        with open(patterns_path, "w") as f:
            json.dump(discovered, f, indent=2)
        if not args.quiet:
            print(f"Discovered patterns saved to {patterns_path}")
    
    # Always output result for parsing by scripts
    result = state.board.result()
    if args.json_result:
        result_data = {
            "result": result,
            "moves": len(state.move_history),
            "outcome": "win" if result == "1-0" else ("loss" if result == "0-1" else "draw" if result == "1/2-1/2" else "unknown"),
            "discovered_patterns": len(discovered) if discovered else 0,
        }
        print(json.dumps(result_data))
    else:
        # Print result on its own line for easy parsing
        print(f"RESULT:{result}")


if __name__ == "__main__":
    main()

