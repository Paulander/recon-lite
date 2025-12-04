#!/usr/bin/env python3
"""M6 Full Game Demo - Play a complete chess game from start to finish.

This demo showcases the M6 goal hierarchy:
- Ultimate goals (WIN/DRAW/SURVIVE) based on position
- Strategic plans (attack, develop, simplify, etc.)
- Phase-aware activation (opening → middlegame → endgame)
- Plan persistence (inertia and decay)
- Fan-in sensor terminals

Usage:
    uv run python demos/persistent/full_game_demo.py
    uv run python demos/persistent/full_game_demo.py --max-moves 100 --output game.json
    uv run python demos/persistent/full_game_demo.py --vs-random  # Play against random moves
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
from recon_lite.plasticity.consolidate import ConsolidationEngine, ConsolidationConfig
from recon_lite.logger import RunLogger


@dataclass
class GameState:
    """State tracking for the game."""
    board: chess.Board
    move_history: List[str] = field(default_factory=list)
    phase_history: List[Dict[str, float]] = field(default_factory=list)
    goal_history: List[str] = field(default_factory=list)
    active_plans: List[str] = field(default_factory=list)
    tick: int = 0


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


def select_move(
    board: chess.Board,
    goal: UltimateGoal,
    phase_weights: Dict[str, float],
    active_plans: List[Tuple[str, float]],
) -> Optional[chess.Move]:
    """
    Select a move based on active plans and goal.
    
    Combines move candidates from different plans weighted by their activation.
    """
    if not board.legal_moves:
        return None
    
    move_scores: Dict[chess.Move, float] = {}
    
    # Get phase for candidate generation
    dominant_phase = max(phase_weights.keys(), key=lambda k: phase_weights[k])
    
    # Collect candidates from opening heuristics
    if dominant_phase == "opening" or phase_weights.get("opening", 0) > 0.3:
        opening_candidates = get_opening_move_candidates(board)
        for move, reason, score in opening_candidates:
            move_scores[move] = move_scores.get(move, 0) + score * phase_weights.get("opening", 0.5)
    
    # Collect candidates from middlegame heuristics
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
    
    # Add base score for captures
    for move in board.legal_moves:
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                value = {
                    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9
                }.get(captured.piece_type, 0)
                move_scores[move] = move_scores.get(move, 0) + value * 0.5
    
    # Fallback: if no candidates, use heuristic eval
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
) -> Tuple[GameState, List[Dict[str, Any]]]:
    """
    Play a complete game using the M6 architecture.
    
    Args:
        max_moves: Maximum number of moves before declaring draw
        vs_random: If True, opponent plays random moves
        verbose: Print progress
        weights_path: Path to general consolidation weights (applies to main graph)
        krk_weights_path: Path to KRK-specific consolidation pack
        kpk_weights_path: Path to KPK-specific consolidation pack
        output_viz: If True, output visualization JSON
        output_basename: Base name for output files
        
    Returns:
        Final game state and list of frame data for visualization
    """
    # Build the graph
    g = build_full_game_graph()
    engine = ReConEngine(g)
    
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
    frames: List[Dict[str, Any]] = []
    persistence_config = PersistenceConfig()
    
    if verbose:
        print("Starting full game demo...")
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
        
        # Update persistence for strategic plans
        goal_plans = get_active_plans_for_goal(ultimate.goal.name, phase.as_dict())
        
        # Apply persistence to plan nodes
        for plan_id, base_weight in goal_plans:
            if plan_id in g.nodes:
                evidence = base_weight / 2.0  # Normalize to 0-1
                apply_persistence_to_node(g.nodes[plan_id], evidence, config=persistence_config)
        
        # Get active plans with their competition weights
        active_plans = get_active_plans(g.nodes, layer="strategic", config=persistence_config)
        
        # Select move
        move = select_move(
            state.board,
            ultimate.goal,
            phase.as_dict(),
            active_plans,
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
        }
        frames.append(frame)
        
        # Make the move
        san = state.board.san(move)
        state.board.push(move)
        state.move_history.append(san)
        state.phase_history.append(phase.as_dict())
        state.goal_history.append(ultimate.goal.name)
        state.active_plans = [p[0] for p in active_plans[:3]]
        
        if verbose and len(state.move_history) % 10 == 0:
            print(f"Move {len(state.move_history)}: {san}")
            print(f"  Phase: {phase.dominant_phase()}, Goal: {ultimate.goal.name}")
            print(f"  Active plans: {state.active_plans}")
        
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
        
        # Reset node states for next iteration
        for node in g.nodes.values():
            if node.state != NodeState.INACTIVE:
                node.state = NodeState.INACTIVE
    
    # Game over
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
    
    return state, frames


def main():
    parser = argparse.ArgumentParser(description="M6 Full Game Demo")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves")
    parser.add_argument("--vs-random", action="store_true", help="Play against random")
    parser.add_argument("--output", type=str, help="Output JSON file for visualization")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output (result still printed)")
    parser.add_argument("--weights", type=str, help="Path to consolidation weights pack to load")
    parser.add_argument("--krk-weights", type=str, help="Path to KRK consolidation weights")
    parser.add_argument("--kpk-weights", type=str, help="Path to KPK consolidation weights")
    parser.add_argument("--json-result", action="store_true", help="Output result as JSON (for parsing)")
    parser.add_argument("--viz", action="store_true", help="Output visualization JSON files")
    parser.add_argument("--output-basename", type=str, default="full_game", help="Base name for output files")
    args = parser.parse_args()
    
    weights_path = Path(args.weights) if args.weights else None
    krk_weights_path = Path(args.krk_weights) if args.krk_weights else None
    kpk_weights_path = Path(args.kpk_weights) if args.kpk_weights else None
    
    state, frames = play_game(
        max_moves=args.max_moves,
        vs_random=args.vs_random,
        verbose=not args.quiet,
        weights_path=weights_path,
        krk_weights_path=krk_weights_path,
        kpk_weights_path=kpk_weights_path,
        output_viz=args.viz,
        output_basename=args.output_basename,
    )
    
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(frames, f, indent=2)
        if not args.quiet:
            print(f"\nVisualization data saved to {out_path}")
    
    # Always output result for parsing by scripts
    result = state.board.result()
    if args.json_result:
        result_data = {
            "result": result,
            "moves": len(state.move_history),
            "outcome": "win" if result == "1-0" else ("loss" if result == "0-1" else "draw" if result == "1/2-1/2" else "unknown"),
        }
        print(json.dumps(result_data))
    else:
        # Print result on its own line for easy parsing
        print(f"RESULT:{result}")


if __name__ == "__main__":
    main()

