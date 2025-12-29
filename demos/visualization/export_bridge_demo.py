#!/usr/bin/env python3
"""
Export a winning bridge game with per-move network state for visualization.

Creates a JSON timeline that can be loaded in the bridge_demo.html viewer.

Usage:
    python demos/visualization/export_bridge_demo.py [--output PATH] [--moves N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess

from recon_lite.graph import Node, NodeType, NodeState, LinkType
from recon_lite.engine import ReConEngine
from recon_lite_chess.graph import build_unified_graph, load_all_weights
from recon_lite_chess.scripts.kqk import is_kqk_position
from recon_lite_chess.sensors.structure import summarize_kpk_material


# Sentinels (from training script)
def kpk_sentinel(env: Dict[str, Any]) -> bool:
    board = env.get("board")
    if not board:
        return False
    summary = summarize_kpk_material(board)
    return bool(summary.get("is_kpk"))


def kqk_sentinel(env: Dict[str, Any]) -> bool:
    board = env.get("board")
    if not board:
        return False
    is_kqk, _ = is_kqk_position(board)
    return is_kqk


def get_node_color(state: NodeState, is_locked_subgraph: bool = False) -> str:
    """Map node state to color for visualization."""
    if is_locked_subgraph:
        return "#22c55e"  # Green for active subgraph
    
    colors = {
        NodeState.INACTIVE: "#64748b",
        NodeState.REQUESTED: "#f59e0b",
        NodeState.WAITING: "#eab308",
        NodeState.TRUE: "#22c55e",
        NodeState.CONFIRMED: "#16a34a",
        NodeState.FAILED: "#ef4444",
    }
    return colors.get(state, "#94a3b8")


def export_frame(
    tick: int,
    board: chess.Board,
    move: Optional[chess.Move],
    engine: ReConEngine,
    env: Dict[str, Any],
    g,
) -> Dict[str, Any]:
    """Export a single frame for visualization."""
    
    # Determine active subgraph
    subgraph_lock = None
    if engine.subgraph_lock:
        subgraph_lock = engine.subgraph_lock.subgraph_root
    
    # Get node states with activation levels
    node_activations = {}
    for nid, node in g.nodes.items():
        is_in_locked = subgraph_lock and nid.startswith(subgraph_lock.replace("_root", "_"))
        activation = 0.0
        if node.state in (NodeState.TRUE, NodeState.CONFIRMED):
            activation = 1.0
        elif node.state in (NodeState.WAITING, NodeState.REQUESTED):
            activation = 0.5
        
        if is_in_locked:
            activation = max(activation, 0.8)
        
        node_activations[nid] = activation
    
    # Get policy suggestions
    kpk_move = env.get("kpk", {}).get("policy", {}).get("suggested_move")
    kqk_move = env.get("kqk", {}).get("policy", {}).get("suggested_move")
    
    # Build frame for macrograph viewer format
    frame = {
        "tick": tick,
        "label": f"{move.uci() if move else 'Start'}" + (f" ({subgraph_lock})" if subgraph_lock else ""),
        "board_fen": board.fen(),
        "move_uci": move.uci() if move else None,
        "subgraph_lock": subgraph_lock,
        "macro_frame": {
            "phase_mix": {
                "kpk": 1.0 if kpk_sentinel({"board": board}) else 0.0,
                "kqk": 1.0 if kqk_sentinel({"board": board}) else 0.0,
            },
            "plan_groups": [
                {
                    "id": "PlanEndgame",
                    "activation": 1.0 if subgraph_lock else 0.3,
                    "plans": [subgraph_lock.replace("_root", "").upper()] if subgraph_lock else []
                },
            ],
            "bindings": {
                f"macro/endgame/{subgraph_lock.replace('_root', '')}": True
            } if subgraph_lock else {},
            "move_synth": {
                "chosen": move.uci() if move else None,
                "source": subgraph_lock if subgraph_lock else "heuristic",
            },
        },
        "node_states": {nid: n.state.name for nid, n in g.nodes.items()},
        "node_activations": node_activations,
    }
    
    return frame


def play_and_export(
    output_path: Path,
    max_moves: int = 80,
    initial_fen: Optional[str] = None,
    verbose: bool = True,
) -> bool:
    """Play a bridge game and export frames for visualization."""
    
    # Use a near-promotion FEN
    if initial_fen is None:
        initial_fen = "8/6P1/7K/8/2k5/8/8/8 w - - 0 1"  # Classic KPK, pawn ready to promote
    
    # Build graph and engine
    g = build_unified_graph(include_endgames=True, include_tactics=True, include_sensors=True)
    load_all_weights(g, weights_dir=Path("weights/latest"))
    engine = ReConEngine(g)
    
    board = chess.Board(initial_fen)
    frames: List[Dict[str, Any]] = []
    
    if verbose:
        print(f"Starting position: {initial_fen}")
    
    # Initial frame
    env = {"board": board}
    frames.append(export_frame(0, board, None, engine, env, g))
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        env = {"board": board}
        
        # Detect endgame and lock subgraph
        if not engine.subgraph_lock:
            is_kqk, kqk_attacker = is_kqk_position(board)
            if is_kqk and kqk_attacker == board.turn:
                engine.lock_subgraph("kqk_root", kqk_sentinel)
            elif kpk_sentinel(env):
                kpk_summary = summarize_kpk_material(board)
                if kpk_summary.get("attacker_color") == board.turn:
                    engine.lock_subgraph("kpk_root", kpk_sentinel)
        
        # Request root and step
        g.nodes["GameRoot"].state = NodeState.REQUESTED
        engine.step(env)
        
        # Get move from policy
        move = None
        for key in ("kqk", "kpk"):
            suggested = env.get(key, {}).get("policy", {}).get("suggested_move")
            if suggested:
                try:
                    candidate = chess.Move.from_uci(suggested)
                    if candidate in board.legal_moves:
                        move = candidate
                        break
                except:
                    pass
        
        # Fallback
        if move is None:
            legal = list(board.legal_moves)
            if not legal:
                break
            # Prefer promotions
            promos = [m for m in legal if m.promotion]
            move = promos[0] if promos else legal[0]
        
        # Make move
        board.push(move)
        move_count += 1
        
        # Handle promotion transition
        if move.promotion and engine.subgraph_lock and engine.subgraph_lock.subgraph_root == "kpk_root":
            engine.unlock_subgraph(goal_achieved=True)
        
        # Export frame
        frames.append(export_frame(move_count, board, move, engine, env, g))
        
        if verbose and move_count <= 5:
            lock = engine.subgraph_lock.subgraph_root if engine.subgraph_lock else "none"
            print(f"  Move {move_count}: {move.uci()}, lock={lock}")
        
        # Opponent move (random)
        if not board.is_game_over():
            opp_moves = list(board.legal_moves)
            if opp_moves:
                opp_move = random.choice(opp_moves)
                board.push(opp_move)
                move_count += 1
                frames.append(export_frame(move_count, board, opp_move, engine, {"board": board}, g))
    
    # Determine result
    if board.is_checkmate():
        result = "checkmate"
        winner = "black" if board.turn else "white"
        if verbose:
            print(f"\n✓ Checkmate! {winner.capitalize()} wins in {move_count} moves")
    elif board.is_game_over():
        result = board.result()
        if verbose:
            print(f"\nGame over: {result}")
    else:
        result = "timeout"
        if verbose:
            print(f"\nGame timeout after {move_count} moves")
    
    # Add game metadata
    output = {
        "version": "bridge_demo_v1",
        "initial_fen": initial_fen,
        "result": result,
        "total_moves": move_count,
        "frames": frames,
    }
    
    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    if verbose:
        print(f"\nExported {len(frames)} frames to {output_path}")
    
    return result == "checkmate"


def main():
    parser = argparse.ArgumentParser(description="Export bridge game for visualization")
    parser.add_argument("--output", "-o", type=Path, default=Path("demos/visualization/sample_data/bridge_demo.json"))
    parser.add_argument("--moves", "-m", type=int, default=80)
    parser.add_argument("--fen", type=str, default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    
    success = play_and_export(
        output_path=args.output,
        max_moves=args.moves,
        initial_fen=args.fen,
        verbose=not args.quiet,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
