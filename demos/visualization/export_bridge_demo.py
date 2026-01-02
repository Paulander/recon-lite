#!/usr/bin/env python3
"""
Export bridge games with network state for visualization.

Creates JSON with:
- Game frames (board position, moves, node states per tick)
- Graph topology (nodes, edges for visualization layout)

Usage:
    # Trained network (default)
    python export_bridge_demo.py --output sample_data/bridge_trained.json
    
    # Untrained network (no weights loaded)
    python export_bridge_demo.py --no-weights --output sample_data/bridge_untrained.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess

from recon_lite.graph import Node, NodeType, NodeState, LinkType, Graph
from recon_lite.engine import ReConEngine
from recon_lite_chess.graph import build_unified_graph, load_all_weights
from recon_lite_chess.scripts.kqk import is_kqk_position
from recon_lite_chess.sensors.structure import summarize_kpk_material


# ============================================================================
# Sentinels for subgraph exit conditions
# ============================================================================

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


def krk_sentinel(env: Dict[str, Any]) -> bool:
    board = env.get("board")
    if not board:
        return False
    pieces = list(board.piece_map().values())
    if len(pieces) != 3:
        return False
    types = [p.piece_type for p in pieces]
    return types.count(chess.KING) == 2 and types.count(chess.ROOK) == 1


# ============================================================================
# Graph Topology Export (for visualization layout)
# ============================================================================

def export_graph_topology(g: Graph, filter_subgraphs: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Export graph topology for visualization.
    
    Layout strategy: SUB = vertical (parent above children), POR = horizontal (left to right)
    """
    # Filter to relevant nodes for bridge demo
    if filter_subgraphs is None:
        filter_subgraphs = {"main", "kpk", "kqk", "krk"}
    
    nodes = []
    edges = []
    
    for nid, node in g.nodes.items():
        subgraph = node.meta.get("subgraph", "main")
        
        # Include main graph + endgame subgraphs
        if subgraph not in filter_subgraphs:
            # Also check node ID prefix
            if not any(nid.startswith(f"{sg}_") for sg in filter_subgraphs if sg != "main"):
                continue
        
        nodes.append({
            "id": nid,
            "type": node.ntype.name.lower(),
            "layer": node.meta.get("layer", "unknown"),
            "subgraph": subgraph,
        })
    
    node_ids = {n["id"] for n in nodes}
    
    for edge in g.edges:
        if edge.src in node_ids and edge.dst in node_ids:
            edges.append({
                "src": edge.src,
                "dst": edge.dst,
                "type": edge.ltype.name.lower(),
                "weight": float(edge.w) if hasattr(edge, 'w') else 1.0,
                "trainable": edge.meta.get("trainable", False) if hasattr(edge, 'meta') else False,
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
    }


def compute_layout(topology: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Compute 2D positions for nodes based on SUB (vertical) and POR (horizontal).
    
    Returns dict mapping node_id -> {x, y}
    """
    nodes = topology["nodes"]
    edges = topology["edges"]
    
    # Build parent-child map from SUB edges
    children: Dict[str, List[str]] = {}
    parents: Dict[str, str] = {}
    
    for edge in edges:
        if edge["type"] == "sub":
            src, dst = edge["src"], edge["dst"]
            children.setdefault(src, []).append(dst)
            if dst not in parents:
                parents[dst] = src
    
    # Find roots (nodes with no parent in SUB structure)
    all_ids = {n["id"] for n in nodes}
    roots = [nid for nid in all_ids if nid not in parents]
    
    # Assign depth (Y coordinate based on SUB hierarchy)
    depths: Dict[str, int] = {}
    
    def assign_depth(nid: str, depth: int):
        if nid in depths:
            return
        depths[nid] = depth
        for child in children.get(nid, []):
            assign_depth(child, depth + 1)
    
    for root in roots:
        assign_depth(root, 0)
    
    # Assign unreached nodes
    for n in nodes:
        if n["id"] not in depths:
            depths[n["id"]] = 5  # Default depth for disconnected
    
    # Group by depth for X positioning
    by_depth: Dict[int, List[str]] = {}
    for nid, d in depths.items():
        by_depth.setdefault(d, []).append(nid)
    
    # Compute positions
    positions: Dict[str, Dict[str, float]] = {}
    y_spacing = 80
    x_spacing = 120
    
    for depth, node_ids in by_depth.items():
        # Sort by subgraph then name for consistent layout
        node_ids.sort(key=lambda nid: (
            next((n.get("subgraph", "") for n in nodes if n["id"] == nid), ""),
            nid
        ))
        
        x_start = -(len(node_ids) - 1) * x_spacing / 2
        for i, nid in enumerate(node_ids):
            positions[nid] = {
                "x": 400 + x_start + i * x_spacing,
                "y": 60 + depth * y_spacing,
            }
    
    return positions


# ============================================================================
# Frame Export
# ============================================================================

def export_frame(
    tick: int,
    board: chess.Board,
    move: Optional[chess.Move],
    engine: ReConEngine,
    env: Dict[str, Any],
    g: Graph,
) -> Dict[str, Any]:
    """Export a single frame for visualization."""
    
    subgraph_lock = None
    if engine.subgraph_lock:
        subgraph_lock = engine.subgraph_lock.subgraph_root
    
    # Node states and activations (use continuous activation.level when available)
    node_states = {}
    node_activations = {}
    # Check if the locked subgraph root has been reached by engine
    subgraph_root_reached = False
    if subgraph_lock and subgraph_lock in g.nodes:
        subgraph_root_reached = g.nodes[subgraph_lock].state != NodeState.INACTIVE
    
    for nid, node in g.nodes.items():
        node_states[nid] = node.state.name
        
        is_in_locked = subgraph_lock and (
            nid == subgraph_lock or 
            nid.startswith(subgraph_lock.replace("_root", "_"))
        )
        
        # Parent path nodes: show connection when locked
        is_in_parent_path = subgraph_lock and nid in ("GameRoot", "WinStrategy", "endgame_gate")
        parent_path_reached = is_in_parent_path and node.state != NodeState.INACTIVE
        
        # Use continuous activation level if available
        activation = 0.0
        if hasattr(node, 'activation') and hasattr(node.activation, 'level'):
            activation = float(node.activation.level)
        elif node.state in (NodeState.TRUE, NodeState.CONFIRMED):
            activation = 1.0
        elif node.state in (NodeState.WAITING, NodeState.REQUESTED):
            activation = 0.5
        
        # Only boost locked subgraph if its root has been reached
        if is_in_locked and subgraph_root_reached:
            activation = max(activation, 0.7)
        elif is_in_locked and not subgraph_root_reached:
            activation = max(activation, 0.3)  # Dim: locked but not yet reached
        elif parent_path_reached:
            activation = max(activation, 0.8)  # Fully reached by propagation
        elif is_in_parent_path:
            activation = max(activation, 0.5)  # Path exists but not yet propagated
        
        node_activations[nid] = round(activation, 3)
    
    # Gate activations
    gate_data = env.get("endgame_gate", {})
    
    # Extract sensor bindings from env (what each sensor is "looking at")
    bindings = {}
    
    # KQK bindings: queen position, king positions
    kqk_data = env.get("kqk", {})
    if kqk_data:
        if "queen_sq" in kqk_data:
            bindings["queen"] = chess.square_name(kqk_data["queen_sq"])
        if "our_king_sq" in kqk_data:
            bindings["our_king"] = chess.square_name(kqk_data["our_king_sq"])
        if "enemy_king_sq" in kqk_data:
            bindings["enemy_king"] = chess.square_name(kqk_data["enemy_king_sq"])
    
    # KRK bindings: rook position, fence rank/file
    krk_data = env.get("krk", {})
    if krk_data:
        if "rook_sq" in krk_data:
            bindings["rook"] = chess.square_name(krk_data["rook_sq"])
        if "fence_rank" in krk_data:
            bindings["fence_rank"] = str(krk_data["fence_rank"] + 1)
        if "fence_file" in krk_data:
            bindings["fence_file"] = chr(ord('a') + krk_data["fence_file"])
    
    # KPK bindings: pawn position
    kpk_data = env.get("kpk", {})
    if kpk_data:
        if "pawn_sq" in kpk_data:
            bindings["pawn"] = chess.square_name(kpk_data["pawn_sq"])
    
    return {
        "tick": tick,
        "label": f"{move.uci() if move else 'Start'}",
        "board_fen": board.fen(),
        "move_uci": move.uci() if move else None,
        "subgraph_lock": subgraph_lock,
        "turn": "white" if board.turn else "black",
        "gate_activations": gate_data.get("activations", {}),
        "gate_decision": gate_data.get("active_endgame"),
        "node_states": node_states,
        "node_activations": node_activations,
        "bindings": bindings,
    }


# ============================================================================
# Game Play and Export
# ============================================================================

def play_and_export(
    output_path: Path,
    max_moves: int = 80,
    initial_fen: Optional[str] = None,
    load_weights: bool = True,
    verbose: bool = True,
) -> bool:
    """Play a bridge game and export for visualization."""
    
    if initial_fen is None:
        initial_fen = "8/6P1/7K/8/2k5/8/8/8 w - - 0 1"
    
    # Build graph
    g = build_unified_graph(include_endgames=True, include_tactics=False, include_sensors=True)
    
    if load_weights:
        weights_dir = Path("weights/latest")
        if weights_dir.exists():
            load_all_weights(g, weights_dir=weights_dir)
            if verbose:
                print("Loaded weights from weights/latest")
        else:
            if verbose:
                print("Warning: weights/latest not found, using default weights")
    else:
        if verbose:
            print("Running with default (untrained) weights")
    
    engine = ReConEngine(g)
    board = chess.Board(initial_fen)
    frames: List[Dict[str, Any]] = []
    
    # Export graph topology
    topology = export_graph_topology(g)
    positions = compute_layout(topology)
    
    if verbose:
        print(f"Starting: {initial_fen}")
        print(f"Graph: {len(topology['nodes'])} nodes, {len(topology['edges'])} edges")
    
    # Sentinel map
    sentinels = {
        "kpk": kpk_sentinel,
        "kqk": kqk_sentinel,
        "krk": krk_sentinel,
    }
    
    # Initial frame - run gate first to get activations
    env = {"board": board}
    gate_node = g.nodes.get("endgame_gate")
    if gate_node and gate_node.predicate:
        gate_node.predicate(gate_node, env)
    frames.append(export_frame(0, board, None, engine, env, g))
    
    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        env = {"board": board}
        
        # Always run gate for visualization (even when locked)
        gate_node = g.nodes.get("endgame_gate")
        if gate_node and gate_node.predicate:
            gate_node.predicate(gate_node, env)
        
        # Lock subgraph based on gate decision (only if not already locked)
        if not engine.subgraph_lock:
            gate_data = env.get("endgame_gate", {})
            active_endgame = gate_data.get("active_endgame")
            
            if active_endgame:
                subgraph_root = f"{active_endgame}_root"
                sentinel = sentinels.get(active_endgame)
                if sentinel and subgraph_root in g.nodes:
                    engine.lock_subgraph(subgraph_root, sentinel)
        
        # Engine step
        g.nodes["GameRoot"].state = NodeState.REQUESTED
        engine.step(env)
        
        # Get move from policy
        move = None
        for key in ("kqk", "kpk", "krk"):
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
            promos = [m for m in legal if m.promotion]
            move = promos[0] if promos else random.choice(legal)
        
        board.push(move)
        move_count += 1
        
        # Handle promotion transition
        if move.promotion and engine.subgraph_lock:
            if engine.subgraph_lock.subgraph_root == "kpk_root":
                engine.unlock_subgraph(goal_achieved=True)
        
        frames.append(export_frame(move_count, board, move, engine, env, g))
        
        if verbose and move_count <= 5:
            lock = engine.subgraph_lock.subgraph_root if engine.subgraph_lock else "none"
            print(f"  {move_count}: {move.uci()} lock={lock}")
        
        # Opponent move
        if not board.is_game_over():
            opp_moves = list(board.legal_moves)
            if opp_moves:
                opp_move = random.choice(opp_moves)
                board.push(opp_move)
                move_count += 1
                frames.append(export_frame(move_count, board, opp_move, engine, {"board": board}, g))
    
    # Result
    if board.is_checkmate():
        result = "checkmate"
        winner = "black" if board.turn else "white"
        if verbose:
            print(f"\n✓ Checkmate! {winner.capitalize()} wins in {move_count} moves")
    elif board.is_game_over():
        result = board.result()
        if verbose:
            print(f"\n{result}")
    else:
        result = "timeout"
        if verbose:
            print(f"\nTimeout after {move_count} moves")
    
    # Build output with topology
    output = {
        "version": "bridge_demo_v2",
        "trained": load_weights,
        "initial_fen": initial_fen,
        "result": result,
        "total_moves": move_count,
        "topology": topology,
        "positions": positions,
        "frames": frames,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    if verbose:
        print(f"Exported {len(frames)} frames to {output_path}")
    
    return result == "checkmate"


def main():
    parser = argparse.ArgumentParser(description="Export bridge game for visualization")
    parser.add_argument("--output", "-o", type=Path, 
                       default=Path("demos/visualization/sample_data/bridge_demo.json"))
    parser.add_argument("--moves", "-m", type=int, default=80)
    parser.add_argument("--fen", type=str, default=None)
    parser.add_argument("--no-weights", action="store_true", 
                       help="Run with untrained (default) weights")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    
    success = play_and_export(
        output_path=args.output,
        max_moves=args.moves,
        initial_fen=args.fen,
        load_weights=not args.no_weights,
        verbose=not args.quiet,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
