"""
Test script for KRK_entry topology runtime execution.

Loads the compiled topology and tests it on KRK mate-in-1 positions.
"""

import sys
import argparse
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.engine import ReConEngine
from recon_lite.graph import Graph, NodeState
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite_chess.baseline_teacher import generate_krk_mate_in_1_position, KRKTeacher


def load_krk_entry_topology(topology_path: Path):
    """Load the compiled KRK_entry topology"""
    print(f"Loading topology: {topology_path}")
    
    # Use builder to create graph from topology
    graph = build_graph_from_topology(topology_path)
    
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    
    return graph


def choose_move_with_engine(graph: Graph, engine: ReConEngine, board: chess.Board, max_ticks: int = 200):
    """
    Run the graph via the ReCon engine until a move is chosen or max_ticks is reached.
    """
    env = {
        "board": board,
        "chosen_move": None,
        "suggested_move": None,
    }

    engine.reset_states()

    # Request the root script node.
    root_id = None
    if "krk_entry" in graph.nodes:
        root_id = "krk_entry"
    elif "ROOT" in graph.nodes:
        root_id = "ROOT"
    else:
        # Fallback: first SCRIPT node without a parent
        for nid, node in graph.nodes.items():
            if node.ntype.name == "SCRIPT" and graph.parent_of(nid) is None:
                root_id = nid
                break

    if root_id:
        graph.nodes[root_id].state = NodeState.REQUESTED

    ticks = 0
    while ticks < max_ticks and env.get("chosen_move") is None:
        ticks += 1
        engine.step(env)

    chosen = env.get("chosen_move") or env.get("suggested_move")
    confidence = env.get("move_confidence", 0.0)
    return chosen, confidence


def test_single_position(graph: Graph, engine: ReConEngine, board: chess.Board):
    """
    Test KRK_entry on a single position.
    
    Returns:
        (move, confidence, is_mate) tuple
    """
    suggested_move, confidence = choose_move_with_engine(graph, engine, board)
    
    if suggested_move is None:
        return None, 0.0, False
    
    # Check if it's mate
    board_copy = board.copy()
    move_obj = suggested_move
    if isinstance(suggested_move, str):
        try:
            move_obj = chess.Move.from_uci(suggested_move)
        except Exception:
            move_obj = None
    if move_obj is None:
        return suggested_move, confidence, False
    board_copy.push(move_obj)
    is_mate = board_copy.is_checkmate()
    
    return suggested_move, confidence, is_mate


def run_evaluation(graph: Graph, num_positions: int = 100):
    """
    Evaluate KRK_entry on multiple mate-in-1 positions.
    
    Returns:
        Statistics dictionary
    """
    print(f"\nEvaluating on {num_positions} positions...")
    print("-" * 70)
    
    teacher = KRKTeacher()
    
    stats = {
        "total": 0,
        "mate_found": 0,
        "no_move": 0,
        "wrong_move": 0,
        "confidences": [],
    }

    engine = ReConEngine(graph)
    
    for i in range(num_positions):
        # Generate position
        board = generate_krk_mate_in_1_position()
        
        # Test
        move, confidence, is_mate = test_single_position(graph, engine, board)
        
        stats["total"] += 1
        
        if move is None:
            stats["no_move"] += 1
        elif is_mate:
            stats["mate_found"] += 1
            stats["confidences"].append(confidence)
        else:
            stats["wrong_move"] += 1
            stats["confidences"].append(confidence)
        
        # Progress
        if (i + 1) % 10 == 0:
            win_rate = stats["mate_found"] / stats["total"] * 100
            print(f"  {i+1:3d}/{num_positions}: {stats['mate_found']:3d} mates ({win_rate:.1f}%)")
    
    return stats


def print_results(stats: dict):
    """Print evaluation results"""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    total = stats["total"]
    mate_found = stats["mate_found"]
    no_move = stats["no_move"]
    wrong_move = stats["wrong_move"]
    
    print(f"\nTotal positions: {total}")
    print(f"  Mate found:    {mate_found:3d} ({mate_found/total*100:.1f}%)")
    print(f"  Wrong move:    {wrong_move:3d} ({wrong_move/total*100:.1f}%)")
    print(f"  No move:       {no_move:3d} ({no_move/total*100:.1f}%)")
    
    if stats["confidences"]:
        import numpy as np
        confidences = np.array(stats["confidences"])
        print(f"\nConfidence scores:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Std:  {np.std(confidences):.3f}")
        print(f"  Min:  {np.min(confidences):.3f}")
        print(f"  Max:  {np.max(confidences):.3f}")
    
    print("\n" + "=" * 70)
    
    if mate_found / total >= 0.9:
        print("✓ SUCCESS: Win rate >= 90%")
    else:
        print("✗ FAILED: Win rate < 90%")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="KRK_entry Runtime Execution Test")
    parser.add_argument("--topology", type=Path, default=Path("topologies/krk_entry_topology.json"),
                        help="Path to topology JSON")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of positions to test")
    args = parser.parse_args()

    print("=" * 70)
    print("KRK_entry Runtime Execution Test")
    print("=" * 70)
    
    # Load topology
    graph = load_krk_entry_topology(args.topology)
    
    # Run evaluation
    stats = run_evaluation(graph, num_positions=args.samples)
    
    # Print results
    print_results(stats)


if __name__ == "__main__":
    main()
