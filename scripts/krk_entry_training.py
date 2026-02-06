"""
Train/evaluate the compiled KRK_entry topology with spawn points.

Runs a simple loop over curriculum positions (default Stage 0),
executes the compiled graph to select a move, and updates spawn points.
"""

import argparse
import random
from pathlib import Path

import chess

from recon_lite.graph import Graph
from recon_lite_chess.graph.builder import build_graph_from_topology, export_topology_from_graph
from recon_lite_chess.spawn_point import SpawnPointManager, SpawnPointConfig
from recon_lite_chess.training.krk_curriculum import generate_krk_curriculum_position, get_stage


def execute_graph(graph: Graph, env: dict) -> None:
    """Execute the compiled topology in a simple order."""
    # Reset blackboard each move
    root = graph.nodes.get("krk_entry")
    if root and "blackboard" in root.meta:
        root.meta["blackboard"].clear()

    # Root: extract features/cache
    if root and root.predicate:
        root.predicate(root, env)

    # Sensors: populate outputs
    for nid, node in graph.nodes.items():
        if "sensor" in nid and node.predicate:
            try:
                node.predicate(node, env)
            except Exception:
                continue

    # Actuators: select moves (run all to choose best)
    for nid, node in graph.nodes.items():
        if "act" in nid and node.predicate:
            try:
                node.predicate(node, env)
            except Exception:
                continue


def select_move(graph: Graph, board: chess.Board) -> tuple[chess.Move | None, dict]:
    """Select a move from the graph; fallback to random legal move."""
    env = {"board": board}
    execute_graph(graph, env)
    move = env.get("suggested_move")
    if move is None:
        legal = list(board.legal_moves)
        return (random.choice(legal) if legal else None), env
    return move, env


def run_cycle(graph: Graph, spawn_mgr: SpawnPointManager, stage_id: int, games: int) -> dict:
    wins = 0
    moves = 0
    for _ in range(games):
        board = generate_krk_curriculum_position(stage_id)
        move, env = select_move(graph, board)
        if move is None:
            continue
        moves += 1

        board_after = board.copy()
        board_after.push(move)
        is_mate = board_after.is_checkmate()
        if is_mate:
            wins += 1

        spawn_mgr.process_position(board, move, is_mate, env=env)

    return {
        "games": games,
        "moves": moves,
        "wins": wins,
        "win_rate": wins / games if games else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KRK_entry training loop")
    parser.add_argument("--topology", type=Path, default=Path("topologies/krk_entry_topology.json"))
    parser.add_argument("--stage-id", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--games-per-cycle", type=int, default=200)
    parser.add_argument("--export", type=Path, default=None)
    args = parser.parse_args()

    stage = get_stage(args.stage_id)
    print(f"Stage {stage.stage_id}: {stage.name} ({stage.description})")

    graph = build_graph_from_topology(args.topology)
    print(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    spawn_mgr = SpawnPointManager(SpawnPointConfig(), graph=graph)
    spawn_mgr.attach_to_legs(graph)

    for cycle in range(1, args.cycles + 1):
        stats = run_cycle(graph, spawn_mgr, args.stage_id, args.games_per_cycle)
        print(
            f"Cycle {cycle}/{args.cycles}: win_rate={stats['win_rate']*100:.1f}% "
            f"({stats['wins']}/{stats['games']})"
        )

    if args.export:
        export_topology_from_graph(graph, args.export, network_name="krk_entry")
        print(f"Exported updated topology to {args.export}")


if __name__ == "__main__":
    main()
