"""
Evaluate Stage-1 backchaining quality.

Tests whether the compiled KRK_entry topology tends to pick moves that
decrease distance to Stage-0 goal prototypes (mate-in-1 goal bank).
"""

import argparse
import pickle
import random
import importlib.util
from pathlib import Path
from typing import Optional

import chess

from recon_lite.engine import ReConEngine
from recon_lite.graph import Graph, NodeState
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite_chess.baseline_teacher import KRKTeacher, can_deliver_mate
from recon_lite_chess.krk_baseline_nodes import _goal_distance_from_features
_baseline_path = Path(__file__).parent / "baseline_to_recon.py"
_spec = importlib.util.spec_from_file_location("baseline_to_recon", _baseline_path)
_baseline_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_baseline_mod)
build_goal_bank = _baseline_mod.build_goal_bank


def generate_random_krk_position(rng: random.Random) -> chess.Board:
    """Generate a random legal KRK position (white to move, not in check)."""
    squares = list(chess.SQUARES)
    while True:
        wk, bk, wr = rng.sample(squares, 3)
        board = chess.Board(None)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
        board.turn = chess.WHITE
        if chess.square_distance(wk, bk) <= 1:
            continue
        if not board.is_valid():
            continue
        if board.is_check():
            continue
        return board


def choose_move_with_engine(
    graph: Graph,
    engine: ReConEngine,
    board: chess.Board,
    max_ticks: int = 200,
    stage_filter: Optional[int] = None,
) -> Optional[str]:
    """Run graph via ReCon engine until a move is suggested."""
    env = {
        "board": board,
        "chosen_move": None,
        "suggested_move": None,
        "blackboard": {"stage_filter": stage_filter} if stage_filter is not None else {},
    }

    engine.reset_states()

    # Request the root script node.
    root_id = None
    if "krk_entry" in graph.nodes:
        root_id = "krk_entry"
    elif "ROOT" in graph.nodes:
        root_id = "ROOT"
    else:
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

    return env.get("chosen_move") or env.get("suggested_move")


def goal_distance(teacher: KRKTeacher, board: chess.Board, goal_bank: dict, min_overlap: float) -> float:
    """Compute weighted goal distance from board features."""
    v = teacher.features(board)
    dist, _overlap = _goal_distance_from_features(
        v, goal_bank, normalize=True, min_overlap=min_overlap
    )
    if dist is None:
        return float("inf")
    return float(dist)


def main():
    parser = argparse.ArgumentParser(description="Stage-1 backchain evaluator")
    parser.add_argument("--topology", type=Path, default=Path("topologies/krk_entry_topology.json"))
    parser.add_argument("--learner", type=Path, default=Path("snapshots/baseline_krk_chain/final_learner.pkl"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min-overlap", type=float, default=8.0)
    parser.add_argument("--lookahead-black", action="store_true", default=True)
    parser.add_argument("--no-lookahead-black", action="store_false", dest="lookahead_black")
    parser.add_argument("--exclude-mate-in-1", action="store_true", default=True)
    parser.add_argument("--include-mate-in-1", action="store_false", dest="exclude_mate_in_1")
    parser.add_argument("--stage-filter", type=int, default=None,
                        help="If set, only actuators from this stage can propose moves")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load learner for goal bank
    if not args.learner.exists():
        raise SystemExit(f"Learner not found: {args.learner}")
    with open(args.learner, "rb") as f:
        learner = pickle.load(f)
    goal_bank = build_goal_bank(learner, label="mate_in_1")
    if not goal_bank:
        raise SystemExit("Goal bank missing; run Stage-0 training first.")

    teacher = KRKTeacher()

    # Load topology
    graph = build_graph_from_topology(args.topology)
    engine = ReConEngine(graph)

    stats = {
        "total": 0,
        "skipped_mate_in_1": 0,
        "no_move": 0,
        "improved": 0,
        "optimal": 0,
        "worsened": 0,
        "avg_reward": 0.0,
    }

    eval_idx = 0
    for i in range(args.samples):
        board = generate_random_krk_position(rng)
        if args.exclude_mate_in_1 and can_deliver_mate(board):
            stats["skipped_mate_in_1"] += 1
            continue

        d0 = goal_distance(teacher, board, goal_bank, args.min_overlap)
        if d0 == float("inf"):
            # If we can't score, skip
            continue

        # Evaluate all legal moves for oracle best improvement
        best_reward = -float("inf")
        move_rewards = {}
        for move in board.legal_moves:
            b1 = board.copy()
            b1.push(move)
            if args.lookahead_black:
                replies = list(b1.legal_moves)
                if replies:
                    d2_list = []
                    for reply in replies:
                        b2 = b1.copy()
                        b2.push(reply)
                        d2_list.append(goal_distance(teacher, b2, goal_bank, args.min_overlap))
                    d1 = max(d2_list) if d2_list else goal_distance(teacher, b1, goal_bank, args.min_overlap)
                else:
                    d1 = goal_distance(teacher, b1, goal_bank, args.min_overlap)
            else:
                d1 = goal_distance(teacher, b1, goal_bank, args.min_overlap)

            reward = d0 - d1
            move_rewards[move.uci()] = reward
            if reward > best_reward:
                best_reward = reward

        chosen = choose_move_with_engine(graph, engine, board, stage_filter=args.stage_filter)
        stats["total"] += 1

        if not chosen:
            stats["no_move"] += 1
            continue

        chosen_reward = move_rewards.get(chosen, -float("inf"))
        stats["avg_reward"] += chosen_reward

        if chosen_reward > args.eps:
            stats["improved"] += 1
        elif chosen_reward < -args.eps:
            stats["worsened"] += 1

        if chosen_reward >= best_reward - args.eps:
            stats["optimal"] += 1

        eval_idx += 1
        if eval_idx % 10 == 0:
            print(f"  {eval_idx:3d}/{args.samples}: improved={stats['improved']} optimal={stats['optimal']}")

    if stats["total"] > 0:
        stats["avg_reward"] /= stats["total"]

    print("\nStage-1 Backchain Evaluation")
    print("-" * 60)
    print(f"Total evaluated: {stats['total']}")
    print(f"Skipped mate-in-1: {stats['skipped_mate_in_1']}")
    print(f"No move: {stats['no_move']}")
    if stats["total"]:
        print(f"Improved: {stats['improved']} ({stats['improved']/stats['total']*100:.1f}%)")
        print(f"Optimal:  {stats['optimal']} ({stats['optimal']/stats['total']*100:.1f}%)")
        print(f"Worsened: {stats['worsened']} ({stats['worsened']/stats['total']*100:.1f}%)")
        print(f"Avg reward (d0-d1): {stats['avg_reward']:.4f}")


if __name__ == "__main__":
    main()
