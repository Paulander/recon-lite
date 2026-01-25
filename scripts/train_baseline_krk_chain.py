"""
Stage-0/1 chained baseline training for KRK.

Stage 0: Learn sensors/actuators from mate-in-1 transitions.
Stage 1: Backchain using goal memories from Stage 0 (move closer to mate-in-1 goals).
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any

import chess
import numpy as np

from recon_lite.learning.baseline import (
    BaselineLearner, Terminal, TerminalRole,
    compute_sensor_xp, should_promote_sensor,
    extract_actuator_patterns, find_similar_actuator,
    TransitionData, apply_sensor,
)
from recon_lite_chess.baseline_teacher import KRKTeacher, generate_krk_mate_in_1_position


def generate_random_krk_position() -> chess.Board:
    """Generate a random legal KRK position (white to move, no check)."""
    squares = list(chess.SQUARES)
    while True:
        wk, bk, wr = random.sample(squares, 3)
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


def collect_goal_memories(learner: BaselineLearner, v0: np.ndarray) -> Dict[int, float] | None:
    """Build a sparse goal memory keyed by sensor id."""
    mature = learner.get_mature_sensors()
    if not mature:
        return None
    goal = {}
    for s in mature:
        goal[s.id] = apply_sensor(s, v0)
    return goal


def goal_distance(current: Dict[int, float], goal: Dict[int, float]) -> float:
    """L2 distance over shared sensor ids."""
    keys = set(current.keys()) & set(goal.keys())
    if not keys:
        return float("inf")
    diffs = [(current[k] - goal[k]) for k in keys]
    return float(np.linalg.norm(diffs))


def compute_sensor_vector_by_ids(learner: BaselineLearner, v: np.ndarray, sensor_ids: List[int]) -> np.ndarray:
    """Compute sensor vector in a fixed, stable sensor-id order."""
    sensor_map = {s.id: s for s in learner.sensors}
    vals = []
    for sid in sensor_ids:
        sensor = sensor_map.get(sid)
        if sensor is None:
            vals.append(0.0)
            continue
        vals.append(apply_sensor(sensor, v))
    return np.array(vals, dtype=np.float32)


def label_transitions_by_goal(
    learner: BaselineLearner,
    board: chess.Board,
    goal_vectors: List[np.ndarray],
    sensor_ids: List[int],
    eps: float = 1e-3,
) -> List[TransitionData]:
    """Label transitions as positive if they move closer to any goal memory."""
    transitions = []
    teacher = KRKTeacher()
    v0 = teacher.features(board)

    s0 = compute_sensor_vector_by_ids(learner, v0, sensor_ids)
    d0 = min((np.linalg.norm(s0 - g) for g in goal_vectors), default=float("inf"))

    for move in board.legal_moves:
        b1 = board.copy()
        b1.push(move)
        v1 = teacher.features(b1)
        s1 = compute_sensor_vector_by_ids(learner, v1, sensor_ids)
        d1 = min((np.linalg.norm(s1 - g) for g in goal_vectors), default=float("inf"))
        label = 1 if d1 < d0 - eps else 0
        transitions.append(TransitionData(v0=v0, v1=v1, label=label, action=move))
    return transitions


def update_learner_from_transitions(learner: BaselineLearner, transitions: List[TransitionData]) -> Dict[str, Any]:
    """Shared update logic for sensors/actuators."""
    sensor_deltas_pos = {s.id: [] for s in learner.sensors}
    sensor_deltas_neg = {s.id: [] for s in learner.sensors}

    for trans in transitions:
        for sensor in learner.sensors:
            t0 = apply_sensor(sensor, trans.v0)
            t1 = apply_sensor(sensor, trans.v1)
            delta_t = t1 - t0
            if trans.label == 1:
                sensor_deltas_pos[sensor.id].append(delta_t)
            else:
                sensor_deltas_neg[sensor.id].append(delta_t)

    # Update sensor XP
    for sensor in learner.sensors:
        xp = compute_sensor_xp(sensor, sensor_deltas_pos[sensor.id], sensor_deltas_neg[sensor.id])
        sensor.xp = xp
        sensor.activations += len(sensor_deltas_pos[sensor.id]) + len(sensor_deltas_neg[sensor.id])
        sensor.cycles_alive += 1
        if len(sensor_deltas_pos[sensor.id]) > 0:
            sensor.good_hits += 1
        if len(sensor_deltas_neg[sensor.id]) > 0:
            sensor.bad_hits += 1

    # Promote/prune
    newly_promoted = []
    for sensor in learner.sensors:
        if should_promote_sensor(sensor):
            sensor.is_mature = True
            newly_promoted.append(sensor.id)

    initial_count = len(learner.sensors)
    learner.sensors = [
        s for s in learner.sensors
        if s.xp > 0.1 or s.cycles_alive < 3 or s.is_mature
    ]
    pruned_count = initial_count - len(learner.sensors)

    # Actuator extraction from positives
    mature_sensors = learner.get_mature_sensors()
    newly_created_actuators = 0
    if len(mature_sensors) >= 3:
        positive_trans = [t for t in transitions if t.label == 1]
        if positive_trans:
            actuator_specs = extract_actuator_patterns(positive_trans, mature_sensors, eps=0.1, top_k=5)
            for spec in actuator_specs:
                existing = find_similar_actuator(learner.actuators, spec, similarity_threshold=0.9)
                if existing:
                    existing.actuator_spec.goal_delta = (
                        0.8 * existing.actuator_spec.goal_delta +
                        0.2 * spec.goal_delta
                    )
                    existing.xp += 0.1
                    existing.activations += 1
                else:
                    actuator = Terminal(
                        id=learner._next_actuator_id,
                        stage=learner.stage,
                        role=TerminalRole.ACTUATOR,
                        actuator_spec=spec
                    )
                    learner._next_actuator_id += 1
                    learner.actuators.append(actuator)
                    newly_created_actuators += 1

    return {
        "newly_promoted": newly_promoted,
        "pruned_count": pruned_count,
        "newly_created_actuators": newly_created_actuators,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-0/1 chained baseline training for KRK")
    parser.add_argument("--stage0-cycles", type=int, default=50)
    parser.add_argument("--stage1-cycles", type=int, default=50)
    parser.add_argument("--samples-per-cycle", type=int, default=50)
    parser.add_argument("--initial-sensors", type=int, default=20)
    parser.add_argument("--spawn-interval", type=int, default=10)
    parser.add_argument("--sensors-per-spawn", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("snapshots/baseline_krk_chain"))
    parser.add_argument("--save-learner", type=Path, default=Path("snapshots/baseline_krk_chain/final_learner.pkl"))
    parser.add_argument("--goal-eps", type=float, default=0.15)
    parser.add_argument("--max-goals", type=int, default=100)
    parser.add_argument("--min-mature-for-goals", type=int, default=8)
    args = parser.parse_args()

    teacher = KRKTeacher()
    learner = BaselineLearner(
        feature_dim=teacher.feature_dim,
        stage=0,
        goal_eps=args.goal_eps,
        max_goals=args.max_goals,
    )

    for _ in range(args.initial_sensors):
        learner.sensors.append(learner.spawn_sensor())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 0: Mate-in-1")
    print("=" * 70)
    goal_sensor_ids: List[int] | None = None

    for cycle in range(args.stage0_cycles):
        transitions = []
        for _ in range(args.samples_per_cycle):
            b0 = generate_krk_mate_in_1_position()
            transitions.extend(teacher.label_transitions(b0))
            # Store goal prototypes only after enough mature sensors
            if len(learner.get_mature_sensors()) >= args.min_mature_for_goals:
                if goal_sensor_ids is None:
                    goal_sensor_ids = [s.id for s in learner.get_mature_sensors()]
                v0 = teacher.features(b0)
                s0 = compute_sensor_vector_by_ids(learner, v0, goal_sensor_ids)
                learner.add_goal_memory(
                    s0,
                    label="mate_in_1",
                    goal_eps=args.goal_eps,
                    max_goals=args.max_goals,
                )

        stats = update_learner_from_transitions(learner, transitions)

        if cycle % 10 == 0 or stats["newly_promoted"] or stats["newly_created_actuators"]:
            mature = len(learner.get_mature_sensors())
            print(f"Cycle {cycle:3d}: sensors={len(learner.sensors)} (mature={mature}) "
                  f"actuators={len(learner.actuators)} goal_prototypes={len(learner.goal_memories)}")

        if cycle % args.spawn_interval == 0 and cycle > 0:
            for _ in range(args.sensors_per_spawn):
                learner.sensors.append(learner.spawn_sensor())

    print("=" * 70)
    print("Stage 1: Backchain to Mate-in-1 goals")
    print("=" * 70)
    learner.stage = 1
    if goal_sensor_ids is None:
        goal_sensor_ids = [s.id for s in learner.get_mature_sensors()]

    for cycle in range(args.stage1_cycles):
        transitions = []
        for _ in range(args.samples_per_cycle):
            b0 = generate_random_krk_position()
            goal_vectors = [
                g.s0 for g in learner.goal_memories
                if g.label == "mate_in_1" and g.s0.shape == (len(goal_sensor_ids),)
            ]
            transitions.extend(label_transitions_by_goal(learner, b0, goal_vectors, goal_sensor_ids))

        stats = update_learner_from_transitions(learner, transitions)

        if cycle % 10 == 0 or stats["newly_promoted"] or stats["newly_created_actuators"]:
            mature = len(learner.get_mature_sensors())
            print(f"Cycle {cycle:3d}: sensors={len(learner.sensors)} (mature={mature}) "
                  f"actuators={len(learner.actuators)} goal_prototypes={len(learner.goal_memories)}")

        if cycle % args.spawn_interval == 0 and cycle > 0:
            for _ in range(args.sensors_per_spawn):
                learner.sensors.append(learner.spawn_sensor())

    # Save learner pickle
    args.save_learner.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_learner, "wb") as f:
        pickle.dump(learner, f)
    print(f"\nSaved learner: {args.save_learner}")


if __name__ == "__main__":
    main()
