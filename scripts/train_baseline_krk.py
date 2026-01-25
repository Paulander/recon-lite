"""
Training script for baseline architecture on KRK mate-in-1.

Implements the complete training loop:
1. Generate labeled transitions from teacher
2. Compute Δt for ALL sensors (candidate + mature)
3. Update sensor XP using explicit formula
4. Promote/prune sensors based on thresholds
5. Learn actuators from positive transitions (using mature sensors only)
6. Collect goal memories from successful starting states
7. Spawn new sensors periodically
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np

from recon_lite.learning.baseline import (
    BaselineLearner, Terminal, TerminalRole, ActuatorSpec,
    compute_sensor_xp, should_promote_sensor,
    extract_actuator_patterns, find_similar_actuator,
    TransitionData
)
from recon_lite_chess.baseline_teacher import KRKTeacher, generate_krk_mate_in_1_position


# ============================================================================
# Training Configuration
# ============================================================================

class TrainingConfig:
    """Configuration for baseline training"""
    
    def __init__(
        self,
        samples_per_cycle: int = 50,
        max_cycles: int = 100,
        initial_sensors: int = 20,
        spawn_interval: int = 10,
        sensors_per_spawn: int = 5,
        output_dir: Path = Path("snapshots/baseline_krk"),
        save_interval: int = 10,
    ):
        self.samples_per_cycle = samples_per_cycle
        self.max_cycles = max_cycles
        self.initial_sensors = initial_sensors
        self.spawn_interval = spawn_interval
        self.sensors_per_spawn = sensors_per_spawn
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.save_learner_path: Path | None = None


# ============================================================================
# Training Loop
# ============================================================================

def train_baseline_krk(config: TrainingConfig) -> Dict:
    """
    Main training loop for baseline architecture on KRK mate-in-1.
    
    Args:
        config: Training configuration
    
    Returns:
        Training statistics and final learner state
    """
    print("=" * 70)
    print("Baseline Architecture Training: KRK Mate-in-1")
    print("=" * 70)
    
    # Initialize teacher and learner
    teacher = KRKTeacher()
    learner = BaselineLearner(feature_dim=teacher.feature_dim, stage=0)
    
    print(f"\nInitialization:")
    print(f"  Feature dimension: {teacher.feature_dim}")
    print(f"  Initial sensors: {config.initial_sensors}")
    
    # Spawn initial sensors
    for _ in range(config.initial_sensors):
        sensor = learner.spawn_sensor()
        learner.sensors.append(sensor)
    
    print(f"  Spawned {len(learner.sensors)} sensors")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training statistics
    stats = {
        "cycles": [],
        "sensor_counts": [],
        "mature_sensor_counts": [],
        "actuator_counts": [],
        "goal_memory_counts": [],
        "avg_sensor_xp": [],
    }
    
    # Main training loop
    print(f"\nStarting training for {config.max_cycles} cycles...")
    print("-" * 70)
    
    for cycle in range(config.max_cycles):
        # ====================================================================
        # 1. Generate labeled transitions
        # ====================================================================
        all_transitions = []
        for _ in range(config.samples_per_cycle):
            x0 = generate_krk_mate_in_1_position()
            transitions = teacher.label_transitions(x0)
            all_transitions.extend(transitions)
        
        positive_count = sum(1 for t in all_transitions if t.label == 1)
        negative_count = len(all_transitions) - positive_count
        
        # ====================================================================
        # 2. Compute Δt for ALL sensors (candidate + mature)
        # ====================================================================
        sensor_deltas_pos = {s.id: [] for s in learner.sensors}
        sensor_deltas_neg = {s.id: [] for s in learner.sensors}
        
        for trans in all_transitions:
            for sensor in learner.sensors:
                from recon_lite.learning.baseline import apply_sensor
                
                t0 = apply_sensor(sensor, trans.v0)
                t1 = apply_sensor(sensor, trans.v1)
                delta_t = t1 - t0
                
                if trans.label == 1:
                    sensor_deltas_pos[sensor.id].append(delta_t)
                else:
                    sensor_deltas_neg[sensor.id].append(delta_t)
        
        # ====================================================================
        # 3. Update sensor XP using explicit formula
        # ====================================================================
        for sensor in learner.sensors:
            xp = compute_sensor_xp(
                sensor,
                sensor_deltas_pos[sensor.id],
                sensor_deltas_neg[sensor.id]
            )
            sensor.xp = xp
            sensor.activations += len(sensor_deltas_pos[sensor.id]) + len(sensor_deltas_neg[sensor.id])
            sensor.cycles_alive += 1
            
            # Track good/bad hits for stats
            if len(sensor_deltas_pos[sensor.id]) > 0:
                sensor.good_hits += 1
            if len(sensor_deltas_neg[sensor.id]) > 0:
                sensor.bad_hits += 1
        
        # ====================================================================
        # 4. Promote sensors based on thresholds
        # ====================================================================
        newly_promoted = []
        for sensor in learner.sensors:
            if should_promote_sensor(sensor):
                sensor.is_mature = True
                newly_promoted.append(sensor.id)
        
        # ====================================================================
        # 5. Prune low-XP sensors
        # ====================================================================
        initial_count = len(learner.sensors)
        learner.sensors = [
            s for s in learner.sensors
            if s.xp > 0.1 or s.cycles_alive < 3 or s.is_mature
        ]
        pruned_count = initial_count - len(learner.sensors)
        
        # ====================================================================
        # 6. Learn actuators from positive transitions (using mature sensors only)
        # ====================================================================
        mature_sensors = learner.get_mature_sensors()
        newly_created_actuators = 0
        
        if len(mature_sensors) >= 3:  # Need at least 3 mature sensors
            positive_trans = [t for t in all_transitions if t.label == 1]
            
            if len(positive_trans) > 0:
                actuator_specs = extract_actuator_patterns(
                    positive_trans,
                    mature_sensors,
                    eps=0.1,
                    top_k=5
                )
                
                # Create/update actuator terminals
                for spec in actuator_specs:
                    # Check if similar actuator exists
                    existing = find_similar_actuator(learner.actuators, spec, similarity_threshold=0.9)
                    
                    if existing:
                        # Update existing (running mean)
                        existing.actuator_spec.goal_delta = (
                            0.8 * existing.actuator_spec.goal_delta +
                            0.2 * spec.goal_delta
                        )
                        existing.xp += 0.1
                        existing.activations += 1
                    else:
                        # Create new actuator
                        actuator = Terminal(
                            id=learner._next_actuator_id,
                            stage=0,
                            role=TerminalRole.ACTUATOR,
                            actuator_spec=spec
                        )
                        learner._next_actuator_id += 1
                        learner.actuators.append(actuator)
                        newly_created_actuators += 1
        
        # ====================================================================
        # 7. Collect goal memories from positive starting states
        # ====================================================================
        newly_created_goals = 0
        if len(mature_sensors) > 0:
            # Group transitions by starting position
            start_positions = {}
            for trans in all_transitions:
                if trans.label == 1:
                    # Use v0 as key (convert to tuple for hashing)
                    key = tuple(trans.v0)
                    if key not in start_positions:
                        start_positions[key] = trans.v0
            
            # Create goal memories
            for v0 in start_positions.values():
                from recon_lite.learning.baseline import apply_sensor
                s0 = np.array([apply_sensor(s, v0) for s in mature_sensors])
                
                # Check if this is a new goal
                initial_goal_count = len(learner.goal_memories)
                learner.add_goal_memory(s0, label="mate_in_1")
                if len(learner.goal_memories) > initial_goal_count:
                    newly_created_goals += 1
        
        # ====================================================================
        # 8. Spawn new sensors periodically
        # ====================================================================
        newly_spawned = 0
        if cycle % config.spawn_interval == 0 and cycle > 0:
            for _ in range(config.sensors_per_spawn):
                sensor = learner.spawn_sensor()
                learner.sensors.append(sensor)
                newly_spawned += 1
        
        # ====================================================================
        # 9. Collect statistics
        # ====================================================================
        mature_count = len(mature_sensors)
        avg_xp = np.mean([s.xp for s in learner.sensors]) if learner.sensors else 0.0
        
        stats["cycles"].append(cycle)
        stats["sensor_counts"].append(len(learner.sensors))
        stats["mature_sensor_counts"].append(mature_count)
        stats["actuator_counts"].append(len(learner.actuators))
        stats["goal_memory_counts"].append(len(learner.goal_memories))
        stats["avg_sensor_xp"].append(avg_xp)
        
        # ====================================================================
        # 10. Report progress
        # ====================================================================
        if cycle % 10 == 0 or newly_promoted or newly_created_actuators > 0:
            print(f"\nCycle {cycle:3d}:")
            print(f"  Transitions: {len(all_transitions)} ({positive_count} pos, {negative_count} neg)")
            print(f"  Sensors: {len(learner.sensors)} ({mature_count} mature, avg XP={avg_xp:.3f})")
            
            if newly_promoted:
                print(f"  ✓ Promoted {len(newly_promoted)} sensors: {newly_promoted[:5]}...")
            
            if pruned_count > 0:
                print(f"  ✗ Pruned {pruned_count} low-XP sensors")
            
            if newly_spawned > 0:
                print(f"  + Spawned {newly_spawned} new sensors")
            
            print(f"  Actuators: {len(learner.actuators)}", end="")
            if newly_created_actuators > 0:
                print(f" (+{newly_created_actuators} new)")
            else:
                print()
            
            print(f"  Goal memories: {len(learner.goal_memories)}", end="")
            if newly_created_goals > 0:
                print(f" (+{newly_created_goals} new)")
            else:
                print()
        
        # ====================================================================
        # 11. Save checkpoint
        # ====================================================================
        if cycle % config.save_interval == 0 and cycle > 0:
            checkpoint_path = config.output_dir / f"checkpoint_cycle_{cycle}.json"
            save_checkpoint(learner, stats, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
    
    # ====================================================================
    # Final report
    # ====================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nFinal Statistics:")
    print(f"  Total sensors: {len(learner.sensors)} ({len(mature_sensors)} mature)")
    print(f"  Total actuators: {len(learner.actuators)}")
    print(f"  Total goal memories: {len(learner.goal_memories)}")
    print(f"  Average sensor XP: {avg_xp:.3f}")
    
    # Show top sensors by XP
    if learner.sensors:
        top_sensors = sorted(learner.sensors, key=lambda s: s.xp, reverse=True)[:5]
        print(f"\nTop 5 Sensors by XP:")
        for i, s in enumerate(top_sensors, 1):
            mature_str = "✓" if s.is_mature else "○"
            print(f"  {i}. Sensor {s.id:3d}: XP={s.xp:.3f} {mature_str} (cycles={s.cycles_alive}, acts={s.activations})")
    
    # Show actuator patterns
    if learner.actuators:
        print(f"\nActuator Patterns:")
        for i, act in enumerate(learner.actuators[:5], 1):
            spec = act.actuator_spec
            print(f"  {i}. Actuator {act.id}: {len(spec.sensor_indices)} sensors, XP={act.xp:.3f}")
            print(f"     Sensors: {spec.sensor_indices[:5]}...")
            print(f"     Goal Δ: {spec.goal_delta[:5]}...")
    
    # Save final checkpoint
    final_path = config.output_dir / "final_checkpoint.json"
    save_checkpoint(learner, stats, final_path)
    print(f"\n💾 Saved final checkpoint: {final_path}")

    if config.save_learner_path:
        config.save_learner_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.save_learner_path, "wb") as f:
            pickle.dump(learner, f)
        print(f"💾 Saved learner pickle: {config.save_learner_path}")
    
    return {
        "learner": learner,
        "stats": stats,
        "config": config
    }


# ============================================================================
# Checkpoint Saving/Loading
# ============================================================================

def save_checkpoint(learner: BaselineLearner, stats: Dict, path: Path):
    """Save learner state and statistics to JSON"""
    checkpoint = {
        "learner": {
            "feature_dim": learner.feature_dim,
            "stage": learner.stage,
            "sensor_count": len(learner.sensors),
            "actuator_count": len(learner.actuators),
            "goal_memory_count": len(learner.goal_memories),
            "mature_sensor_count": len(learner.get_mature_sensors()),
        },
        "stats": stats,
    }
    
    with open(path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train baseline architecture on KRK mate-in-1")
    parser.add_argument("--samples-per-cycle", type=int, default=50,
                       help="Number of positions to sample per cycle")
    parser.add_argument("--max-cycles", type=int, default=100,
                       help="Maximum number of training cycles")
    parser.add_argument("--initial-sensors", type=int, default=20,
                       help="Number of initial sensor terminals")
    parser.add_argument("--spawn-interval", type=int, default=10,
                       help="Spawn new sensors every N cycles")
    parser.add_argument("--sensors-per-spawn", type=int, default=5,
                       help="Number of sensors to spawn each interval")
    parser.add_argument("--output-dir", type=Path, default=Path("snapshots/baseline_krk"),
                       help="Output directory for checkpoints")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save checkpoint every N cycles")
    parser.add_argument("--save-learner", type=Path, default=None,
                       help="Optional path to save BaselineLearner pickle")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        samples_per_cycle=args.samples_per_cycle,
        max_cycles=args.max_cycles,
        initial_sensors=args.initial_sensors,
        spawn_interval=args.spawn_interval,
        sensors_per_spawn=args.sensors_per_spawn,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
    )
    if args.save_learner:
        config.save_learner_path = args.save_learner
    
    result = train_baseline_krk(config)
    
    print("\n✓ Training complete!")
    return result


if __name__ == "__main__":
    main()
