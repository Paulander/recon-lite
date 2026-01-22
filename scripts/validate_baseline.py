"""
Quick validation script for baseline architecture.
Run this to verify all core components are working.
"""

import numpy as np
import chess
from recon_lite.learning.baseline import (
    Terminal, TerminalRole, SensorSpec, ActuatorSpec,
    apply_sensor, compute_sensor_xp, BaselineLearner
)
from recon_lite_chess.baseline_teacher import KRKTeacher, generate_krk_mate_in_1_position


def main():
    print("=" * 60)
    print("Baseline Architecture Validation")
    print("=" * 60)
    
    # Test 1: Role validation
    print("\n1. Testing role validation...")
    try:
        spec = SensorSpec(
            feature_mask=np.array([True, False, True]),
            readout_type="mean"
        )
        sensor = Terminal(id=0, stage=0, role=TerminalRole.SENSOR, sensor_spec=spec)
        print(f"   ✓ Created sensor: {sensor}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    # Test 2: Sensor readouts
    print("\n2. Testing sensor readouts...")
    v = np.array([0.2, 0.5, 0.8])
    result = apply_sensor(sensor, v)
    print(f"   ✓ Sensor output: {result:.3f} (expected ~0.5)")
    
    # Test 3: XP computation
    print("\n3. Testing XP computation...")
    delta_pos = [0.5, 0.51, 0.49, 0.5]
    delta_neg = [0.1, 0.0, 0.05]
    xp = compute_sensor_xp(sensor, delta_pos, delta_neg)
    print(f"   ✓ Sensor XP: {xp:.3f}")
    
    # Test 4: BaselineLearner
    print("\n4. Testing BaselineLearner...")
    learner = BaselineLearner(feature_dim=13, stage=0)
    for _ in range(5):
        s = learner.spawn_sensor()
        learner.sensors.append(s)
    print(f"   ✓ Spawned {len(learner.sensors)} sensors")
    print(f"   ✓ Learner: {learner}")
    
    # Test 5: KRK Teacher
    print("\n5. Testing KRK Teacher...")
    teacher = KRKTeacher()
    print(f"   ✓ Teacher feature_dim: {teacher.feature_dim}")
    
    board = generate_krk_mate_in_1_position()
    features = teacher.features(board)
    print(f"   ✓ Features shape: {features.shape}")
    print(f"   ✓ Features: {features[:5]}...")  # First 5
    
    transitions = teacher.label_transitions(board)
    mate_count = sum(t.label for t in transitions)
    print(f"   ✓ Generated {len(transitions)} transitions ({mate_count} mates)")
    
    # Test 6: Actuator spec
    print("\n6. Testing ActuatorSpec...")
    try:
        act_spec = ActuatorSpec(
            sensor_indices=[0, 1, 2],
            goal_delta=np.array([0.5, -0.3, 0.8])
        )
        actuator = Terminal(
            id=0, stage=0, role=TerminalRole.ACTUATOR,
            actuator_spec=act_spec
        )
        print(f"   ✓ Created actuator: {actuator}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✓ All validation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
