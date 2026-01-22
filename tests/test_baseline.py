"""
Unit tests for baseline architecture components.

Tests:
- Role invariant validation
- Sensor readouts (all types)
- XP computation
- Actuator pattern extraction
- Sensor spawning
- KRK teacher features
"""

import pytest
import numpy as np
import chess
from recon_lite.learning.baseline import (
    Terminal, TerminalRole, SensorSpec, ActuatorSpec, GoalMemory,
    apply_sensor, compute_sensor_xp, should_promote_sensor,
    extract_actuator_patterns, spawn_sensor, BaselineLearner,
    TransitionData,
    XP_PROMOTE_THRESHOLD, MIN_CYCLES_FOR_PROMOTION, MIN_ACTIVATIONS_FOR_PROMOTION
)
from recon_lite_chess.baseline_teacher import KRKTeacher, generate_krk_mate_in_1_position


# ============================================================================
# Test Role Invariant Validation
# ============================================================================

def test_sensor_terminal_valid():
    """Test that SENSOR terminal with sensor_spec is valid"""
    spec = SensorSpec(
        feature_mask=np.array([True, False, True]),
        readout_type="mean"
    )
    terminal = Terminal(
        id=0,
        stage=0,
        role=TerminalRole.SENSOR,
        sensor_spec=spec
    )
    assert terminal.role == TerminalRole.SENSOR
    assert terminal.sensor_spec is not None
    assert terminal.actuator_spec is None


def test_actuator_terminal_valid():
    """Test that ACTUATOR terminal with actuator_spec is valid"""
    spec = ActuatorSpec(
        sensor_indices=[0, 1, 2],
        goal_delta=np.array([0.5, -0.3, 0.8])
    )
    terminal = Terminal(
        id=0,
        stage=0,
        role=TerminalRole.ACTUATOR,
        actuator_spec=spec
    )
    assert terminal.role == TerminalRole.ACTUATOR
    assert terminal.actuator_spec is not None
    assert terminal.sensor_spec is None


def test_sensor_without_spec_raises():
    """Test that SENSOR without sensor_spec raises ValueError"""
    with pytest.raises(ValueError, match="SENSOR terminal must have sensor_spec"):
        Terminal(
            id=0,
            stage=0,
            role=TerminalRole.SENSOR
        )


def test_sensor_with_actuator_spec_raises():
    """Test that SENSOR with actuator_spec raises ValueError"""
    with pytest.raises(ValueError, match="SENSOR terminal cannot have actuator_spec"):
        Terminal(
            id=0,
            stage=0,
            role=TerminalRole.SENSOR,
            sensor_spec=SensorSpec(
                feature_mask=np.array([True]),
                readout_type="identity"
            ),
            actuator_spec=ActuatorSpec(
                sensor_indices=[0],
                goal_delta=np.array([0.5])
            )
        )


# ============================================================================
# Test Sensor Readouts
# ============================================================================

def test_identity_readout():
    """Test identity readout"""
    spec = SensorSpec(
        feature_mask=np.array([False, True, False]),
        readout_type="identity"
    )
    sensor = Terminal(id=0, stage=0, role=TerminalRole.SENSOR, sensor_spec=spec)
    v = np.array([0.1, 0.7, 0.3])
    
    result = apply_sensor(sensor, v)
    assert result == 0.7


def test_sum_readout():
    """Test sum readout"""
    spec = SensorSpec(
        feature_mask=np.array([True, True, False]),
        readout_type="sum"
    )
    sensor = Terminal(id=0, stage=0, role=TerminalRole.SENSOR, sensor_spec=spec)
    v = np.array([0.2, 0.3, 0.5])
    
    result = apply_sensor(sensor, v)
    assert np.isclose(result, 0.5)


def test_mean_readout():
    """Test mean readout"""
    spec = SensorSpec(
        feature_mask=np.array([True, True, False]),
        readout_type="mean"
    )
    sensor = Terminal(id=0, stage=0, role=TerminalRole.SENSOR, sensor_spec=spec)
    v = np.array([0.2, 0.4, 0.6])
    
    result = apply_sensor(sensor, v)
    assert np.isclose(result, 0.3)


def test_threshold_readout():
    """Test threshold readout"""
    spec = SensorSpec(
        feature_mask=np.array([True, True]),
        readout_type="threshold",
        readout_params={"threshold": 0.5}
    )
    sensor = Terminal(id=0, stage=0, role=TerminalRole.SENSOR, sensor_spec=spec)
    
    # Above threshold
    v1 = np.array([0.6, 0.7])
    assert apply_sensor(sensor, v1) == 1.0
    
    # Below threshold
    v2 = np.array([0.3, 0.4])
    assert apply_sensor(sensor, v2) == 0.0


# ============================================================================
# Test XP Computation
# ============================================================================

def test_xp_high_stability():
    """Test XP with high stability (low variance on positives)"""
    sensor = Terminal(
        id=0, stage=0, role=TerminalRole.SENSOR,
        sensor_spec=SensorSpec(feature_mask=np.array([True]), readout_type="identity")
    )
    
    # Low variance on positives
    delta_pos = [0.5, 0.51, 0.49, 0.5, 0.5]
    delta_neg = [0.0, 0.1, -0.1]
    
    xp = compute_sensor_xp(sensor, delta_pos, delta_neg)
    
    # Should have high stability component
    assert xp > 0.5


def test_xp_high_separation():
    """Test XP with high separation (different pos vs neg)"""
    sensor = Terminal(
        id=0, stage=0, role=TerminalRole.SENSOR,
        sensor_spec=SensorSpec(feature_mask=np.array([True]), readout_type="identity")
    )
    
    # Clear separation
    delta_pos = [0.8, 0.9, 0.85, 0.9]
    delta_neg = [0.1, 0.0, 0.05, 0.1]
    
    xp = compute_sensor_xp(sensor, delta_pos, delta_neg)
    
    # Should have high separation component
    assert xp > 0.6


def test_promotion_criteria():
    """Test sensor promotion criteria"""
    sensor = Terminal(
        id=0, stage=0, role=TerminalRole.SENSOR,
        sensor_spec=SensorSpec(feature_mask=np.array([True]), readout_type="identity")
    )
    
    # Not enough XP
    sensor.xp = 0.5
    sensor.cycles_alive = 10
    sensor.activations = 30
    assert not should_promote_sensor(sensor)
    
    # Not enough cycles
    sensor.xp = 0.8
    sensor.cycles_alive = 3
    sensor.activations = 30
    assert not should_promote_sensor(sensor)
    
    # Not enough activations
    sensor.xp = 0.8
    sensor.cycles_alive = 10
    sensor.activations = 10
    assert not should_promote_sensor(sensor)
    
    # All criteria met
    sensor.xp = 0.8
    sensor.cycles_alive = 10
    sensor.activations = 30
    assert should_promote_sensor(sensor)


# ============================================================================
# Test Actuator Pattern Extraction
# ============================================================================

def test_extract_actuator_patterns():
    """Test actuator pattern extraction from transitions"""
    # Create mature sensors
    sensors = [
        Terminal(
            id=i, stage=0, role=TerminalRole.SENSOR,
            sensor_spec=SensorSpec(
                feature_mask=np.array([i == j for j in range(5)]),
                readout_type="identity"
            ),
            is_mature=True
        )
        for i in range(5)
    ]
    
    # Create transitions with consistent pattern
    transitions = [
        TransitionData(
            v0=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            v1=np.array([0.5, -0.3, 0.0, 0.8, 0.0]),
            label=1
        )
        for _ in range(3)
    ]
    
    patterns = extract_actuator_patterns(transitions, sensors, eps=0.1, top_k=5)
    
    assert len(patterns) > 0
    # Should extract pattern with sensors 0, 1, 3 (significant deltas)
    assert len(patterns[0].sensor_indices) == 3


# ============================================================================
# Test Sensor Spawning
# ============================================================================

def test_spawn_sensor():
    """Test sensor spawning with random mask"""
    sensor = spawn_sensor(
        id=0,
        stage=0,
        feature_dim=10,
        allowed_features=list(range(10)),
        mask_size_range=(1, 4)
    )
    
    assert sensor.role == TerminalRole.SENSOR
    assert sensor.sensor_spec is not None
    assert np.sum(sensor.sensor_spec.feature_mask) >= 1
    assert np.sum(sensor.sensor_spec.feature_mask) <= 4


# ============================================================================
# Test BaselineLearner
# ============================================================================

def test_baseline_learner_init():
    """Test BaselineLearner initialization"""
    learner = BaselineLearner(feature_dim=10, stage=0)
    
    assert learner.feature_dim == 10
    assert learner.stage == 0
    assert len(learner.sensors) == 0
    assert len(learner.actuators) == 0
    assert len(learner.goal_memories) == 0


def test_baseline_learner_spawn_sensor():
    """Test sensor spawning through learner"""
    learner = BaselineLearner(feature_dim=10, stage=0)
    
    sensor1 = learner.spawn_sensor()
    sensor2 = learner.spawn_sensor()
    
    assert sensor1.id == 0
    assert sensor2.id == 1
    assert sensor1.id != sensor2.id


def test_baseline_learner_add_goal_memory():
    """Test goal memory addition"""
    learner = BaselineLearner(feature_dim=10, stage=0)
    
    s0 = np.array([0.5, 0.3, 0.8])
    goal1 = learner.add_goal_memory(s0, "mate_in_1")
    
    assert goal1.label == "mate_in_1"
    assert len(learner.goal_memories) == 1
    
    # Adding similar goal should increment count
    goal2 = learner.add_goal_memory(s0, "mate_in_1")
    assert goal2.id == goal1.id
    assert goal2.count == 2
    assert len(learner.goal_memories) == 1


# ============================================================================
# Test KRK Teacher
# ============================================================================

def test_krk_teacher_feature_dim():
    """Test KRK teacher feature dimension"""
    teacher = KRKTeacher()
    assert teacher.feature_dim == 13


def test_krk_teacher_features():
    """Test KRK feature extraction"""
    teacher = KRKTeacher()
    board = chess.Board("7k/5K2/6R1/8/8/8/8/8 w - - 0 1")
    
    features = teacher.features(board)
    
    assert len(features) == 13
    assert all(0.0 <= f <= 1.0 for f in features)


def test_krk_teacher_label_transitions():
    """Test transition labeling for mate-in-1"""
    teacher = KRKTeacher()
    board = chess.Board("7k/5K2/6R1/8/8/8/8/8 w - - 0 1")
    
    transitions = teacher.label_transitions(board)
    
    assert len(transitions) > 0
    # At least one transition should be labeled as mate
    assert any(t.label == 1 for t in transitions)


def test_generate_mate_in_1():
    """Test mate-in-1 position generation"""
    board = generate_krk_mate_in_1_position()
    
    # Should be a valid KRK position
    assert board.is_valid()
    
    # Should have exactly 3 pieces
    assert len(board.piece_map()) == 3
    
    # Should have mate-in-1 available
    has_mate = False
    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        if test_board.is_checkmate():
            has_mate = True
            break
    
    assert has_mate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
