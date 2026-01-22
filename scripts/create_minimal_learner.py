"""
Create minimal learner pickle for graph compilation testing.

This creates a placeholder learner with the right structure.
For production, modify train_baseline_krk.py to save the actual learner.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.learning.baseline import BaselineLearner, Terminal, TerminalRole, SensorSpec, ActuatorSpec
from recon_lite_chess.baseline_teacher import KRKTeacher
import numpy as np

print("Creating minimal learner for compilation...")

teacher = KRKTeacher()
learner = BaselineLearner(feature_dim=teacher.feature_dim, stage=0)

# Create 40 sensors (25 mature)
for i in range(40):
    sensor = learner.spawn_sensor()
    sensor.id = i
    sensor.xp = 0.8 if i < 25 else 0.3
    sensor.is_mature = (i < 25)
    sensor.cycles_alive = 50
    sensor.activations = 20000 if sensor.is_mature else 5000
    learner.sensors.append(sensor)

# Create 8 actuators (from checkpoint data)
actuator_patterns = [
    ([6, 7, 11, 18, 19], [-0.20, 0.20, 0.37, -0.98, -1.03]),
    ([6, 7, 11, 19, 20], [-0.25, 0.29, -0.81, -1.0, -1.0]),
    ([6, 7, 19, 20, 27], [0.29, -0.81, -1.0, -1.0, -1.0]),
    ([6, 7, 18, 19, 27], [0.25, 0.38, -1.0, -1.0, -1.03]),
    ([6, 19, 20, 27, 28], [-0.81, -1.0, -1.0, -1.0, -1.14]),
    ([7, 11, 18, 19, 20], [0.20, 0.37, -0.98, -1.03, -1.0]),
    ([6, 11, 18, 19, 27], [-0.20, 0.37, -0.98, -1.03, -1.0]),
    ([7, 18, 19, 20, 27], [0.38, -1.0, -1.0, -1.0, -1.03]),
]

for i, (indices, deltas) in enumerate(actuator_patterns):
    spec = ActuatorSpec(
        sensor_indices=indices,
        goal_delta=np.array(deltas),
        match_mode="cosine"
    )
    
    actuator = Terminal(
        id=i,
        stage=0,
        role=TerminalRole.ACTUATOR,
        actuator_spec=spec
    )
    actuator.xp = 1.5
    actuator.cycles_alive = 50
    actuator.activations = 1000
    learner.actuators.append(actuator)

# Save
output_path = Path("snapshots/baseline_krk_final/final_learner.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(learner, f)

print(f"✓ Saved minimal learner: {output_path}")
print(f"  Sensors: {len(learner.sensors)} ({len([s for s in learner.sensors if s.is_mature])} mature)")
print(f"  Actuators: {len(learner.actuators)}")
print("\nNote: This is placeholder data for testing compilation.")
print("For production, modify train_baseline_krk.py to save the actual learner object.")
