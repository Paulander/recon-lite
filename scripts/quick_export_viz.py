"""
Quick script to export baseline visualization from training run.

Since the training script doesn't save learner state yet, we'll re-run
a minimal version to recreate the final state and export it.
"""

import json
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.learning.baseline import BaselineLearner
from recon_lite.learning.baseline_viz import (
    export_baseline_to_topology,
    create_baseline_visualization_index
)
from recon_lite_chess.baseline_teacher import KRKTeacher, generate_krk_mate_in_1_position


def recreate_and_export():
    """
    Recreate learner from checkpoint stats and export visualization.
    
    Note: This is a workaround. Ideally, training script should save learner state.
    """
    print("=" * 70)
    print("Baseline Visualization Export (from final checkpoint)")
    print("=" * 70)
    
    # Load checkpoint stats
    checkpoint_path = Path("snapshots/baseline_krk_viz/final_checkpoint.json")
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
    
    learner_info = checkpoint["learner"]
    print(f"  Sensors: {learner_info['sensor_count']} ({learner_info['mature_sensor_count']} mature)")
    print(f"  Actuators: {learner_info['actuator_count']}")
    print(f"  Goal memories: {learner_info['goal_memory_count']}")
    
    # Create a minimal learner for visualization
    # (We don't have the actual sensor/actuator objects, so create placeholders)
    teacher = KRKTeacher()
    learner = BaselineLearner(feature_dim=teacher.feature_dim, stage=0)
    
    # Spawn sensors to match count
    print(f"\nCreating placeholder sensors...")
    for i in range(learner_info['sensor_count']):
        sensor = learner.spawn_sensor()
        sensor.is_mature = (i < learner_info['mature_sensor_count'])
        sensor.xp = 0.8 if sensor.is_mature else 0.3
        sensor.cycles_alive = 50
        sensor.activations = 20000 if sensor.is_mature else 5000
        learner.sensors.append(sensor)
    
    # Create placeholder actuators
    print(f"Creating placeholder actuators...")
    from recon_lite.learning.baseline import Terminal, TerminalRole, ActuatorSpec
    import numpy as np
    
    for i in range(learner_info['actuator_count']):
        # Create actuator with random pattern
        mature_ids = [s.id for s in learner.sensors if s.is_mature]
        sensor_indices = list(np.random.choice(mature_ids, size=min(5, len(mature_ids)), replace=False))
        
        spec = ActuatorSpec(
            sensor_indices=sensor_indices,
            goal_delta=np.random.choice([-1.0, 1.0], size=len(sensor_indices)),
            match_mode="l2"
        )
        
        actuator = Terminal(
            id=i,
            stage=0,
            role=TerminalRole.ACTUATOR,
            actuator_spec=spec
        )
        actuator.xp = 1.5
        learner.actuators.append(actuator)
    
    # Export visualization
    viz_dir = Path("snapshots/baseline_evolution")
    viz_path = viz_dir / "baseline" / "cycle_0050.json"
    
    print(f"\nExporting topology to: {viz_path}")
    topology = export_baseline_to_topology(learner, 50, viz_path)
    print(f"  Nodes: {len(topology['nodes'])}")
    print(f"  Edges: {len(topology['edges'])}")
    
    # Create index
    index_path = create_baseline_visualization_index(viz_dir, "baseline")
    print(f"\nCreated visualization index: {index_path}")
    
    print("\n" + "=" * 70)
    print("✓ Export complete!")
    print("=" * 70)
    print(f"\n🌐 Open in browser: file://{index_path.absolute()}")
    print(f"\nNote: This uses placeholder data. For accurate visualization,")
    print(f"modify train_baseline_krk.py to save learner state with pickle.")


if __name__ == "__main__":
    recreate_and_export()
