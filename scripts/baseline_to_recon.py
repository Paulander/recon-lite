"""
Baseline to ReCoN Graph Compiler

Converts learned baseline patterns (sensors/actuators) into proper ReCoN graph
structure following Article.md constraints.

Output: krk_entry_topology.json with:
- Root → Hub → Legs hierarchy
- 3-part micro-scripts (precond → actuator → postcond)
- Stable IDs (not indices)
- Blackboard caching
- Parallel Legs (SUB, not POR)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from recon_lite.learning.baseline import BaselineLearner, SensorSpec, ActuatorSpec


def compile_baseline_to_topology(
    learner_path: Path,
    output_path: Path
) -> Dict:
    """
    Main compilation function.
    
    Args:
        learner_path: Path to pickled BaselineLearner
        output_path: Path to save topology JSON
    
    Returns:
        Topology dictionary
    """
    # Load learner
    with open(learner_path, 'rb') as f:
        learner = pickle.load(f)
    
    print(f"Loaded learner: {len(learner.sensors)} sensors, {len(learner.actuators)} actuators")
    
    # Filter mature sensors
    mature_sensors = [s for s in learner.sensors if s.is_mature]
    print(f"Mature sensors: {len(mature_sensors)}")
    
    # Build topology
    topology = {
        "nodes": {},
        "edges": [],
        "meta": {
            "origin": "baseline_compilation",
            "mature_sensors": len(mature_sensors),
            "total_actuators": len(learner.actuators),
            "baseline_xp_avg": float(np.mean([s.xp for s in mature_sensors]))
        }
    }
    
    # Create Root
    create_root_node(topology)
    
    # Create Hub
    create_hub_node(topology)
    
    # Create Legs (one per actuator)
    for actuator in learner.actuators:
        create_leg_micro_script(topology, actuator, mature_sensors)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"\n✓ Saved topology: {output_path}")
    print(f"  Nodes: {len(topology['nodes'])}")
    print(f"  Edges: {len(topology['edges'])}")
    
    return topology


def create_root_node(topology: Dict):
    """Create KRK_entry root node with blackboard cache"""
    topology["nodes"]["krk_entry"] = {
        "id": "krk_entry",
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_krk_entry_root",
        "meta": {
            "blackboard": {},  # Will cache features + sensor outputs
            "description": "KRK entry point with feature extraction"
        }
    }
    
    print("Created root: krk_entry")


def create_hub_node(topology: Dict):
    """Create Hub with bandit selection"""
    topology["nodes"]["krk_hub"] = {
        "id": "krk_hub",
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_krk_hub",
        "meta": {
            "bandit_enabled": True,
            "description": "Bandit selector for Leg alternatives"
        }
    }
    
    # Edge: Root → Hub
    topology["edges"].append({
        "source": "krk_entry",
        "target": "krk_hub",
        "type": "SUB",
        "weight": 1.0
    })
    
    print("Created hub: krk_hub")


def create_leg_micro_script(
    topology: Dict,
    actuator: Any,
    sensors: List[Any]
):
    """
    Create 3-part micro-script for one actuator pattern.
    
    Structure:
    Leg (SCRIPT)
    ├─ SUB → Precondition (SCRIPT, and-gate)
    │   ├─ SUB → sensor_X (TERMINAL)
    │   └─ SUB → sensor_Y (TERMINAL)
    ├─ SUB → Actuator (TERMINAL)
    │   └─ POR from Precondition
    └─ SUB → Postcondition (SCRIPT, and-gate)
        ├─ SUB → sensor_X_post (TERMINAL)
        └─ SUB → sensor_Y_post (TERMINAL)
        └─ POR from Actuator
    """
    leg_id = f"leg_{actuator.id}"
    precond_id = f"precond_{actuator.id}"
    actuator_id = f"actuator_{actuator.id}"
    postcond_id = f"postcond_{actuator.id}"
    
    # Leg SCRIPT
    topology["nodes"][leg_id] = {
        "id": leg_id,
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_leg_script",
        "meta": {
            "actuator_id": actuator.id,
            "description": f"Leg for actuator pattern {actuator.id}"
        }
    }
    
    # Edge: Hub → Leg (parallel alternative)
    topology["edges"].append({
        "source": "krk_hub",
        "target": leg_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    # Part 1: Precondition gate
    topology["nodes"][precond_id] = {
        "id": precond_id,
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_and_gate",
        "meta": {
            "aggregation": "and",
            "description": "Precondition sensors must all fire"
        }
    }
    
    topology["edges"].append({
        "source": leg_id,
        "target": precond_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    # Add precondition sensors
    sensor_map = {s.id: s for s in sensors}
    for sensor_idx in actuator.actuator_spec.sensor_indices:
        if sensor_idx in sensor_map:
            sensor = sensor_map[sensor_idx]
            sensor_id = f"sensor_{sensor.id}"
            
            create_sensor_terminal(topology, sensor_id, sensor)
            
            topology["edges"].append({
                "source": precond_id,
                "target": sensor_id,
                "type": "SUB",
                "weight": 1.0
            })
    
    # Part 2: Actuator terminal
    create_actuator_terminal(topology, actuator_id, actuator, sensors)
    
    topology["edges"].append({
        "source": leg_id,
        "target": actuator_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    topology["edges"].append({
        "source": precond_id,
        "target": actuator_id,
        "type": "POR",  # Sequence: precond → actuator
        "weight": 1.0
    })
    
    topology["edges"].append({
        "source": actuator_id,
        "target": precond_id,
        "type": "RET",
        "weight": 1.0
    })
    
    # Part 3: Postcondition gate (verify Δs)
    topology["nodes"][postcond_id] = {
        "id": postcond_id,
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_and_gate",
        "meta": {
            "aggregation": "and",
            "description": "Postcondition verification"
        }
    }
    
    topology["edges"].append({
        "source": leg_id,
        "target": postcond_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    topology["edges"].append({
        "source": actuator_id,
        "target": postcond_id,
        "type": "POR",  # Sequence: actuator → postcond
        "weight": 1.0
    })
    
    topology["edges"].append({
        "source": postcond_id,
        "target": actuator_id,
        "type": "RET",
        "weight": 1.0
    })
    
    # Add postcondition sensors (same as precondition, different instances)
    for sensor_idx in actuator.actuator_spec.sensor_indices:
        if sensor_idx in sensor_map:
            sensor = sensor_map[sensor_idx]
            sensor_post_id = f"sensor_{sensor.id}_post_{actuator.id}"
            
            create_sensor_terminal(topology, sensor_post_id, sensor)
            
            topology["edges"].append({
                "source": postcond_id,
                "target": sensor_post_id,
                "type": "SUB",
                "weight": 1.0
            })
    
    print(f"Created leg: {leg_id} with {len(actuator.actuator_spec.sensor_indices)} sensors")


def create_sensor_terminal(topology: Dict, sensor_id: str, sensor: Any):
    """Create TERMINAL node for sensor with stable IDs"""
    if sensor_id in topology["nodes"]:
        return  # Already created
    
    # Get feature keys (stable, not indices)
    feature_mask_keys = get_feature_keys_from_mask(sensor.sensor_spec.feature_mask)
    
    topology["nodes"][sensor_id] = {
        "id": sensor_id,
        "type": "TERMINAL",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_sensor_terminal",
        "meta": {
            "origin": "baseline",
            "stage": sensor.stage,
            "baseline_xp": float(sensor.xp),
            "readout_type": sensor.sensor_spec.readout_type,
            "feature_mask_keys": feature_mask_keys,
            "readout_params": sensor.sensor_spec.readout_params,
            "is_mature": sensor.is_mature,
            "activations": sensor.activations,
            "cycles_alive": sensor.cycles_alive
        }
    }


def create_actuator_terminal(
    topology: Dict,
    actuator_id: str,
    actuator: Any,
    sensors: List[Any]
):
    """Create TERMINAL node for actuator with stable target IDs"""
    sensor_map = {s.id: s for s in sensors}
    
    # Build targets list (stable IDs)
    targets = []
    goal_delta = {}
    
    for idx, delta_val in zip(
        actuator.actuator_spec.sensor_indices,
        actuator.actuator_spec.goal_delta
    ):
        if idx in sensor_map:
            sensor_id = f"sensor_{idx}"
            targets.append(sensor_id)
            goal_delta[sensor_id] = float(delta_val)
    
    topology["nodes"][actuator_id] = {
        "id": actuator_id,
        "type": "TERMINAL",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_actuator_terminal",
        "meta": {
            "origin": "baseline",
            "stage": actuator.stage,
            "baseline_xp": float(actuator.xp),
            "targets": targets,  # Stable IDs
            "goal_delta": goal_delta,  # Keyed by stable IDs
            "match_mode": actuator.actuator_spec.match_mode,
            "activations": actuator.activations,
            "cycles_alive": actuator.cycles_alive
        }
    }


def get_feature_keys_from_mask(feature_mask: np.ndarray) -> List[str]:
    """
    Convert boolean feature mask to stable feature keys.
    
    For now, use indices as keys. In future, map to named features.
    """
    indices = np.where(feature_mask)[0]
    return [f"feature_{i}" for i in indices]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile baseline to ReCoN topology")
    parser.add_argument("--learner", type=Path, required=True,
                       help="Path to pickled learner (e.g., final_learner.pkl)")
    parser.add_argument("--output", type=Path, default=Path("topologies/krk_entry_topology.json"),
                       help="Output topology JSON path")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Baseline → ReCoN Graph Compiler")
    print("=" * 70)
    
    topology = compile_baseline_to_topology(args.learner, args.output)
    
    print("\n" + "=" * 70)
    print("✓ Compilation complete!")
    print("=" * 70)
