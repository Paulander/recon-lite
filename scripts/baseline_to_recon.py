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
    goal_bank = build_goal_bank(learner, label="mate_in_1")
    topology = {
        "nodes": {},
        "edges": [],
        "meta": {
            "origin": "baseline_compilation",
            "mature_sensors": len(mature_sensors),
            "total_actuators": len(learner.actuators),
            "baseline_xp_avg": float(np.mean([s.xp for s in mature_sensors])),
            "goal_bank": goal_bank,
            "goal_label": "mate_in_1",
            "goal_normalize": bool(getattr(learner, "normalize_goals", True)),
            "goal_weight": 0.7,
            "goal_lookahead": "max",
            "goal_min_overlap": 8,
            "goal_handoff_threshold": 0.2,
        }
    }
    
    # Create Root
    create_root_node(topology, goal_bank)
    
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


def create_root_node(topology: Dict, goal_bank: Dict | None = None):
    """Create KRK_entry root node with blackboard cache"""
    topology["nodes"]["krk_entry"] = {
        "id": "krk_entry",
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_krk_entry_root",
        "meta": {
            "blackboard": {},  # Will cache features + sensor outputs
            "goal_bank": topology.get("meta", {}).get("goal_bank"),
            "goal_label": topology.get("meta", {}).get("goal_label", "mate_in_1"),
            "goal_normalize": topology.get("meta", {}).get("goal_normalize", True),
            "goal_weight": topology.get("meta", {}).get("goal_weight", 0.7),
            "goal_lookahead": topology.get("meta", {}).get("goal_lookahead", "max"),
            "goal_min_overlap": topology.get("meta", {}).get("goal_min_overlap", 8),
            "goal_handoff_threshold": topology.get("meta", {}).get("goal_handoff_threshold", 0.2),
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
        "src": "krk_entry",
        "dst": "krk_hub",
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
    act_script_id = f"act_script_{actuator.id}"
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
        "src": "krk_hub",
        "dst": leg_id,
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
        "src": leg_id,
        "dst": precond_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    # Add precondition sensors
    sensor_map = {s.id: s for s in sensors}
    for sensor_idx in actuator.actuator_spec.sensor_indices:
        sensor = None
        # Actuator spec indices are relative to the mature sensor list
        if 0 <= sensor_idx < len(sensors):
            sensor = sensors[sensor_idx]
        elif sensor_idx in sensor_map:
            # Fallback: treat as absolute sensor id
            sensor = sensor_map[sensor_idx]
        if sensor is not None:
            sensor_id = f"sensor_{sensor.id}"
            
            create_sensor_terminal(topology, sensor_id, sensor)
            
            topology["edges"].append({
                "src": precond_id,
                "dst": sensor_id,
                "type": "SUB",
                "weight": 1.0
            })
    
    # Part 2: Actuator script wrapper (SCRIPT)
    topology["nodes"][act_script_id] = {
        "id": act_script_id,
        "type": "SCRIPT",
        "factory": "recon_lite_chess.krk_baseline_nodes:create_act_script",
        "meta": {
            "description": "Actuator wrapper (SCRIPT)"
        }
    }
    
    topology["edges"].append({
        "src": leg_id,
        "dst": act_script_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    # Actuator terminal (SUB under actuator script)
    create_actuator_terminal(topology, actuator_id, actuator, sensors)
    
    topology["edges"].append({
        "src": act_script_id,
        "dst": actuator_id,
        "type": "SUB",
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
        "src": leg_id,
        "dst": postcond_id,
        "type": "SUB",
        "weight": 1.0
    })
    
    # POR sequencing between scripts only
    topology["edges"].append({
        "src": precond_id,
        "dst": act_script_id,
        "type": "POR",
        "weight": 1.0
    })
    
    topology["edges"].append({
        "src": act_script_id,
        "dst": postcond_id,
        "type": "POR",
        "weight": 1.0
    })
    
    # Add postcondition sensors (same as precondition, different instances)
    for sensor_idx in actuator.actuator_spec.sensor_indices:
        sensor = None
        if 0 <= sensor_idx < len(sensors):
            sensor = sensors[sensor_idx]
        elif sensor_idx in sensor_map:
            sensor = sensor_map[sensor_idx]
        if sensor is not None:
            sensor_post_id = f"sensor_{sensor.id}_post_{actuator.id}"
            
            create_sensor_terminal(topology, sensor_post_id, sensor)
            
            topology["edges"].append({
                "src": postcond_id,
                "dst": sensor_post_id,
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
        sensor = None
        if 0 <= idx < len(sensors):
            sensor = sensors[idx]
        elif idx in sensor_map:
            sensor = sensor_map[idx]
        if sensor is not None:
            sensor_id = f"sensor_{sensor.id}"
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


def build_goal_bank(learner: BaselineLearner, label: str = "mate_in_1") -> Dict[str, Any] | None:
    """
    Build a compact goal bank for runtime scoring.

    Returns a dict with:
      - label
      - goals: list of {values: {sensor_id: value}, count}
      - sensor_specs: map of sensor_id -> spec
      - goal_eps: merge threshold used in training
    """
    goals = [g for g in learner.goal_memories if g.label == label]
    if not goals:
        return None

    # Prefer goals with explicit sensor_ids
    sensor_ids = None
    for g in goals:
        if getattr(g, "sensor_ids", None):
            sensor_ids = g.sensor_ids
            break

    if sensor_ids is None:
        # Best-effort fallback: cannot safely align goal vectors without sensor ids
        print("⚠️  Warning: goal prototypes missing sensor_ids; skipping goal bank export.")
        return None

    sensor_map = {s.id: s for s in learner.sensors}
    sensor_specs: Dict[str, Any] = {}
    for sid in sensor_ids:
        sensor = sensor_map.get(sid)
        if sensor is None:
            continue
        sensor_specs[f"sensor_{sid}"] = {
            "readout_type": sensor.sensor_spec.readout_type,
            "feature_mask_keys": get_feature_keys_from_mask(sensor.sensor_spec.feature_mask),
            "readout_params": sensor.sensor_spec.readout_params,
        }

    goals_payload = []
    for g in goals:
        if getattr(g, "sensor_ids", None) != sensor_ids:
            continue
        values = {f"sensor_{sid}": float(val) for sid, val in zip(sensor_ids, g.s0.tolist())}
        goals_payload.append({
            "values": values,
            "count": int(getattr(g, "count", 1)),
        })

    if not goals_payload:
        return None

    return {
        "label": label,
        "sensor_ids": list(sensor_ids),
        "goals": goals_payload,
        "sensor_specs": sensor_specs,
        "goal_eps": float(getattr(learner, "goal_eps", 0.15)),
    }


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
