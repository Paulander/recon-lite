"""
Visualization adapter for baseline architecture.

Converts baseline learner state (sensors, actuators, goal memories) into
graph topology format compatible with existing ReCoN visualizers.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from recon_lite.learning.baseline import BaselineLearner, Terminal, TerminalRole


def export_baseline_to_topology(learner: BaselineLearner, 
                                 cycle: int,
                                 output_path: Path) -> Dict:
    """
    Export baseline learner state as graph topology JSON.
    
    Creates a virtual graph where:
    - Sensors are TERMINAL nodes (group="sensor")
    - Actuators are TERMINAL nodes (group="actuator")
    - Goal memories are represented as metadata
    - Edges show sensor->actuator connections
    
    Args:
        learner: BaselineLearner instance
        cycle: Training cycle number
        output_path: Path to save JSON file
    
    Returns:
        Topology dictionary
    """
    nodes = {}
    edges = []
    
    # Add sensor nodes
    for sensor in learner.sensors:
        nodes[f"sensor_{sensor.id}"] = {
            "id": f"sensor_{sensor.id}",
            "type": "TERMINAL",
            "group": "sensor",
            "factory": None,
            "meta": {
                "sensor_id": int(sensor.id),
                "is_mature": bool(sensor.is_mature),
                "xp": float(sensor.xp),
                "cycles_alive": int(sensor.cycles_alive),
                "activations": int(sensor.activations),
                "readout_type": str(sensor.sensor_spec.readout_type),
                "feature_count": int(np.sum(sensor.sensor_spec.feature_mask)),
            }
        }
    
    # Add actuator nodes
    for actuator in learner.actuators:
        nodes[f"actuator_{actuator.id}"] = {
            "id": f"actuator_{actuator.id}",
            "type": "TERMINAL",
            "group": "actuator",
            "factory": None,
            "meta": {
                "actuator_id": int(actuator.id),
                "xp": float(actuator.xp),
                "sensor_count": int(len(actuator.actuator_spec.sensor_indices)),
                "sensor_indices": [int(i) for i in actuator.actuator_spec.sensor_indices],
                "goal_delta_magnitude": float(np.linalg.norm(actuator.actuator_spec.goal_delta)),
            }
        }
        
        # Add edges from sensors to actuators
        for sensor_id in actuator.actuator_spec.sensor_indices:
            edges.append({
                "source": f"sensor_{int(sensor_id)}",
                "target": f"actuator_{actuator.id}",
                "type": "SUB",  # Actuator depends on sensor
                "weight": 1.0
            })
    
    # Create topology
    topology = {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "cycle": cycle,
            "total_sensors": len(learner.sensors),
            "mature_sensors": len(learner.get_mature_sensors()),
            "total_actuators": len(learner.actuators),
            "goal_memories": len(learner.goal_memories),
            "architecture": "baseline",
        }
    }
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(topology, f, indent=2)
    
    return topology


def export_baseline_evolution(learner: BaselineLearner,
                              stats: Dict,
                              output_dir: Path,
                              stage: str = "baseline") -> List[Path]:
    """
    Export baseline training evolution as series of topology snapshots.
    
    Creates one JSON file per checkpoint cycle, compatible with
    topology_timelapse.html visualizer.
    
    Args:
        learner: Final BaselineLearner state
        stats: Training statistics dict
        output_dir: Directory to save snapshots
        stage: Stage name (default "baseline")
    
    Returns:
        List of created file paths
    """
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # We only have final state, so create a single snapshot
    # In future, modify training loop to save learner state at each checkpoint
    cycle = stats["cycles"][-1] if stats["cycles"] else 0
    
    output_path = stage_dir / f"cycle_{cycle:04d}.json"
    export_baseline_to_topology(learner, cycle, output_path)
    created_files.append(output_path)
    
    print(f"Exported baseline topology to: {output_path}")
    
    return created_files


def create_baseline_visualization_index(output_dir: Path,
                                        stage: str = "baseline") -> Path:
    """
    Create an index.html that points to the topology visualizer.
    
    Args:
        output_dir: Directory containing snapshots
        stage: Stage name
    
    Returns:
        Path to created index.html
    """
    index_path = output_dir / "index.html"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Baseline Architecture Visualization</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
        .link-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .link-card {{
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            text-decoration: none;
            color: #333;
            transition: all 0.2s;
        }}
        .link-card:hover {{
            border-color: #007bff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,123,255,0.15);
        }}
        .link-card h3 {{
            margin: 0 0 10px 0;
            color: #007bff;
        }}
        .link-card p {{
            margin: 0;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🧠 Baseline Architecture Visualization</h1>
        <p>Explore the learned sensor-actuator topology from KRK mate-in-1 training.</p>
        
        <div class="link-grid">
            <a href="../../demos/visualization/topology_timelapse.html?snapshot_dir=../../snapshots/baseline_evolution/{stage}" class="link-card">
                <h3>📊 Topology Viewer</h3>
                <p>Interactive graph visualization of sensors and actuators</p>
            </a>
            
            <a href="../../demos/visualization/consolidation_dashboard.html?snapshot_dir=../../snapshots/baseline_evolution/{stage}" class="link-card">
                <h3>📈 Training Dashboard</h3>
                <p>View sensor XP, actuator patterns, and learning dynamics</p>
            </a>
            
            <a href="{stage}/cycle_0050.json" class="link-card">
                <h3>📄 Raw JSON Data</h3>
                <p>View the exported topology data structure</p>
            </a>
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e9ecef;">
            <h3>Architecture Summary</h3>
            <ul>
                <li><strong>Sensors:</strong> Terminal nodes that read from feature space</li>
                <li><strong>Actuators:</strong> Terminal nodes encoding goal patterns in sensor space</li>
                <li><strong>Edges:</strong> Show which sensors each actuator depends on</li>
                <li><strong>XP:</strong> Experience metric (stability + separation)</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    return index_path


if __name__ == "__main__":
    # Example usage
    print("Baseline visualization adapter")
    print("Import this module and use export_baseline_to_topology() or export_baseline_evolution()")
