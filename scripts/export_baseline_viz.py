"""
Export baseline training results to visualization format.

Run this after training to create topology snapshots compatible with
existing ReCoN visualizers.
"""

import argparse
import pickle
from pathlib import Path

from recon_lite.learning.baseline_viz import (
    export_baseline_to_topology,
    create_baseline_visualization_index
)


def main():
    parser = argparse.ArgumentParser(description="Export baseline results for visualization")
    parser.add_argument("--checkpoint", type=Path, required=True,
                       help="Path to final checkpoint (e.g., snapshots/baseline_krk/final_checkpoint.json)")
    parser.add_argument("--learner-pkl", type=Path, required=True,
                       help="Path to pickled learner state (must save this during training)")
    parser.add_argument("--output-dir", type=Path, default=Path("snapshots/baseline_evolution"),
                       help="Output directory for visualization files")
    parser.add_argument("--stage", type=str, default="baseline",
                       help="Stage name for visualization")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Baseline Visualization Export")
    print("=" * 70)
    
    # Load learner state
    print(f"\nLoading learner from: {args.learner_pkl}")
    with open(args.learner_pkl, 'rb') as f:
        learner = pickle.load(f)
    
    print(f"  Sensors: {len(learner.sensors)} ({len(learner.get_mature_sensors())} mature)")
    print(f"  Actuators: {len(learner.actuators)}")
    print(f"  Goal memories: {len(learner.goal_memories)}")
    
    # Export topology
    stage_dir = args.output_dir / args.stage
    output_path = stage_dir / "cycle_0050.json"
    
    print(f"\nExporting topology to: {output_path}")
    topology = export_baseline_to_topology(learner, 50, output_path)
    
    print(f"  Nodes: {len(topology['nodes'])}")
    print(f"  Edges: {len(topology['edges'])}")
    
    # Create index
    index_path = create_baseline_visualization_index(args.output_dir, args.stage)
    print(f"\nCreated visualization index: {index_path}")
    
    print("\n" + "=" * 70)
    print("✓ Export complete!")
    print("=" * 70)
    print(f"\nOpen in browser: file://{index_path.absolute()}")


if __name__ == "__main__":
    main()
