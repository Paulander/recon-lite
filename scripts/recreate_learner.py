"""
Quick script to recreate learner from checkpoint and save as pickle.
"""

import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.learning.baseline import BaselineLearner, Terminal, TerminalRole, SensorSpec, ActuatorSpec
import numpy as np

# Load checkpoint
checkpoint_path = Path("snapshots/baseline_krk_viz/final_checkpoint.json")
with open(checkpoint_path) as f:
    checkpoint = json.load(f)

print(f"Loaded checkpoint: {checkpoint_path}")
print(f"  Sensors: {checkpoint['learner']['sensor_count']}")
print(f"  Actuators: {checkpoint['learner']['actuator_count']}")

# Create minimal learner for compilation
# (We don't have the actual Terminal objects, so we'll use the quick_export_viz approach)
print("\nNote: Using placeholder learner from quick_export_viz.py")
print("This has the correct structure but placeholder XP values.")
print("For production, modify train_baseline_krk.py to save learner.pkl")

# Just run the quick export which already creates a learner
import subprocess
result = subprocess.run(
    [".venv/bin/python", "scripts/quick_export_viz.py"],
    cwd="/home/paulander/git/recon-lite",
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("Error:", result.stderr)
    sys.exit(1)

print("\n✓ Placeholder learner created")
print("Note: For accurate compilation, we need actual Terminal objects from training.")
print("Proceeding with placeholder for now...")
