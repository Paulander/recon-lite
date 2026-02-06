"""Quick script to save learner pickle from last training run"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.learning.baseline import BaselineLearner, Terminal, TerminalRole, SensorSpec, ActuatorSpec
import numpy as np

# We need to recreate the learner from the training run
# Since we don't have it saved, let's just re-run the last part of training
# Or use the quick export approach

print("Creating learner from training...")
print("Note: Re-running training is recommended for accurate data")
print("For now, using placeholder approach")

# Import and run quick export
import subprocess
result = subprocess.run(
    ["wsl", "bash", "-c", "cd /home/paulander/git/recon-lite && .venv/bin/python scripts/quick_export_viz.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("Error:", result.stderr)
