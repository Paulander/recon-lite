#!/usr/bin/env python3
"""Quick test for bridge sweep imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports...")
from scripts.run_krk_bridge_sweep import create_krk_bridge_sweep
print("✅ Imports OK")

configs = create_krk_bridge_sweep()
print(f"Configs: {[c.trial_name for c in configs]}")

# Check if transfer source exists
for c in configs:
    if c.transfer_from:
        if c.transfer_from.exists():
            print(f"✅ Transfer source exists: {c.transfer_from}")
        else:
            print(f"❌ Transfer source missing: {c.transfer_from}")

