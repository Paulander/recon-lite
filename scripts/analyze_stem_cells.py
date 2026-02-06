#!/usr/bin/env python3
"""Quick analysis of stem cells from extended run."""
import json
from pathlib import Path
from collections import Counter

# Load final stem cells
path = Path("snapshots/evolution/krk_extended/stem_cells_stage10.json")
if not path.exists():
    print(f"File not found: {path}")
    exit(1)

data = json.loads(path.read_text())

cells = data.get("cells", {})
print(f"=== STEM CELL ANALYSIS ===")
print(f"Total cells: {len(cells)}")
print(f"Next ID: {data.get('next_id', 0)}")

# Count states
states = Counter(c.get("state", "UNKNOWN") for c in cells.values())
print(f"\nBy State:")
for state, count in sorted(states.items()):
    print(f"  {state}: {count}")

# Count samples
total_samples = sum(len(c.get("samples", [])) for c in cells.values())
print(f"\nTotal samples collected: {total_samples}")

# Count cells with trial_node_id (promoted to TRIAL at some point)
trial_promoted = [c for c in cells.values() if c.get("trial_node_id")]
print(f"Cells with trial_node_id: {len(trial_promoted)}")

# Check for pack-spawned cells
pack_cells = [c for c in cells.values() if c.get("metadata", {}).get("spawned_pack")]
print(f"Pack-spawned cells: {len(pack_cells)}")

# Check for exploration children
exploration_cells = [c for c in cells.values() if c.get("metadata", {}).get("spawn_reason") == "failure_exploration"]
print(f"Failure-exploration spawned: {len(exploration_cells)}")

# XP distribution for TRIAL cells
trial_cells = [c for c in cells.values() if c.get("state") == "TRIAL"]
if trial_cells:
    xps = [c.get("xp", 0) for c in trial_cells]
    print(f"\nTRIAL cell XP: min={min(xps)}, max={max(xps)}, avg={sum(xps)/len(xps):.1f}")
    high_xp = [c for c in trial_cells if c.get("xp", 0) >= 100]
    print(f"High XP cells (>=100): {len(high_xp)}")

# Check win_coactivation for potential AND-gates
coactivations = data.get("win_coactivation", {})
if coactivations:
    top_pairs = sorted(coactivations.items(), key=lambda x: -x[1])[:5]
    print(f"\nTop win-coactivation pairs:")
    for pair, count in top_pairs:
        print(f"  {pair}: {count}")

print("\n=== END ANALYSIS ===")
