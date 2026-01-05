#!/usr/bin/env python3
"""Analyze KRK transfer experiment results."""
import json
import sys
from pathlib import Path

def analyze_stem_cells(path: Path):
    with open(path) as f:
        data = json.load(f)
    
    cells = data.get('cells', {})
    print(f"\nTotal cells: {len(cells)}")
    
    # Count transferred cells
    transferred = [
        (cid, c) for cid, c in cells.items()
        if c.get('metadata', {}).get('origin') == 'kpk_transfer'
    ]
    print(f"Transferred (kpk_transfer origin): {len(transferred)}")
    
    # Check states and sample counts for transferred
    print("\nTransferred Cell Details:")
    for cid, cell in transferred[:10]:
        state = cell.get('state', 'UNKNOWN')
        samples = len(cell.get('samples', []))
        xp = cell.get('xp', 0)
        trial_node = cell.get('trial_node_id', 'None')
        print(f"  {cid}: state={state}, samples={samples}, xp={xp}, trial_node={trial_node}")
    
    # Count by state
    state_counts = {}
    for c in cells.values():
        s = c.get('state', 'UNKNOWN')
        state_counts[s] = state_counts.get(s, 0) + 1
    print(f"\nBy State: {state_counts}")


if __name__ == "__main__":
    base = Path("snapshots/krk_bridge_experiment")
    latest = sorted(base.iterdir())[-1] if base.exists() else None
    
    if latest:
        print(f"Analyzing: {latest}")
        
        for trial in ["blank_slate", "experienced_hector"]:
            stem_path = latest / trial / "snapshots" / "stem_cells.json"
            if stem_path.exists():
                print(f"\n{'='*60}")
                print(f"TRIAL: {trial}")
                print(f"{'='*60}")
                analyze_stem_cells(stem_path)
    else:
        print("No experiment found")

