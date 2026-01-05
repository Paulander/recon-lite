#!/usr/bin/env python3
"""Inspect transferred cells to understand why reuse ratio is 0%."""

import json
from pathlib import Path

STEM_CELLS_PATH = Path("snapshots/krk_bridge_experiment/20260105_163449/experienced_hector/snapshots/stem_cells.json")

def main():
    with open(STEM_CELLS_PATH) as f:
        data = json.load(f)
    
    print("=" * 60)
    print("TRANSFERRED CELLS INSPECTION")
    print("=" * 60)
    
    # Handle both dict and list formats
    cells_data = data.get("cells", {})
    if isinstance(cells_data, dict):
        cells = list(cells_data.values())
    else:
        cells = cells_data
    
    transferred = [c for c in cells if isinstance(c, dict) and c.get("metadata", {}).get("origin") == "kpk_transfer"]
    
    print(f"\nTotal cells: {len(cells)}")
    print(f"Transferred cells: {len(transferred)}")
    
    print("\n" + "=" * 60)
    print("TRANSFERRED CELL DETAILS")
    print("=" * 60)
    
    for cell in transferred[:5]:
        print(f"\nCell: {cell['cell_id']}")
        print(f"  State: {cell['state']}")
        print(f"  XP: {cell.get('xp', 0)}")
        print(f"  Trial Node: {cell.get('trial_node_id', 'None')}")
        centroid = cell.get('pattern_centroid')
        if centroid:
            if isinstance(centroid, list):
                print(f"  Pattern Centroid: {centroid[:5]}... (len={len(centroid)})")
            else:
                print(f"  Pattern Centroid: {str(centroid)[:100]}")
        else:
            print(f"  Pattern Centroid: None")
        
        samples = cell.get("samples", [])
        print(f"  Samples: {len(samples)}")
        
        if samples:
            s = samples[0]
            features = s.get("features", {})
            print(f"  Sample 0 features type: {type(features)}")
            if isinstance(features, dict):
                print(f"  Feature keys: {list(features.keys())[:10]}")
            elif isinstance(features, list):
                print(f"  Feature vector length: {len(features)}")
    
    print("\n" + "=" * 60)
    print("LOCAL (NEW) CELLS")
    print("=" * 60)
    
    local = [c for c in cells if isinstance(c, dict) and c.get("metadata", {}).get("origin") != "kpk_transfer"][:3]
    for cell in local:
        print(f"\nCell: {cell['cell_id']}")
        print(f"  State: {cell['state']}")
        samples = cell.get("samples", [])
        print(f"  Samples: {len(samples)}")
        if samples:
            s = samples[0]
            features = s.get("features", {})
            if isinstance(features, dict):
                print(f"  Feature keys: {list(features.keys())[:10]}")
            elif isinstance(features, list):
                print(f"  Feature vector length: {len(features)}")

if __name__ == "__main__":
    main()

