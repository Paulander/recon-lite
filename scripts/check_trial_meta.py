#!/usr/bin/env python3
"""Check if TRIAL nodes have subgraph metadata."""
import json
from pathlib import Path

topo_path = Path("snapshots/evolution/trial_activation_test/snapshots/cycle_0003.json")
if topo_path.exists():
    with open(topo_path) as f:
        d = json.load(f)
    
    nodes = d.get("nodes", {})
    for nid, node in nodes.items():
        if nid.startswith("TRIAL"):
            meta = node.get("meta", {})
            subgraph = meta.get("subgraph", "NOT SET")
            print(f"{nid}: subgraph={subgraph}")
else:
    print(f"File not found: {topo_path}")

