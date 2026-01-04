#!/usr/bin/env python3
"""Clean up stale TRIAL nodes from topology.json."""

import json
from pathlib import Path

topo_path = Path("topologies/kpk_topology.json")
with open(topo_path) as f:
    data = json.load(f)

# Remove TRIAL nodes
original_nodes = len(data["nodes"])
data["nodes"] = [n for n in data["nodes"] if not n["id"].startswith("TRIAL_")]
removed_nodes = original_nodes - len(data["nodes"])
print(f"Removed {removed_nodes} TRIAL nodes")

# Remove edges to TRIAL nodes
original_edges = len(data["edges"])
data["edges"] = [e for e in data["edges"] if not e["dst"].startswith("TRIAL_") and not e["src"].startswith("TRIAL_")]
removed_edges = original_edges - len(data["edges"])
print(f"Removed {removed_edges} edges to TRIAL nodes")

# Clear evolution history
data["evolution_history"] = []

# Save
with open(topo_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Cleaned topology: {len(data['nodes'])} nodes, {len(data['edges'])} edges")
