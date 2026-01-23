"""
Convert baseline topology from dict format to list format for TopologyRegistry.
"""

import json
from pathlib import Path

# Load the dict-format topology
input_path = Path("topologies/krk_entry_topology.json")
output_path = Path("topologies/krk_entry_topology_fixed.json")

with open(input_path) as f:
    data = json.load(f)

# Convert nodes from dict to list
nodes_list = []
if isinstance(data["nodes"], dict):
    for node_id, node_data in data["nodes"].items():
        # Add group field if missing
        if "group" not in node_data:
            if "root" in node_id or "entry" in node_id:
                node_data["group"] = "root"
            elif "hub" in node_id:
                node_data["group"] = "hub"
            elif "leg" in node_id:
                node_data["group"] = "leg"
            elif "precond" in node_id or "postcond" in node_id:
                node_data["group"] = "gate"
            elif "sensor" in node_id:
                node_data["group"] = "sensor"
            elif "actuator" in node_id:
                node_data["group"] = "actuator"
            else:
                node_data["group"] = "generic"
        
        nodes_list.append(node_data)

# Convert edges: source/target → src/dst
edges_list = []
for edge in data["edges"]:
    edges_list.append({
        "src": edge.get("source", edge.get("src")),
        "dst": edge.get("target", edge.get("dst")),
        "type": edge["type"],
        "weight": edge.get("weight", 1.0),
        "consolidate": edge.get("consolidate", True),
    })

# Create new topology
new_topology = {
    "version": "1.0",
    "network": "krk_entry",
    "created": "2026-01-23T00:00:00",
    "nodes": nodes_list,
    "edges": edges_list,
    "stem_cells": [],
    "promoted_nodes": [],
    "evolution_history": [],
}

# Save
with open(output_path, 'w') as f:
    json.dump(new_topology, f, indent=2)

print(f"✓ Converted topology:")
print(f"  Input:  {input_path}")
print(f"  Output: {output_path}")
print(f"  Nodes:  {len(nodes_list)}")
print(f"  Edges:  {len(edges_list)}")
