"""
Fix topology by removing illegal POR/RET edges to TERMINAL nodes.

According to Article.md:
- TERMINAL nodes can only receive SUB and send SUR
- POR/RET require SCRIPT nodes

We'll remove POR edges involving TERMINAL nodes to make the topology valid.
The sequencing will be handled implicitly by the Leg SCRIPT execution order.
"""

import json
from pathlib import Path

# Load topology
input_path = Path("topologies/krk_entry_topology.json")
output_path = Path("topologies/krk_entry_topology.json")

with open(input_path) as f:
    data = json.load(f)

# Get node types
nodes = {n["id"]: n["type"] for n in data["nodes"]}

# Filter out POR/RET edges that involve TERMINAL nodes
original_edge_count = len(data["edges"])
valid_edges = []
removed_edges = []

for edge in data["edges"]:
    src_type = nodes.get(edge["src"], "?")
    dst_type = nodes.get(edge["dst"], "?")
    
    if edge["type"] in ["POR", "RET"]:
        # POR/RET can only be between SCRIPT nodes
        if src_type == "TERMINAL" or dst_type == "TERMINAL":
            removed_edges.append(edge)
            continue
    
    valid_edges.append(edge)

data["edges"] = valid_edges

# Save
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Fixed topology:")
print(f"  Original edges: {original_edge_count}")
print(f"  Valid edges:    {len(valid_edges)}")
print(f"  Removed edges:  {len(removed_edges)}")
print()
print("Removed edges:")
for e in removed_edges:
    print(f"  {e['type']}: {e['src']} -> {e['dst']}")
