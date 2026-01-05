#!/usr/bin/env python3
"""Inspect topology to understand depth issue."""

import json
from pathlib import Path

topo_path = Path("snapshots/evolution/clean_structural_spurt/stage5/snapshots/cycle_0015.json")
with open(topo_path) as f:
    topo = json.load(f)

print("=" * 60)
print("TOPOLOGY INSPECTION - Stage 5 Final")
print("=" * 60)

# Handle both dict and list formats for nodes/edges
nodes_data = topo.get("nodes", {})
if isinstance(nodes_data, dict):
    nodes = list(nodes_data.values())
else:
    nodes = nodes_data

edges_data = topo.get("edges", [])
if isinstance(edges_data, dict):
    edges = list(edges_data.values())
else:
    edges = edges_data

print(f"\nTotal NODES: {len(nodes)}")
print(f"Total EDGES: {len(edges)}")

print("\n--- SCRIPT Nodes ---")
for node in nodes:
    if isinstance(node, dict):
        ntype = node.get("type", "UNKNOWN")
        nid = node.get("id", "N/A")
        group = node.get("group", "")
        if ntype == "SCRIPT":
            print(f"  {nid}: group={group}")

print("\n--- CLUSTER/HOISTED/AND Nodes ---")
for node in nodes:
    if isinstance(node, dict):
        nid = node.get("id", "N/A")
        ntype = node.get("type", "")
        if "CLUSTER" in nid.upper() or "HOISTED" in nid.upper() or "AND" in nid.upper():
            print(f"  {nid}: type={ntype}")

print("\n--- TRIAL Nodes ---")
trial_nodes = [n for n in nodes if isinstance(n, dict) and "TRIAL" in n.get("id", "")]
print(f"Total TRIAL nodes: {len(trial_nodes)}")
for node in trial_nodes[:5]:
    print(f"  {node.get('id')}: type={node.get('type')}")

print("\n--- Edge Types ---")
edge_types = {}
for edge in edges:
    if isinstance(edge, dict):
        ltype = edge.get("type", "EXCITE")
        edge_types[ltype] = edge_types.get(ltype, 0) + 1
for ltype, count in edge_types.items():
    print(f"  {ltype}: {count}")

print("\n--- Sample Edges (first 15) ---")
for edge in edges[:15]:
    if isinstance(edge, dict):
        src = edge.get("from", "?")
        dst = edge.get("to", "?")
        weight = edge.get("weight", 1.0)
        ltype = edge.get("type", "EXCITE")
        print(f"  {src} -> {dst} ({ltype}, weight={weight:.2f})")

