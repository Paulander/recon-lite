#!/usr/bin/env python3
"""Analyze topology depth structure."""
import json
import sys
from collections import defaultdict

# Load snapshot
snapshot_path = sys.argv[1] if len(sys.argv) > 1 else "snapshots/evolution/with_packs/stage7/cycle_0030.json"
with open(snapshot_path) as f:
    data = json.load(f)

nodes = data.get("nodes", {})
edges = data.get("edges", {})

# Build SUB parent map (child -> parent) - case insensitive!
sub_parent = {}
for edge_key, edge in edges.items():
    if edge.get("type", "").upper() == "SUB":
        sub_parent[edge["dst"]] = edge["src"]

# Find roots (nodes with no SUB parent)
roots = [nid for nid in nodes if nid not in sub_parent]

# Compute depth from root via SUB edges
def get_depth(node_id, cache={}):
    if node_id in cache:
        return cache[node_id]
    if node_id not in sub_parent:
        cache[node_id] = 0
        return 0
    parent = sub_parent[node_id]
    depth = 1 + get_depth(parent, cache)
    cache[node_id] = depth
    return depth

# Analyze
depth_dist = defaultdict(list)
for nid in nodes:
    d = get_depth(nid)
    depth_dist[d].append(nid)

print("="*60)
print("TOPOLOGY DEPTH ANALYSIS (SUB hierarchy)")
print("="*60)

# Find max depth with nodes
max_depth = max(depth_dist.keys()) if depth_dist else 0
print(f"\nMax depth: {max_depth}")
print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")

print("\n" + "-"*60)
print("DEPTH DISTRIBUTION")
print("-"*60)
for d in range(max_depth + 1):
    node_list = depth_dist[d]
    print(f"\nDepth {d}: {len(node_list)} nodes")
    for nid in sorted(node_list)[:8]:
        parent = sub_parent.get(nid, "ROOT")
        tier = nodes[nid].get("meta", {}).get("tier", "")
        print(f"    {nid[:50]:50s} <- {parent[:30]} [{tier}]")
    if len(node_list) > 8:
        print(f"    ... and {len(node_list)-8} more")

# Check for pack nodes
print("\n" + "-"*60)
print("PACK/GATE NODES")
print("-"*60)
pack_nodes = [nid for nid in nodes if any(x in nid.lower() for x in ["pack_", "and_", "or_", "_gate"])]
print(f"Found {len(pack_nodes)} pack-related nodes")
for nid in pack_nodes[:10]:
    print(f"  {nid}")

# ASCII tree for depth visualization
print("\n" + "-"*60)
print("ASCII TREE (first few branches)")
print("-"*60)

def print_tree(node_id, indent=0, max_depth=3, printed=set()):
    if indent > max_depth or node_id in printed:
        return
    printed.add(node_id)
    tier = nodes.get(node_id, {}).get("meta", {}).get("tier", "")
    tier_mark = "[T]" if tier == "trial" else ""
    print("  " * indent + f"└─ {node_id} {tier_mark}")
    
    # Find children
    children = [e["dst"] for e in edges.values() if e["src"] == node_id and e["type"] == "sub"]
    for child in sorted(children)[:4]:
        print_tree(child, indent + 1, max_depth, printed)
    if len(children) > 4:
        print("  " * (indent+1) + f"... and {len(children)-4} more children")

# Start from known roots
for root in ["kpk_root", "root"] + roots[:2]:
    if root in nodes:
        print(f"\nFrom {root}:")
        print_tree(root)
