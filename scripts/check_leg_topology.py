#!/usr/bin/env python3
"""Check leg node topology to understand gating issue."""
import json
from pathlib import Path

topo_path = Path("snapshots/evolution/stage11_logic_gauntlet/topology_gated.json")
if not topo_path.exists():
    topo_path = Path("topologies/kpk_legs_topology.json")

with open(topo_path) as f:
    topo = json.load(f)

print(f"Analyzing: {topo_path}")
print()

print("=== ALL EDGES WITH TYPES ===")
edges = topo.get("edges", [])
for e in edges:
    src = e.get("src", "")
    dst = e.get("dst", "")
    etype = e.get("type", "SUB")
    if "_leg" in src or "_leg" in dst:
        print(f"  {src} --{etype}--> {dst}")

print()
print("=== LEG NODE METADATA ===")
for node in topo.get("nodes", []):
    if node.get("id", "").endswith("_leg"):
        print(f"  {node['id']}: {node.get('meta', {})}")

print()
print("=== SUB CHILDREN ANALYSIS (what gating checks) ===")
# Find SUB children only (not POR)
leg_sub_children = {}
for e in edges:
    if e.get("type", "SUB") == "SUB":
        src = e.get("src", "")
        if src.endswith("_leg"):
            dst = e.get("dst", "")
            if src not in leg_sub_children:
                leg_sub_children[src] = []
            leg_sub_children[src].append(dst)

for leg in ["kpk_pawn_leg", "kpk_king_leg"]:
    children = leg_sub_children.get(leg, [])
    if children:
        print(f"  {leg} SUB children: {children}")
    else:
        print(f"  ⚠️ {leg} has NO SUB children!")

if not any(leg_sub_children.values()):
    print()
    print("  ⚠️ PROBLEM CONFIRMED: Legs have no SUB children!")
    print("  Gating (require_child_confirm) will ALWAYS pass because")
    print("  Graph.children() only returns SUB children, not POR successors.")
    print()
    print("  The POR edges (kpk_pawn_leg -> kpk_king_leg -> kpk_arbiter)")
    print("  are for SEQUENCING, not hierarchy!")
    print()
    print("  FIX OPTIONS:")
    print("  1. Add TRIAL nodes as SUB children of the legs")
    print("  2. Or: Change gating to also check POR successors")
