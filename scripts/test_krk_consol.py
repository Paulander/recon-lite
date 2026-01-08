#!/usr/bin/env python3
"""Simple test to debug KRK recon mode crash."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite.graph import LinkType

print("Loading topology...")
graph = build_graph_from_topology(Path("topologies/krk_topology.json"))
print(f"Graph: {len(graph.nodes)} nodes, edges type: {type(graph.edges)}")

# Show edge type
if hasattr(graph.edges, 'values'):
    print("graph.edges is a dict")
    edges = list(graph.edges.values())[:3]
else:
    print("graph.edges is a list")
    edges = list(graph.edges)[:3]

for e in edges:
    print(f"  Edge: {e.src} -> {e.dst} ({e.ltype})")

# Try consolidation init
print("\nTesting consolidation init...")
from recon_lite.plasticity.consolidate import ConsolidationEngine

consolidation = ConsolidationEngine()
edge_list = graph.edges.values() if isinstance(graph.edges, dict) else graph.edges
krk_edges = [
    f"{e.src}->{e.dst}:{e.ltype.name}"
    for e in edge_list
    if e.ltype in (LinkType.POR, LinkType.SUB) and ("krk" in e.src.lower() or "krk" in e.dst.lower())
]
print(f"Found {len(krk_edges)} KRK edges")
consolidation.init_from_graph(graph, edge_whitelist=krk_edges)
print("Consolidation init SUCCESS!")
