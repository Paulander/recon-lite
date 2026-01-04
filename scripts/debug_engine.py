#!/usr/bin/env python3
"""Diagnostic test for engine and game loop."""

import chess

print("=== Graph Building Test ===")
from recon_lite.models.registry import TopologyRegistry
from recon_lite_chess.graph.builder import build_graph_from_topology

registry = TopologyRegistry("topologies/kpk_topology.json")
print("Registry nodes:", len(registry.get_all_node_ids()))

graph = build_graph_from_topology("topologies/kpk_topology.json", registry)
print("Graph nodes:", len(graph.nodes))
for nid, node in list(graph.nodes.items())[:3]:
    print(f"  {nid}: {node.ntype}")

print("\n=== Engine Test ===")
from recon_lite.engine import ReConEngine
from recon_lite.graph import NodeState

engine = ReConEngine(graph)

# Test position: pawn on 7th rank
board = chess.Board("8/4P3/4K3/8/8/8/8/7k w - - 0 1")
print("Position:", board.fen())

env = {"board": board, "our_color": chess.WHITE, "move_count": 0}

# CRITICAL: Lock into KPK subgraph (this was missing!)
def kpk_sentinel(env):
    return env.get("board") is not None

if "kpk_root" in graph.nodes:
    engine.lock_subgraph("kpk_root", kpk_sentinel)
    print("Locked into kpk_root subgraph")
else:
    print("WARNING: kpk_root not found in graph!")

# Step engine a few times
for i in range(20):
    engine.step(env)
    kpk_policy = env.get("kpk", {}).get("policy", {})
    if "suggested_move" in kpk_policy:
        print(f"Step {i}: Suggested move = {kpk_policy['suggested_move']}")
        break
else:
    print("No suggested move after 20 steps")
    active = [n.nid for n in graph.nodes.values() if n.state in (NodeState.ACTIVE, NodeState.REQUESTED)]
    print(f"Active/Requested nodes: {active}")
    print(f"env.kpk: {env.get('kpk', {})}")


