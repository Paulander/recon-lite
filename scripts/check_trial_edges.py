#!/usr/bin/env python3
"""Check if TRIAL nodes have edges from kpk_detect."""
import json
from pathlib import Path

def main():
    base = Path("snapshots/evolution/clean_structural_spurt")
    
    # Check Stage 1 Cycle 5 topology
    topo_path = base / "stage1/snapshots/cycle_0005.json"
    if not topo_path.exists():
        print(f"Topology not found: {topo_path}")
        return
    
    with open(topo_path) as f:
        topo = json.load(f)
    
    nodes = topo.get("nodes", {})
    edges = topo.get("edges", {})
    
    trial_nodes = [k for k in nodes.keys() if k.startswith("TRIAL")]
    
    print(f"=== Edge Analysis for {topo_path} ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    print(f"TRIAL nodes: {len(trial_nodes)}")
    
    # Check edges to TRIAL nodes
    edges_to_trial = []
    edges_from_detect = []
    edges_from_execute = []
    
    # Handle both dict and list formats
    if isinstance(edges, dict):
        edge_list = list(edges.values())
    else:
        edge_list = edges
    
    for edge in edge_list:
        if not isinstance(edge, dict):
            continue
        
        dst = edge.get("dst", edge.get("to", ""))
        src = edge.get("src", edge.get("from", ""))
        
        if dst.startswith("TRIAL"):
            edges_to_trial.append((src, dst, edge.get("weight", 1.0)))
            if src == "kpk_detect":
                edges_from_detect.append((src, dst, edge.get("weight", 1.0)))
            elif src == "kpk_execute":
                edges_from_execute.append((src, dst, edge.get("weight", 1.0)))
    
    print(f"\nEdges TO TRIAL nodes: {len(edges_to_trial)}")
    print(f"  From kpk_detect: {len(edges_from_detect)}")
    print(f"  From kpk_execute: {len(edges_from_execute)}")
    
    if edges_from_detect[:5]:
        print("\nSample edges from kpk_detect to TRIAL:")
        for src, dst, weight in edges_from_detect[:5]:
            print(f"  {src} -> {dst} (weight={weight})")
    
    if not edges_from_detect and not edges_from_execute:
        print("\n!!! NO EDGES from kpk_detect or kpk_execute to TRIAL nodes !!!")
        print("This is why TRIAL nodes are not activating!")
        
        # Check where edges are coming from
        other_sources = set(src for src, dst, _ in edges_to_trial)
        if other_sources:
            print(f"\nEdges are coming from: {other_sources}")
    
    # Check what edges exist from kpk_detect
    print("\n=== All edges FROM kpk_detect ===")
    from_detect = [(src, dst, edge.get("weight", 1.0)) 
                   for edge in edge_list 
                   if isinstance(edge, dict) and edge.get("src", edge.get("from", "")) == "kpk_detect"]
    
    for src, dst, weight in from_detect[:15]:
        marker = " [TRIAL]" if dst.startswith("TRIAL") else ""
        print(f"  {src} -> {dst} (weight={weight}){marker}")

if __name__ == "__main__":
    main()

