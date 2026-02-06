#!/usr/bin/env python3
"""Check if TRIAL nodes exist in the topology and traces."""
import json
from pathlib import Path

def main():
    base = Path("snapshots/evolution/clean_structural_spurt")
    
    # Check Stage 1 Cycle 5 topology
    topo_path = base / "stage1/snapshots/cycle_0005.json"
    if topo_path.exists():
        with open(topo_path) as f:
            topo = json.load(f)
        
        nodes = topo.get("nodes", {})
        trial_nodes = [k for k in nodes.keys() if k.startswith("TRIAL")]
        cluster_nodes = [k for k in nodes.keys() if k.startswith("cluster")]
        
        print(f"=== Topology: {topo_path} ===")
        print(f"Total nodes: {len(nodes)}")
        print(f"TRIAL nodes: {len(trial_nodes)}")
        print(f"Cluster nodes: {len(cluster_nodes)}")
        
        if trial_nodes:
            print("\nTRIAL nodes found:")
            for n in trial_nodes[:10]:
                print(f"  - {n}")
        
        if cluster_nodes:
            print("\nCluster nodes found:")
            for n in cluster_nodes[:10]:
                print(f"  - {n}")
    else:
        print(f"Topology not found: {topo_path}")
    
    # Check traces for active_nodes
    trace_path = base / "stage1/traces/cycle_0005.jsonl"
    if trace_path.exists():
        print(f"\n=== Traces: {trace_path} ===")
        with open(trace_path) as f:
            lines = f.readlines()[:5]
        
        all_active = set()
        for line in lines:
            ep = json.loads(line)
            for tick in ep.get("ticks", []):
                all_active.update(tick.get("active_nodes", []))
        
        trial_active = [n for n in all_active if n.startswith("TRIAL")]
        cluster_active = [n for n in all_active if n.startswith("cluster")]
        
        print(f"Total unique active nodes: {len(all_active)}")
        print(f"TRIAL nodes active: {len(trial_active)}")
        print(f"Cluster nodes active: {len(cluster_active)}")
        
        print("\nAll active nodes:")
        for n in sorted(all_active):
            print(f"  - {n}")
    else:
        print(f"Traces not found: {trace_path}")

if __name__ == "__main__":
    main()

