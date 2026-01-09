#!/usr/bin/env python3
"""Analyze topology depth from training snapshot."""
import json
import sys
sys.path.insert(0, 'src')

snapshot_path = "snapshots/evolution/pure_verified/stage7/cycle_0010.json"

with open(snapshot_path) as f:
    d = json.load(f)
    nodes = d.get('nodes', {})
    edges = d.get('edges', [])
    
    print('=== TOPOLOGY DEPTH ANALYSIS ===')
    print(f'Total nodes: {len(nodes)}')
    
    # Count by type
    types = {}
    stem_nodes = []
    for nid, n in nodes.items():
        ntype = n.get('type', 'unknown')
        types[ntype] = types.get(ntype, 0) + 1
        if 'stem' in nid.lower():
            stem_nodes.append(nid)
    
    print(f'Node types: {types}')
    print(f'Stem cells in graph: {len(stem_nodes)}')
    
    # Calculate depth via BFS from root
    children = {}
    for e in edges:
        if isinstance(e, dict):
            src, dst = e.get('source'), e.get('target')
        else:
            src, dst = e[:2] if len(e) >= 2 else (None, None)
        if src:
            children.setdefault(src, []).append(dst)
    
    # BFS
    depths = {'kpk_root': 0}
    queue = ['kpk_root']
    max_depth = 0
    while queue:
        nid = queue.pop(0)
        for child in children.get(nid, []):
            if child and child not in depths:
                depths[child] = depths[nid] + 1
                max_depth = max(max_depth, depths[child])
                queue.append(child)
    
    print(f'\nMax depth from root: {max_depth}')
    print(f'Nodes at depth > 2: {sum(1 for d in depths.values() if d > 2)}')
    
    # Show depth distribution
    depth_dist = {}
    for nid, d in depths.items():
        depth_dist[d] = depth_dist.get(d, 0) + 1
    print(f'Depth distribution: {dict(sorted(depth_dist.items()))}')
