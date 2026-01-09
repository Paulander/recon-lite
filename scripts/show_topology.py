#!/usr/bin/env python3
"""Show topology structure."""
import json

snapshot_path = "snapshots/evolution/pure_verified/stage7/cycle_0010.json"

with open(snapshot_path) as f:
    d = json.load(f)

nodes = d.get('nodes', {})
edges = d.get('edges', [])

print('=== TOPOLOGY STRUCTURE ===')
print(f'Total nodes: {len(nodes)}')
print()

print('SCRIPT nodes:')
for nid, n in nodes.items():
    if n.get('type') == 'SCRIPT':
        print(f'  {nid}')

print()
print('TERMINAL nodes (strategies/arbiter):')
for nid, n in nodes.items():
    if n.get('type') == 'TERMINAL' and 'stem' not in nid:
        meta = n.get('meta', {})
        learned = meta.get('learned_weights', {})
        print(f'  {nid}')
        if learned:
            print(f'    learned_weights: {dict(list(learned.items())[:3])}...')

print()
print(f'Stem cells: {sum(1 for nid in nodes if "stem" in nid)}')
print()

# Show stem cell status breakdown
stem_status = {}
for nid, n in nodes.items():
    if 'stem' in nid:
        status = n.get('meta', {}).get('status', 'UNKNOWN')
        stem_status[status] = stem_status.get(status, 0) + 1
print(f'Stem cell status: {stem_status}')

print()
print(f'Total edges: {len(edges)}')
