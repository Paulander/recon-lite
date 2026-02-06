#!/usr/bin/env python3
"""Find TRIAL edges in snapshot."""
import json

with open('/home/paulander/git/recon-lite/snapshots/evolution/with_packs/stage7/cycle_0030.json') as f:
    data = json.load(f)

edges = data.get('edges', {})
nodes = data.get('nodes', {})

# Find edges involving TRIAL nodes
trial_edges = []
for key, e in edges.items():
    if 'TRIAL' in e.get('dst','') or 'TRIAL' in e.get('src',''):
        trial_edges.append(e)

print(f"Total edges: {len(edges)}")
print(f"TRIAL-related edges: {len(trial_edges)}")

# Check if kpk_detect has any outgoing SUB edges
kpk_detect_children = [e for e in edges.values() if e.get('src') == 'kpk_detect' and e.get('type') == 'sub']
print(f"\nkpk_detect SUB children: {len(kpk_detect_children)}")
for e in kpk_detect_children[:5]:
    print(f"  -> {e['dst']}")

# All SUB edges
sub_edges = [e for e in edges.values() if e.get('type') == 'sub' or e.get('type') == 'SUB']
print(f"\nTotal SUB edges: {len(sub_edges)}")
for e in sub_edges[:10]:
    print(f"  {e['src']} --SUB--> {e['dst']}")
