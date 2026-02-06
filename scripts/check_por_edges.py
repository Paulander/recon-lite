"""Check POR edges in topology"""
from pathlib import Path
from recon_lite.models.registry import TopologyRegistry

registry = TopologyRegistry(Path('topologies/krk_entry_topology.json'))
edges = registry.get_all_edges()
nodes = {n.id: n for n in [registry.get_node(nid) for nid in registry.get_all_node_ids()]}

print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")
print()

# Find POR edges
por_edges = [e for e in edges if e.type == 'POR']
print(f"POR edges: {len(por_edges)}")

for e in por_edges:
    src_node = nodes.get(e.src)
    dst_node = nodes.get(e.dst)
    src_type = src_node.type if src_node else "?"
    dst_type = dst_node.type if dst_node else "?"
    print(f"  POR: {e.src} ({src_type}) -> {e.dst} ({dst_type})")
    
    # Check if either is TERMINAL
    if src_type == "TERMINAL" or dst_type == "TERMINAL":
        print(f"    *** PROBLEM: TERMINAL nodes cannot have POR edges ***")
