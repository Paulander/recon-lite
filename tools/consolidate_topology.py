#!/usr/bin/env python3
"""
Topology Consolidation Tool (M5 Deduplication Pruner)
Pure Python version (no numpy required)
"""

import json
import os
import argparse
import math

def compute_similarity(sig1, sig2):
    """Cosine similarity between two signatures (Pure Python)."""
    if sig1 is None or sig2 is None: return 0.0
    if len(sig1) != len(sig2): return 0.0
    
    dot = sum(a * b for a, b in zip(sig1, sig2))
    norm1 = math.sqrt(sum(a * a for a in sig1))
    norm2 = math.sqrt(sum(b * b for b in sig2))
    
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(dot / (norm1 * norm2))

def consolidate(input_path, output_path, threshold=0.95):
    print(f"📦 Loading topology from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    if isinstance(edges, dict):
        edges = list(edges.values())

    # 1. Identify merge candidates (TERMINAL nodes with signatures)
    terminals = [
        (nid, n["meta"].get("pattern_signature"))
        for nid, n in nodes.items()
        if n.get("type") == "TERMINAL" and n.get("meta", {}).get("pattern_signature")
    ]
    
    print(f"🔍 Found {len(terminals)} terminal nodes with signatures.")
    
    merge_map = {} # source_id -> target_id
    processed = set()

    for i, (id1, sig1) in enumerate(terminals):
        if id1 in processed or id1 in merge_map: continue
        
        for j in range(i + 1, len(terminals)):
            id2, sig2 = terminals[j]
            if id2 in processed or id2 in merge_map: continue
            
            sim = compute_similarity(sig1, sig2)
            if sim >= threshold:
                print(f"  🤝 Merging {id2} -> {id1} (sim: {sim:.3f})")
                merge_map[id2] = id1
        
        processed.add(id1)

    if not merge_map:
        print("✅ No duplicates found above threshold. Nothing to do.")
        return

    print(f"💡 Merging {len(merge_map)} redundant nodes...")

    # 2. Rebuild Nodes (Remove merged ones)
    new_nodes = {}
    for nid, node in nodes.items():
        if nid not in merge_map:
            new_nodes[nid] = node
        else:
            target_id = merge_map[nid]
            target_node = nodes[target_id]
            
            # Combine XP/Samples
            xp1 = node.get("meta", {}).get("xp", 50)
            xp2 = target_node.get("meta", {}).get("xp", 50)
            target_node["meta"]["xp"] = max(xp1, xp2)
            
            samples1 = node.get("meta", {}).get("sample_count", 0)
            samples2 = target_node.get("meta", {}).get("sample_count", 0)
            target_node["meta"]["sample_count"] = samples1 + samples2

    # 3. Rebuild Edges (Rewire targets)
    new_edges = []
    seen_edges = set()

    for edge in edges:
        src = edge["src"]
        dst = edge["dst"]
        ltype = edge.get("type") or edge.get("ltype") or "SUB"
        
        new_src = merge_map.get(src, src)
        new_dst = merge_map.get(dst, dst)
        
        if new_src == new_dst:
            continue
            
        edge_key = f"{new_src}->{new_dst}:{ltype}"
        if edge_key not in seen_edges:
            new_edge = dict(edge)
            new_edge["src"] = new_src
            new_edge["dst"] = new_dst
            new_edges.append(new_edge)
            seen_edges.add(edge_key)

    # 4. Save
    output_data = {
        "nodes": new_nodes,
        "edges": new_edges,
        "meta": data.get("meta", {})
    }
    output_data["meta"]["consolidated"] = True
    output_data["meta"]["nodes_merged"] = len(merge_map)
    output_data["meta"]["original_node_count"] = len(nodes)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✨ Consolidation complete. Saved to {output_path}")
    print(f"📉 Node count reduced from {len(nodes)} to {len(new_nodes)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to topology JSON")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--threshold", type=float, default=0.95, help="Similarity threshold (0-1)")
    args = parser.parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_merged{ext}"
    
    consolidate(input_path, output_path, args.threshold)
