#!/usr/bin/env python3
"""Human Annotation Tool for Discovered Patterns.

Displays signature heatmaps for unlabeled promoted nodes and allows
human-provided naming. Updates topology.json with the new labels.

Usage:
    python tools/annotate_discovery.py topologies/kpk_topology.json
    
    # With specific output directory
    python tools/annotate_discovery.py topologies/kpk_topology.json --signatures signatures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_topology(path: Path) -> dict:
    """Load topology JSON."""
    with open(path) as f:
        return json.load(f)


def save_topology(path: Path, data: dict):
    """Save topology JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_unlabeled_promotions(data: dict) -> list:
    """Get promoted nodes without human labels."""
    promotions = data.get("promoted_nodes", [])
    return [p for p in promotions if not p.get("human_label")]


def display_signature(node_id: str, signature_dir: Path) -> bool:
    """Display the signature image for a node."""
    sig_path = signature_dir / f"{node_id}.png"
    
    if not sig_path.exists():
        print(f"  [No signature image found at {sig_path}]")
        return False
    
    # Try to display with PIL
    try:
        from PIL import Image
        img = Image.open(sig_path)
        img.show()
        return True
    except ImportError:
        pass
    
    # Try system viewer
    try:
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            subprocess.run(["start", str(sig_path)], shell=True)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(sig_path)])
        else:
            subprocess.run(["xdg-open", str(sig_path)])
        return True
    except Exception:
        pass
    
    print(f"  [Signature at: {sig_path}]")
    return False


def display_pattern_info(promo: dict, nodes: list):
    """Display information about a promoted pattern."""
    node_id = promo.get("node_id", "unknown")
    cell_id = promo.get("cell_id", "unknown")
    parent_id = promo.get("parent_id", "unknown")
    tick = promo.get("tick", 0)
    
    print("\n" + "=" * 60)
    print(f"Node ID: {node_id}")
    print(f"Source Stem Cell: {cell_id}")
    print(f"Parent Node: {parent_id}")
    print(f"Promoted at tick: {tick}")
    
    # Find node meta
    for n in nodes:
        if n.get("id") == node_id:
            meta = n.get("meta", {})
            if "consistency" in meta:
                print(f"Pattern consistency: {meta['consistency']:.2%}")
            if "sample_count" in meta:
                print(f"Sample count: {meta['sample_count']}")
            if "avg_reward" in meta:
                print(f"Average reward: {meta['avg_reward']:.3f}")
            break
    
    # Show signature if available
    signature = promo.get("pattern_signature")
    if signature:
        print(f"Signature vector: [{len(signature)} dimensions]")
        # Show top 5 values
        if len(signature) > 5:
            top_indices = sorted(range(len(signature)), 
                                key=lambda i: abs(signature[i]), 
                                reverse=True)[:5]
            print(f"  Top activations: {[(i, f'{signature[i]:.3f}') for i in top_indices]}")
    
    print("=" * 60)


def rename_node(data: dict, old_id: str, new_id: str) -> bool:
    """
    Rename a node throughout the topology.
    
    Updates:
    - Node ID in nodes list
    - Edge src/dst references
    - Promotion record
    """
    # Check if new ID already exists
    for n in data.get("nodes", []):
        if n.get("id") == new_id:
            print(f"Error: Node {new_id} already exists!")
            return False
    
    # Rename in nodes list
    for n in data.get("nodes", []):
        if n.get("id") == old_id:
            n["id"] = new_id
            break
    
    # Update edges
    for e in data.get("edges", []):
        if e.get("src") == old_id:
            e["src"] = new_id
        if e.get("dst") == old_id:
            e["dst"] = new_id
    
    # Update promotion records
    for p in data.get("promoted_nodes", []):
        if p.get("node_id") == old_id:
            p["node_id"] = new_id
    
    return True


def annotate_interactive(
    topology_path: Path,
    signature_dir: Path,
):
    """Run interactive annotation session."""
    data = load_topology(topology_path)
    
    unlabeled = get_unlabeled_promotions(data)
    
    if not unlabeled:
        print("No unlabeled promoted nodes found.")
        return
    
    print(f"\nFound {len(unlabeled)} unlabeled promoted nodes.")
    print("For each pattern, you can:")
    print("  - Type a descriptive name (e.g., 'Horizontal_Opposition')")
    print("  - Press Enter to skip")
    print("  - Type 'q' to quit")
    
    nodes = data.get("nodes", [])
    modified = False
    
    for i, promo in enumerate(unlabeled, 1):
        print(f"\n[{i}/{len(unlabeled)}]")
        
        node_id = promo.get("node_id", "unknown")
        display_pattern_info(promo, nodes)
        display_signature(node_id, signature_dir)
        
        # Get user input
        label = input("\nEnter label (or Enter to skip, 'q' to quit): ").strip()
        
        if label.lower() == 'q':
            print("Quitting...")
            break
        
        if label:
            # Validate label (no spaces, reasonable characters)
            clean_label = label.replace(" ", "_").replace("-", "_")
            clean_label = "".join(c for c in clean_label if c.isalnum() or c == "_")
            
            if clean_label:
                # Update label
                promo["human_label"] = clean_label
                
                # Optionally rename the node
                rename_prompt = input(f"Rename node from '{node_id}' to '{clean_label}'? (y/N): ").strip().lower()
                if rename_prompt == 'y':
                    if rename_node(data, node_id, clean_label):
                        print(f"  ✓ Renamed to {clean_label}")
                    else:
                        print(f"  ✗ Rename failed, keeping original ID")
                else:
                    print(f"  ✓ Label set: {clean_label}")
                
                modified = True
    
    if modified:
        # Save changes
        save_topology(topology_path, data)
        print(f"\n✓ Saved changes to {topology_path}")
    else:
        print("\nNo changes made.")


def list_unlabeled(topology_path: Path):
    """List all unlabeled promoted nodes."""
    data = load_topology(topology_path)
    unlabeled = get_unlabeled_promotions(data)
    
    if not unlabeled:
        print("No unlabeled promoted nodes.")
        return
    
    print(f"\n{len(unlabeled)} unlabeled promoted nodes:")
    for p in unlabeled:
        node_id = p.get("node_id", "unknown")
        parent_id = p.get("parent_id", "unknown")
        tick = p.get("tick", 0)
        print(f"  - {node_id} (parent: {parent_id}, tick: {tick})")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate discovered patterns with human-readable labels"
    )
    parser.add_argument(
        "topology",
        type=Path,
        help="Path to topology.json file"
    )
    parser.add_argument(
        "--signatures", "-s",
        type=Path,
        default=Path("signatures"),
        help="Directory containing signature PNG files"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List unlabeled nodes without interactive annotation"
    )
    
    args = parser.parse_args()
    
    if not args.topology.exists():
        print(f"Error: Topology file not found: {args.topology}")
        sys.exit(1)
    
    if args.list:
        list_unlabeled(args.topology)
    else:
        annotate_interactive(args.topology, args.signatures)


if __name__ == "__main__":
    main()
