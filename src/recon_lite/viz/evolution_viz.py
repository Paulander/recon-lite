"""Evolution Visualization for Graph Topology Changes.

Visualizes network evolution over training cycles, highlighting
new nodes, new edges, and removed components.

Usage:
    from recon_lite.viz.evolution_viz import render_evolution_snapshot
    
    path = render_evolution_snapshot(
        topology=registry.get_snapshot(),
        diff=diff_topologies(old_snap, new_snap),
        output_path=Path("snapshots/cycle_001.png")
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False


def diff_topologies(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two topology snapshots.
    
    Args:
        old: Previous topology snapshot (from registry.get_snapshot())
        new: Current topology snapshot
        
    Returns:
        Diff dict with added/removed nodes and edges, weight changes
    """
    old_nodes = set(old.get("nodes", {}).keys())
    new_nodes = set(new.get("nodes", {}).keys())
    
    old_edges = set(old.get("edges", {}).keys())
    new_edges = set(new.get("edges", {}).keys())
    
    added_nodes = list(new_nodes - old_nodes)
    removed_nodes = list(old_nodes - new_nodes)
    added_edges = list(new_edges - old_edges)
    removed_edges = list(old_edges - new_edges)
    
    # Detect weight changes
    weight_changes = []
    for edge_key in old_edges & new_edges:
        old_w = old["edges"][edge_key].get("weight", 1.0)
        new_w = new["edges"][edge_key].get("weight", 1.0)
        if abs(old_w - new_w) > 0.001:
            weight_changes.append({
                "edge": edge_key,
                "old": old_w,
                "new": new_w,
            })
    
    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "added_edges": added_edges,
        "removed_edges": removed_edges,
        "weight_changes": weight_changes,
        "timestamp": datetime.now().isoformat(),
    }


def render_evolution_snapshot(
    topology: Dict[str, Any],
    diff: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    highlight_new: bool = True,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Render graph as PNG with optional diff highlighting.
    
    - New nodes: Green
    - New edges: Blue  
    - Pruned nodes: Red (if showing removals)
    
    Args:
        topology: Current topology snapshot
        diff: Optional diff from diff_topologies()
        output_path: Where to save PNG
        highlight_new: Whether to highlight new elements
        title: Optional title
        
    Returns:
        Path to saved PNG, or None if failed
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        return None
    
    # Build networkx graph
    G = nx.DiGraph()
    
    nodes = topology.get("nodes", {})
    edges = topology.get("edges", {})
    
    # Prepare highlight sets
    new_nodes: Set[str] = set()
    new_edges: Set[str] = set()
    removed_nodes: Set[str] = set()
    
    if diff and highlight_new:
        new_nodes = set(diff.get("added_nodes", []))
        new_edges = set(diff.get("added_edges", []))
        removed_nodes = set(diff.get("removed_nodes", []))
    
    # Add nodes
    for node_id, node_data in nodes.items():
        group = node_data.get("group", "generic")
        ntype = node_data.get("type", "SCRIPT")
        G.add_node(node_id, group=group, ntype=ntype)
    
    # Add edges
    for edge_key, edge_data in edges.items():
        src = edge_data.get("src", "")
        dst = edge_data.get("dst", "")
        etype = edge_data.get("type", "SUB")
        weight = edge_data.get("weight", 1.0)
        
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, edge_key=edge_key, etype=etype, weight=weight)
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except Exception:
        pos = nx.shell_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Node colors based on status and group
    node_colors = []
    for node_id in G.nodes():
        if node_id in new_nodes:
            node_colors.append('#2ECC71')  # Green for new
        elif node_id in removed_nodes:
            node_colors.append('#E74C3C')  # Red for removed
        else:
            group = G.nodes[node_id].get("group", "generic")
            if group == "stem_promoted":
                node_colors.append('#9B59B6')  # Purple for stem promoted
            elif group == "sensor":
                node_colors.append('#3498DB')  # Blue for sensors
            elif group == "actuator":
                node_colors.append('#E67E22')  # Orange for actuators
            elif group == "backbone":
                node_colors.append('#34495E')  # Dark gray for backbone
            else:
                node_colors.append('#95A5A6')  # Light gray for generic
    
    # Node shapes based on type
    terminals = [n for n in G.nodes() if G.nodes[n].get("ntype") == "TERMINAL"]
    scripts = [n for n in G.nodes() if G.nodes[n].get("ntype") != "TERMINAL"]
    
    # Draw edges first
    edge_colors = []
    for u, v, data in G.edges(data=True):
        edge_key = data.get("edge_key", f"{u}->{v}")
        if edge_key in new_edges:
            edge_colors.append('#3498DB')  # Blue for new edges
        else:
            weight = data.get("weight", 1.0)
            # Fade color based on weight
            edge_colors.append((0.5, 0.5, 0.5, max(0.2, min(1.0, weight))))
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
        alpha=0.7,
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=scripts,
        node_color=[node_colors[list(G.nodes()).index(n)] for n in scripts],
        node_shape='s',  # Square for scripts
        node_size=800,
        alpha=0.9,
    )
    
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=terminals,
        node_color=[node_colors[list(G.nodes()).index(n)] for n in terminals],
        node_shape='o',  # Circle for terminals
        node_size=500,
        alpha=0.9,
    )
    
    # Labels
    labels = {n: n.replace("_", "\n") for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=6,
        font_weight='bold',
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ECC71', label='New Node'),
        mpatches.Patch(color='#3498DB', label='Sensor'),
        mpatches.Patch(color='#9B59B6', label='Stem Promoted'),
        mpatches.Patch(color='#E67E22', label='Actuator'),
        mpatches.Patch(color='#34495E', label='Backbone'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        node_count = len(G.nodes())
        edge_count = len(G.edges())
        ax.set_title(f"Network Topology ({node_count} nodes, {edge_count} edges)", fontsize=14)
    
    # Stats annotation
    if diff:
        stats_text = (
            f"Changes: +{len(new_nodes)} nodes, +{len(new_edges)} edges\n"
            f"Weight changes: {len(diff.get('weight_changes', []))}"
        )
        ax.annotate(
            stats_text,
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return output_path
    else:
        plt.close(fig)
        return None


def render_evolution_timeline(
    snapshot_dir: Path,
    output_path: Path,
    max_snapshots: int = 6,
) -> Optional[Path]:
    """
    Create side-by-side comparison of topology evolution over epochs.
    
    Args:
        snapshot_dir: Directory containing topology snapshot JSON files
        output_path: Where to save the combined PNG
        max_snapshots: Maximum snapshots to include
        
    Returns:
        Path to saved PNG
    """
    if not HAS_MATPLOTLIB:
        return None
    
    import json
    
    snapshot_dir = Path(snapshot_dir)
    snapshot_files = sorted(snapshot_dir.glob("*.json"))[:max_snapshots]
    
    if not snapshot_files:
        return None
    
    # Load snapshots
    snapshots = []
    for sf in snapshot_files:
        try:
            with open(sf) as f:
                data = json.load(f)
            snapshots.append((sf.stem, data))
        except Exception:
            continue
    
    if not snapshots:
        return None
    
    # Create multi-panel figure
    n_cols = min(3, len(snapshots))
    n_rows = (len(snapshots) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, (name, snapshot) in enumerate(snapshots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        
        # Convert snapshot to topology format
        topology = {
            "nodes": snapshot.get("nodes", {}),
            "edges": snapshot.get("edges", {}),
        }
        
        # Build and draw mini graph
        _draw_mini_graph(ax, topology, name)
    
    # Hide unused axes
    for idx in range(len(snapshots), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis('off')
    
    fig.suptitle("Network Evolution Timeline", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return output_path


def _draw_mini_graph(ax, topology: Dict[str, Any], title: str):
    """Draw a mini graph in a subplot."""
    if not HAS_NETWORKX:
        ax.text(0.5, 0.5, "networkx required", transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    
    G = nx.DiGraph()
    
    nodes = topology.get("nodes", {})
    edges = topology.get("edges", {})
    
    for node_id in nodes:
        G.add_node(node_id)
    
    for edge_key, edge_data in edges.items():
        src = edge_data.get("src", "")
        dst = edge_data.get("dst", "")
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst)
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "Empty graph", transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')
        return
    
    try:
        pos = nx.spring_layout(G, seed=42)
    except Exception:
        pos = nx.circular_layout(G)
    
    nx.draw_networkx(
        G, pos, ax=ax,
        node_size=100,
        font_size=5,
        arrows=True,
        arrowsize=5,
        node_color='#3498DB',
        edge_color='#95A5A6',
        alpha=0.8,
    )
    
    ax.set_title(f"{title}\n({len(G.nodes())} nodes)")
    ax.axis('off')


def save_topology_snapshot(
    registry: Any,  # TopologyRegistry
    output_dir: Path,
    cycle: int,
) -> Path:
    """
    Save a topology snapshot for a training cycle.
    
    Args:
        registry: TopologyRegistry
        output_dir: Directory for snapshots
        cycle: Cycle number
        
    Returns:
        Path to saved JSON snapshot
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot = registry.get_snapshot()
    snapshot["cycle"] = cycle
    
    output_path = output_dir / f"cycle_{cycle:04d}.json"
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    
    return output_path
