"""Graph Builder from Topology JSON.

Builds ReCoN graphs dynamically from topology.json files instead of
hardcoded class structures.

Usage:
    from recon_lite_chess.graph.builder import build_graph_from_topology
    
    graph = build_graph_from_topology(Path("topologies/kpk_topology.json"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from recon_lite.graph import Graph, Node, NodeType, LinkType
from recon_lite.models.registry import TopologyRegistry, NodeSpec, EdgeSpec


def _node_type_from_str(s: str) -> NodeType:
    """Convert string to NodeType enum."""
    return NodeType.TERMINAL if s.upper() == "TERMINAL" else NodeType.SCRIPT


def _link_type_from_str(s: str) -> LinkType:
    """Convert string to LinkType enum."""
    mapping = {
        "SUB": LinkType.SUB,
        "SUR": LinkType.SUR,
        "POR": LinkType.POR,
        "RET": LinkType.RET,
    }
    return mapping.get(s.upper(), LinkType.SUB)


def build_graph_from_topology(
    topology_path: Path,
    registry: Optional[TopologyRegistry] = None,
) -> Graph:
    """
    Build a ReCoN Graph from a topology.json file.
    
    Args:
        topology_path: Path to the topology JSON file
        registry: Optional pre-loaded registry (will load from path if None)
        
    Returns:
        Constructed Graph with all nodes and edges
    """
    if registry is None:
        registry = TopologyRegistry(Path(topology_path))
    
    g = Graph()
    
    # Phase 1: Create all nodes
    for node_id in registry.get_all_node_ids():
        node_spec = registry.get_node(node_id)
        if node_spec is None:
            continue
        
        node = _create_node_from_spec(node_spec, registry)
        g.add_node(node)
    
    # Phase 2: Create all edges
    for edge_spec in registry.get_all_edges():
        ltype = _link_type_from_str(edge_spec.type)
        
        # Only add edge if both endpoints exist
        if edge_spec.src in g.nodes and edge_spec.dst in g.nodes:
            g.add_edge(edge_spec.src, edge_spec.dst, ltype)
            
            # Update weight if not default
            if edge_spec.weight != 1.0:
                _set_edge_weight(g, edge_spec.src, edge_spec.dst, ltype, edge_spec.weight)
            
            # Mark for consolidation
            edge = _get_edge(g, edge_spec.src, edge_spec.dst, ltype)
            if edge:
                edge.meta["consolidate"] = edge_spec.consolidate
                edge.meta["confirmation_count"] = edge_spec.confirmation_count
    
    return g


def _create_node_from_spec(spec: NodeSpec, registry: TopologyRegistry) -> Node:
    """Create a Node from a NodeSpec, optionally using factory function."""
    ntype = _node_type_from_str(spec.type)
    
    # Try to get factory-created node
    factory = registry.get_node_factory(spec.id)
    if factory is not None:
        try:
            node = factory(spec.id)
            node.meta["factory"] = spec.factory
            # Preserve any pattern signature in meta
            if spec.pattern_signature is not None:
                node.meta["pattern_signature"] = spec.pattern_signature
            if spec.weight_source is not None:
                node.meta["weight_source"] = spec.weight_source
            node.meta.update(spec.meta)
            return node
        except Exception as e:
            print(f"Warning: Factory {spec.factory} failed for {spec.id}: {e}")
    
    # Fallback: create basic node without predicate
    node = Node(nid=spec.id, ntype=ntype)
    node.meta["factory"] = spec.factory
    if spec.pattern_signature is not None:
        node.meta["pattern_signature"] = spec.pattern_signature
    if spec.weight_source is not None:
        node.meta["weight_source"] = spec.weight_source
    node.meta.update(spec.meta)
    node.meta["group"] = spec.group
    
    return node


def _get_edge(g: Graph, src: str, dst: str, ltype: LinkType):
    """Get an edge by endpoints and type."""
    for edge in g.edges:
        if edge.src == src and edge.dst == dst and edge.ltype == ltype:
            return edge
    return None


def _set_edge_weight(g: Graph, src: str, dst: str, ltype: LinkType, weight: float):
    """Set weight on an edge."""
    edge = _get_edge(g, src, dst, ltype)
    if edge:
        edge.w = weight


def export_topology_from_graph(
    graph: Graph,
    output_path: Path,
    network_name: str = "exported",
) -> TopologyRegistry:
    """
    Export current Graph to topology.json format.
    
    Args:
        graph: The graph to export
        output_path: Where to save the topology JSON
        network_name: Name for the network
        
    Returns:
        The created TopologyRegistry
    """
    from datetime import datetime
    
    registry = TopologyRegistry(output_path)
    registry.data["network"] = network_name
    registry.data["created"] = datetime.now().isoformat()
    
    def _to_jsonable(val: Any):
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None:
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.floating, np.integer)):
                return val.item()
        if isinstance(val, dict):
            return {k: _to_jsonable(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_to_jsonable(v) for v in val]
        return val
    
    # Export nodes
    for nid, node in graph.nodes.items():
        node_spec = {
            "id": nid,
            "type": "TERMINAL" if node.ntype == NodeType.TERMINAL else "SCRIPT",
            "group": node.meta.get("group", "generic"),
            "factory": node.meta.get("factory"),
            "meta": {
                k: _to_jsonable(v)
                for k, v in node.meta.items()
                if k not in ("group", "factory", "blackboard")
            },
        }
        
        # Include pattern signature if present
        if "pattern_signature" in node.meta:
            node_spec["pattern_signature"] = node.meta["pattern_signature"]
        if "weight_source" in node.meta:
            node_spec["weight_source"] = node.meta["weight_source"]
        
        registry._nodes[nid] = NodeSpec.from_dict(node_spec)
    
    # Export edges
    for edge in graph.edges:
        edge_spec = EdgeSpec(
            src=edge.src,
            dst=edge.dst,
            type=edge.ltype.value.upper() if hasattr(edge.ltype, 'value') else str(edge.ltype).upper(),
            weight=float(edge.w) if hasattr(edge, 'w') else 1.0,
            consolidate=edge.meta.get("consolidate", True),
            confirmation_count=edge.meta.get("confirmation_count", 0),
        )
        registry._edges[edge_spec.key] = edge_spec
    
    registry.save()
    return registry


def refresh_graph_from_registry(
    graph: Graph,
    registry: TopologyRegistry,
) -> Dict[str, str]:
    """
    Hot-reload topology changes from registry into existing graph.
    
    - Adds new nodes that exist in registry but not in graph
    - Updates edge weights from registry
    - Does NOT remove nodes (pruning happens separately)
    
    Args:
        graph: The graph to update
        registry: The registry with new topology
        
    Returns:
        Dict mapping node/edge IDs to their status ("added", "updated")
    """
    changes: Dict[str, str] = {}
    
    # Add new nodes
    for node_id in registry.get_all_node_ids():
        if node_id not in graph.nodes:
            spec = registry.get_node(node_id)
            if spec:
                node = _create_node_from_spec(spec, registry)
                graph.add_node(node)
                changes[node_id] = "added"
    
    # Add/update edges
    for edge_spec in registry.get_all_edges():
        ltype = _link_type_from_str(edge_spec.type)
        
        # Check if edge exists
        existing = _get_edge(graph, edge_spec.src, edge_spec.dst, ltype)
        
        if existing is None:
            # Add new edge if endpoints exist
            if edge_spec.src in graph.nodes and edge_spec.dst in graph.nodes:
                graph.add_edge(edge_spec.src, edge_spec.dst, ltype)
                edge = _get_edge(graph, edge_spec.src, edge_spec.dst, ltype)
                if edge:
                    edge.w = edge_spec.weight
                    edge.meta["consolidate"] = edge_spec.consolidate
                changes[edge_spec.key] = "added"
        else:
            # Update existing edge weight if changed
            if abs(existing.w - edge_spec.weight) > 0.001:
                existing.w = edge_spec.weight
                changes[edge_spec.key] = "updated"
    
    return changes


def remove_node_from_graph(graph: Graph, node_id: str) -> bool:
    """
    Remove a node and its edges from a graph.
    
    Args:
        graph: The graph to modify
        node_id: ID of node to remove
        
    Returns:
        True if removed, False if not found
    """
    if node_id not in graph.nodes:
        return False
    
    # Remove edges involving this node
    graph.edges = [
        e for e in graph.edges
        if e.src != node_id and e.dst != node_id
    ]
    
    # Update out/inc index
    keys_to_remove = [
        k for k in graph.out.keys()
        if k[0] == node_id
    ]
    for k in keys_to_remove:
        del graph.out[k]
    
    keys_to_remove = [
        k for k in graph.inc.keys()
        if k[0] == node_id
    ]
    for k in keys_to_remove:
        del graph.inc[k]
    
    # Remove node
    del graph.nodes[node_id]
    
    return True
