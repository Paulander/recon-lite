"""
Lightweight graph snapshot exporter for visualization.

Exports nodes/edges and basic metadata (type, layer, subgraph, weight) to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recon_lite import Graph


def export_graph_snapshot(graph: Graph, path: Path, *, meta: dict[str, Any] | None = None) -> None:
    """
    Save a minimal snapshot of the graph.

    Args:
        graph: ReCoN Graph to export
        path: Output JSON path
        meta: Optional metadata to include (e.g., {"game": i})
    """
    data = {
        "meta": meta or {},
        "nodes": [],
        "edges": [],
    }

    for nid, node in graph.nodes.items():
        data["nodes"].append({
            "id": nid,
            "type": getattr(node, "ntype", None).name if hasattr(node, "ntype") else None,
            "layer": node.meta.get("layer") if getattr(node, "meta", None) else None,
            "subgraph": node.meta.get("subgraph") if getattr(node, "meta", None) else None,
        })

    for edge in graph.edges:
        data["edges"].append({
            "src": edge.src,
            "dst": edge.dst,
            "type": edge.ltype.name if hasattr(edge, "ltype") else None,
            "w": float(getattr(edge, "w", 1.0) or 1.0),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


__all__ = ["export_graph_snapshot"]
