"""
Macrograph spec loading utilities.

The macrograph describes the coarse, top-level ReCoN layout (game control,
phase/plan hubs, feature hubs, synthesis, learning). It is intentionally
agnostic of low-level node semantics so that sub-networks (e.g., the KRK
endgame recon) can be mounted underneath plan groups without rewriting the
overall skeleton.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class MacroNode:
    nid: str
    ntype: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroEdge:
    src: str
    dst: str
    kind: str
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroGraphSpec:
    version: str
    nodes: Dict[str, MacroNode]
    edges: List[MacroEdge]
    notes: Dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> Iterable[str]:
        return self.nodes.keys()

    def subgraph_mounts(self) -> Dict[str, str]:
        return {
            node.nid: node.meta["mount"]
            for node in self.nodes.values()
            if node.meta.get("mount")
        }


class MacrographError(RuntimeError):
    """Raised when the macrograph spec cannot be parsed or validated."""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise MacrographError(f"Macrograph spec not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MacrographError(f"Invalid JSON in macrograph spec: {path}") from exc


def load_macrograph(spec_path: str | Path) -> MacroGraphSpec:
    """
    Load the macrograph specification from disk.

    Args:
        spec_path: Path to a JSON specification (see specs/macrograph_v0.json).

    Returns:
        MacroGraphSpec with node/edge metadata ready for instantiation.
    """
    path = Path(spec_path)
    payload = _load_json(path)

    version = str(payload.get("version", "0.0"))
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])

    if not raw_nodes:
        raise MacrographError(f"Macrograph spec {path} defines no nodes.")

    nodes: Dict[str, MacroNode] = {}
    for item in raw_nodes:
        nid = item.get("id")
        ntype = item.get("type", "script")
        if not nid:
            raise MacrographError(f"Macrograph spec {path} has a node without id.")
        if nid in nodes:
            raise MacrographError(f"Duplicate macrograph node id: {nid}")
        meta = {
            key: value
            for key, value in item.items()
            if key not in {"id", "type"}
        }
        nodes[nid] = MacroNode(nid=nid, ntype=ntype, meta=meta)

    edges: List[MacroEdge] = []
    for item in raw_edges:
        src = item.get("from")
        dst = item.get("to")
        kind = item.get("kind", "sub")
        weight = float(item.get("weight", 1.0))
        if src not in nodes:
            raise MacrographError(f"Macrograph edge references unknown src '{src}'.")
        if dst not in nodes:
            raise MacrographError(f"Macrograph edge references unknown dst '{dst}'.")
        meta = {key: value for key, value in item.items() if key not in {"from", "to", "kind", "weight"}}
        edges.append(MacroEdge(src=src, dst=dst, kind=kind, weight=weight, meta=meta))

    notes = payload.get("notes", {})
    return MacroGraphSpec(version=version, nodes=nodes, edges=edges, notes=notes)


def describe_macrograph(spec: MacroGraphSpec) -> str:
    """
    Produce a human-readable summary of the macrograph specification.

    This assists with CLI inspection and sanity checks.
    """
    lines = [
        f"Macrograph version {spec.version}",
        f"Nodes ({len(spec.nodes)}):"
    ]
    for node in spec.nodes.values():
        meta_summary = ", ".join(f"{k}={v}" for k, v in node.meta.items()) or "no-meta"
        lines.append(f"  - {node.nid} [{node.ntype}] ({meta_summary})")

    lines.append(f"Edges ({len(spec.edges)}):")
    for edge in spec.edges:
        lines.append(f"  - {edge.src} --{edge.kind}/{edge.weight}--> {edge.dst}")

    if spec.notes:
        lines.append("Notes:")
        for key, val in spec.notes.items():
            lines.append(f"  * {key}: {val}")
    return "\n".join(lines)
