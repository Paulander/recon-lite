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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from recon_lite import Graph, LinkType, Node, NodeType
from recon_lite.graph import NodeState


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


def _resolve_weight_pack_path(spec_path: Path, entry: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        return entry_path
    for anchor in spec_path.parents:
        candidate = anchor / entry_path
        if candidate.exists():
            return candidate
    return spec_path.parent / entry_path


def _load_macro_weight_pack(spec: MacroGraphSpec, spec_path: Path) -> Optional[Dict[str, Any]]:
    packs = spec.notes.get("weight_packs") or spec.notes.get("sidecars")
    if not packs:
        return None
    for entry in packs:
        if "macro_weight_pack" in entry or "macro_weights" in entry:
            target = _resolve_weight_pack_path(spec_path, entry)
            return _load_json(target)
    return None


def _apply_macro_weight_pack(graph: Graph, payload: Mapping[str, Any]) -> None:
    por_edges = payload.get("por_edges", {})
    for key, value in por_edges.items():
        if not isinstance(key, str) or "->" not in key:
            raise MacrographError(f"Invalid POR edge key '{key}' in macro weight pack.")
        src, dst = (part.strip() for part in key.split("->", 1))
        try:
            graph.set_por_weight(src, dst, float(value))
        except (KeyError, ValueError) as exc:
            raise MacrographError(f"Failed to set POR weight for edge {src}->{dst}: {exc}") from exc

    por_policies = payload.get("por_policies", {})
    for nid, cfg in por_policies.items():
        if not isinstance(cfg, Mapping):
            raise MacrographError(f"POR policy for node '{nid}' must be a mapping.")
        policy = cfg.get("policy")
        if not policy:
            raise MacrographError(f"POR policy for node '{nid}' is missing a 'policy' entry.")
        k = cfg.get("k")
        theta = cfg.get("theta")
        try:
            graph.set_por_policy(
                nid,
                policy=policy,
                k=None if k is None else int(k),
                theta=None if theta is None else float(theta),
            )
        except KeyError as exc:
            raise MacrographError(f"POR policy references unknown node '{nid}'.") from exc

    confirm_policies = payload.get("confirm_policies", {})
    for nid, cfg in confirm_policies.items():
        if not isinstance(cfg, Mapping):
            raise MacrographError(f"Confirm policy for node '{nid}' must be a mapping.")
        policy = cfg.get("policy")
        if not policy:
            raise MacrographError(f"Confirm policy for node '{nid}' is missing a 'policy' entry.")
        k = cfg.get("k")
        try:
            graph.set_confirm_policy(
                nid,
                policy=policy,
                k=None if k is None else int(k),
            )
        except KeyError as exc:
            raise MacrographError(f"Confirm policy references unknown node '{nid}'.") from exc

    version = payload.get("version")
    if version is not None:
        setattr(graph, "macro_weight_pack_version", str(version))

# ---------------------------------------------------------------------------
# Runtime instantiation helpers

EDGE_KIND_TO_LINKTYPE: Dict[str, LinkType] = {
    "sub": LinkType.SUB,
    # For the macro skeleton we treat non-hierarchical relationships as POR edges.
    # They preserve connectivity without imposing parent constraints. Future work
    # can refine this mapping once specialised edge handling is implemented.
    "request": LinkType.POR,
    "confirm": LinkType.POR,
    "feature": LinkType.POR,
    "eval": LinkType.POR,
    "gate": LinkType.POR,
    "tune": LinkType.POR,
    "goal": LinkType.POR,
}


def _edge_kind_to_linktype(kind: str) -> LinkType:
    try:
        return EDGE_KIND_TO_LINKTYPE[kind]
    except KeyError as exc:
        raise MacrographError(f"Unsupported edge kind '{kind}' in macrograph spec.") from exc


def _copy_node(node: Node) -> Node:
    """
    Create a shallow copy of a node suitable for mounting into the macrograph.
    Activation state and tick metadata are reset; predicates and meta are retained.
    """
    return Node(
        nid=node.nid,
        ntype=node.ntype,
        predicate=node.predicate,
        meta=dict(node.meta),
    )


def _mount_subgraph(parent_graph: Graph, mount_id: str, child_graph: Graph) -> None:
    """
    Merge an existing child graph (e.g., KRK network) into the parent graph and
    attach its root to the mount node via a SUB edge.
    """
    if mount_id not in parent_graph.nodes:
        raise MacrographError(f"Mount node '{mount_id}' not present in macrograph.")

    for node in child_graph.nodes.values():
        if node.nid in parent_graph.nodes:
            raise MacrographError(f"Cannot mount subgraph; node id collision: {node.nid}")
        parent_graph.add_node(_copy_node(node))

    for edge in child_graph.edges:
        parent_graph.add_edge(edge.src, edge.dst, edge.ltype)
        parent_graph.edges[-1].w = edge.w

    # Identify candidate roots (nodes without parents in child graph). For KRK this
    # will be 'krk_root'.
    root_candidates = [nid for nid, parent in child_graph.parent.items() if parent is None]
    if not root_candidates:
        raise MacrographError("Subgraph mount requires at least one root node.")

    for root in root_candidates:
        if root not in parent_graph.nodes:
            raise MacrographError(f"Expected subgraph root '{root}' missing after merge.")
        parent_graph.add_edge(mount_id, root, LinkType.SUB)


def instantiate_macrograph(
    spec_path: str | Path,
    *,
    krk_builder: Optional[Callable[[], Graph]] = None,
    kpk_builder: Optional[Callable[[], Graph]] = None,
    mount_builders: Optional[Mapping[str, Callable[[], Graph]]] = None,
) -> Graph:
    """
    Instantiate a `Graph` from the macrograph specification, optionally mounting
    the KRK sub-network (or any builder supplied via `krk_builder`).
    """
    spec_path = Path(spec_path)
    spec = load_macrograph(spec_path)
    graph = Graph()

    # Materialise nodes. We default to SCRIPT nodes but tag macro-specific type info.
    for node in spec.nodes.values():
        macro_type = node.ntype
        ntype = NodeType.SCRIPT
        macro_node = Node(nid=node.nid, ntype=ntype, meta={"macro_type": macro_type, **node.meta})
        graph.add_node(macro_node)

    # Materialise edges with LinkType mapping; store weights.
    for edge in spec.edges:
        ltype = _edge_kind_to_linktype(edge.kind)
        graph.add_edge(edge.src, edge.dst, ltype)
        graph.edges[-1].w = edge.weight

    weight_payload = _load_macro_weight_pack(spec, spec_path)
    if weight_payload:
        _apply_macro_weight_pack(graph, weight_payload)

    # Optional subgraph mounts (KRK, KPK, ...).
    mounts = spec.subgraph_mounts()
    builder_map: Dict[str, Callable[[], Graph]] = {}
    if mount_builders:
        builder_map.update(mount_builders)
    if krk_builder:
        builder_map.setdefault("krk", krk_builder)
    if kpk_builder:
        builder_map.setdefault("kpk", kpk_builder)

    for mount_id, mount_key in mounts.items():
        builder = builder_map.get(mount_key)
        if builder is None:
            continue
        subgraph = builder()
        _mount_subgraph(graph, mount_id, subgraph)

    return graph
