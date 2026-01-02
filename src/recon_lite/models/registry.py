"""Topology Registry for Dynamic Graph Loading.

Manages node/edge registration from topology.json files, enabling
dynamic graph construction and hot-reloading during training.

Usage:
    from recon_lite.models.registry import TopologyRegistry
    
    registry = TopologyRegistry(Path("topologies/kpk_topology.json"))
    
    # Add a new node
    registry.add_node({
        "id": "SC_Opposition_42",
        "type": "TERMINAL",
        "group": "stem_promoted",
        "factory": "recon_lite.nodes.stem_cell:create_pattern_sensor",
        "pattern_signature": [0.1, 0.3, ...],
    })
    
    # Persist changes
    registry.save()
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class NodeSpec:
    """Specification for a node in the topology."""
    id: str
    type: str  # "SCRIPT" or "TERMINAL"
    group: str
    factory: Optional[str] = None
    pattern_signature: Optional[List[float]] = None
    weight_source: Optional[str] = None  # stem cell ID if promoted
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type,
            "group": self.group,
            "factory": self.factory,
            "meta": self.meta,
        }
        if self.pattern_signature is not None:
            d["pattern_signature"] = self.pattern_signature
        if self.weight_source is not None:
            d["weight_source"] = self.weight_source
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeSpec":
        return cls(
            id=data["id"],
            type=data["type"],
            group=data.get("group", "generic"),
            factory=data.get("factory"),
            pattern_signature=data.get("pattern_signature"),
            weight_source=data.get("weight_source"),
            meta=data.get("meta", {}),
        )


@dataclass
class EdgeSpec:
    """Specification for an edge in the topology."""
    src: str
    dst: str
    type: str  # "SUB", "POR", "SUR", "RET"
    weight: float = 1.0
    consolidate: bool = True
    confirmation_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "weight": self.weight,
            "consolidate": self.consolidate,
            "confirmation_count": self.confirmation_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeSpec":
        return cls(
            src=data["src"],
            dst=data["dst"],
            type=data["type"],
            weight=data.get("weight", 1.0),
            consolidate=data.get("consolidate", True),
            confirmation_count=data.get("confirmation_count", 0),
            meta=data.get("meta", {}),
        )
    
    @property
    def key(self) -> str:
        """Canonical edge key."""
        return f"{self.src}->{self.dst}:{self.type}"


@dataclass
class EvolutionEvent:
    """Record of a structural change."""
    tick: int
    event_type: str  # "node_added", "node_removed", "edge_added", "edge_removed", "edge_weight_updated"
    target_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "event_type": self.event_type,
            "target_id": self.target_id,
            "details": self.details,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionEvent":
        return cls(
            tick=data["tick"],
            event_type=data["event_type"],
            target_id=data["target_id"],
            details=data.get("details", {}),
            timestamp=data.get("timestamp", ""),
        )


class TopologyRegistry:
    """Manages dynamic node/edge registration from topology.json."""
    
    def __init__(self, topology_path: Path):
        self.path = Path(topology_path)
        self.data = self._load()
        self._factory_cache: Dict[str, Callable] = {}
        self._nodes: Dict[str, NodeSpec] = {}
        self._edges: Dict[str, EdgeSpec] = {}
        self._parse_data()
    
    def _load(self) -> Dict[str, Any]:
        """Load topology from JSON."""
        if not self.path.exists():
            return {
                "version": "1.0",
                "network": "unknown",
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "nodes": [],
                "edges": [],
                "stem_cells": [],
                "promoted_nodes": [],
                "evolution_history": [],
            }
        with open(self.path) as f:
            return json.load(f)
    
    def _parse_data(self):
        """Parse loaded data into NodeSpec and EdgeSpec objects."""
        self._nodes.clear()
        self._edges.clear()
        
        for node_data in self.data.get("nodes", []):
            spec = NodeSpec.from_dict(node_data)
            self._nodes[spec.id] = spec
        
        for edge_data in self.data.get("edges", []):
            spec = EdgeSpec.from_dict(edge_data)
            self._edges[spec.key] = spec
    
    def save(self):
        """Persist current topology to JSON."""
        # Update data from specs
        self.data["nodes"] = [n.to_dict() for n in self._nodes.values()]
        self.data["edges"] = [e.to_dict() for e in self._edges.values()]
        self.data["last_modified"] = datetime.now().isoformat()
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    # -------------------------------------------------------------------------
    # Node operations
    # -------------------------------------------------------------------------
    
    def add_node(self, node_spec: Dict[str, Any], tick: int = 0) -> str:
        """
        Add a new node specification.
        
        Args:
            node_spec: Node specification dict
            tick: Current tick for evolution logging
            
        Returns:
            Node ID
        """
        spec = NodeSpec.from_dict(node_spec)
        if spec.id in self._nodes:
            raise ValueError(f"Node {spec.id} already exists")
        
        self._nodes[spec.id] = spec
        
        # Log evolution event
        self._log_evolution(tick, "node_added", spec.id, {
            "type": spec.type,
            "group": spec.group,
            "factory": spec.factory,
        })
        
        return spec.id
    
    def remove_node(self, node_id: str, tick: int = 0):
        """Remove a node (for pruning)."""
        if node_id not in self._nodes:
            return
        
        spec = self._nodes.pop(node_id)
        
        # Remove related edges
        edges_to_remove = [
            key for key, edge in self._edges.items()
            if edge.src == node_id or edge.dst == node_id
        ]
        for key in edges_to_remove:
            del self._edges[key]
        
        self._log_evolution(tick, "node_removed", node_id, {
            "type": spec.type,
            "group": spec.group,
        })
    
    def get_node(self, node_id: str) -> Optional[NodeSpec]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_nodes_by_group(self, group: str) -> List[NodeSpec]:
        """Get all nodes in a group."""
        return [n for n in self._nodes.values() if n.group == group]
    
    def get_all_node_ids(self) -> List[str]:
        """Get all node IDs."""
        return list(self._nodes.keys())
    
    # -------------------------------------------------------------------------
    # Edge operations
    # -------------------------------------------------------------------------
    
    def add_edge(
        self,
        src: str,
        dst: str,
        ltype: str,
        weight: float = 1.0,
        consolidate: bool = True,
        tick: int = 0,
    ) -> str:
        """Add a new edge."""
        spec = EdgeSpec(
            src=src,
            dst=dst,
            type=ltype,
            weight=weight,
            consolidate=consolidate,
        )
        
        if spec.key in self._edges:
            raise ValueError(f"Edge {spec.key} already exists")
        
        self._edges[spec.key] = spec
        
        self._log_evolution(tick, "edge_added", spec.key, {
            "src": src,
            "dst": dst,
            "type": ltype,
            "weight": weight,
        })
        
        return spec.key
    
    def remove_edge(self, src: str, dst: str, ltype: str, tick: int = 0):
        """Remove an edge."""
        key = f"{src}->{dst}:{ltype}"
        if key not in self._edges:
            return
        
        del self._edges[key]
        
        self._log_evolution(tick, "edge_removed", key, {
            "src": src,
            "dst": dst,
            "type": ltype,
        })
    
    def update_edge_weight(self, src: str, dst: str, ltype: str, weight: float, tick: int = 0):
        """Update edge weight."""
        key = f"{src}->{dst}:{ltype}"
        if key not in self._edges:
            return
        
        old_weight = self._edges[key].weight
        self._edges[key].weight = weight
        
        self._log_evolution(tick, "edge_weight_updated", key, {
            "old_weight": old_weight,
            "new_weight": weight,
        })
    
    def get_edge(self, src: str, dst: str, ltype: str) -> Optional[EdgeSpec]:
        """Get an edge by (src, dst, type)."""
        key = f"{src}->{dst}:{ltype}"
        return self._edges.get(key)
    
    def get_all_edges(self) -> List[EdgeSpec]:
        """Get all edges."""
        return list(self._edges.values())
    
    def get_edges_from(self, src: str) -> List[EdgeSpec]:
        """Get all edges originating from a node."""
        return [e for e in self._edges.values() if e.src == src]
    
    def get_edges_to(self, dst: str) -> List[EdgeSpec]:
        """Get all edges targeting a node."""
        return [e for e in self._edges.values() if e.dst == dst]
    
    # -------------------------------------------------------------------------
    # Factory resolution
    # -------------------------------------------------------------------------
    
    def get_node_factory(self, node_id: str) -> Optional[Callable]:
        """
        Resolve factory function for a node.
        
        Factory strings are in format: "module.path:function_name"
        """
        node = self._nodes.get(node_id)
        if not node or not node.factory:
            return None
        
        if node.factory in self._factory_cache:
            return self._factory_cache[node.factory]
        
        try:
            module_path, func_name = node.factory.rsplit(":", 1)
            module = importlib.import_module(module_path)
            factory = getattr(module, func_name)
            self._factory_cache[node.factory] = factory
            return factory
        except (ValueError, ImportError, AttributeError) as e:
            print(f"Warning: Could not resolve factory {node.factory}: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Stem cell tracking
    # -------------------------------------------------------------------------
    
    def register_stem_cell(self, cell_id: str, state: str, sample_count: int):
        """Register or update a stem cell in the topology."""
        cells = self.data.setdefault("stem_cells", [])
        
        # Update existing or add new
        for cell in cells:
            if cell["cell_id"] == cell_id:
                cell["state"] = state
                cell["samples"] = sample_count
                return
        
        cells.append({
            "cell_id": cell_id,
            "state": state,
            "samples": sample_count,
        })
    
    def remove_stem_cell(self, cell_id: str):
        """Remove a stem cell from tracking."""
        cells = self.data.get("stem_cells", [])
        self.data["stem_cells"] = [c for c in cells if c["cell_id"] != cell_id]
    
    def get_stem_cells(self) -> List[Dict[str, Any]]:
        """Get all tracked stem cells."""
        return self.data.get("stem_cells", [])
    
    def record_promotion(
        self,
        cell_id: str,
        new_node_id: str,
        parent_id: str,
        tick: int,
        pattern_signature: Optional[List[float]] = None,
    ):
        """Record a stem cell promotion."""
        promotions = self.data.setdefault("promoted_nodes", [])
        promotions.append({
            "cell_id": cell_id,
            "node_id": new_node_id,
            "parent_id": parent_id,
            "tick": tick,
            "pattern_signature": pattern_signature,
            "human_label": None,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Remove from stem cells
        self.remove_stem_cell(cell_id)
    
    def get_promotions(self) -> List[Dict[str, Any]]:
        """Get all promotion records."""
        return self.data.get("promoted_nodes", [])
    
    def set_human_label(self, node_id: str, label: str):
        """Set human-provided label for a promoted node."""
        for promo in self.data.get("promoted_nodes", []):
            if promo["node_id"] == node_id:
                promo["human_label"] = label
                break
        
        # Also update the node meta
        if node_id in self._nodes:
            self._nodes[node_id].meta["human_label"] = label
    
    # -------------------------------------------------------------------------
    # Evolution history
    # -------------------------------------------------------------------------
    
    def _log_evolution(self, tick: int, event_type: str, target_id: str, details: Dict[str, Any]):
        """Log an evolution event."""
        history = self.data.setdefault("evolution_history", [])
        event = EvolutionEvent(
            tick=tick,
            event_type=event_type,
            target_id=target_id,
            details=details,
        )
        history.append(event.to_dict())
    
    def get_evolution_history(self) -> List[EvolutionEvent]:
        """Get all evolution events."""
        return [
            EvolutionEvent.from_dict(e)
            for e in self.data.get("evolution_history", [])
        ]
    
    # -------------------------------------------------------------------------
    # Snapshot support
    # -------------------------------------------------------------------------
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current topology for comparison."""
        return {
            "nodes": {n.id: n.to_dict() for n in self._nodes.values()},
            "edges": {e.key: e.to_dict() for e in self._edges.values()},
            "timestamp": datetime.now().isoformat(),
        }
    
    @property
    def network_name(self) -> str:
        """Get the network name."""
        return self.data.get("network", "unknown")
    
    @property
    def version(self) -> str:
        """Get the topology version."""
        return self.data.get("version", "1.0")
