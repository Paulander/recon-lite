from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Callable, Any

from .core.activations import ActivationState


# This file defines the graph data structure and the core node and edge types.
# Chess-specific node types are defined in chess_nodes.py for domain separation.
# The Node class is designed to be extensible - chess nodes inherit from it.

class NodeType(Enum):
    SCRIPT = auto()
    TERMINAL = auto() #"Sensors", Leaf nodes. Could be anything from physical photo detector/thermometer to or a sophisticated AI model connected to a virtual world. 


class NodeState(Enum):
    INACTIVE = auto()
    REQUESTED = auto()
    ACTIVE = auto()         # Post-request, sending wait
    SUPPRESSED = auto()     # Por-inhibited
    WAITING = auto()
    TRUE = auto()
    CONFIRMED = auto()
    FAILED = auto()

# Sub/Sur - Hierarchy links. Sub points at children, Sur points at parent. Top down hierarchy, with corresponding back links. "Logical order"
# POR/RET - Sequence links.(successor→predecessor) encode temporal/sequential constraints.                      "Temporal/causal/real time order"

# Memory "trick" -- "Start with SUB( nodes == children), SU belongs with SU so sub/sur, sur = "on" the parent is "on top of" children. 
# Then POR (Predecessor→Successor). successor RETurns when it's done (temporal. )
# So, the graph is a tree with SUB/SUR and a DAG with POR/RET.

class LinkType(Enum):
    SUB = "sub" # subgraph - points at children ↓  (top down request)
    SUR = "sur" # parent - points at parent ↑      (bottom up confirmation)

    POR = "por" # predecessor - points at successor → (temporal/causal/real time order)
    RET = "ret" # successor - points at predecessor ← (temporal/causal/real time order)


@dataclass
class Node:
    nid: str
    ntype: NodeType
    state: NodeState = NodeState.INACTIVE
    activation: ActivationState = field(default_factory=ActivationState)  # Continuous activation per node (scalar for now).
    predicate: Optional[Callable[['Node', Any], Tuple[bool, bool]]] = None
    tick_entered: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    ltype: LinkType
    w: Any = field(default_factory=lambda: 1.0)  # Allow scalar weight without forcing numpy.
    meta: Dict[str, Any] = field(default_factory=dict)  # Metadata for trainability, subgraph, etc.

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.out: Dict[Tuple[str, LinkType], List[str]] = {}
        self.inc: Dict[Tuple[str, LinkType], List[str]] = {}
        # Single parent for scripts (1:1), but terminals can have multiple (fan-in)
        self.parent: Dict[str, Optional[str]] = {}
        # For fan-in terminals: track all parents that can query them
        self.parents_fanin: Dict[str, List[str]] = {}

    def add_node(self, node: Node):
        if node.nid in self.nodes:
            raise ValueError(f"Duplicate node id: {node.nid}")
        self.nodes[node.nid] = node
        self.parent[node.nid] = None
        # Initialize fan-in list for terminals
        if node.ntype == NodeType.TERMINAL:
            self.parents_fanin[node.nid] = []

    def add_edge(self, src: str, dst: str, ltype: LinkType):
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must be existing nodes")
        src_node = self.nodes[src]
        dst_node = self.nodes[dst]

        # --- Article compliance enforcement for link types ---
        # Terminals: can only be TARGETED by SUB, and ORIGINATE SUR
        if src_node.ntype == NodeType.TERMINAL:
            if ltype != LinkType.SUR:
                raise ValueError(
                    f"Illegal edge: terminal node '{src}' may only originate SUR links (got {ltype.name})."
                )
        if dst_node.ntype == NodeType.TERMINAL:
            if ltype != LinkType.SUB:
                raise ValueError(
                    f"Illegal edge: terminal node '{dst}' may only be targeted by SUB links (got {ltype.name})."
                )

        # POR/RET sequences must connect scripts only
        if ltype in (LinkType.POR, LinkType.RET):
            if not (src_node.ntype == NodeType.SCRIPT and dst_node.ntype == NodeType.SCRIPT):
                raise ValueError(
                    f"Illegal {ltype.name} edge: both src '{src}' and dst '{dst}' must be SCRIPT nodes."
                )

        e = Edge(src, dst, ltype)
        self.edges.append(e)
        self.out.setdefault((src, ltype), []).append(dst)
        self.inc.setdefault((dst, ltype), []).append(src)
        if ltype == LinkType.SUB:
            dst_node = self.nodes[dst]
            # Fan-in allowed for TERMINAL nodes (sensors can have multiple parents)
            # Scripts must have exactly one parent (1:1)
            if dst_node.ntype == NodeType.TERMINAL:
                # Track all parents for fan-in terminals
                if dst not in self.parents_fanin:
                    self.parents_fanin[dst] = []
                self.parents_fanin[dst].append(src)
                # Keep first parent as primary for backward compatibility
                if self.parent[dst] is None:
                    self.parent[dst] = src
            else:
                # Script nodes: enforce single parent
                if self.parent[dst] is not None:
                    raise ValueError(f"Script node {dst} already has a parent {self.parent[dst]}")
                self.parent[dst] = src

    def children(self, nid: str) -> List[str]:
        return list(self.out.get((nid, LinkType.SUB), []))

    def parent_of(self, nid: str) -> Optional[str]:
        """Return the primary parent (first parent for fan-in terminals)."""
        return self.parent.get(nid, None)

    def all_parents(self, nid: str) -> List[str]:
        """Return all parents for fan-in terminals, or [parent] for scripts."""
        if nid in self.parents_fanin and self.parents_fanin[nid]:
            return list(self.parents_fanin[nid])
        p = self.parent.get(nid)
        return [p] if p else []

    def is_fanin_terminal(self, nid: str) -> bool:
        """Check if a node is a terminal with multiple parents (fan-in)."""
        return (
            nid in self.nodes
            and self.nodes[nid].ntype == NodeType.TERMINAL
            and len(self.parents_fanin.get(nid, [])) > 1
        )

    def predecessors(self, nid: str) -> List[str]:
        return list(self.inc.get((nid, LinkType.POR), []))

    def successors(self, nid: str) -> List[str]:
        return list(self.out.get((nid, LinkType.POR), []))

    def is_last_in_sequence(self, nid: str) -> bool:
        return len(self.successors(nid)) == 0

    # --- Validation utilities ---
    def validate_article_compliance(self) -> None:
        """
        Validate core article constraints:
          - Terminal nodes only targeted by SUB, and only originate SUR
          - POR/RET edges connect scripts only
          - Every script node has at least one SUB child
        Raises ValueError on violations.
        """
        # Edge-based checks
        for e in self.edges:
            src_node = self.nodes[e.src]
            dst_node = self.nodes[e.dst]
            if src_node.ntype == NodeType.TERMINAL and e.ltype != LinkType.SUR:
                raise ValueError(
                    f"Article violation: terminal '{e.src}' originates non-SUR link {e.ltype.name} to '{e.dst}'."
                )
            if dst_node.ntype == NodeType.TERMINAL and e.ltype != LinkType.SUB:
                raise ValueError(
                    f"Article violation: terminal '{e.dst}' targeted by non-SUB link {e.ltype.name} from '{e.src}'."
                )
            if e.ltype in (LinkType.POR, LinkType.RET):
                if not (src_node.ntype == NodeType.SCRIPT and dst_node.ntype == NodeType.SCRIPT):
                    raise ValueError(
                        f"Article violation: {e.ltype.name} edge must connect scripts (got {src_node.ntype.name}->{dst_node.ntype.name})."
                    )

        # Script children check
        for nid, n in self.nodes.items():
            if n.ntype == NodeType.SCRIPT:
                if len(self.children(nid)) == 0:
                    raise ValueError(
                        f"Article violation: script node '{nid}' has no SUB children."
                    )

    # --- Helper utilities for policy configuration ---
    def set_por_policy(self, nid: str, *, policy: str, k: Optional[int] = None, theta: Optional[float] = None) -> None:
        """
        Configure predecessor gating policy for the given node (POR incoming edges):
          policy ∈ {"and","or","xor","k_of_n","weighted"}
          k      : integer threshold for k_of_n
          theta  : float threshold for weighted (sum of satisfied predecessor weights)
        """
        if nid not in self.nodes:
            raise KeyError(f"Unknown node id: {nid}")
        node = self.nodes[nid]
        node.meta["por_policy"] = str(policy).lower()
        if k is not None:
            node.meta["por_k"] = int(k)
        if theta is not None:
            node.meta["por_theta"] = float(theta)

    def set_confirm_policy(self, nid: str, *, policy: str, k: Optional[int] = None) -> None:
        """
        Configure confirmation aggregation across root child chains for a parent script:
          policy ∈ {"and","or","xor","k_of_n"}
          k      : integer threshold for k_of_n
        """
        if nid not in self.nodes:
            raise KeyError(f"Unknown node id: {nid}")
        node = self.nodes[nid]
        node.meta["confirm_policy"] = str(policy).lower()
        if k is not None:
            node.meta["confirm_k"] = int(k)

    def set_por_weight(self, src: str, dst: str, weight: float) -> None:
        """Set the weight on a POR edge (src -> dst). Creates no edges; raises if edge missing."""
        found = False
        for e in self.edges:
            if e.src == src and e.dst == dst and e.ltype == LinkType.POR:
                e.w = float(weight)
                found = True
                break
        if not found:
            raise KeyError(f"No POR edge {src} -> {dst} to set weight on")

    # -------------------------------------------------------------------------
    # Hot-reload / Dynamic topology methods
    # -------------------------------------------------------------------------

    def refresh_bindings(self, registry: "TopologyRegistry") -> Dict[str, str]:
        """
        Hot-reload topology changes from registry.
        
        - Adds new nodes that exist in registry but not in graph
        - Updates edge weights from registry
        - Does NOT remove nodes (pruning happens via separate call)
        
        Args:
            registry: TopologyRegistry with new topology
            
        Returns:
            Dict mapping node/edge IDs to their status ("added", "updated")
        
        Note: Import is done inline to avoid circular imports.
        """
        from recon_lite_chess.graph.builder import refresh_graph_from_registry
        return refresh_graph_from_registry(self, registry)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and its edges from the graph.
        
        Args:
            node_id: ID of the node to remove
            
        Returns:
            True if removed, False if not found
        """
        if node_id not in self.nodes:
            return False
        
        # Remove edges involving this node
        self.edges = [
            e for e in self.edges
            if e.src != node_id and e.dst != node_id
        ]
        
        # Update out index
        keys_to_remove = [k for k in self.out.keys() if k[0] == node_id]
        for k in keys_to_remove:
            del self.out[k]
        
        # Also remove from out lists
        for key, dsts in list(self.out.items()):
            self.out[key] = [d for d in dsts if d != node_id]
        
        # Update inc index
        keys_to_remove = [k for k in self.inc.keys() if k[0] == node_id]
        for k in keys_to_remove:
            del self.inc[k]
        
        # Also remove from inc lists
        for key, srcs in list(self.inc.items()):
            self.inc[key] = [s for s in srcs if s != node_id]
        
        # Update parent tracking
        if node_id in self.parent:
            del self.parent[node_id]
        
        # Remove from other nodes' parent references
        for nid in list(self.parent.keys()):
            if self.parent[nid] == node_id:
                self.parent[nid] = None
        
        # Update fan-in tracking
        if node_id in self.parents_fanin:
            del self.parents_fanin[node_id]
        
        for nid in list(self.parents_fanin.keys()):
            self.parents_fanin[nid] = [
                p for p in self.parents_fanin[nid] if p != node_id
            ]
        
        # Finally remove the node
        del self.nodes[node_id]
        
        return True

    def get_edge(self, src: str, dst: str, ltype: LinkType) -> Optional[Edge]:
        """Get an edge by source, destination, and type."""
        for e in self.edges:
            if e.src == src and e.dst == dst and e.ltype == ltype:
                return e
        return None

    def to_snapshot(self) -> Dict[str, Any]:
        """
        Export the full graph state to a snapshot dictionary.
        
        This captures ALL nodes and edges (including dynamically spawned packs)
        for persistence across training cycles.
        
        Returns:
            Dict with "nodes" and "edges" keys, suitable for JSON serialization.
        """
        snapshot = {"nodes": {}, "edges": {}}
        
        # Export all nodes
        for nid, node in self.nodes.items():
            node_entry = {
                "id": nid,
                "type": node.ntype.name,  # "SCRIPT" or "TERMINAL"
                "meta": node.meta.copy() if node.meta else {},
            }
            # Include factory if present in meta
            if "factory" in node.meta:
                node_entry["factory"] = node.meta["factory"]
            snapshot["nodes"][nid] = node_entry
        
        # Export all edges
        for edge in self.edges:
            edge_key = f"{edge.src}->{edge.dst}:{edge.ltype.name}"
            edge_entry = {
                "src": edge.src,
                "dst": edge.dst,
                "type": edge.ltype.name,  # "SUB", "SUR", "POR", "RET"
                "weight": edge.w if isinstance(edge.w, (int, float)) else 1.0,
            }
            # Include meta if present
            if edge.meta:
                edge_entry.update(edge.meta)
            snapshot["edges"][edge_key] = edge_entry
        
        return snapshot

    # =========================================================================
    # CONTINUOUS ACTIVATION PROPAGATION (Section 3.1)
    # =========================================================================
    
    def get_sur_children(self, nid: str) -> List[Tuple[str, float]]:
        """
        Get all nodes that send SUR (confirmation) links TO this node.
        
        Returns:
            List of (child_nid, weight) tuples
        """
        children = []
        for edge in self.edges:
            if edge.dst == nid and edge.ltype == LinkType.SUR:
                w = edge.w if hasattr(edge, 'w') else 1.0
                children.append((edge.src, w))
        return children
    
    def get_sub_children(self, nid: str) -> List[Tuple[str, float]]:
        """
        Get all nodes that are targeted by SUB links from this node.
        
        Returns:
            List of (child_nid, weight) tuples
        """
        children = []
        for edge in self.edges:
            if edge.src == nid and edge.ltype == LinkType.SUB:
                w = edge.w if hasattr(edge, 'w') else 1.0
                children.append((edge.dst, w))
        return children

    def compute_z_sur(self, nid: str) -> float:
        """
        Compute weighted sum of child activations for a node.
        
        z_i = Σ(w_ij * a_j) for all children j
        
        Args:
            nid: Node ID to compute z for
            
        Returns:
            Weighted sum of child activations
        """
        z = 0.0
        
        # Get children via SUB links (this node's sub-nodes)
        for child_nid, weight in self.get_sub_children(nid):
            child = self.nodes.get(child_nid)
            if child:
                z += weight * child.activation.value
        
        return z
    
    def propagate_activation(self, eta: float = 0.1) -> Dict[str, float]:
        """
        Single propagation step for all nodes.
        
        For each SCRIPT node with children, compute:
            z = Σ(w_ij * a_j)  [weighted sum of child activations]
            a_new = a_old + eta * k * (z - a_old)  [exponential smoothing]
        
        If a node's predicate has set meta["activation"], use that as source.
        Terminal nodes (sensors) set their own activation via predicate.
        
        Args:
            eta: Learning rate / smoothing factor (default 0.1)
            
        Returns:
            Dict of node_id -> new activation value
        """
        new_activations = {}
        
        for nid, node in self.nodes.items():
            # Check if predicate has set an activation value
            if "activation" in node.meta:
                # Predicate explicitly set activation - use it
                target = node.meta["activation"]
                node.activation.nudge(target, eta)
                new_activations[nid] = node.activation.value
            elif node.ntype == NodeType.SCRIPT:
                # Compute target from children based on aggregation mode
                children = self.get_sub_children(nid)
                aggregation = node.meta.get("aggregation", "avg")
                
                if aggregation == "and" and children:
                    # TRUE AND GATE: Uses min() - fires ONLY when ALL children active
                    child_activations = []
                    for child_id in children:
                        child = self.nodes.get(child_id)
                        if child:
                            child_activations.append(child.activation.value)
                    z = min(child_activations) if child_activations else 0.0
                else:
                    # Default: Weighted average (OR-like behavior)
                    z = self.compute_z_sur(nid)
                    if children:
                        z /= len(children)
                
                # Smooth update
                new_val = node.activation.nudge(z, eta)
                new_activations[nid] = new_val
            else:
                # Terminal nodes maintain their current activation
                new_activations[nid] = node.activation.value
        
        return new_activations
    
    def propagate_microtick(self, num_steps: int = 5, eta: float = 0.1) -> Dict[str, float]:
        """
        Run multiple propagation steps to settle activations.
        
        This implements the microtick loop that allows activations to
        propagate from terminals up through the script hierarchy.
        
        Args:
            num_steps: Number of microticks to run
            eta: Smoothing factor per step
            
        Returns:
            Final activation values
        """
        for _ in range(num_steps):
            self.propagate_activation(eta)
        
        return {nid: node.activation.value for nid, node in self.nodes.items()}
    
    def reset_activations(self, value: float = 0.0):
        """Reset all node activations to a given value."""
        for node in self.nodes.values():
            node.activation.reset(value)

