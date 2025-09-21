from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np  # For activations


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
    activation: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # a ∈ R^n - activation function. Can implement boolean AND/OR or continuous functions.
    predicate: Optional[Callable[['Node', Any], Tuple[bool, bool]]] = None
    tick_entered: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    ltype: LinkType
    w: np.ndarray = field(default_factory=lambda: np.array([1.0]))  # Weights \in R^n; default scalar 1.0. 

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.out: Dict[Tuple[str, LinkType], List[str]] = {}
        self.inc: Dict[Tuple[str, LinkType], List[str]] = {}
        self.parent: Dict[str, Optional[str]] = {}

    def add_node(self, node: Node):
        if node.nid in self.nodes:
            raise ValueError(f"Duplicate node id: {node.nid}")
        self.nodes[node.nid] = node
        self.parent[node.nid] = None

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
            if self.parent[dst] is not None:
                raise ValueError(f"Node {dst} already has a parent {self.parent[dst]}")
            self.parent[dst] = src

    def children(self, nid: str) -> List[str]:
        return list(self.out.get((nid, LinkType.SUB), []))

    def parent_of(self, nid: str) -> Optional[str]:
        return self.parent.get(nid, None)

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
