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
