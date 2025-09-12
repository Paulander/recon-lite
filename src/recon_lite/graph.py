from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Callable, Any


class NodeType(Enum):
    SCRIPT = auto()
    TERMINAL = auto()


class NodeState(Enum):
    INACTIVE = auto()
    REQUESTED = auto()
    WAITING = auto()
    TRUE = auto()
    CONFIRMED = auto()
    FAILED = auto()


class LinkType(Enum):
    SUB = "sub"
    POR = "por"


@dataclass
class Node:
    nid: str
    ntype: NodeType
    state: NodeState = NodeState.INACTIVE
    predicate: Optional[Callable[['Node', Any], Tuple[bool, bool]]] = None
    tick_entered: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    ltype: LinkType


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
