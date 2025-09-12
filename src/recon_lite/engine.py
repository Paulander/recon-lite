from typing import Dict, Any, List, Optional
from .graph import Graph, NodeType, NodeState


class ReConEngine:
    def __init__(self, graph: Graph):
        self.g = graph
        self.tick = 0
        self.logs: List[Dict[str, Any]] = []

    def snapshot(self, note: str = "") -> Dict[str, Any]:
        snap = {
            "tick": self.tick,
            "note": note,
            "nodes": {nid: n.state.name for nid, n in self.g.nodes.items()}
        }
        self.logs.append(snap)
        return snap

    def _all_por_predecessors_true(self, nid: str) -> bool:
        preds = self.g.predecessors(nid)
        if not preds:
            return True
        return all(self.g.nodes[p].state in (NodeState.TRUE, NodeState.CONFIRMED) for p in preds)

    def _request_child_if_ready(self, child_id: str, now_requested: Dict[str, bool]):
        child = self.g.nodes[child_id]
        if child.state == NodeState.INACTIVE and self._all_por_predecessors_true(child_id):
            child.state = NodeState.REQUESTED
            child.tick_entered = self.tick
            now_requested[child_id] = True

    def _children_confirmed_sequence_done(self, parent_id: str) -> bool:
        roots = [c for c in self.g.children(parent_id) if not self.g.predecessors(c)]
        if not roots:
            return True
        for r in roots:
            cur = r
            last = cur
            visited = set()
            while True:
                succ = self.g.successors(cur)
                visited.add(cur)
                if not succ:
                    last = cur
                    break
                cur = succ[0]
                if cur in visited:
                    break
            if self.g.nodes[last].state != NodeState.CONFIRMED:
                return False
        return True

    def step(self, env: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        self.tick += 1
        env = env or {}
        now_requested: Dict[str, bool] = {}

        # Terminals
        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.TERMINAL:
                if node.state == NodeState.REQUESTED:
                    node.state = NodeState.WAITING
                    node.tick_entered = self.tick
                elif node.state == NodeState.WAITING:
                    if node.predicate is None:
                        node.state = NodeState.TRUE
                    else:
                        done, success = node.predicate(node, env)
                        if done:
                            node.state = NodeState.TRUE if success else NodeState.FAILED
                elif node.state == NodeState.TRUE:
                    node.state = NodeState.CONFIRMED

        # Scripts
        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.SCRIPT:
                if node.state == NodeState.REQUESTED:
                    node.state = NodeState.WAITING
                    node.tick_entered = self.tick

                if node.state in (NodeState.REQUESTED, NodeState.WAITING):
                    for child_id in self.g.children(nid):
                        self._request_child_if_ready(child_id, now_requested)

                if node.state in (NodeState.REQUESTED, NodeState.WAITING, NodeState.TRUE):
                    if self._children_confirmed_sequence_done(nid):
                        node.state = NodeState.TRUE

        # Confirm scripts
        for nid, node in self.g.nodes.items():
            if node.ntype == NodeType.SCRIPT and node.state == NodeState.TRUE:
                node.state = NodeState.CONFIRMED

        return now_requested
