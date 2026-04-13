"""Article-style symbolic ReCoN message-passing executor.

``FormalReConEngine`` is deliberately separate from ``ReConEngine``. The
existing engine remains the pragmatic, high-level executor used by the
grid-world example. This module implements the explicit SUB/SUR/POR/RET message
semantics from the Bach/Herger ReCoN state-machine description.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

from .graph import Graph, LinkType, Node, NodeState, NodeType


class FormalMessage(Enum):
    """Messages emitted by the symbolic ReCoN state machine."""

    REQUEST = "request"
    WAIT = "wait"
    CONFIRM = "confirm"
    FAIL = "fail"
    INHIBIT_REQUEST = "inhibit_request"
    INHIBIT_CONFIRM = "inhibit_confirm"


@dataclass(frozen=True)
class EdgeMessage:
    """One message sent over one graph edge during a formal tick."""

    tick: int
    src: str
    dst: str
    link_type: LinkType
    message: FormalMessage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "src": self.src,
            "dst": self.dst,
            "link_type": self.link_type.name,
            "message": self.message.value,
        }


class FormalReConEngine:
    """
    Explicit symbolic ReCoN executor.

    Each ``step`` is a two-phase update:

    1. emit messages from every node based on its state at tick start
    2. group incoming messages by target and compute all next states

    This mirrors the message-passing definition. The compact neural equations
    are a separate implementation milestone and are not mixed into this class.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        validate_pairs: bool = True,
    ) -> None:
        if validate_pairs:
            graph.validate_formal_pairs()
        self.g = graph
        self.tick = 0
        self.trace: List[Dict[str, Any]] = []
        self._external_requests = set()

    def request(self, nid: str) -> None:
        """Request validation of a root script."""
        if nid not in self.g.nodes:
            raise KeyError(f"Unknown node id: {nid}")
        self._external_requests.add(nid)
        node = self.g.nodes[nid]
        if node.state == NodeState.INACTIVE:
            node.state = NodeState.REQUESTED
            node.tick_entered = self.tick

    def clear_request(self, nid: str) -> None:
        """Remove an external root request."""
        self._external_requests.discard(nid)

    def step(self, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run one formal two-phase tick and return its trace frame."""
        env = env or {}
        self.tick += 1
        states_before = self._node_states()
        messages = self._emit_messages()
        incoming = self._group_by_target(messages)
        next_states = {
            nid: self._next_state(node, incoming.get(nid, []), env)
            for nid, node in self.g.nodes.items()
        }

        for nid, state in next_states.items():
            node = self.g.nodes[nid]
            if node.state != state:
                node.state = state
                node.tick_entered = self.tick

        frame = {
            "tick": self.tick,
            "states_before": states_before,
            "messages": [message.to_dict() for message in messages],
            "states_after": self._node_states(),
            "activations": {
                nid: round(float(node.activation.value), 6)
                for nid, node in self.g.nodes.items()
            },
        }
        self.trace.append(frame)
        return frame

    def run(
        self,
        *,
        max_ticks: int = 32,
        env: Optional[Dict[str, Any]] = None,
        until: Optional[Callable[["FormalReConEngine"], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Run formal ticks until ``max_ticks`` or an optional stop predicate."""
        for _ in range(max(0, max_ticks)):
            self.step(env)
            if until is not None and until(self):
                break
        return self.trace

    def to_trace(self, *, name: str = "formal-recon") -> Dict[str, Any]:
        """Return a visualizer-neutral JSON-ready formal trace."""
        return {
            "schema_version": 1,
            "example": name,
            "engine": "FormalReConEngine",
            "graph": self.g.to_snapshot(),
            "frames": list(self.trace),
        }

    def _emit_messages(self) -> List[EdgeMessage]:
        messages: List[EdgeMessage] = []
        for edge in self.g.edges:
            message = self._message_for(edge.ltype, self.g.nodes[edge.src].state)
            if message is not None:
                messages.append(
                    EdgeMessage(
                        tick=self.tick,
                        src=edge.src,
                        dst=edge.dst,
                        link_type=edge.ltype,
                        message=message,
                    )
                )
        return messages

    def _message_for(self, link_type: LinkType, state: NodeState) -> Optional[FormalMessage]:
        if state in (NodeState.REQUESTED, NodeState.ACTIVE, NodeState.SUPPRESSED, NodeState.WAITING, NodeState.FAILED):
            if link_type == LinkType.POR:
                return FormalMessage.INHIBIT_REQUEST
        if state in (
            NodeState.REQUESTED,
            NodeState.ACTIVE,
            NodeState.SUPPRESSED,
            NodeState.WAITING,
            NodeState.TRUE,
            NodeState.CONFIRMED,
            NodeState.FAILED,
        ):
            if link_type == LinkType.RET:
                return FormalMessage.INHIBIT_CONFIRM
        if state in (NodeState.ACTIVE, NodeState.WAITING):
            if link_type == LinkType.SUB:
                return FormalMessage.REQUEST
        if state in (NodeState.REQUESTED, NodeState.ACTIVE, NodeState.WAITING):
            if link_type == LinkType.SUR:
                return FormalMessage.WAIT
        if state == NodeState.CONFIRMED and link_type == LinkType.SUR:
            return FormalMessage.CONFIRM
        if state == NodeState.FAILED and link_type == LinkType.SUR:
            return FormalMessage.FAIL
        return None

    def _group_by_target(self, messages: Iterable[EdgeMessage]) -> Dict[str, List[EdgeMessage]]:
        grouped: Dict[str, List[EdgeMessage]] = {}
        for message in messages:
            grouped.setdefault(message.dst, []).append(message)
        return grouped

    def _next_state(
        self,
        node: Node,
        incoming: List[EdgeMessage],
        env: Dict[str, Any],
    ) -> NodeState:
        request = self._has(incoming, FormalMessage.REQUEST) or node.nid in self._external_requests
        inhibit_request = self._has(incoming, FormalMessage.INHIBIT_REQUEST)
        inhibit_confirm = self._has(incoming, FormalMessage.INHIBIT_CONFIRM)
        confirm = self._has(incoming, FormalMessage.CONFIRM)
        fail = self._has(incoming, FormalMessage.FAIL)
        wait = self._has(incoming, FormalMessage.WAIT)

        if node.ntype == NodeType.TERMINAL:
            return self._next_terminal_state(node, request, inhibit_request, env)

        if node.state == NodeState.INACTIVE:
            return NodeState.REQUESTED if request else NodeState.INACTIVE
        if node.state == NodeState.REQUESTED:
            return NodeState.SUPPRESSED if inhibit_request else NodeState.ACTIVE
        if node.state == NodeState.SUPPRESSED:
            if not request:
                return NodeState.INACTIVE
            return NodeState.SUPPRESSED if inhibit_request else NodeState.REQUESTED
        if node.state == NodeState.ACTIVE:
            return NodeState.WAITING
        if node.state == NodeState.WAITING:
            if confirm:
                return NodeState.TRUE
            if fail and not wait:
                return NodeState.FAILED
            return NodeState.WAITING
        if node.state == NodeState.TRUE:
            return NodeState.TRUE if inhibit_confirm else NodeState.CONFIRMED
        return node.state

    def _next_terminal_state(
        self,
        node: Node,
        request: bool,
        inhibit_request: bool,
        env: Dict[str, Any],
    ) -> NodeState:
        if node.state == NodeState.INACTIVE:
            return NodeState.REQUESTED if request else NodeState.INACTIVE
        if node.state == NodeState.REQUESTED:
            return NodeState.SUPPRESSED if inhibit_request else NodeState.ACTIVE
        if node.state == NodeState.SUPPRESSED:
            if not request:
                return NodeState.INACTIVE
            return NodeState.SUPPRESSED if inhibit_request else NodeState.REQUESTED
        if node.state in (NodeState.ACTIVE, NodeState.WAITING):
            return self._evaluate_terminal(node, env)
        if node.state == NodeState.TRUE:
            return NodeState.CONFIRMED
        return node.state

    def _evaluate_terminal(self, node: Node, env: Dict[str, Any]) -> NodeState:
        if node.predicate is None:
            return NodeState.TRUE
        try:
            env["__graph__"] = self.g
            done, success = node.predicate(node, env)
        except Exception:
            return NodeState.FAILED
        if not done:
            return NodeState.WAITING
        return NodeState.TRUE if success else NodeState.FAILED

    def _node_states(self) -> Dict[str, str]:
        return {nid: node.state.name for nid, node in self.g.nodes.items()}

    def _has(self, incoming: List[EdgeMessage], message: FormalMessage) -> bool:
        return any(edge_message.message == message for edge_message in incoming)
