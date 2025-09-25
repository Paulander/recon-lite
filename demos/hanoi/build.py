"""Utilities for constructing a Hanoi ReCon graph."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:  # Prefer the canonical recon.* imports, fall back if unavailable locally.
    from recon.graph import Graph, Node, NodeType, NodeState, LinkType  # type: ignore
    from recon.engine import ReConEngine  # type: ignore
except ImportError:  # pragma: no cover - fallback for local development
    from recon_lite.graph import Graph, Node, NodeType, NodeState, LinkType
    from recon_lite.engine import ReConEngine

from .env import Hanoi

MovePlan = List[Tuple[str, Tuple[int, int]]]
Predicate = Callable[[Node, Dict[str, object]], Tuple[bool, bool]]


def plan_hanoi(n: int, src: int = 0, aux: int = 1, dst: int = 2) -> MovePlan:
    """Return a flat list of Hanoi moves described as ("MOVE", (src, dst))."""
    if n <= 0:
        return []
    if n == 1:
        return [("MOVE", (src, dst))]
    plan: MovePlan = []
    plan.extend(plan_hanoi(n - 1, src, dst, aux))
    plan.append(("MOVE", (src, dst)))
    plan.extend(plan_hanoi(n - 1, aux, src, dst))
    return plan


def make_move_predicate(src: int, dst: int) -> Predicate:
    """Create a terminal predicate that moves one disc when first evaluated."""

    def predicate(node: Node, env: Dict[str, object]) -> Tuple[bool, bool]:
        hanoi = env["hanoi"]  # type: ignore[index]
        assert isinstance(hanoi, Hanoi)
        if not node.meta.get("executed"):
            success = hanoi.move(src, dst)
            node.meta["executed"] = True
            node.meta["move"] = (src, dst)
            node.meta["success"] = success
            node.meta["after_state"] = [peg.copy() for peg in hanoi.pegs]
        return True, bool(node.meta.get("success", False))

    return predicate


def build_graph(n: int) -> Tuple[Graph, ReConEngine]:
    """Construct the Hanoi execution graph and accompanying engine."""
    graph = Graph()
    root = Node("ROOT", NodeType.SCRIPT)
    graph.add_node(root)

    plan = plan_hanoi(n)
    previous_script: str | None = None
    for idx, (_, (src, dst)) in enumerate(plan, start=1):
        script_id = f"S{idx:03d}"
        terminal_id = f"{script_id}_MOVE"
        script = Node(script_id, NodeType.SCRIPT)
        terminal = Node(terminal_id, NodeType.TERMINAL, predicate=make_move_predicate(src, dst))
        graph.add_node(script)
        graph.add_node(terminal)
        graph.add_edge("ROOT", script_id, LinkType.SUB)
        graph.add_edge(script_id, terminal_id, LinkType.SUB)
        if previous_script is not None:
            graph.add_edge(previous_script, script_id, LinkType.POR)
        previous_script = script_id

    root.state = NodeState.REQUESTED

    engine = ReConEngine(graph)
    engine.env = {"hanoi": Hanoi(n)}  # type: ignore[attr-defined]
    return graph, engine
