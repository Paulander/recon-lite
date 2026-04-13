"""Generate a tiny formal ReCoN trace for the static HTML viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from recon_lite import FormalReConEngine, Graph, Node, NodeState, NodeType


TracePath = Union[str, Path]


def _terminal_success(node: Node, _env: Dict[str, Any]) -> tuple[bool, bool]:
    node.meta["activation"] = 1.0
    return True, True


def build_sequence_graph() -> Graph:
    """Build root -> A -> B -> C with explicit SUB/SUR and POR/RET pairs."""
    graph = Graph()
    for nid in ["root", "A", "B", "C"]:
        graph.add_node(Node(nid, NodeType.SCRIPT))
    for nid in ["A_done", "B_done", "C_done"]:
        graph.add_node(Node(nid, NodeType.TERMINAL, predicate=_terminal_success))

    for child in ["A", "B", "C"]:
        graph.add_hierarchy_pair("root", child)
        graph.add_hierarchy_pair(child, f"{child}_done")
    graph.add_sequence_pair("A", "B")
    graph.add_sequence_pair("B", "C")
    return graph


def generate_trace(*, max_ticks: int = 32) -> Dict[str, Any]:
    graph = build_sequence_graph()
    engine = FormalReConEngine(graph)
    engine.request("root")
    engine.run(
        max_ticks=max_ticks,
        until=lambda formal: formal.g.nodes["root"].state == NodeState.CONFIRMED,
    )
    trace = engine.to_trace(name="formal-sequence")
    trace["metadata"] = {
        "description": "root requests A, B, C; POR/RET makes them execute as A then B then C",
        "max_ticks": max_ticks,
        "final_root_state": graph.nodes["root"].state.name,
    }
    return trace


def write_trace(path: TracePath, trace: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a formal ReCoN sequence trace.")
    parser.add_argument("--trace-json", required=True, help="Write formal trace JSON to this path.")
    parser.add_argument("--max-ticks", type=int, default=32)
    args = parser.parse_args(list(argv) if argv is not None else None)

    trace = generate_trace(max_ticks=args.max_ticks)
    write_trace(args.trace_json, trace)
    print(
        "formal_trace "
        f"frames={len(trace['frames'])} "
        f"root={trace['metadata']['final_root_state']} "
        f"path={args.trace_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
