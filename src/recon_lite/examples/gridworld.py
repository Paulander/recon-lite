"""Beginner-friendly grid-world example for ReCoN execution and tracing.

The example is intentionally small: a single agent moves across a 5x5 grid
toward a goal. The interesting part is not the path planner, but the ReCoN
control loop around it:

* ``root`` requests ``sense`` and ``act``.
* ``sense`` runs a terminal predicate that observes the current grid state.
* ``act`` runs after ``sense`` confirms, because a POR edge encodes that
  temporal dependency.
* ``move_agent`` mutates the grid by taking one step toward the goal.

The same graph can run in discrete mode or continuous mode. Continuous mode
does not change the visible path in this toy example; it adds activation
settling between engine ticks so traces can show interpretable latent state.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from recon_lite import (
    ActivationMode,
    EngineConfig,
    FormalReConEngine,
    Graph,
    LinkType,
    Node,
    NodeState,
    NodeType,
    ReConEngine,
)
from recon_lite.binding.manager import BindingInstance, BindingTable


TracePath = Union[str, Path]


@dataclass
class GridState:
    """Minimal world state for the example agent."""

    width: int = 5
    height: int = 5
    agent: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)

    def signature(self) -> str:
        """Return the identity of the current state for binding invalidation."""
        return f"{self.width}x{self.height}:agent={self.agent}:goal={self.goal}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "agent": list(self.agent),
            "goal": list(self.goal),
            "signature": self.signature(),
        }

    def step_toward_goal(self) -> None:
        """Move one cell along a simple deterministic Manhattan path."""
        ax, ay = self.agent
        gx, gy = self.goal
        if ax < gx:
            ax += 1
        elif ax > gx:
            ax -= 1
        elif ay < gy:
            ay += 1
        elif ay > gy:
            ay -= 1
        self.agent = (ax, ay)


def build_graph() -> Graph:
    """Build a tiny reusable ReCoN graph for the grid-world loop."""
    graph = Graph()

    # Script nodes coordinate work. They do not touch the world directly.
    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(Node("sense", NodeType.SCRIPT))
    graph.add_node(Node("act", NodeType.SCRIPT))

    # Terminal nodes are where ReCoN meets an environment. These predicates can
    # read sensors, call tools, run models, or mutate a simulated world.
    graph.add_node(Node("goal_sensor", NodeType.TERMINAL, predicate=_sense_goal))
    graph.add_node(Node("move_agent", NodeType.TERMINAL, predicate=_move_agent))

    # SUB edges express hierarchy: root asks sense and act to do their jobs.
    graph.add_edge("root", "sense", LinkType.SUB)
    graph.add_edge("root", "act", LinkType.SUB)
    graph.add_edge("sense", "goal_sensor", LinkType.SUB)
    graph.add_edge("act", "move_agent", LinkType.SUB)

    # POR edges express ordering: act may run only after sense has confirmed.
    graph.add_edge("sense", "act", LinkType.POR)
    return graph


def build_formal_graph() -> Graph:
    """Build the same grid-world graph with explicit article-style edge pairs."""
    graph = Graph()

    graph.add_node(Node("root", NodeType.SCRIPT))
    graph.add_node(Node("sense", NodeType.SCRIPT))
    graph.add_node(Node("act", NodeType.SCRIPT))
    graph.add_node(Node("goal_sensor", NodeType.TERMINAL, predicate=_sense_goal))
    graph.add_node(Node("move_agent", NodeType.TERMINAL, predicate=_move_agent))

    graph.add_hierarchy_pair("root", "sense")
    graph.add_hierarchy_pair("root", "act")
    graph.add_hierarchy_pair("sense", "goal_sensor")
    graph.add_hierarchy_pair("act", "move_agent")
    graph.add_sequence_pair("sense", "act")
    return graph


def _sense_goal(node: Node, env: dict) -> tuple[bool, bool]:
    """Observe the grid and reserve bindings for the agent and goal cells."""
    grid: GridState = env["grid"]
    bindings: BindingTable = env["bindings"]
    reached = grid.agent == grid.goal

    # Terminal predicates may publish activation targets. Continuous mode then
    # settles neighboring activations during microticks.
    node.meta["activation"] = 1.0 if reached else 0.25

    # Bindings keep feature uses explicit. Here the features are trivial, but
    # the same pattern scales to richer environments where hypotheses should
    # not silently reuse the same object or region.
    with bindings.begin_tick("grid/sense") as session:
        session.reserve(BindingInstance("agent", {f"cell:{grid.agent[0]},{grid.agent[1]}"}, node.nid))
        session.reserve(BindingInstance("goal", {f"cell:{grid.goal[0]},{grid.goal[1]}"}, node.nid))
    return True, True


def _move_agent(node: Node, env: dict) -> tuple[bool, bool]:
    """Actuator predicate: move the agent once when requested."""
    grid: GridState = env["grid"]
    before = grid.agent
    if before != grid.goal:
        grid.step_toward_goal()
    node.meta["activation"] = 1.0 if grid.agent != before else 0.5
    return True, True


def run_simulation(
    *,
    mode: ActivationMode = ActivationMode.DISCRETE,
    seed: int = 0,
    steps: int = 10,
    microticks: int = 0,
    engine_kind: str = "formal",
    explain: bool = False,
    trace_json: Optional[TracePath] = None,
) -> List[str]:
    """Run the example and return the lines that the CLI should print."""
    random.seed(seed)
    grid = GridState()
    bindings = BindingTable()
    engine_kind = _parse_engine_kind(engine_kind)
    graph = build_formal_graph() if engine_kind == "formal" else build_graph()
    config = EngineConfig(
        activation_mode=mode,
        microtick_steps=microticks,
        record_activation_history=mode == ActivationMode.CONTINUOUS,
    )
    engine = FormalReConEngine(graph) if engine_kind == "formal" else ReConEngine(graph, config=config)
    _request_root(engine_kind, engine, graph)
    lines: List[str] = []
    trace = _new_trace(
        mode=mode,
        engine_kind=engine_kind,
        seed=seed,
        steps=steps,
        microticks=microticks,
        graph=graph,
    )

    for step_idx in range(max(0, steps)):
        signature_before = grid.signature()
        world_before = grid.to_dict()

        # Bindings are valid only for the state they were created in. When the
        # grid signature changes, the table clears stale object reservations.
        bindings_invalidated = bindings.invalidate_on_signature(signature_before)

        # Each visible grid move is an outer step. Inside it the ReCoN engine may
        # need several ticks to request children, run predicates, and confirm
        # scripts. Resetting node states makes each outer step a fresh request.
        _reset_engine(engine_kind, engine, graph)
        env = {"grid": grid, "bindings": bindings}
        tick_frames: List[Dict[str, Any]] = []

        for _ in range(20):
            frame = _step_engine(engine_kind, engine, env)
            tick_frames.append(_tick_frame(engine_kind, engine, env, frame))
            if _outer_step_complete(engine_kind, graph):
                break

        summary = _format_summary_line(
            step_idx=step_idx,
            mode=mode,
            engine_kind=engine_kind,
            grid=grid,
            bindings=bindings,
            tick_frames=tick_frames,
        )
        lines.append(summary)
        if explain:
            lines.extend(_format_explanation(grid=grid, graph=graph))

        trace["steps"].append(
            {
                "step": step_idx + 1,
                "signature_before": signature_before,
                "binding_invalidated": bindings_invalidated,
                "world_before": world_before,
                "world_after": grid.to_dict(),
                "world": grid.to_dict(),
                "bindings": bindings.snapshot(),
                "ticks": tick_frames,
            }
        )

        if grid.agent == grid.goal:
            break

    if trace_json is not None:
        _write_trace(trace_json, trace)

    return lines


def _new_trace(
    *,
    mode: ActivationMode,
    engine_kind: str,
    seed: int,
    steps: int,
    microticks: int,
    graph: Graph,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "example": "gridworld",
        "metadata": {
            "engine": engine_kind,
            "mode": mode.value,
            "seed": seed,
            "steps_requested": steps,
            "microticks": microticks,
        },
        "graph": graph.to_snapshot(),
        "steps": [],
    }


def _tick_frame(engine_kind: str, engine: Any, env: Dict[str, Any], raw_frame: Any) -> Dict[str, Any]:
    if engine_kind == "formal":
        return {
            "tick": engine.tick,
            "new_requests": [],
            "nodes": _node_states(engine.g),
            "activations": _activation_values(engine.g),
            "bindings": env["bindings"].snapshot(),
            "messages": raw_frame["messages"],
            "formal_states_before": raw_frame["states_before"],
        }

    now_requested = raw_frame
    frame: Dict[str, Any] = {
        "tick": engine.tick,
        "new_requests": sorted(nid for nid, requested in now_requested.items() if requested),
        "nodes": _node_states(engine.g),
        "activations": _activation_values(engine.g),
        "bindings": env["bindings"].snapshot(),
    }
    if "activation_history" in env:
        frame["activation_history"] = env["activation_history"]
    if "microtick_history" in env:
        frame["microtick_history"] = env["microtick_history"]
    return frame


def _node_states(graph: Graph) -> Dict[str, str]:
    return {nid: node.state.name for nid, node in graph.nodes.items()}


def _activation_values(graph: Graph) -> Dict[str, float]:
    return {nid: round(float(node.activation.value), 6) for nid, node in graph.nodes.items()}


def _format_summary_line(
    *,
    step_idx: int,
    mode: ActivationMode,
    engine_kind: str,
    grid: GridState,
    bindings: BindingTable,
    tick_frames: List[Dict[str, Any]],
) -> str:
    confirmed = sorted(
        nid
        for nid, state in tick_frames[-1]["nodes"].items()
        if state == NodeState.CONFIRMED.name
    ) if tick_frames else []
    confirmed_text = ",".join(confirmed) if confirmed else "-"
    return (
        f"step={step_idx + 1} engine={engine_kind} mode={mode.value} agent={grid.agent} goal={grid.goal} "
        f"engine_ticks={len(tick_frames)} confirmed={confirmed_text} bindings={len(bindings.snapshot())}"
    )


def _format_explanation(*, grid: GridState, graph: Graph) -> List[str]:
    lines = ["  grid:"]
    lines.extend(f"    {row}" for row in _ascii_grid(grid))
    lines.append(f"  activations: {_activation_values(graph)}")
    lines.append(f"  node_states: {_node_states(graph)}")
    return lines


def _ascii_grid(grid: GridState) -> List[str]:
    rows: List[str] = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            pos = (x, y)
            if pos == grid.agent == grid.goal:
                row.append("*")
            elif pos == grid.agent:
                row.append("A")
            elif pos == grid.goal:
                row.append("G")
            else:
                row.append(".")
        rows.append(" ".join(row))
    return rows


def _write_trace(path: TracePath, trace: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _parse_mode(value: str) -> ActivationMode:
    return ActivationMode(value)


def _parse_engine_kind(value: str) -> str:
    if value not in ("pragmatic", "formal"):
        raise ValueError(f"Unknown engine kind: {value}")
    return value


def _request_root(engine_kind: str, engine: Any, graph: Graph) -> None:
    if engine_kind == "formal":
        engine.request("root")
    else:
        graph.nodes["root"].state = NodeState.REQUESTED


def _reset_engine(engine_kind: str, engine: Any, graph: Graph) -> None:
    if engine_kind == "formal":
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
            node.tick_entered = -1
        engine.trace.clear()
        engine.tick = 0
        engine.clear_request("root")
        engine.request("root")
    else:
        engine.reset_states()
        graph.nodes["root"].state = NodeState.REQUESTED


def _step_engine(engine_kind: str, engine: Any, env: Dict[str, Any]) -> Any:
    if engine_kind == "formal":
        return engine.step(env)
    return engine.step(env)


def _outer_step_complete(engine_kind: str, graph: Graph) -> bool:
    if engine_kind == "formal":
        return graph.nodes["root"].state in (NodeState.CONFIRMED, NodeState.FAILED)
    return graph.nodes["move_agent"].state == NodeState.CONFIRMED


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ReCoN Lite grid-world example.")
    parser.add_argument("--engine", choices=["formal", "pragmatic"], default="formal")
    parser.add_argument("--mode", choices=[mode.value for mode in ActivationMode], default=ActivationMode.DISCRETE.value)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--microticks", type=int, default=0)
    parser.add_argument("--explain", "--verbose", action="store_true", help="Print grid, node states, and activations after each visible step.")
    parser.add_argument("--trace-json", help="Write a visualization-friendly JSON trace to this path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for line in run_simulation(
        mode=_parse_mode(args.mode),
        seed=args.seed,
        steps=args.steps,
        microticks=args.microticks,
        engine_kind=args.engine,
        explain=args.explain,
        trace_json=args.trace_json,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
