#!/usr/bin/env python3
"""Synthetic dataset generator for the mock_v2 visualization demo.

This script produces a large, topology-aware network with scripted
and terminal nodes, along with a sequence of frames (and optional
deltas) that capture activity over a series of ticks.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

# Available runtime states for nodes.
STATES: Sequence[str] = (
    "INACTIVE",
    "REQUESTED",
    "WAITING",
    "TRUE",
    "CONFIRMED",
    "FAILED",
)

# Default qualitative styles and notes to keep playback varied.
STYLES = [
    "SHARP",
    "BALANCED",
    "SOLID",
    "GAMBIT",
    "RESOURCEFUL",
    "TECHNICAL",
]

NOTES = [
    "Focusing search budget on critical lines.",
    "Reinforcing safety rails before escalation.",
    "Leaning on positional heuristics for stability.",
    "Conferring with endgame tablebase expert.",
    "Routing aggressive tactic burst against defender.",
    "Waiting on arbiter confirmation for risky line.",
    "Adapting plan based on watcher feedback.",
    "Tracking new request saturation for this phase.",
]

THOUGHTS = [
    "Phase blend feels healthy.",
    "Need fresher tactical ideas.",
    "Watcher flagged elevated risk budget.",
    "Terminal backlog within tolerance.",
    "True confirmations stalling, revisit plan.",
    "Arbiter sync smooth across experts.",
]


class NodeBuilder:
    """Utility for assembling nodes grouped by drawing layers."""

    def __init__(self) -> None:
        self.nodes: List[Dict] = []
        self.layers: Dict[int, List[Dict]] = {}

    def add(self, node_id: str, label: str, node_type: str, layer: int) -> Dict:
        node = {"id": node_id, "label": label, "type": node_type, "_layer": layer}
        self.nodes.append(node)
        self.layers.setdefault(layer, []).append(node)
        return node

    def assign_positions(
        self,
        *,
        base_radius: float = 14.0,
        radius_step: float = 18.0,
        height_step: float = 12.0,
        rng: random.Random,
    ) -> None:
        """Populate the `pos` attribute of every node."""
        for layer, nodes in sorted(self.layers.items()):
            count = len(nodes)
            if count == 0:
                continue

            # Grow radius so densely populated rings stay readable.
            radius = (
                base_radius
                + radius_step * layer
                + math.log(count + 1, 1.5) * 4.0
            )
            y = layer * height_step

            if count == 1:
                nodes[0]["pos"] = [0.0, y, 0.0]
                continue

            # Apply a small ring rotation so overlaps shift between layers.
            rotation = rng.random() * math.pi * 2.0
            for idx, node in enumerate(nodes):
                angle = (idx / count) * math.tau + rotation
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                node["pos"] = [
                    round(x, 3),
                    round(y, 3),
                    round(z, 3),
                ]

        # Remove the internal helper attribute.
        for node in self.nodes:
            node.pop("_layer", None)


def build_topology(
    *,
    tactic_instances: int,
    expert_clones: int,
    rng: random.Random,
) -> Tuple[Dict, List[str]]:
    """Create the topology dictionary plus convenience lists of node ids."""
    builder = NodeBuilder()
    edges: List[Dict] = []

    def add_edge(src: str, dst: str, edge_type: str = "sub", label: Optional[str] = None) -> None:
        entry = {"from": src, "to": dst, "type": edge_type}
        if label:
            entry["label"] = label
        edges.append(entry)

    # Spine scripts (layer 0).
    spine_ids: List[str] = []
    spine_order = [
        ("script.root", "ROOT", "SCRIPT"),
        ("script.phase_router", "PHASE_ROUTER", "SCRIPT"),
        ("script.expert_switch", "EXPERT_SWITCH", "SCRIPT"),
        ("script.arbiter", "ARBITER", "SCRIPT"),
        ("script.wait_loop", "WAIT_LOOP", "SCRIPT"),
    ]
    for idx, (node_id, label, node_type) in enumerate(spine_order):
        builder.add(node_id, label, node_type, layer=0)
        spine_ids.append(node_id)
        if idx > 0:
            add_edge(spine_ids[idx - 1], node_id, "sub")

    # Phase scripts (layer 1).
    phase_names = ["OPENING", "MIDDLEGAME", "ENDGAME", "CRITICAL"]
    phase_ids: List[str] = []
    for name in phase_names:
        node_id = f"script.phase.{name.lower()}"
        builder.add(node_id, f"PHASE_{name}", "SCRIPT", layer=1)
        phase_ids.append(node_id)
        add_edge("script.phase_router", node_id, "por")
        add_edge(node_id, "script.expert_switch", "sub", label="handoff")

    # Experts (layer 2).
    expert_bases = [
        "BOOK",
        "TACTICS",
        "POSITIONAL",
        "SEARCH_LITE",
        "ENDGAME_TB",
        "MIXTURE",
    ]
    expert_ids: List[str] = []
    for base in expert_bases:
        clones = max(1, expert_clones)
        for clone_idx in range(1, clones + 1):
            node_id = f"script.expert.{base.lower()}.{clone_idx:02d}"
            label = f"{base}_EXPERT_{clone_idx}"
            builder.add(node_id, label, "SCRIPT", layer=2)
            expert_ids.append(node_id)
            add_edge("script.expert_switch", node_id, "sub")
            add_edge(node_id, "script.arbiter", "sub", label="proposal")

    # Watchers (layer 3).
    watcher_bases = [
        "REPETITION",
        "FIFTY_MOVE",
        "RISK_BUDGET",
        "PLAN_COMMIT",
        "STYLE_PROFILE",
    ]
    watcher_ids: List[str] = []
    for base in watcher_bases:
        node_id = f"terminal.watcher.{base.lower()}"
        builder.add(node_id, f"WATCH_{base}", "TERMINAL", layer=3)
        watcher_ids.append(node_id)
        add_edge("script.wait_loop", node_id, "sub")
        add_edge(node_id, "script.phase_router", "por", label="feedback")

    # Safety terminals (layer 3).
    safety_bases = ["SAFETY_GUARD", "SANITY_CHECK", "FAILSAFE"]
    safety_ids: List[str] = []
    for base in safety_bases:
        node_id = f"terminal.safety.{base.lower()}"
        builder.add(node_id, base, "TERMINAL", layer=3)
        safety_ids.append(node_id)
        add_edge("script.arbiter", node_id, "sub")
        add_edge(node_id, "script.wait_loop", "sub", label="reset")

    # Endgame themes (layer 4).
    endgame_bases = [
        "ZUGZWANG",
        "OPPOSITION",
        "TRIANGULATION",
        "BRIDGE",
        "LUCENA",
        "PHILIDOR",
    ]
    endgame_ids: List[str] = []
    for base in endgame_bases:
        node_id = f"terminal.endgame.{base.lower()}"
        builder.add(node_id, f"ENDGAME_{base}", "TERMINAL", layer=4)
        endgame_ids.append(node_id)
        parent = rng.choice(expert_ids)
        add_edge(parent, node_id, "sub")

    # Tactic terminals (layer 5).
    tactic_bases = [
        "FORK",
        "PIN",
        "SKEWER",
        "XRAY",
        "OVERLOAD",
        "DEFLECTION",
        "DISCOVERED",
        "ZWISCHENZUG",
        "TRAPPED_PIECE",
        "BACK_RANK",
        "SMOTHERED",
        "DOUBLE_ATTACK",
        "ATTRACTION",
    ]

    tactic_ids: List[str] = []
    per_base = max(1, tactic_instances)
    for base in tactic_bases:
        for idx in range(1, per_base + 1):
            node_id = f"terminal.tactic.{base.lower()}.{idx:03d}"
            label = f"{base}_{idx:03d}"
            builder.add(node_id, label, "TERMINAL", layer=5)
            tactic_ids.append(node_id)
            parent = rng.choice(expert_ids)
            add_edge(parent, node_id, "sub")

    all_ids = (
        spine_ids
        + phase_ids
        + expert_ids
        + watcher_ids
        + safety_ids
        + endgame_ids
        + tactic_ids
    )

    builder.assign_positions(rng=rng)
    topology = {"nodes": builder.nodes, "edges": edges}
    return topology, all_ids


def pick_phase_weights(
    phase_names: Sequence[str],
    rng: random.Random,
) -> Dict[str, float]:
    """Create a normalized phase weight dictionary."""
    raw = [rng.random() ** 1.2 + 0.05 for _ in phase_names]
    total = sum(raw)
    weights = [value / total for value in raw]
    return {
        phase.lower(): round(weight, 3)
        for phase, weight in zip(phase_names, weights)
    }


def evolve_states(
    node_ids: Sequence[str],
    *,
    ticks: int,
    rng: random.Random,
    include_deltas: bool,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """Generate the frames (and optional deltas) for the timeline."""
    states = {node_id: "INACTIVE" for node_id in node_ids}
    phase_names = ["OPENING", "MIDDLEGAME", "ENDGAME", "CRITICAL"]
    frames: List[Dict] = []
    deltas: List[Dict] = []
    previous_states = states.copy()

    for tick in range(ticks):
        frame_note = rng.choice(NOTES)
        frame_thought = rng.choice(THOUGHTS)
        style = rng.choice(STYLES)
        risk = round(min(0.95, max(0.05, rng.gauss(0.45, 0.2))), 3)

        inactive_nodes = [node_id for node_id, state in states.items() if state == "INACTIVE"]
        request_count = min(len(inactive_nodes), rng.randint(4, 12))
        new_requests = rng.sample(inactive_nodes, request_count) if request_count else []
        for node_id in new_requests:
            states[node_id] = "REQUESTED"

        # Progress existing nodes.
        for node_id, current in list(states.items()):
            roll = rng.random()
            if current == "REQUESTED":
                if roll < 0.5:
                    states[node_id] = "WAITING"
                elif roll < 0.6:
                    states[node_id] = "FAILED"
            elif current == "WAITING":
                if roll < 0.4:
                    states[node_id] = "TRUE"
                elif roll < 0.5:
                    states[node_id] = "FAILED"
            elif current == "TRUE":
                if roll < 0.5:
                    states[node_id] = "CONFIRMED"
                elif roll < 0.55:
                    states[node_id] = "FAILED"
            elif current == "CONFIRMED":
                if roll < 0.15:
                    states[node_id] = "INACTIVE"
            elif current == "FAILED":
                if roll < 0.6:
                    states[node_id] = "INACTIVE"

        # Occasionally boost a watcher to demonstrate cross-layer activity.
        if rng.random() < 0.4:
            watcher_candidates = [node_id for node_id in node_ids if node_id.startswith("terminal.watcher.")]
            if watcher_candidates:
                boosted = rng.choice(watcher_candidates)
                states[boosted] = rng.choice(("WAITING", "TRUE", "CONFIRMED"))

        env = {
            "style": style,
            "risk": risk,
            "phase_weights": pick_phase_weights(phase_names, rng),
        }
        frame_states = dict(states)

        frame_entry = {
            "tick": tick,
            "note": frame_note,
            "thoughts": frame_thought,
            "env": env,
            "states": frame_states,
        }
        if new_requests:
            frame_entry["new_requests"] = new_requests
        frames.append(frame_entry)

        if include_deltas:
            changes = {
                node_id: state
                for node_id, state in frame_states.items()
                if previous_states.get(node_id) != state
            }
            if changes or new_requests or frame_note or frame_thought:
                delta_entry = {
                    "tick": tick,
                    "changes": changes,
                }
                if new_requests:
                    delta_entry["new_requests"] = new_requests
                if frame_note:
                    delta_entry["note"] = frame_note
                if frame_thought:
                    delta_entry["thoughts"] = frame_thought
                deltas.append(delta_entry)
            previous_states = frame_states

    return frames, deltas if include_deltas else None


def write_json(path: pathlib.Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate topology + frames data for the mock_v2 visualization.",
    )
    parser.add_argument(
        "--tactic-instances",
        type=int,
        default=8,
        help="Number of instances per tactic terminal (default: 8).",
    )
    parser.add_argument(
        "--expert-clones",
        type=int,
        default=2,
        help="Number of clones per expert archetype (default: 2).",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=24,
        help="Number of timeline ticks to emit (default: 24).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic output.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Path to the combined dataset JSON (topology + frames).",
    )
    parser.add_argument(
        "--topology-out",
        type=pathlib.Path,
        default=None,
        help="Optional standalone topology JSON output path.",
    )
    parser.add_argument(
        "--frames-out",
        type=pathlib.Path,
        default=None,
        help="Optional standalone frames JSON output path.",
    )
    parser.add_argument(
        "--deltas-out",
        type=pathlib.Path,
        default=None,
        help="Optional standalone deltas JSON output path (requires --include-deltas).",
    )
    parser.add_argument(
        "--include-deltas",
        action="store_true",
        help="Emit a deltas array capturing state changes between frames.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write JSON without indentation (useful for very large datasets).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    topology, all_node_ids = build_topology(
        tactic_instances=args.tactic_instances,
        expert_clones=args.expert_clones,
        rng=rng,
    )
    frames, deltas = evolve_states(
        all_node_ids,
        ticks=args.ticks,
        rng=rng,
        include_deltas=args.include_deltas,
    )

    dataset = {
        "topology": topology,
        "frames": frames,
    }
    if deltas is not None:
        dataset["deltas"] = deltas

    indent = None if args.compact else 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=indent)

    if args.topology_out:
        write_json(args.topology_out, topology)
    if args.frames_out:
        write_json(args.frames_out, {"frames": frames})
    if args.deltas_out and deltas is not None:
        write_json(args.deltas_out, {"deltas": deltas})


if __name__ == "__main__":
    main()
