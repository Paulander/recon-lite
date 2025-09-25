"""CLI runner for the Tower of Hanoi ReCon demo."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from recon.graph import NodeState  # type: ignore
except ImportError:  # pragma: no cover
    from recon_lite.graph import NodeState

from .build import build_graph
from .env import Hanoi


def _expected_moves(n: int) -> int:
    return (1 << n) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tower of Hanoi ReCon demo.")
    parser.add_argument("--n", type=int, default=3, help="Number of discs to solve for (default: 3)")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Optional tick interval for emitting engine snapshots (0 disables)",
    )
    args = parser.parse_args()

    graph, engine = build_graph(args.n)
    env: Dict[str, object] = getattr(engine, "env", {"hanoi": Hanoi(args.n)})
    hanoi = env["hanoi"]
    assert isinstance(hanoi, Hanoi)

    expected = _expected_moves(args.n)
    safety_limit = max(expected * 4, 16)

    print("Initial state:")
    print(hanoi)
    print()

    root = graph.nodes["ROOT"]
    while engine.tick < safety_limit and root.state not in (NodeState.CONFIRMED, NodeState.FAILED):
        engine.step(env)
        if args.log_every > 0 and engine.tick % args.log_every == 0:
            engine.snapshot(note=f"tick {engine.tick}")
        print(f"Tick {engine.tick:03d}")
        print(hanoi)
        print()

    print("Final state:")
    print(hanoi)
    print()

    print(f"Goal reached: {hanoi.is_goal()}")
    print(f"Moves executed: {hanoi.moves}")
    print(f"Expected moves: {expected}")
    print(f"Root state: {root.state.name}")


if __name__ == "__main__":
    main()
