#!/usr/bin/env python3
"""Integration harness for the persistent KRK demo."""

import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.persistent.krk_persistent_demo import play_persistent_game


def _clean_outputs(basename: str) -> None:
    out_dir = Path("demos/outputs")
    for suffix in ("_viz.json", "_debug.json", "_visualization.json"):
        path = out_dir / f"{basename}{suffix}"
        if path.exists():
            path.unlink()


def run_integration() -> bool:
    basename = "krk_persistent_test"
    _clean_outputs(basename)
    start_fen = "7k/6K1/8/8/8/8/R7/8 w - - 0 1"
    result = play_persistent_game(
        initial_fen=start_fen,
        max_plies=4,
        tick_watchdog=120,
        split_logs=True,
        output_basename=basename,
        skip_opponent=True,
        single_phase="phase2",
        seed=0,
    )

    viz_path = Path("demos/outputs") / f"{basename}_viz.json"
    debug_path = Path("demos/outputs") / f"{basename}_debug.json"

    ok = True
    if result.get("plies") != 4:
        print(f"✗ Expected 4 plies, got {result.get('plies')}")
        ok = False
    if result.get("rook_lost"):
        print("✗ Rook was lost during integration run")
        ok = False
    if not viz_path.exists() or not debug_path.exists():
        print("✗ Expected viz/debug log files not found")
        ok = False

    _clean_outputs(basename)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_integration() else 1)
