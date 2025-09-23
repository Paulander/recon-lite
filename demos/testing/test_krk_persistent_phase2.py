#!/usr/bin/env python3
"""Validate that the persistent KRK demo selects shrink moves in phase 2."""

import sys
from pathlib import Path
import chess

# Make project importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.testing.test_krk_box_minimization import generate_test_positions
from demos.persistent.krk_persistent_demo import preview_decision


def _validate_persistent_shrink(board: chess.Board, idx: int) -> bool:
    result = preview_decision(board, tick_watchdog=80, target_phase="phase2")
    decision = result.get("decision")

    print(f"\nPosition {idx}: {board.fen()}")

    if not decision:
        print("✗ No decision returned by persistent demo")
        return False

    phase = decision.get("phase")
    move = decision.get("move")
    print(f"  Proposed move: {move} from {phase}")

    if phase != "phase2":
        print("✗ Expected phase2 proposer")
        return False

    metrics = decision.get("validation")
    if not metrics:
        print("✗ Missing validation metrics for phase2 move")
        return False

    failure = metrics.get("failure")
    if failure:
        print(f"✗ Validation failure: {failure}")
        return False

    initial_ms = metrics["initial_min_side"]
    worst_ms = metrics["worst_min_side"]
    if worst_ms > initial_ms:
        print("✗ Worst-case min-side regressed")
        return False
    if initial_ms > 1 and worst_ms >= initial_ms:
        print("✗ Worst-case min-side did not shrink")
        return False

    initial_area = metrics["initial_area"]
    worst_area = metrics["worst_area"]
    if initial_area > 1 and worst_area >= initial_area:
        print("✗ Worst-case area did not shrink")
        return False

    print("✓ Persistent demo proposed a safe shrink move")
    return True


def run_all() -> bool:
    positions = generate_test_positions()
    success = True
    for idx, board in enumerate(positions, 1):
        if not _validate_persistent_shrink(board, idx):
            success = False
    return success


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
