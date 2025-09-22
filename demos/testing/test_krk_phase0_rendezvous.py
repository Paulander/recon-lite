#!/usr/bin/env python3
"""Phase 0 rendezvous tests for the persistent KRK demo."""

import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.persistent.krk_persistent_demo import preview_decision


PHASE0_POSITIONS = [
    chess.Board("8/2k5/8/8/8/8/6K1/R7 w - - 0 1"),
    chess.Board("3k4/8/8/8/8/6K1/8/R7 w - - 0 1"),
]


def _assert_phase0(board: chess.Board) -> bool:
    decision = preview_decision(board, target_phase="phase0")["decision"]
    if not decision:
        print("✗ No decision produced")
        return False
    if decision.get("phase") != "phase0":
        print(f"✗ Expected phase0, got {decision.get('phase')}")
        return False
    move = chess.Move.from_uci(decision["move"])
    if move not in board.legal_moves:
        print("✗ Proposed move is illegal")
        return False
    print(f"✓ Phase0 proposal {decision['move']} accepted")
    return True


def run_all() -> bool:
    ok = True
    for idx, board in enumerate(PHASE0_POSITIONS, 1):
        print(f"\nPhase0 scenario {idx}: {board.fen()}")
        if not _assert_phase0(board.copy()):
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
