#!/usr/bin/env python3
"""Phase 4 mate-in-one tests for the persistent KRK demo."""

import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.persistent.krk_persistent_demo import preview_decision


PHASE4_POSITIONS = [
    chess.Board("7k/6R1/6K1/8/8/8/8/8 w - - 0 1"),
    chess.Board("6k1/8/6K1/8/5R2/8/8/8 w - - 0 1"),
]


def _assert_phase4(board: chess.Board) -> bool:
    decision = preview_decision(board, target_phase="phase4")["decision"]
    if not decision:
        print("✗ No decision produced")
        return False
    if decision.get("phase") != "phase4":
        print(f"✗ Expected phase4, got {decision.get('phase')}")
        return False
    move = chess.Move.from_uci(decision["move"])
    if move not in board.legal_moves:
        print("✗ Illegal mate move")
        return False
    print(f"✓ Phase4 proposal {decision['move']} accepted")
    return True


def run_all() -> bool:
    ok = True
    for idx, board in enumerate(PHASE4_POSITIONS, 1):
        print(f"\nPhase4 scenario {idx}: {board.fen()}")
        if not _assert_phase4(board.copy()):
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
