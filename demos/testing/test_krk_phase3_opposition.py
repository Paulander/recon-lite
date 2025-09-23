#!/usr/bin/env python3
"""Phase 3 opposition tests for the persistent KRK demo."""

import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.persistent.krk_persistent_demo import preview_decision
PHASE3_POSITIONS = [
    chess.Board("7k/6R1/5K2/8/8/8/8/8 w - - 0 1"),
    chess.Board("7k/5R2/6K1/8/8/8/8/8 w - - 0 1"),
]


def _assert_phase3(board: chess.Board) -> bool:
    decision = preview_decision(board, target_phase="phase3")["decision"]
    if not decision:
        print("✗ No decision produced")
        return False
    if decision.get("phase") != "phase3":
        print(f"✗ Expected phase3, got {decision.get('phase')}")
        return False
    move = chess.Move.from_uci(decision["move"])
    if move not in board.legal_moves:
        print("✗ Illegal move suggested")
        return False
    print(f"✓ Phase3 proposal {decision['move']} establishes opposition")
    return True


def run_all() -> bool:
    ok = True
    for idx, board in enumerate(PHASE3_POSITIONS, 1):
        print(f"\nPhase3 scenario {idx}: {board.fen()}")
        if not _assert_phase3(board.copy()):
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
