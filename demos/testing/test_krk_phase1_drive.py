#!/usr/bin/env python3
"""Phase 1 drive-to-edge tests for the persistent KRK demo."""

import sys
from pathlib import Path
import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.persistent.krk_persistent_demo import preview_decision
from recon_lite_chess.predicates import dist_to_edge


PHASE1_POSITIONS = [
    chess.Board("2k5/8/R7/6K1/8/8/8/8 w - - 0 1"),
    chess.Board("3k4/8/1R6/6K1/8/8/8/8 w - - 0 1"),
]


def _assert_phase1(board: chess.Board) -> bool:
    decision = preview_decision(board, target_phase="phase1")["decision"]
    if not decision:
        print("✗ No decision produced")
        return False
    if decision.get("phase") != "phase1":
        print(f"✗ Expected phase1, got {decision.get('phase')}")
        return False
    move = chess.Move.from_uci(decision["move"])
    if move not in board.legal_moves:
        print("✗ Illegal move suggested")
        return False
    bk_before = dist_to_edge(board.king(chess.BLACK))
    board_after = board.copy()
    board_after.push(move)
    bk_after = dist_to_edge(board_after.king(chess.BLACK))
    if bk_after > bk_before:
        print("✗ Enemy king moved away from edge")
        return False
    print(f"✓ Phase1 proposal {decision['move']} accepted (dist {bk_before}->{bk_after})")
    return True


def run_all() -> bool:
    ok = True
    for idx, board in enumerate(PHASE1_POSITIONS, 1):
        print(f"\nPhase1 scenario {idx}: {board.fen()}")
        if not _assert_phase1(board.copy()):
            ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
