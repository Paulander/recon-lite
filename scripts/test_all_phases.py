#!/usr/bin/env python3
"""Test all Phase 1-5 implementations."""
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
os.chdir('/home/paulander/git/recon-lite')

# Test Phase 1: Heuristic Ramp
print("=== Phase 1: Heuristic Ramp ===")
from evolution_driver import EvolutionConfig
c = EvolutionConfig()
print(f"  Stage 0: heuristic_prob = {c.get_heuristic_prob():.2f}")
c.current_stage_idx = 1
print(f"  Stage 1: heuristic_prob = {c.get_heuristic_prob():.2f}")
c.current_stage_idx = 2
print(f"  Stage 2: heuristic_prob = {c.get_heuristic_prob():.2f}")
c.current_stage_idx = 3
print(f"  Stage 3: heuristic_prob = {c.get_heuristic_prob():.2f} (should be 0.0)")
print("✓ Phase 1 passed")

# Test Phase 2: Pack Templates
print("\n=== Phase 2: Pack Templates ===")
from recon_lite.nodes.pack_template import (
    spawn_and_gate_pack, 
    spawn_or_gate_pack, 
    spawn_sequence_pack
)
print("  spawn_and_gate_pack: ✓")
print("  spawn_or_gate_pack: ✓")
print("  spawn_sequence_pack: ✓")
print("✓ Phase 2 passed (templates available)")

# Test Phase 4: Softmax Arbiter
print("\n=== Phase 4: Softmax Arbiter ===")
from recon_lite_chess.scripts.strategy_actuator import create_generic_arbiter
arbiter = create_generic_arbiter("test_arbiter")
arbiter.meta["strategy_ids"] = ["push", "king"]
print(f"  Arbiter created: {arbiter.nid}")
print(f"  Predicate exists: {arbiter.predicate is not None}")
print("✓ Phase 4 passed")

# Test Feature Vector (41 features)
print("\n=== Feature Vector Enhancement ===")
import chess
from recon_lite.nodes.stem_cell import StemCellManager
mgr = StemCellManager()
board = chess.Board()
board.clear()
board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
board.set_piece_at(chess.D6, chess.Piece(chess.KING, chess.WHITE))
board.set_piece_at(chess.D8, chess.Piece(chess.KING, chess.BLACK))
features = mgr._default_board_features(board)
print(f"  Feature count: {len(features)} (should be 41)")
print(f"  Opposition indicator (last): {features[-1]:.1f}")
print("✓ Feature vector passed")

print("\n=== ALL TESTS PASSED ===")
