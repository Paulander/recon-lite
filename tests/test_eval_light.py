"""
Tests for M4 expanded heuristic evaluation.
"""

import sys
from pathlib import Path

import pytest
import chess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recon_lite_chess.eval.heuristic import (
    eval_position,
    eval_position_fast,
    eval_position_full,
    compute_reward_tick,
    _material_score,
    _king_safety_score,
    _mobility_score,
    _pawn_structure_score,
    _piece_activity_score,
    _tactical_tension_score,
    _endgame_factor,
)


class TestMaterialScore:
    def test_equal_material(self):
        board = chess.Board()  # Starting position
        score = _material_score(board)
        assert score == 0.0

    def test_white_up_pawn(self):
        # White has extra pawn
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        board.remove_piece_at(chess.A7)  # Remove black pawn
        score = _material_score(board)
        assert score == 1.0  # 1 pawn unit

    def test_black_up_queen(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
        score = _material_score(board)
        assert score == -9.0  # Queen = 9 pawns


class TestKingSafetyScore:
    def test_castled_king_safer(self):
        # King on g1 (castled) vs king on e1 (not castled)
        board_castled = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        board_castled.push_san("O-O")

        board_not_castled = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        safety_castled = _king_safety_score(board_castled)
        safety_not_castled = _king_safety_score(board_not_castled)

        # Castled should be safer (higher score for white)
        assert safety_castled > safety_not_castled


class TestMobilityScore:
    def test_starting_position(self):
        board = chess.Board()
        score = _mobility_score(board)
        # Should be roughly equal at start
        assert abs(score) < 0.5


class TestPawnStructureScore:
    def test_doubled_pawns_penalty(self):
        # Position with doubled pawns for white
        board = chess.Board("8/8/8/8/8/P7/P7/K6k w - - 0 1")
        score = _pawn_structure_score(board)
        # Doubled pawns should give negative score for white
        assert score < 0

    def test_isolated_pawn_penalty(self):
        # White has doubled isolated pawns (clear structural weakness)
        # vs black with connected pawns
        board = chess.Board("8/2p1p3/8/3P4/3P4/8/8/K6k w - - 0 1")
        score = _pawn_structure_score(board)
        # White has doubled isolated pawns, black has connected pawns
        # Doubled penalty + isolated penalty should outweigh any bonuses
        assert score < 0

    def test_passed_pawn_bonus(self):
        # Passed pawn on 6th rank
        board = chess.Board("8/3P4/8/8/8/8/8/K6k w - - 0 1")
        score = _pawn_structure_score(board)
        # Passed pawn on 7th rank should give positive score
        assert score > 0

    def test_no_pawns(self):
        board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
        score = _pawn_structure_score(board)
        assert score == 0.0


class TestPieceActivityScore:
    def test_centralized_knight(self):
        # Knight on e4 (center) vs knight on a1 (corner)
        board_center = chess.Board("8/8/8/8/4N3/8/8/K6k w - - 0 1")
        board_corner = chess.Board("N7/8/8/8/8/8/8/K6k w - - 0 1")

        score_center = _piece_activity_score(board_center)
        score_corner = _piece_activity_score(board_corner)

        assert score_center > score_corner

    def test_bishop_pair_bonus(self):
        # Two bishops vs one bishop
        board_pair = chess.Board("8/8/8/8/8/8/8/KBB4k w - - 0 1")
        board_single = chess.Board("8/8/8/8/8/8/8/KB5k w - - 0 1")

        score_pair = _piece_activity_score(board_pair)
        score_single = _piece_activity_score(board_single)

        assert score_pair > score_single

    def test_rook_on_open_file(self):
        # Rook on open file vs rook on closed file
        board_open = chess.Board("8/8/8/8/8/8/8/R3K2k w - - 0 1")
        board_closed = chess.Board("8/p7/8/8/8/8/P7/R3K2k w - - 0 1")

        score_open = _piece_activity_score(board_open)
        score_closed = _piece_activity_score(board_closed)

        assert score_open > score_closed


class TestTacticalTensionScore:
    def test_hanging_piece_penalty(self):
        # Knight attacked by pawn, not defended
        board = chess.Board("8/8/8/4p3/3N4/8/8/K6k w - - 0 1")
        board.turn = chess.WHITE  # White to move, but knight is attacked

        score = _tactical_tension_score(board)
        # Should be negative for white (hanging knight)
        assert score < 0

    def test_defended_piece_ok(self):
        # Knight defended by pawn
        board = chess.Board("8/8/8/4p3/3N4/2P5/8/K6k w - - 0 1")

        score = _tactical_tension_score(board)
        # Should be less negative than hanging piece
        board_hanging = chess.Board("8/8/8/4p3/3N4/8/8/K6k w - - 0 1")
        score_hanging = _tactical_tension_score(board_hanging)

        assert score >= score_hanging


class TestEndgameFactor:
    def test_starting_position_is_not_endgame(self):
        board = chess.Board()
        factor = _endgame_factor(board)
        assert factor == 0.0

    def test_krk_is_endgame(self):
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        factor = _endgame_factor(board)
        assert factor == 1.0

    def test_middlegame_is_partial(self):
        # Middlegame position: rooks and pawns only (material ~2600cp)
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        factor = _endgame_factor(board)
        assert 0.0 < factor < 1.0


class TestEvalPosition:
    def test_starting_position_is_equal(self):
        board = chess.Board()
        score = eval_position(board)
        assert abs(score) < 1.0  # Should be roughly equal

    def test_checkmate_for_white(self):
        # White delivers back-rank checkmate
        board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
        board.push_san("Ra8#")
        score = eval_position(board)
        assert score == 100.0  # Max positive for white win

    def test_checkmate_for_black(self):
        # Black delivers back-rank checkmate
        board = chess.Board("3r4/8/8/8/8/8/5PPP/7K b - - 0 1")
        board.push_san("Rd1#")
        score = eval_position(board)
        assert score == -100.0  # Max negative for black win

    def test_stalemate_is_zero(self):
        # Stalemate position
        board = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        score = eval_position(board)
        assert score == 0.0

    def test_fast_vs_full(self):
        board = chess.Board()
        fast = eval_position_fast(board)
        full = eval_position_full(board)

        # Both should be roughly equal for starting position
        assert abs(fast) < 1.0
        assert abs(full) < 1.0


class TestComputeRewardTick:
    def test_positive_improvement(self):
        reward = compute_reward_tick(0.0, 1.0, r_max=2.0)
        assert reward == 1.0

    def test_negative_change(self):
        reward = compute_reward_tick(1.0, 0.0, r_max=2.0)
        assert reward == -1.0

    def test_clipping_positive(self):
        reward = compute_reward_tick(0.0, 5.0, r_max=2.0)
        assert reward == 2.0

    def test_clipping_negative(self):
        reward = compute_reward_tick(5.0, 0.0, r_max=2.0)
        assert reward == -2.0

    def test_no_change(self):
        reward = compute_reward_tick(1.0, 1.0, r_max=2.0)
        assert reward == 0.0


class TestEvalManager:
    def test_heuristic_mode(self):
        from src.recon_lite_chess.eval.manager import (
            EvalManager,
            EvalConfig,
            EvalMode,
        )

        config = EvalConfig(mode=EvalMode.HEURISTIC)
        manager = EvalManager(config)

        board = chess.Board()
        result = manager.evaluate(board)

        assert result.source == EvalMode.HEURISTIC
        assert abs(result.score) < 1.0

    def test_caching(self):
        from src.recon_lite_chess.eval.manager import (
            EvalManager,
            EvalConfig,
            EvalMode,
        )

        config = EvalConfig(mode=EvalMode.HEURISTIC, cache_enabled=True)
        manager = EvalManager(config)

        board = chess.Board()

        # First call
        result1 = manager.evaluate(board)
        assert not result1.cached

        # Second call should be cached
        result2 = manager.evaluate(board)
        assert result2.cached

        stats = manager.get_stats()
        assert stats["cache_hits"] == 1

    def test_cache_disabled(self):
        from src.recon_lite_chess.eval.manager import (
            EvalManager,
            EvalConfig,
            EvalMode,
        )

        config = EvalConfig(mode=EvalMode.HEURISTIC, cache_enabled=False)
        manager = EvalManager(config)

        board = chess.Board()

        result1 = manager.evaluate(board)
        result2 = manager.evaluate(board)

        assert not result1.cached
        assert not result2.cached


class TestKRKSpecificPositions:
    """Test evaluation on KRK endgame positions."""

    def test_enemy_king_in_corner_is_good(self):
        # Enemy king in corner - test the KRK bonus directly
        from src.recon_lite_chess.eval.heuristic import _krk_specific_score

        board_corner = chess.Board("k7/8/8/8/8/8/8/R3K3 w - - 0 1")
        board_center = chess.Board("8/8/8/3k4/8/8/8/R3K3 w - - 0 1")

        # KRK-specific bonus should be higher for corner position
        krk_corner = _krk_specific_score(board_corner)
        krk_center = _krk_specific_score(board_center)

        assert krk_corner > krk_center

    def test_enemy_king_on_edge_is_good(self):
        # Enemy king on edge - test the KRK bonus directly
        from src.recon_lite_chess.eval.heuristic import _krk_specific_score

        board_edge = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        board_center = chess.Board("8/8/8/3k4/8/8/8/R3K3 w - - 0 1")

        # KRK-specific bonus should be higher for edge position
        krk_edge = _krk_specific_score(board_edge)
        krk_center = _krk_specific_score(board_center)

        assert krk_edge > krk_center

