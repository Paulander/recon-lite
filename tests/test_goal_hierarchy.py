"""Tests for M6 goal hierarchy."""

import pytest
import chess

from recon_lite_chess.goals.ultimate import (
    UltimateGoal,
    assess_ultimate_goal,
    create_ultimate_goal_node,
)
from recon_lite_chess.goals.strategic import (
    STRATEGIC_PLANS,
    get_active_plans_for_goal,
)
from recon_lite_chess.sensors.material import assess_material, MaterialCategory
from recon_lite_chess.sensors.phase import estimate_phase


class TestUltimateGoal:
    """Tests for ultimate goal assessment."""
    
    def test_winning_position_returns_win_goal(self):
        """Position with significant material advantage should return WIN."""
        # White has queen vs nothing
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        assessment = assess_ultimate_goal(board, side_to_move=chess.WHITE)
        
        assert assessment.goal == UltimateGoal.WIN
        assert assessment.confidence >= 0.6
    
    def test_losing_position_returns_survive_goal(self):
        """Position with significant material deficit should return SURVIVE."""
        # White has nothing vs queen
        board = chess.Board("4k2q/8/8/8/8/8/8/4K3 w - - 0 1")
        assessment = assess_ultimate_goal(board, side_to_move=chess.WHITE)
        
        assert assessment.goal == UltimateGoal.SURVIVE
        assert assessment.confidence >= 0.6
    
    def test_equal_position_returns_draw_goal(self):
        """Equal material position should return DRAW."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        assessment = assess_ultimate_goal(board, side_to_move=chess.WHITE)
        
        assert assessment.goal == UltimateGoal.DRAW
        assert assessment.confidence >= 0.5
    
    def test_krk_attacker_returns_win(self):
        """KRK endgame attacker should return WIN."""
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        assessment = assess_ultimate_goal(board, side_to_move=chess.WHITE)
        
        assert assessment.goal == UltimateGoal.WIN
    
    def test_krk_defender_returns_survive(self):
        """KRK endgame defender should return SURVIVE."""
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 b - - 0 1")
        assessment = assess_ultimate_goal(board, side_to_move=chess.BLACK)
        
        assert assessment.goal == UltimateGoal.SURVIVE
    
    def test_ultimate_goal_node_creation(self):
        """Test creating the terminal node."""
        node = create_ultimate_goal_node()
        
        assert node.nid == "UltimateGoal"
        assert node.predicate is not None


class TestStrategicPlans:
    """Tests for strategic plan layer."""
    
    def test_all_plans_have_required_fields(self):
        """All strategic plans should have required fields."""
        for plan_id, plan in STRATEGIC_PLANS.items():
            assert plan.id == plan_id
            assert plan.name
            assert plan.description
            assert plan.category is not None
            assert plan.base_weight > 0
    
    def test_phase_boost_adjusts_weight(self):
        """Phase boost should adjust plan weight."""
        develop = STRATEGIC_PLANS["Develop"]
        
        opening_weights = {"opening": 1.0, "middlegame": 0.0, "endgame": 0.0}
        endgame_weights = {"opening": 0.0, "middlegame": 0.0, "endgame": 1.0}
        
        opening_weight = develop.phase_adjusted_weight(opening_weights)
        endgame_weight = develop.phase_adjusted_weight(endgame_weights)
        
        # Develop should be much stronger in opening
        assert opening_weight > endgame_weight
    
    def test_get_active_plans_for_win_goal(self):
        """WIN goal should prioritize attacking plans."""
        plans = get_active_plans_for_goal("WIN")
        plan_ids = [p[0] for p in plans]
        
        assert "AttackKing" in plan_ids
        assert "WinMaterial" in plan_ids
    
    def test_get_active_plans_for_survive_goal(self):
        """SURVIVE goal should prioritize defensive plans."""
        plans = get_active_plans_for_goal("SURVIVE")
        plan_ids = [p[0] for p in plans]
        
        assert "DefendWeakness" in plan_ids


class TestMaterialSensor:
    """Tests for material sensor."""
    
    def test_starting_position_is_equal(self):
        """Starting position should be equal."""
        board = chess.Board()
        assessment = assess_material(board)
        
        assert assessment.category == MaterialCategory.EQUAL
        assert abs(assessment.balance) < 0.5
    
    def test_krk_pattern_detected(self):
        """KRK pattern should be detected."""
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        assessment = assess_material(board)
        
        assert assessment.pattern == "KRK"
        assert assessment.category == MaterialCategory.KRK
    
    def test_queen_advantage_is_kqk_pattern(self):
        """KQK position should be categorized as KQK pattern."""
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        assessment = assess_material(board)
        
        # KQK is a specific endgame pattern
        assert assessment.category == MaterialCategory.KQK
        assert assessment.pattern == "KQK"
        assert assessment.balance >= 9.0
    
    def test_large_material_advantage_is_winning(self):
        """Large material advantage (non-pattern) should be WINNING."""
        # Queen + rook vs nothing - not a recognized pattern
        board = chess.Board("4k3/8/8/8/8/8/8/R3K2Q w - - 0 1")
        assessment = assess_material(board)
        
        assert assessment.category == MaterialCategory.WINNING
        assert assessment.balance >= 9.0


class TestPhaseSensor:
    """Tests for phase sensor."""
    
    def test_starting_position_is_opening(self):
        """Starting position should be opening-weighted."""
        board = chess.Board()
        phase = estimate_phase(board)
        
        assert phase.opening > phase.endgame
        assert phase.dominant_phase() == "opening"
    
    def test_krk_is_endgame(self):
        """KRK position should be endgame-weighted."""
        board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        phase = estimate_phase(board)
        
        assert phase.endgame > phase.opening
        assert phase.dominant_phase() == "endgame"
    
    def test_phase_weights_sum_to_one(self):
        """Phase weights should approximately sum to 1."""
        board = chess.Board()
        phase = estimate_phase(board)
        
        total = phase.opening + phase.middlegame + phase.endgame
        assert 0.99 <= total <= 1.01

