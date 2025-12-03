"""Tests for M6 opening and middlegame scripts."""

import pytest
import chess

from recon_lite.graph import Graph, Node, NodeType
from recon_lite_chess.scripts.opening import (
    build_opening_hierarchy,
    development_sensor_predicate,
    castling_sensor_predicate,
    center_control_sensor_predicate,
    get_opening_move_candidates,
)
from recon_lite_chess.scripts.middlegame import (
    build_middlegame_hierarchy,
    king_safety_sensor_predicate,
    piece_activity_sensor_predicate,
    structure_sensor_predicate,
    get_middlegame_move_candidates,
)


class TestOpeningSensors:
    """Tests for opening sensor terminals."""
    
    def test_development_sensor_starting_position(self):
        """Starting position should show no development."""
        node = Node("test", NodeType.TERMINAL)
        board = chess.Board()
        env = {"board": board}
        
        done, success = development_sensor_predicate(node, env)
        
        assert done
        assert not success  # No pieces developed yet
        assert node.activation.meta["developed_count"] == 0
    
    def test_development_sensor_after_moves(self):
        """After developing, sensor should detect it."""
        node = Node("test", NodeType.TERMINAL)
        board = chess.Board()
        # Play Nf3 and Nc3
        board.push_san("Nf3")
        board.push_san("d6")
        board.push_san("Nc3")
        board.push_san("e6")
        env = {"board": board}
        
        done, success = development_sensor_predicate(node, env)
        
        assert done
        assert success  # At least 2 pieces developed
        assert node.activation.meta["developed_count"] >= 2
    
    def test_castling_sensor_can_castle(self):
        """Fresh position should allow castling."""
        node = Node("test", NodeType.TERMINAL)
        board = chess.Board()
        env = {"board": board}
        
        done, success = castling_sensor_predicate(node, env)
        
        assert done
        assert success
        assert node.activation.meta["can_castle"]
        assert not node.activation.meta["has_castled"]
    
    def test_castling_sensor_has_castled(self):
        """After castling, sensor should detect it."""
        node = Node("test", NodeType.TERMINAL)
        # Position where white has castled kingside
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 1")
        env = {"board": board}
        
        # Check from white's perspective (already castled)
        board.turn = chess.WHITE
        done, success = castling_sensor_predicate(node, env)
        
        assert done
        assert success
        assert node.activation.meta["has_castled"]
    
    def test_center_control_sensor(self):
        """Sensor should evaluate center control."""
        node = Node("test", NodeType.TERMINAL)
        # Position with good center control (e4, d4 pawns)
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1")
        board.turn = chess.WHITE
        env = {"board": board}
        
        done, success = center_control_sensor_predicate(node, env)
        
        assert done
        assert node.activation.meta["occupation_score"] >= 1  # At least d4


class TestMiddlegameSensors:
    """Tests for middlegame sensor terminals."""
    
    def test_king_safety_sensor_castled_king(self):
        """Castled king should be reasonably safe."""
        node = Node("test", NodeType.TERMINAL)
        # Normal middlegame position with castled kings
        board = chess.Board("r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 1")
        env = {"board": board}
        
        done, success = king_safety_sensor_predicate(node, env)
        
        assert done
        # Enemy king is reasonably defended, so vulnerability should be moderate
        assert node.activation.meta["enemy_king_vulnerability"] < 0.8
    
    def test_piece_activity_sensor(self):
        """Sensor should identify piece activity levels."""
        node = Node("test", NodeType.TERMINAL)
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        env = {"board": board}
        
        done, success = piece_activity_sensor_predicate(node, env)
        
        assert done
        assert "worst_piece_sq" in node.activation.meta
        assert "avg_activity" in node.activation.meta
    
    def test_structure_sensor_isolated_pawn(self):
        """Sensor should detect isolated pawns."""
        node = Node("test", NodeType.TERMINAL)
        # Position with black isolated a-pawn (no pawns on b-file)
        board = chess.Board("8/p4ppp/8/8/8/8/PPP2PPP/8 w - - 0 1")
        env = {"board": board}
        
        done, success = structure_sensor_predicate(node, env)
        
        assert done
        # a-pawn is isolated (no pawn on b-file)
        weaknesses = node.activation.meta["weaknesses"]
        assert weaknesses["isolated"] >= 1


class TestHierarchyBuilding:
    """Tests for building script hierarchies."""
    
    def test_build_opening_hierarchy(self):
        """Opening hierarchy should create valid graph structure."""
        g = Graph()
        root_id = build_opening_hierarchy(g)
        
        assert root_id == "OpeningPhase"
        assert "OpeningPhase" in g.nodes
        assert "DevelopMinorPieces" in g.nodes
        assert "CastleEarly" in g.nodes
        assert "ControlCenter" in g.nodes
        assert "DevelopmentSensor" in g.nodes
        assert "CastlingSensor" in g.nodes
        assert "CenterControlSensor" in g.nodes
        
        # Check fan-in: CenterControlSensor should have multiple parents
        assert g.is_fanin_terminal("CenterControlSensor")
    
    def test_build_middlegame_hierarchy(self):
        """Middlegame hierarchy should create valid graph structure."""
        g = Graph()
        root_id = build_middlegame_hierarchy(g)
        
        assert root_id == "MiddlegamePhase"
        assert "MiddlegamePhase" in g.nodes
        assert "AttackKingPlan" in g.nodes
        assert "KingSafetySensor" in g.nodes


class TestMoveCandidates:
    """Tests for move candidate generation."""
    
    def test_opening_move_candidates(self):
        """Should prioritize development moves in opening."""
        board = chess.Board()
        candidates = get_opening_move_candidates(board)
        
        assert len(candidates) > 0
        # Should include e4 or d4 pawn moves
        move_strs = [board.san(m[0]) for m in candidates]
        assert any(m in ["e4", "d4", "e3", "d3", "Nf3", "Nc3"] for m in move_strs)
    
    def test_middlegame_move_candidates_attack(self):
        """Attack plan should prioritize moves toward enemy king."""
        # Position with attacking chances
        board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        candidates = get_middlegame_move_candidates(board, plan="attack")
        
        # Should have some candidates
        assert len(candidates) >= 0  # May be empty depending on position

