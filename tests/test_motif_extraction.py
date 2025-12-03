"""Tests for M5.1 motif extraction module."""

import json
import tempfile
from pathlib import Path

import chess
import pytest

from recon_lite.motifs.descriptors import (
    BindingDescriptor,
    MotifDataset,
    MotifType,
)
from recon_lite.motifs.extractors import (
    extract_3x3_patch,
    extract_king_zone,
    extract_pawn_chain,
    extract_hanging_pieces,
    extract_tactical_features,
    extract_all_features,
)


class TestBindingDescriptor:
    """Tests for BindingDescriptor dataclass."""
    
    def test_create_descriptor(self):
        """Test creating a basic descriptor."""
        desc = BindingDescriptor(
            dtype=MotifType.TACTICAL.value,
            pattern_key="fork_knight",
            context={"test": "value"},
            active_nodes=["node1", "node2"],
            outcome_score=0.5,
            source_tick=10,
            source_episode="ep_001",
        )
        
        assert desc.dtype == "tactical"
        assert desc.pattern_key == "fork_knight"
        assert desc.outcome_score == 0.5
        assert len(desc.active_nodes) == 2
    
    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        desc = BindingDescriptor(
            dtype=MotifType.ENDGAME.value,
            pattern_key="opposition",
            context={"king_dist": 2},
            active_nodes=["krk_root"],
            outcome_score=1.0,
            source_tick=5,
            source_episode="ep_002",
            fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            confidence=0.8,
        )
        
        d = desc.to_dict()
        restored = BindingDescriptor.from_dict(d)
        
        assert restored.dtype == desc.dtype
        assert restored.pattern_key == desc.pattern_key
        assert restored.fen == desc.fen
        assert restored.confidence == desc.confidence
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        desc = BindingDescriptor(
            dtype=MotifType.STRUCTURAL.value,
            pattern_key="passed_pawn",
            context={},
            active_nodes=[],
            outcome_score=0.3,
            source_tick=1,
            source_episode="test",
        )
        
        json_str = desc.to_json()
        restored = BindingDescriptor.from_json(json_str)
        
        assert restored.pattern_key == "passed_pawn"


class TestMotifDataset:
    """Tests for MotifDataset class."""
    
    def test_add_and_filter(self):
        """Test adding motifs and filtering."""
        dataset = MotifDataset()
        
        dataset.add(BindingDescriptor(
            dtype=MotifType.TACTICAL.value,
            pattern_key="fork",
            context={},
            active_nodes=[],
            outcome_score=1.0,
            source_tick=1,
            source_episode="ep1",
        ))
        dataset.add(BindingDescriptor(
            dtype=MotifType.ENDGAME.value,
            pattern_key="opposition",
            context={},
            active_nodes=[],
            outcome_score=-0.5,
            source_tick=2,
            source_episode="ep1",
        ))
        dataset.add(BindingDescriptor(
            dtype=MotifType.TACTICAL.value,
            pattern_key="pin",
            context={},
            active_nodes=[],
            outcome_score=0.8,
            source_tick=3,
            source_episode="ep2",
        ))
        
        assert len(dataset) == 3
        assert len(dataset.filter_by_type("tactical")) == 2
        assert len(dataset.filter_by_type("endgame")) == 1
        assert len(dataset.filter_by_outcome(0.5)) == 2
    
    def test_group_by_pattern(self):
        """Test grouping by pattern."""
        dataset = MotifDataset()
        
        for i in range(5):
            dataset.add(BindingDescriptor(
                dtype=MotifType.TACTICAL.value,
                pattern_key="fork" if i % 2 == 0 else "pin",
                context={},
                active_nodes=[],
                outcome_score=float(i),
                source_tick=i,
                source_episode="ep",
            ))
        
        groups = dataset.group_by_pattern()
        assert len(groups["fork"]) == 3
        assert len(groups["pin"]) == 2
    
    def test_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = MotifDataset()
        dataset.metadata["test"] = "value"
        
        dataset.add(BindingDescriptor(
            dtype=MotifType.TACTICAL.value,
            pattern_key="test_pattern",
            context={"key": "value"},
            active_nodes=["node1"],
            outcome_score=0.5,
            source_tick=10,
            source_episode="ep",
        ))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_motifs.jsonl"
            dataset.save(path)
            
            loaded = MotifDataset.load(path)
            
            assert len(loaded) == 1
            assert loaded.metadata.get("test") == "value"
            assert loaded.motifs[0].pattern_key == "test_pattern"
    
    def test_statistics(self):
        """Test statistics computation."""
        dataset = MotifDataset()
        
        dataset.add(BindingDescriptor(
            dtype=MotifType.TACTICAL.value,
            pattern_key="fork",
            context={},
            active_nodes=[],
            outcome_score=1.0,
            source_tick=1,
            source_episode="ep",
        ))
        dataset.add(BindingDescriptor(
            dtype=MotifType.TACTICAL.value,
            pattern_key="fork",
            context={},
            active_nodes=[],
            outcome_score=-0.5,
            source_tick=2,
            source_episode="ep",
        ))
        
        stats = dataset.statistics()
        
        assert stats["count"] == 2
        assert stats["avg_outcome"] == 0.25
        assert stats["positive_ratio"] == 0.5


class TestExtract3x3Patch:
    """Tests for 3x3 patch extraction."""
    
    def test_center_patch(self):
        """Test extracting patch from center of board."""
        board = chess.Board()
        patch = extract_3x3_patch(board, chess.E4)
        
        assert patch["center"] == "e4"
        assert "pieces" in patch
        assert len(patch["pieces"]) == 9
    
    def test_corner_patch(self):
        """Test extracting patch from corner (handles edges)."""
        board = chess.Board()
        patch = extract_3x3_patch(board, chess.A1)
        
        assert patch["center"] == "a1"
        # Should have pieces for valid squares only


class TestExtractKingZone:
    """Tests for king zone extraction."""
    
    def test_starting_position_white(self):
        """Test white king zone in starting position."""
        board = chess.Board()
        zone = extract_king_zone(board, chess.WHITE)
        
        assert zone["king_square"] == "e1"
        assert zone["pawn_shield"] >= 0
        assert isinstance(zone["zone_squares"], list)
    
    def test_castled_king(self):
        """Test king zone after castling."""
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        board.push_san("O-O")
        
        zone = extract_king_zone(board, chess.WHITE)
        
        assert zone["king_square"] == "g1"


class TestExtractPawnChain:
    """Tests for pawn structure extraction."""
    
    def test_starting_pawns(self):
        """Test pawn structure in starting position."""
        board = chess.Board()
        pawn = extract_pawn_chain(board)
        
        assert len(pawn["white_pawns"]) == 8
        assert len(pawn["black_pawns"]) == 8
        assert len(pawn["isolated_pawns"]) == 0
        assert pawn["white_pawn_islands"] == 1
    
    def test_isolated_pawn(self):
        """Test detecting isolated pawn."""
        # White pawn on d4 with no adjacent pawns
        board = chess.Board("8/8/8/8/3P4/8/8/4K2k w - - 0 1")
        pawn = extract_pawn_chain(board)
        
        assert "d4" in pawn["isolated_pawns"]
    
    def test_passed_pawn(self):
        """Test detecting passed pawn."""
        # White pawn on e5 with no black pawns that can stop it
        board = chess.Board("8/8/8/4P3/8/8/8/4K2k w - - 0 1")
        pawn = extract_pawn_chain(board)
        
        assert "e5" in pawn["passed_pawns"]


class TestExtractHangingPieces:
    """Tests for hanging piece detection."""
    
    def test_no_hanging_start(self):
        """Test no hanging pieces in starting position."""
        board = chess.Board()
        hanging = extract_hanging_pieces(board)
        
        # All pieces are defended in starting position
        assert len(hanging["white_en_prise"]) == 0
        assert len(hanging["black_en_prise"]) == 0
    
    def test_hanging_piece(self):
        """Test detecting a hanging piece."""
        # Simple position: black queen on d5 is completely undefended
        board = chess.Board("4k3/8/8/3q4/8/8/8/4K3 w - - 0 1")
        
        hanging = extract_hanging_pieces(board)
        
        # The queen on d5 is undefended (hanging)
        assert "d5" in hanging["black_hanging"]


class TestExtractTacticalFeatures:
    """Tests for tactical feature extraction."""
    
    def test_basic_tactical(self):
        """Test basic tactical extraction."""
        board = chess.Board()
        tactical = extract_tactical_features(board)
        
        assert "potential_forks" in tactical
        assert "pins" in tactical
        assert "checks_available" in tactical
    
    def test_pinned_piece(self):
        """Test detecting a pinned piece."""
        # Position where black knight on e4 is pinned by white rook on e1
        # The knight is between the rook (e1) and the black king (e8)
        board = chess.Board("4k3/8/8/8/4n3/8/8/4RK2 w - - 0 1")
        tactical = extract_tactical_features(board)
        
        # e4 knight should be detected as pinned (along e-file to king on e8)
        assert "e4" in tactical["pins"]


class TestExtractAllFeatures:
    """Tests for combined feature extraction."""
    
    def test_extract_all(self):
        """Test extracting all features."""
        board = chess.Board()
        features = extract_all_features(board)
        
        assert "king_zone_white" in features
        assert "king_zone_black" in features
        assert "pawn_structure" in features
        assert "hanging_pieces" in features
        assert "tactical" in features
        assert "material" in features
        assert features["turn"] == "white"
    
    def test_material_count(self):
        """Test material counting."""
        board = chess.Board()
        features = extract_all_features(board)
        
        # Starting position: 8 pawns + 2 knights (6) + 2 bishops (6) + 2 rooks (10) + queen (9) = 39
        assert features["material"]["white"] == 39
        assert features["material"]["black"] == 39
        assert features["material"]["diff"] == 0

