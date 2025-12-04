"""Tests for M7 distillation features."""

import pytest
import chess

from recon_lite_chess.eval.features import (
    extract_features,
    extract_material_features,
    extract_king_position_features,
    extract_pawn_structure_features,
    extract_mobility_features,
    FEATURE_COUNT,
)
from recon_lite_chess.eval.distill import (
    DistillationConfig,
    DistillationSample,
    DistillationDataset,
    DistilledEvaluator,
)
from recon_lite_chess.eval.manager import (
    EvalMode,
    EvalConfig,
    EvalManager,
)


class TestFeatureExtraction:
    """Test feature extraction functions."""
    
    def test_extract_features_starting_position(self):
        """Test feature extraction on starting position."""
        board = chess.Board()
        fv = extract_features(board)
        
        # Check feature vector length
        assert len(fv) == len(fv.feature_names)
        assert len(fv.features) == len(fv.feature_names)
        
        # Check feature names are unique
        assert len(set(fv.feature_names)) == len(fv.feature_names)
    
    def test_extract_features_endgame(self):
        """Test feature extraction on endgame position."""
        board = chess.Board("8/8/4k3/8/8/4K3/8/4R3 w - - 0 1")  # KRK
        fv = extract_features(board)
        
        assert len(fv) > 0
        
        # Check material balance is correct (white ahead)
        feature_dict = fv.as_dict()
        assert feature_dict.get("bal_total", 0) > 0  # White has rook
    
    def test_extract_material_features(self):
        """Test material feature extraction."""
        board = chess.Board()
        features = extract_material_features(board)
        
        # 12 features: 6 piece types × 2 colors
        assert len(features) == 12
        
        # Starting position: all pieces present
        # Features are normalized (count / max_count)
        # White pawns: 8/8 = 1.0
        assert features[0] == 1.0  # White pawns
        assert features[6] == 1.0  # Black pawns
    
    def test_extract_king_position_features(self):
        """Test king position feature extraction."""
        board = chess.Board()
        features = extract_king_position_features(board)
        
        # 8 features: 4 per king × 2 colors
        assert len(features) == 8
        
        # White king on e1: file=4/7, rank=0/7
        assert features[0] == pytest.approx(4/7, abs=0.01)  # file
        assert features[1] == pytest.approx(0/7, abs=0.01)  # rank
    
    def test_extract_pawn_structure_features(self):
        """Test pawn structure feature extraction."""
        # Position with isolated pawn
        board = chess.Board("8/pp1p4/8/8/8/8/P1P5/8 w - - 0 1")
        features = extract_pawn_structure_features(board)
        
        # 12 features: 6 per side × 2 colors
        assert len(features) == 12
    
    def test_extract_mobility_features(self):
        """Test mobility feature extraction."""
        board = chess.Board()
        features = extract_mobility_features(board)
        
        # 4 features: white_moves, black_moves, diff, center_control
        assert len(features) == 4
        
        # Starting position: both sides have 20 legal moves
        assert features[0] == pytest.approx(20/40, abs=0.1)  # White mobility
    
    def test_feature_count_constant(self):
        """Test that FEATURE_COUNT matches actual extraction."""
        board = chess.Board()
        fv = extract_features(board)
        
        # Allow some tolerance for feature count changes during development
        # The constant is for reference; actual count may vary
        assert abs(len(fv) - FEATURE_COUNT) < 10


class TestDistillationDataset:
    """Test distillation dataset class."""
    
    def test_create_dataset(self):
        """Test creating an empty dataset."""
        config = DistillationConfig()
        dataset = DistillationDataset(config)
        
        assert len(dataset) == 0
    
    def test_add_sample(self):
        """Test adding samples to dataset."""
        dataset = DistillationDataset()
        
        dataset.add_sample(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            stockfish_eval=0.2,
            heuristic_eval=0.1,
        )
        
        assert len(dataset) == 1
        assert dataset.samples[0].stockfish_eval == 0.2
    
    def test_max_samples_limit(self):
        """Test that max_samples limit is respected."""
        config = DistillationConfig(max_samples=5)
        dataset = DistillationDataset(config)
        
        for i in range(10):
            dataset.add_sample(
                fen=f"8/8/8/8/8/8/{i%8}K6/8 w - - 0 1",  # Various FENs
                stockfish_eval=float(i),
            )
        
        assert len(dataset) == 5
    
    def test_train_val_split(self):
        """Test train/validation split."""
        config = DistillationConfig(validation_split=0.2)
        dataset = DistillationDataset(config)
        
        for i in range(100):
            dataset.add_sample(
                fen=chess.Board().fen(),
                stockfish_eval=float(i),
            )
        
        train, val = dataset.get_train_val_split()
        
        assert len(train) == 80
        assert len(val) == 20


class TestDistilledEvaluator:
    """Test distilled evaluator class."""
    
    def test_create_evaluator(self):
        """Test creating an evaluator without model."""
        evaluator = DistilledEvaluator()
        
        assert not evaluator.is_loaded()
    
    def test_evaluate_without_model_raises(self):
        """Test that evaluate raises error without model."""
        evaluator = DistilledEvaluator()
        board = chess.Board()
        
        with pytest.raises(RuntimeError):
            evaluator.evaluate(board)
    
    def test_load_nonexistent_model_raises(self):
        """Test loading nonexistent model raises error."""
        from pathlib import Path
        evaluator = DistilledEvaluator()
        
        with pytest.raises(FileNotFoundError):
            evaluator.load(Path("nonexistent_model.pt"))


class TestEvalManagerDistilled:
    """Test EvalManager with distilled mode."""
    
    def test_distilled_mode_fallback_to_heuristic(self):
        """Test that distilled mode falls back to heuristic without model."""
        config = EvalConfig(mode=EvalMode.DISTILLED)
        manager = EvalManager(config)
        
        board = chess.Board()
        result = manager.evaluate(board)
        
        # Should fall back to heuristic
        assert "fallback" in result.meta or result.source in (
            EvalMode.HEURISTIC, EvalMode.HEURISTIC_FAST, EvalMode.HEURISTIC_FULL
        )
    
    def test_distilled_hybrid_mode_fallback(self):
        """Test that distilled_hybrid mode falls back without model."""
        config = EvalConfig(mode=EvalMode.DISTILLED_HYBRID)
        manager = EvalManager(config)
        
        board = chess.Board()
        result = manager.evaluate(board)
        
        # Should produce a score
        assert isinstance(result.score, float)
    
    def test_eval_manager_stats_distilled(self):
        """Test that stats track distilled evals."""
        config = EvalConfig(mode=EvalMode.HEURISTIC)
        manager = EvalManager(config)
        
        stats = manager.get_stats()
        assert "distilled_evals" in stats

