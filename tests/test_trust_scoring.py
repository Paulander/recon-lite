"""Tests for M5.3 trust scoring module."""

import tempfile
from pathlib import Path

import pytest

from recon_lite.trust.scoring import (
    TrustConfig,
    TrustAction,
    NodeTrustScore,
    EdgeTrustScore,
    TrustReport,
    compute_node_trust,
    compute_edge_trust,
    recommend_action,
    compute_trust_report,
)


class TestTrustConfig:
    """Tests for TrustConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrustConfig()
        
        assert config.alpha_activation == 0.3
        assert config.beta_reward == 0.4
        assert config.gamma_variance == 0.3
        assert config.freeze_threshold == 0.3
        assert config.promote_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TrustConfig(
            freeze_threshold=0.2,
            promote_threshold=0.9,
            min_generations=5,
        )
        
        assert config.freeze_threshold == 0.2
        assert config.promote_threshold == 0.9
        assert config.min_generations == 5


class TestNodeTrustScore:
    """Tests for NodeTrustScore."""
    
    def test_mean_reward_calculation(self):
        """Test mean reward calculation."""
        score = NodeTrustScore(
            node_id="test",
            total_reward=10.0,
            reward_samples=5,
        )
        
        assert score.mean_reward == 2.0
    
    def test_mean_reward_zero_samples(self):
        """Test mean reward with zero samples."""
        score = NodeTrustScore(node_id="test")
        
        assert score.mean_reward == 0.0
    
    def test_to_dict(self):
        """Test serialization."""
        score = NodeTrustScore(
            node_id="test_node",
            activation_count=50,
            total_reward=5.0,
            reward_samples=10,
            trust_score=0.75,
        )
        
        d = score.to_dict()
        
        assert d["node_id"] == "test_node"
        assert d["activation_count"] == 50
        assert d["mean_reward"] == 0.5


class TestEdgeTrustScore:
    """Tests for EdgeTrustScore."""
    
    def test_mean_contribution(self):
        """Test mean contribution calculation."""
        score = EdgeTrustScore(
            edge_key="a->b:POR",
            contribution_total=2.0,
            contribution_samples=4,
        )
        
        assert score.mean_contribution == 0.5
    
    def test_weight_stability_stable(self):
        """Test stability with stable weights."""
        score = EdgeTrustScore(
            edge_key="a->b:POR",
            weight_history=[1.0, 1.0, 1.0, 1.0],
        )
        
        assert score.weight_stability == 1.0
    
    def test_weight_stability_unstable(self):
        """Test stability with varying weights."""
        score = EdgeTrustScore(
            edge_key="a->b:POR",
            weight_history=[0.5, 1.5, 0.5, 1.5],
        )
        
        # High variance relative to mean should give lower stability
        assert score.weight_stability < 1.0


class TestTrustReport:
    """Tests for TrustReport."""
    
    def test_add_scores(self):
        """Test adding scores to report."""
        report = TrustReport()
        
        report.add_node_score(NodeTrustScore(
            node_id="node1",
            trust_score=0.9,
            recommended_action=TrustAction.PROMOTE.value,
        ))
        report.add_edge_score(EdgeTrustScore(
            edge_key="a->b:POR",
            trust_score=0.2,
            recommended_action=TrustAction.FREEZE.value,
        ))
        
        assert len(report.node_scores) == 1
        assert len(report.edge_scores) == 1
    
    def test_get_candidates(self):
        """Test getting candidate lists."""
        report = TrustReport()
        
        report.add_node_score(NodeTrustScore(
            node_id="high_trust",
            trust_score=0.9,
            recommended_action=TrustAction.PROMOTE.value,
        ))
        report.add_node_score(NodeTrustScore(
            node_id="low_trust",
            trust_score=0.1,
            recommended_action=TrustAction.REMOVE.value,
        ))
        report.add_edge_score(EdgeTrustScore(
            edge_key="freeze_me",
            trust_score=0.25,
            recommended_action=TrustAction.FREEZE.value,
        ))
        
        assert "node:high_trust" in report.get_promote_candidates()
        assert "node:low_trust" in report.get_remove_candidates()
        assert "edge:freeze_me" in report.get_freeze_candidates()
    
    def test_save_and_load(self):
        """Test saving and loading report."""
        report = TrustReport(generation=5)
        report.config = TrustConfig()
        
        report.add_node_score(NodeTrustScore(
            node_id="test_node",
            activation_count=100,
            total_reward=10.0,
            reward_samples=20,
            trust_score=0.8,
        ))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trust.json"
            report.save(path)
            
            loaded = TrustReport.load(path)
            
            assert loaded.generation == 5
            assert "test_node" in loaded.node_scores
            assert loaded.node_scores["test_node"].activation_count == 100


class TestComputeNodeTrust:
    """Tests for compute_node_trust function."""
    
    def test_insufficient_data(self):
        """Test neutral score with insufficient data."""
        config = TrustConfig(min_activations=10)
        score = NodeTrustScore(
            node_id="test",
            activation_count=5,  # Below threshold
        )
        
        trust = compute_node_trust(score, config)
        
        assert trust == 0.5  # Neutral
    
    def test_high_activation_positive_reward(self):
        """Test high trust with good performance."""
        config = TrustConfig()
        score = NodeTrustScore(
            node_id="test",
            activation_count=100,
            total_reward=50.0,
            reward_samples=50,
            reward_variance=0.1,
        )
        
        trust = compute_node_trust(score, config)
        
        assert trust > 0.5  # Should be above neutral
    
    def test_low_activation_negative_reward(self):
        """Test low trust with poor performance."""
        config = TrustConfig()
        score = NodeTrustScore(
            node_id="test",
            activation_count=15,
            total_reward=-20.0,
            reward_samples=15,
            reward_variance=2.0,
        )
        
        trust = compute_node_trust(score, config)
        
        assert trust < 0.5  # Should be below neutral


class TestComputeEdgeTrust:
    """Tests for compute_edge_trust function."""
    
    def test_insufficient_data(self):
        """Test neutral score with insufficient data."""
        config = TrustConfig(min_activations=10)
        score = EdgeTrustScore(
            edge_key="a->b:POR",
            fire_count=5,  # Below threshold
        )
        
        trust = compute_edge_trust(score, config)
        
        assert trust == 0.5  # Neutral
    
    def test_stable_high_contribution(self):
        """Test high trust with stable, positive contribution."""
        config = TrustConfig()
        score = EdgeTrustScore(
            edge_key="a->b:POR",
            fire_count=100,
            contribution_total=50.0,
            contribution_samples=50,
            weight_history=[1.0, 1.0, 1.0, 1.0],
        )
        
        trust = compute_edge_trust(score, config)
        
        assert trust > 0.5


class TestRecommendAction:
    """Tests for recommend_action function."""
    
    def test_insufficient_generations(self):
        """Test no action with insufficient generations."""
        config = TrustConfig(min_generations=3)
        
        action = recommend_action(0.1, 2, config)  # Only 2 generations
        
        assert action == TrustAction.NONE.value
    
    def test_freeze_recommendation(self):
        """Test freeze recommendation."""
        config = TrustConfig(freeze_threshold=0.3, remove_threshold=0.1)
        
        action = recommend_action(0.2, 5, config)
        
        assert action == TrustAction.FREEZE.value
    
    def test_remove_recommendation(self):
        """Test remove recommendation."""
        config = TrustConfig(remove_threshold=0.1)
        
        action = recommend_action(0.05, 5, config)
        
        assert action == TrustAction.REMOVE.value
    
    def test_promote_recommendation(self):
        """Test promote recommendation."""
        config = TrustConfig(promote_threshold=0.8)
        
        action = recommend_action(0.9, 5, config)
        
        assert action == TrustAction.PROMOTE.value
    
    def test_no_action_normal_trust(self):
        """Test no action with normal trust."""
        config = TrustConfig(
            freeze_threshold=0.3,
            promote_threshold=0.8,
        )
        
        action = recommend_action(0.5, 5, config)
        
        assert action == TrustAction.NONE.value


class TestComputeTrustReport:
    """Tests for compute_trust_report function."""
    
    def test_empty_traces(self):
        """Test with no trace data."""
        report = compute_trust_report([])
        
        assert report.generation == 1
        assert len(report.node_scores) == 0
        assert len(report.edge_scores) == 0
    
    def test_with_trace_data(self):
        """Test with trace frames."""
        frames = [
            {
                "nodes": {"node1": "TRUE", "node2": "INACTIVE"},
                "env": {"m3_reward_tick": 0.5},
            },
            {
                "nodes": {"node1": "CONFIRMED", "node2": "WAITING"},
                "env": {"m3_reward_tick": -0.2},
            },
        ]
        
        report = compute_trust_report(frames)
        
        assert "node1" in report.node_scores
        assert "node2" in report.node_scores
        assert report.node_scores["node1"].activation_count == 2
        assert report.node_scores["node2"].activation_count == 1
    
    def test_incremental_tracking(self):
        """Test incremental generation tracking."""
        previous = TrustReport(generation=3)
        previous.add_node_score(NodeTrustScore(
            node_id="existing",
            activation_count=50,
            generations_tracked=3,
        ))
        
        report = compute_trust_report([], previous_report=previous)
        
        assert report.generation == 4
        assert report.node_scores["existing"].generations_tracked == 4
    
    def test_with_weight_deltas(self):
        """Test edge tracking from weight deltas."""
        frames = [
            {
                "nodes": {},
                "env": {
                    "m3_weight_deltas": {
                        "a->b:POR": 0.1,
                        "b->c:SUB": -0.05,
                    },
                },
            },
        ]
        
        report = compute_trust_report(frames)
        
        assert "a->b:POR" in report.edge_scores
        assert "b->c:SUB" in report.edge_scores
        assert report.edge_scores["a->b:POR"].fire_count == 1

