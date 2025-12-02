"""
Tests for M4 consolidation module.
"""

import sys
from pathlib import Path
import tempfile
import json

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recon_lite.trace_db import (
    EpisodeSummary,
    BanditArmSummary,
    EpisodeRecord,
    outcome_to_score,
)
from src.recon_lite.plasticity.consolidate import (
    ConsolidationConfig,
    EdgeConsolidationState,
    ConsolidationEngine,
)
from src.recon_lite.graph import Graph, Node, Edge, LinkType, NodeState, NodeType


def make_test_graph() -> Graph:
    """Create a simple test graph with POR edges."""
    g = Graph()
    g.add_node(Node("root", NodeType.SCRIPT, state=NodeState.INACTIVE))
    g.add_node(Node("phase1", NodeType.SCRIPT, state=NodeState.INACTIVE))
    g.add_node(Node("phase2", NodeType.SCRIPT, state=NodeState.INACTIVE))
    g.add_node(Node("check", NodeType.SCRIPT, state=NodeState.INACTIVE))
    g.add_node(Node("move", NodeType.SCRIPT, state=NodeState.INACTIVE))

    g.add_edge("root", "phase1", LinkType.SUB)
    g.add_edge("root", "phase2", LinkType.SUB)
    g.add_edge("phase1", "phase2", LinkType.POR)
    g.add_edge("check", "move", LinkType.POR)

    return g


def make_test_summary(
    edge_deltas: dict = None,
    avg_reward: float = 0.0,
    outcome: float = 0.0,
) -> EpisodeSummary:
    """Create a test episode summary."""
    return EpisodeSummary(
        edge_delta_sums=edge_deltas or {},
        avg_reward_tick=avg_reward,
        outcome_score=outcome,
    )


class TestConsolidationConfig:
    def test_default_config(self):
        config = ConsolidationConfig()
        assert config.eta_consolidate == 0.01
        assert config.min_episodes == 10
        assert config.outcome_weight == 0.5
        assert config.max_base_delta == 0.5
        assert config.enabled is True

    def test_custom_config(self):
        config = ConsolidationConfig(
            eta_consolidate=0.05,
            min_episodes=5,
            outcome_weight=0.7,
        )
        assert config.eta_consolidate == 0.05
        assert config.min_episodes == 5
        assert config.outcome_weight == 0.7


class TestEdgeConsolidationState:
    def test_initial_state(self):
        state = EdgeConsolidationState(edge_key="a->b:POR", w_base=1.0)
        assert state.edge_key == "a->b:POR"
        assert state.w_base == 1.0
        assert state.accumulated_weighted_delta == 0.0
        assert state.episode_count == 0

    def test_mean_weighted_delta(self):
        state = EdgeConsolidationState(edge_key="a->b:POR", w_base=1.0)
        state.accumulated_weighted_delta = 0.5
        state.episode_count = 5
        assert state.mean_weighted_delta() == 0.1

    def test_mean_weighted_delta_zero_episodes(self):
        state = EdgeConsolidationState(edge_key="a->b:POR", w_base=1.0)
        assert state.mean_weighted_delta() == 0.0


class TestConsolidationEngine:
    def test_init_from_graph(self):
        g = make_test_graph()
        engine = ConsolidationEngine()
        engine.init_from_graph(g)

        # Should track POR and SUB edges
        assert len(engine.edge_states) >= 2
        assert "phase1->phase2:POR" in engine.edge_states
        assert "check->move:POR" in engine.edge_states

    def test_init_with_whitelist(self):
        g = make_test_graph()
        engine = ConsolidationEngine()
        engine.init_from_graph(g, edge_whitelist=["phase1->phase2:POR"])

        assert len(engine.edge_states) == 1
        assert "phase1->phase2:POR" in engine.edge_states

    def test_accumulate_episode_basic(self):
        engine = ConsolidationEngine()
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.0
        )

        summary = make_test_summary(
            edge_deltas={"a->b:POR": 0.1},
            avg_reward=0.5,
            outcome=1.0,  # win
        )
        engine.accumulate_episode(summary)

        assert engine.total_episodes == 1
        state = engine.edge_states["a->b:POR"]
        assert state.episode_count == 1
        # delta_episode = 1.0 * 0.5 + 0.5 * 0.5 = 0.75
        # weighted_delta = 0.1 * 0.75 = 0.075
        assert abs(state.accumulated_weighted_delta - 0.075) < 0.001

    def test_accumulate_episode_creates_new_edge(self):
        engine = ConsolidationEngine()

        summary = make_test_summary(
            edge_deltas={"new->edge:POR": 0.2},
            avg_reward=0.0,
            outcome=0.0,
        )
        engine.accumulate_episode(summary)

        assert "new->edge:POR" in engine.edge_states
        assert engine.edge_states["new->edge:POR"].episode_count == 1

    def test_should_apply_threshold(self):
        config = ConsolidationConfig(min_episodes=5)
        engine = ConsolidationEngine(config)

        for _ in range(4):
            engine.accumulate_episode(make_test_summary())

        assert not engine.should_apply()

        engine.accumulate_episode(make_test_summary())
        assert engine.should_apply()

    def test_apply_to_graph(self):
        g = make_test_graph()
        config = ConsolidationConfig(
            eta_consolidate=0.1,
            min_episodes=1,
        )
        engine = ConsolidationEngine(config)
        engine.init_from_graph(g)

        # Accumulate a positive episode
        summary = make_test_summary(
            edge_deltas={"phase1->phase2:POR": 1.0},
            avg_reward=1.0,
            outcome=1.0,
        )
        engine.accumulate_episode(summary)

        # Apply consolidation
        deltas = engine.apply_to_graph(g)

        assert "phase1->phase2:POR" in deltas
        assert deltas["phase1->phase2:POR"] > 0

        # Check w_base was updated
        state = engine.edge_states["phase1->phase2:POR"]
        assert state.w_base > 1.0

        # Check episode count was reset
        assert state.episode_count == 0
        assert engine.episodes_since_apply == 0

    def test_apply_respects_max_base_delta(self):
        config = ConsolidationConfig(
            eta_consolidate=1.0,  # Very high learning rate
            min_episodes=1,
            max_base_delta=0.1,  # But capped
        )
        engine = ConsolidationEngine(config)
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.0
        )

        # Accumulate a huge positive episode
        summary = make_test_summary(
            edge_deltas={"a->b:POR": 10.0},
            avg_reward=1.0,
            outcome=1.0,
        )
        engine.accumulate_episode(summary)

        g = make_test_graph()
        g.add_node(Node("a", NodeType.SCRIPT, state=NodeState.INACTIVE))
        g.add_node(Node("b", NodeType.SCRIPT, state=NodeState.INACTIVE))
        g.add_edge("a", "b", LinkType.POR)

        deltas = engine.apply_to_graph(g)

        # Delta should be capped
        assert abs(deltas.get("a->b:POR", 0)) <= 0.1 + 0.001

    def test_apply_respects_weight_bounds(self):
        config = ConsolidationConfig(
            eta_consolidate=1.0,
            min_episodes=1,
            max_base_delta=10.0,  # No delta cap
            w_min=0.5,
            w_max=2.0,
        )
        engine = ConsolidationEngine(config)
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.5
        )

        # Try to push above max
        summary = make_test_summary(
            edge_deltas={"a->b:POR": 10.0},
            avg_reward=1.0,
            outcome=1.0,
        )
        engine.accumulate_episode(summary)

        g = make_test_graph()
        g.add_node(Node("a", NodeType.SCRIPT, state=NodeState.INACTIVE))
        g.add_node(Node("b", NodeType.SCRIPT, state=NodeState.INACTIVE))
        g.add_edge("a", "b", LinkType.POR)
        # Set initial weight to 1.5
        for e in g.edges:
            if e.src == "a" and e.dst == "b":
                e.w = 1.5

        engine.apply_to_graph(g)

        state = engine.edge_states["a->b:POR"]
        assert state.w_base <= 2.0

    def test_save_and_load_state(self):
        engine = ConsolidationEngine()
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.5, w_init=1.0
        )
        engine.total_episodes = 10

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            engine.save_state(path)

            # Load into new engine
            engine2 = ConsolidationEngine()
            engine2.load_state(path)

            assert engine2.total_episodes == 10
            assert "a->b:POR" in engine2.edge_states
            assert engine2.edge_states["a->b:POR"].w_base == 1.5
            assert engine2.edge_states["a->b:POR"].w_init == 1.0
        finally:
            path.unlink()

    def test_export_w_base_pack(self):
        engine = ConsolidationEngine()
        engine.edge_states["phase1->phase2:POR"] = EdgeConsolidationState(
            edge_key="phase1->phase2:POR", w_base=1.2
        )
        engine.edge_states["check->move:SUB"] = EdgeConsolidationState(
            edge_key="check->move:SUB", w_base=0.8
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".swp", delete=False) as f:
            path = Path(f.name)

        try:
            engine.export_w_base_pack(path)

            with path.open() as fh:
                data = json.load(fh)

            # Only POR edges should be in the pack
            assert "por_edges" in data
            assert "phase1->phase2" in data["por_edges"]
            assert data["por_edges"]["phase1->phase2"] == 1.2
        finally:
            path.unlink()

    def test_get_all_w_base(self):
        engine = ConsolidationEngine()
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.1
        )
        engine.edge_states["c->d:SUB"] = EdgeConsolidationState(
            edge_key="c->d:SUB", w_base=0.9
        )

        w_base = engine.get_all_w_base()
        assert w_base["a->b:POR"] == 1.1
        assert w_base["c->d:SUB"] == 0.9

    def test_snapshot(self):
        engine = ConsolidationEngine()
        engine.total_episodes = 5
        engine.episodes_since_apply = 3
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.1
        )

        snapshot = engine.snapshot()
        assert snapshot["total_episodes"] == 5
        assert snapshot["episodes_since_apply"] == 3
        assert snapshot["edges_tracked"] == 1


class TestEpisodeSummary:
    def test_to_dict_and_from_dict(self):
        summary = EpisodeSummary(
            edge_delta_sums={"a->b:POR": 0.1},
            bandit_stats={
                "parent": {
                    "child1": BanditArmSummary(pulls=5, sum_reward=2.5, mean_reward=0.5)
                }
            },
            avg_reward_tick=0.3,
            total_reward_tick=1.5,
            reward_tick_count=5,
            phase_usage={"phase1": 10, "phase2": 5},
            outcome_score=1.0,
        )

        d = summary.to_dict()
        restored = EpisodeSummary.from_dict(d)

        assert restored.edge_delta_sums == {"a->b:POR": 0.1}
        assert restored.avg_reward_tick == 0.3
        assert restored.outcome_score == 1.0
        assert "parent" in restored.bandit_stats
        assert restored.bandit_stats["parent"]["child1"].pulls == 5


class TestOutcomeToScore:
    def test_white_win(self):
        assert outcome_to_score("1-0") == 1.0

    def test_black_win(self):
        assert outcome_to_score("0-1") == -1.0

    def test_draw(self):
        assert outcome_to_score("1/2-1/2") == 0.0
        assert outcome_to_score("draw") == 0.0

    def test_none(self):
        assert outcome_to_score(None) == 0.0

    def test_unknown(self):
        assert outcome_to_score("unknown") == 0.0


class TestConsolidationDisabled:
    def test_accumulate_does_nothing_when_disabled(self):
        config = ConsolidationConfig(enabled=False)
        engine = ConsolidationEngine(config)

        summary = make_test_summary(edge_deltas={"a->b:POR": 0.1})
        engine.accumulate_episode(summary)

        assert engine.total_episodes == 0

    def test_apply_does_nothing_when_disabled(self):
        config = ConsolidationConfig(enabled=False)
        engine = ConsolidationEngine(config)
        engine.edge_states["a->b:POR"] = EdgeConsolidationState(
            edge_key="a->b:POR", w_base=1.0
        )

        g = make_test_graph()
        deltas = engine.apply_to_graph(g)

        assert deltas == {}

