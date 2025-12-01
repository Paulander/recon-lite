"""
Unit tests for M3 plasticity, bandit, and modulation modules.
"""

import math
import pytest

from recon_lite.graph import Graph, Node, NodeType, LinkType
from recon_lite.plasticity.fast import (
    EdgePlasticityState,
    PlasticityConfig,
    init_plasticity_state,
    update_eligibility,
    apply_fast_update,
    reset_episode,
    snapshot_plasticity,
)
from recon_lite.plasticity.bandit import (
    BanditArmState,
    BanditConfig,
    init_bandit_state,
    ucb_score,
    choose_child,
    assign_reward,
    reset_bandit_episode,
    snapshot_bandit,
)
from recon_lite.plasticity.modulation import (
    ModulationConfig,
    Modulators,
    compute_modulators,
)


# ============================================================================
# Plasticity Tests
# ============================================================================


def _make_test_graph():
    """Create a minimal test graph with POR and SUB edges."""
    g = Graph()
    # Scripts
    g.add_node(Node(nid="parent", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="child1", ntype=NodeType.SCRIPT))
    g.add_node(Node(nid="child2", ntype=NodeType.SCRIPT))
    # Terminals
    g.add_node(Node(nid="terminal1", ntype=NodeType.TERMINAL))
    g.add_node(Node(nid="terminal2", ntype=NodeType.TERMINAL))

    # SUB edges
    g.add_edge("parent", "child1", LinkType.SUB)
    g.add_edge("parent", "child2", LinkType.SUB)
    g.add_edge("child1", "terminal1", LinkType.SUB)
    g.add_edge("child2", "terminal2", LinkType.SUB)

    # POR edge between children
    g.add_edge("child1", "child2", LinkType.POR)

    return g


def test_init_plasticity_state_default():
    """Test that init_plasticity_state tracks POR and SUB edges."""
    g = _make_test_graph()
    state = init_plasticity_state(g)

    # Should track POR and SUB edges
    assert len(state) > 0

    # Check that POR edge is tracked
    por_key = "child1->child2:POR"
    assert por_key in state
    assert state[por_key].w_init == 1.0
    assert state[por_key].eligibility == 0.0


def test_init_plasticity_state_whitelist():
    """Test that whitelist restricts tracked edges."""
    g = _make_test_graph()
    whitelist = [("child1", "child2", LinkType.POR)]
    state = init_plasticity_state(g, whitelist)

    # Should only track the whitelisted edge
    assert len(state) == 1
    assert "child1->child2:POR" in state


def test_update_eligibility():
    """Test that eligibility traces update correctly."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    por_key = "child1->child2:POR"

    # Initial eligibility is 0
    assert state[por_key].eligibility == 0.0

    # Fire the edge
    fired_edges = [{"src": "child1", "dst": "child2", "ltype": "POR"}]
    update_eligibility(state, fired_edges, lambda_decay=0.8)

    # Eligibility should be 1.0 (0 * 0.8 + 1.0)
    assert state[por_key].eligibility == 1.0

    # Fire again
    update_eligibility(state, fired_edges, lambda_decay=0.8)

    # Eligibility should be 1.8 (1.0 * 0.8 + 1.0)
    assert abs(state[por_key].eligibility - 1.8) < 0.001

    # Decay without firing
    update_eligibility(state, [], lambda_decay=0.8)

    # Eligibility should be 1.44 (1.8 * 0.8)
    assert abs(state[por_key].eligibility - 1.44) < 0.001


def test_apply_fast_update_positive_reward():
    """Test that positive reward strengthens edges."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    config = PlasticityConfig(eta_tick=0.1, r_max=2.0, w_min=0.1, w_max=3.0)
    por_key = "child1->child2:POR"

    # Set eligibility
    state[por_key].eligibility = 1.0

    # Get initial weight
    initial_weight = None
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            initial_weight = float(e.w)
            break
    assert initial_weight == 1.0

    # Apply positive reward
    deltas = apply_fast_update(state, g, reward_tick=1.0, eta_eff=0.1, config=config)

    # Weight should increase
    new_weight = None
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            new_weight = float(e.w)
            break

    assert new_weight > initial_weight
    assert abs(new_weight - 1.1) < 0.001  # 1.0 + 0.1 * 1.0 * 1.0


def test_apply_fast_update_negative_reward():
    """Test that negative reward weakens edges."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    config = PlasticityConfig(eta_tick=0.1, r_max=2.0, w_min=0.1, w_max=3.0)
    por_key = "child1->child2:POR"

    # Set eligibility
    state[por_key].eligibility = 1.0

    # Apply negative reward
    deltas = apply_fast_update(state, g, reward_tick=-1.0, eta_eff=0.1, config=config)

    # Weight should decrease
    new_weight = None
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            new_weight = float(e.w)
            break

    assert new_weight < 1.0
    assert abs(new_weight - 0.9) < 0.001  # 1.0 - 0.1 * 1.0 * 1.0


def test_apply_fast_update_respects_bounds():
    """Test that weight bounds are enforced."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    config = PlasticityConfig(eta_tick=1.0, r_max=10.0, w_min=0.5, w_max=1.5)
    por_key = "child1->child2:POR"

    # Set high eligibility
    state[por_key].eligibility = 10.0

    # Apply large positive reward
    apply_fast_update(state, g, reward_tick=10.0, eta_eff=1.0, config=config)

    # Weight should be clamped to w_max
    new_weight = None
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            new_weight = float(e.w)
            break

    assert new_weight == config.w_max


def test_reset_episode():
    """Test that reset_episode restores initial weights."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    config = PlasticityConfig(eta_tick=0.1, r_max=2.0)
    por_key = "child1->child2:POR"

    # Modify weights
    state[por_key].eligibility = 1.0
    apply_fast_update(state, g, reward_tick=1.0, eta_eff=0.1, config=config)

    # Verify weight changed
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            assert float(e.w) != 1.0
            break

    # Reset
    reset_episode(state, g)

    # Weight should be restored
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            assert float(e.w) == 1.0
            break

    # Eligibility should be reset
    assert state[por_key].eligibility == 0.0


def test_plasticity_disabled():
    """Test that plasticity does nothing when disabled."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    config = PlasticityConfig(enabled=False)
    por_key = "child1->child2:POR"

    state[por_key].eligibility = 1.0
    deltas = apply_fast_update(state, g, reward_tick=1.0, eta_eff=0.1, config=config)

    # No deltas should be returned
    assert deltas == {}

    # Weight should be unchanged
    for e in g.edges:
        if e.src == "child1" and e.dst == "child2" and e.ltype == LinkType.POR:
            assert float(e.w) == 1.0
            break


# ============================================================================
# Bandit Tests
# ============================================================================


def test_init_bandit_state():
    """Test that bandit state initializes correctly."""
    parent_children = {
        "p1_move": ["king_drive_moves", "confinement_moves", "barrier_placement_moves"],
    }
    state = init_bandit_state(parent_children)

    assert "p1_move" in state
    assert len(state["p1_move"]) == 3
    assert all(arm.pulls == 0 for arm in state["p1_move"].values())


def test_ucb_score_untried_arm():
    """Test that untried arms have infinite UCB score."""
    arm = BanditArmState(child_id="test", pulls=0)
    score = ucb_score(arm, total_pulls=10, c_explore=1.0)
    assert score == float("inf")


def test_ucb_score_computation():
    """Test UCB score computation for tried arms."""
    arm = BanditArmState(child_id="test", pulls=10, sum_reward=5.0)
    score = ucb_score(arm, total_pulls=100, c_explore=1.0)

    # mean = 5.0 / 10 = 0.5
    # exploration = 1.0 * sqrt(2 * ln(101) / 10) ≈ 0.96
    expected_mean = 0.5
    expected_explore = 1.0 * math.sqrt(2.0 * math.log(101) / 10)
    expected = expected_mean + expected_explore

    assert abs(score - expected) < 0.01


def test_choose_child_untried_first():
    """Test that untried children are chosen first."""
    parent_children = {"parent": ["child1", "child2", "child3"]}
    state = init_bandit_state(parent_children)
    config = BanditConfig(c_explore=1.0, min_pulls_before_ucb=1)

    # First choice should be child1 (first untried)
    chosen = choose_child("parent", state, c_explore_eff=1.0, config=config)
    assert chosen == "child1"


def test_choose_child_ucb_selection():
    """Test that UCB selection favors higher-reward arms."""
    parent_children = {"parent": ["good", "bad"]}
    state = init_bandit_state(parent_children)
    config = BanditConfig(c_explore=0.1, min_pulls_before_ucb=1)  # Low exploration

    # Give "good" arm better rewards
    for _ in range(10):
        assign_reward("parent", "good", 1.0, state)
        assign_reward("parent", "bad", -1.0, state)

    # With low exploration, should prefer "good"
    chosen = choose_child("parent", state, c_explore_eff=0.1, config=config)
    assert chosen == "good"


def test_assign_reward():
    """Test that reward assignment updates arm statistics."""
    parent_children = {"parent": ["child1"]}
    state = init_bandit_state(parent_children)

    # Assign rewards
    assign_reward("parent", "child1", 1.0, state)
    assign_reward("parent", "child1", 2.0, state)

    arm = state["parent"]["child1"]
    assert arm.pulls == 2
    assert arm.sum_reward == 3.0
    assert arm.last_reward == 2.0
    assert abs(arm.mean_reward() - 1.5) < 0.001


def test_reset_bandit_episode():
    """Test that bandit state resets correctly."""
    parent_children = {"parent": ["child1", "child2"]}
    state = init_bandit_state(parent_children)

    # Add some data
    assign_reward("parent", "child1", 1.0, state)
    assign_reward("parent", "child2", 2.0, state)

    # Reset
    reset_bandit_episode(state)

    # All arms should be reset
    for arm in state["parent"].values():
        assert arm.pulls == 0
        assert arm.sum_reward == 0.0
        assert arm.last_reward == 0.0


def test_bandit_disabled():
    """Test that bandit does nothing when disabled."""
    parent_children = {"parent": ["child1"]}
    state = init_bandit_state(parent_children)
    config = BanditConfig(enabled=False)

    chosen = choose_child("parent", state, c_explore_eff=1.0, config=config)
    assert chosen is None


# ============================================================================
# Modulation Tests
# ============================================================================


def test_compute_modulators_default():
    """Test modulators with empty goal_vector."""
    modulators = compute_modulators({})

    assert 0.0 <= modulators.risk <= 1.0
    assert 0.0 <= modulators.urgency <= 1.0
    assert modulators.eta_tick_eff > 0
    assert modulators.c_explore_eff > 0


def test_compute_modulators_high_risk():
    """Test that negative material increases risk."""
    config = ModulationConfig(alpha_risk=0.5, eta_base=0.1)

    low_risk = compute_modulators({"material": 5.0}, config)
    high_risk = compute_modulators({"material": -5.0}, config)

    assert high_risk.risk > low_risk.risk
    assert high_risk.eta_tick_eff > low_risk.eta_tick_eff


def test_compute_modulators_high_urgency():
    """Test that low phase progress increases urgency."""
    config = ModulationConfig(alpha_urgency=0.5, c_explore_base=1.0)

    low_urgency = compute_modulators({"phase_progress": 0.9}, config)
    high_urgency = compute_modulators({"phase_progress": 0.1}, config)

    assert high_urgency.urgency > low_urgency.urgency
    assert high_urgency.c_explore_eff > low_urgency.c_explore_eff


def test_modulators_scaling():
    """Test that modulation scales correctly."""
    config = ModulationConfig(
        alpha_risk=1.0,
        alpha_urgency=1.0,
        eta_base=0.1,
        c_explore_base=1.0,
    )

    # Max risk and urgency
    goal_vector = {
        "material": -10.0,  # Very negative
        "defense_pressure": 1.0,
        "king_safety": 0.0,
        "phase_progress": 0.0,
        "box_area": 64.0,
        "attack_pressure": 1.0,
    }

    modulators = compute_modulators(goal_vector, config)

    # With high risk, eta should be significantly above base
    assert modulators.eta_tick_eff > config.eta_base

    # With high urgency, c_explore should be significantly above base
    assert modulators.c_explore_eff > config.c_explore_base


def test_modulators_to_dict():
    """Test that modulators serialize correctly."""
    modulators = Modulators(
        risk=0.5,
        urgency=0.3,
        eta_tick_eff=0.075,
        c_explore_eff=1.15,
    )

    d = modulators.to_dict()
    assert d["risk"] == 0.5
    assert d["urgency"] == 0.3
    assert d["eta_tick_eff"] == 0.075
    assert d["c_explore_eff"] == 1.15


# ============================================================================
# Snapshot Tests
# ============================================================================


def test_snapshot_plasticity():
    """Test plasticity snapshot for logging."""
    g = _make_test_graph()
    state = init_plasticity_state(g)
    por_key = "child1->child2:POR"

    # Add some data
    state[por_key].eligibility = 0.5
    state[por_key].delta_sum = 0.1

    snap = snapshot_plasticity(state)

    assert por_key in snap
    assert snap[por_key]["eligibility"] == 0.5
    assert snap[por_key]["delta_sum"] == 0.1


def test_snapshot_bandit():
    """Test bandit snapshot for logging."""
    parent_children = {"parent": ["child1", "child2"]}
    state = init_bandit_state(parent_children)

    # Add some data
    assign_reward("parent", "child1", 1.0, state)

    snap = snapshot_bandit(state)

    assert "parent" in snap
    assert "child1" in snap["parent"]
    assert snap["parent"]["child1"]["pulls"] == 1

