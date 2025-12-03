"""Tests for M6 plan persistence."""

import pytest
from recon_lite.graph import Node, NodeType
from recon_lite.dynamics.persistence import (
    PersistenceConfig,
    PersistenceState,
    update_persistence,
    apply_persistence_to_node,
    is_plan_active,
    get_active_plans,
    compute_plan_competition,
    InterruptType,
)


class TestPersistenceState:
    """Tests for persistence state management."""
    
    def test_default_state(self):
        """Default state should have reasonable defaults."""
        state = PersistenceState()
        
        assert state.accumulated == 0.0
        assert state.inertia == 0.7
        assert state.decay_rate == 0.1
        assert state.ticks_active == 0
    
    def test_state_serialization(self):
        """State should round-trip through dict."""
        state = PersistenceState(
            accumulated=0.5,
            inertia=0.8,
            decay_rate=0.15,
            ticks_active=5,
        )
        
        d = state.as_dict()
        restored = PersistenceState.from_dict(d)
        
        assert restored.accumulated == state.accumulated
        assert restored.inertia == state.inertia
        assert restored.decay_rate == state.decay_rate
        assert restored.ticks_active == state.ticks_active


class TestPersistenceUpdate:
    """Tests for persistence update logic."""
    
    def test_evidence_increases_activation(self):
        """New evidence should increase accumulated activation."""
        state = PersistenceState(accumulated=0.3, inertia=0.5)
        
        new_state = update_persistence(state, evidence=0.8)
        
        # Should increase (0.8 * (1 - 0.5) = 0.4 added, minus decay)
        assert new_state.accumulated > state.accumulated
    
    def test_decay_reduces_activation(self):
        """Without evidence, activation should decay."""
        state = PersistenceState(accumulated=0.8, decay_rate=0.1)
        
        new_state = update_persistence(state, evidence=0.0)
        
        # Should decay by 10%
        assert new_state.accumulated < state.accumulated
        assert abs(new_state.accumulated - 0.72) < 0.01
    
    def test_high_inertia_reduces_responsiveness(self):
        """High inertia should make plan less responsive to evidence."""
        state_low_inertia = PersistenceState(accumulated=0.5, inertia=0.2)
        state_high_inertia = PersistenceState(accumulated=0.5, inertia=0.9)
        
        new_low = update_persistence(state_low_inertia, evidence=1.0)
        new_high = update_persistence(state_high_inertia, evidence=1.0)
        
        # Low inertia should respond more
        assert new_low.accumulated > new_high.accumulated
    
    def test_interrupt_reduces_activation(self):
        """Strong interrupt should reduce activation."""
        state = PersistenceState(accumulated=0.9)
        config = PersistenceConfig(interrupt_threshold=0.5, interrupt_factor=0.3)
        
        new_state = update_persistence(
            state,
            evidence=0.5,
            interrupt_signals={InterruptType.TACTICAL_URGENCY: 0.8},
            config=config,
        )
        
        # Should be interrupted and reduced
        assert new_state.interrupted
        assert new_state.accumulated < state.accumulated
    
    def test_weak_interrupt_ignored(self):
        """Weak interrupt should not affect activation."""
        state = PersistenceState(accumulated=0.5)
        config = PersistenceConfig(interrupt_threshold=0.7)
        
        new_state = update_persistence(
            state,
            evidence=0.5,
            interrupt_signals={InterruptType.TACTICAL_URGENCY: 0.3},  # Below threshold
            config=config,
        )
        
        assert not new_state.interrupted
    
    def test_activation_clamped(self):
        """Activation should be clamped to [0, 1]."""
        state = PersistenceState(accumulated=0.95, inertia=0.0)
        
        new_state = update_persistence(state, evidence=1.0)
        
        assert new_state.accumulated <= 1.0
        
        state2 = PersistenceState(accumulated=0.05, decay_rate=0.5)
        new_state2 = update_persistence(state2, evidence=0.0)
        
        assert new_state2.accumulated >= 0.0


class TestNodePersistence:
    """Tests for applying persistence to nodes."""
    
    def test_apply_persistence_to_node(self):
        """Persistence should be saved to node activation."""
        node = Node("TestPlan", NodeType.SCRIPT, meta={"layer": "strategic"})
        node.activation.value = 0.3
        
        state = apply_persistence_to_node(node, evidence=0.6)
        
        # State should be stored in meta
        assert "persistence" in node.activation.meta
        assert node.activation.value == state.accumulated
    
    def test_is_plan_active(self):
        """is_plan_active should respect threshold."""
        node = Node("TestPlan", NodeType.SCRIPT)
        node.activation.value = 0.5
        node.activation.meta["persistence"] = {"accumulated": 0.5, "inertia": 0.7, "decay_rate": 0.1}
        
        config = PersistenceConfig(activation_threshold=0.3)
        assert is_plan_active(node, config)
        
        config_high = PersistenceConfig(activation_threshold=0.7)
        assert not is_plan_active(node, config_high)
    
    def test_get_active_plans(self):
        """Should return active plans sorted by activation."""
        nodes = {
            "PlanA": Node("PlanA", NodeType.SCRIPT, meta={"layer": "strategic"}),
            "PlanB": Node("PlanB", NodeType.SCRIPT, meta={"layer": "strategic"}),
            "PlanC": Node("PlanC", NodeType.SCRIPT, meta={"layer": "tactical"}),  # Different layer
        }
        
        nodes["PlanA"].activation.meta["persistence"] = {"accumulated": 0.8, "inertia": 0.7, "decay_rate": 0.1}
        nodes["PlanB"].activation.meta["persistence"] = {"accumulated": 0.5, "inertia": 0.7, "decay_rate": 0.1}
        nodes["PlanC"].activation.meta["persistence"] = {"accumulated": 0.9, "inertia": 0.7, "decay_rate": 0.1}
        
        active = get_active_plans(nodes, layer="strategic")
        
        assert len(active) == 2
        assert active[0][0] == "PlanA"  # Highest activation
        assert active[1][0] == "PlanB"


class TestPlanCompetition:
    """Tests for soft-max plan competition."""
    
    def test_competition_sums_to_one(self):
        """Competition weights should sum to 1.0."""
        nodes = {
            "PlanA": Node("PlanA", NodeType.SCRIPT, meta={"layer": "strategic"}),
            "PlanB": Node("PlanB", NodeType.SCRIPT, meta={"layer": "strategic"}),
        }
        
        nodes["PlanA"].activation.meta["persistence"] = {"accumulated": 0.6, "inertia": 0.7, "decay_rate": 0.1}
        nodes["PlanB"].activation.meta["persistence"] = {"accumulated": 0.4, "inertia": 0.7, "decay_rate": 0.1}
        
        weights = compute_plan_competition(nodes, layer="strategic")
        
        total = sum(weights.values())
        assert 0.99 <= total <= 1.01
    
    def test_higher_activation_gets_higher_weight(self):
        """Plan with higher activation should get higher competition weight."""
        nodes = {
            "PlanA": Node("PlanA", NodeType.SCRIPT, meta={"layer": "strategic"}),
            "PlanB": Node("PlanB", NodeType.SCRIPT, meta={"layer": "strategic"}),
        }
        
        nodes["PlanA"].activation.meta["persistence"] = {"accumulated": 0.8, "inertia": 0.7, "decay_rate": 0.1}
        nodes["PlanB"].activation.meta["persistence"] = {"accumulated": 0.3, "inertia": 0.7, "decay_rate": 0.1}
        
        weights = compute_plan_competition(nodes, layer="strategic")
        
        assert weights["PlanA"] > weights["PlanB"]

