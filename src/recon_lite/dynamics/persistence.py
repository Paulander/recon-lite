"""Plan Persistence for M6 Goal Hierarchy.

Implements persistence (hysteresis) for strategic plans using the
node's activation field, which is allowed by the ReCoN paper:
"activation can be used to store additional state"

Persistence model:
- accumulated += evidence * (1 - inertia)  # new info modulated by stickiness
- accumulated *= (1 - decay_rate)          # temporal decay
- if interrupt > threshold: accumulated *= interrupt_factor

This allows plans to maintain activation over time, creating "commitment"
to a strategy rather than constantly switching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple, List

from ..graph import Node, NodeType


class InterruptType(Enum):
    """Types of interrupt signals that can override plan persistence."""
    TACTICAL_URGENCY = auto()  # A tactic is too good to ignore
    THREAT_DETECTED = auto()   # Must respond to immediate threat  
    BLUNDER_RISK = auto()      # Current plan leads to disaster
    OPPORTUNITY = auto()       # Better option suddenly available


@dataclass
class PersistenceConfig:
    """Configuration for plan persistence behavior."""
    default_inertia: float = 0.7      # How sticky plans are (0=responsive, 1=very sticky)
    default_decay: float = 0.1        # How fast activation fades without reinforcement
    activation_threshold: float = 0.3  # Minimum activation to consider a plan "active"
    interrupt_threshold: float = 0.7   # How strong an interrupt must be to override
    interrupt_factor: float = 0.5      # Multiplier when interrupted (0=instant switch)
    max_concurrent_plans: int = 3      # Soft limit on simultaneous active plans


@dataclass 
class PersistenceState:
    """
    Persistence state stored in node.activation.meta.
    
    This leverages the ReCoN paper's allowance for real-valued activation
    to store additional state.
    """
    accumulated: float = 0.0    # Accumulated evidence for this plan
    inertia: float = 0.7        # How sticky (learnable parameter)
    decay_rate: float = 0.1     # How fast it fades
    last_evidence: float = 0.0  # Evidence from last tick (for debugging)
    ticks_active: int = 0       # How many ticks this plan has been active
    interrupted: bool = False   # Whether currently interrupted
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "accumulated": self.accumulated,
            "inertia": self.inertia,
            "decay_rate": self.decay_rate,
            "last_evidence": self.last_evidence,
            "ticks_active": self.ticks_active,
            "interrupted": self.interrupted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersistenceState":
        return cls(
            accumulated=d.get("accumulated", 0.0),
            inertia=d.get("inertia", 0.7),
            decay_rate=d.get("decay_rate", 0.1),
            last_evidence=d.get("last_evidence", 0.0),
            ticks_active=d.get("ticks_active", 0),
            interrupted=d.get("interrupted", False),
        )


def get_persistence_state(node: Node) -> PersistenceState:
    """Get or initialize persistence state from a node."""
    meta = node.activation.meta
    if "persistence" in meta:
        return PersistenceState.from_dict(meta["persistence"])
    
    # Initialize from node meta defaults
    inertia = node.meta.get("inertia", 0.7)
    decay = node.meta.get("decay", 0.1)
    
    return PersistenceState(
        accumulated=node.activation.value,
        inertia=inertia,
        decay_rate=decay,
    )


def save_persistence_state(node: Node, state: PersistenceState) -> None:
    """Save persistence state back to node."""
    node.activation.meta["persistence"] = state.as_dict()
    # Also update the main activation value
    node.activation.value = state.accumulated


def update_persistence(
    state: PersistenceState,
    evidence: float,
    interrupt_signals: Optional[Dict[InterruptType, float]] = None,
    config: Optional[PersistenceConfig] = None,
) -> PersistenceState:
    """
    Update persistence state based on new evidence and interrupts.
    
    Args:
        state: Current persistence state
        evidence: New evidence for this plan (0-1, from sensors/confirmations)
        interrupt_signals: Dict of interrupt types to their strength (0-1)
        config: Persistence configuration
        
    Returns:
        Updated persistence state
    """
    config = config or PersistenceConfig()
    interrupt_signals = interrupt_signals or {}
    
    # Apply decay first (temporal fading)
    new_accumulated = state.accumulated * (1.0 - state.decay_rate)
    
    # Incorporate new evidence, modulated by inertia
    # High inertia = less responsive to new evidence
    evidence_contribution = evidence * (1.0 - state.inertia)
    new_accumulated += evidence_contribution
    
    # Check for interrupts
    interrupted = False
    for itype, strength in interrupt_signals.items():
        if strength >= config.interrupt_threshold:
            new_accumulated *= config.interrupt_factor
            interrupted = True
            break
    
    # Clamp to [0, 1]
    new_accumulated = max(0.0, min(1.0, new_accumulated))
    
    # Update ticks_active
    if new_accumulated >= config.activation_threshold:
        new_ticks = state.ticks_active + 1
    else:
        new_ticks = 0
    
    return PersistenceState(
        accumulated=new_accumulated,
        inertia=state.inertia,
        decay_rate=state.decay_rate,
        last_evidence=evidence,
        ticks_active=new_ticks,
        interrupted=interrupted,
    )


def apply_persistence_to_node(
    node: Node,
    evidence: float,
    interrupt_signals: Optional[Dict[InterruptType, float]] = None,
    config: Optional[PersistenceConfig] = None,
) -> PersistenceState:
    """
    Apply persistence update to a node and save the result.
    
    Args:
        node: The node to update
        evidence: New evidence for this plan
        interrupt_signals: Dict of interrupt types to their strength
        config: Persistence configuration
        
    Returns:
        Updated persistence state
    """
    state = get_persistence_state(node)
    new_state = update_persistence(state, evidence, interrupt_signals, config)
    save_persistence_state(node, new_state)
    return new_state


def is_plan_active(node: Node, config: Optional[PersistenceConfig] = None) -> bool:
    """Check if a plan's accumulated activation exceeds threshold."""
    config = config or PersistenceConfig()
    state = get_persistence_state(node)
    return state.accumulated >= config.activation_threshold


def get_active_plans(
    nodes: Dict[str, Node],
    layer: str = "strategic",
    config: Optional[PersistenceConfig] = None,
) -> List[Tuple[str, float]]:
    """
    Get all active plans sorted by accumulated activation.
    
    Args:
        nodes: Dict of node ID to Node
        layer: Which layer to filter by (from node.meta["layer"])
        config: Persistence configuration
        
    Returns:
        List of (node_id, accumulated) tuples, sorted descending by activation
    """
    config = config or PersistenceConfig()
    active = []
    
    for nid, node in nodes.items():
        if node.meta.get("layer") != layer:
            continue
        
        state = get_persistence_state(node)
        if state.accumulated >= config.activation_threshold:
            active.append((nid, state.accumulated))
    
    return sorted(active, key=lambda x: x[1], reverse=True)


def create_interrupt_terminal(itype: InterruptType) -> Node:
    """Create a terminal node that detects interrupt conditions."""
    
    def predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check for interrupt condition."""
        # Check env for interrupt signals
        interrupts = env.get("interrupt_signals", {})
        
        if itype == InterruptType.TACTICAL_URGENCY:
            # High-value tactic available
            tactics = env.get("tactical_opportunities", [])
            if any(t.get("value", 0) > 3.0 for t in tactics):
                interrupts[itype] = 0.9
                node.activation.value = 0.9
                env["interrupt_signals"] = interrupts
                return True, True
        
        elif itype == InterruptType.THREAT_DETECTED:
            # King in danger or major threat
            threat_level = env.get("threat_level", 0.0)
            if threat_level > 0.7:
                interrupts[itype] = threat_level
                node.activation.value = threat_level
                env["interrupt_signals"] = interrupts
                return True, True
        
        elif itype == InterruptType.BLUNDER_RISK:
            # Current plan leads to material loss
            blunder_risk = env.get("blunder_risk", 0.0)
            if blunder_risk > 0.6:
                interrupts[itype] = blunder_risk
                node.activation.value = blunder_risk
                env["interrupt_signals"] = interrupts
                return True, True
        
        # No interrupt triggered
        node.activation.value = 0.0
        return True, False
    
    return Node(
        nid=f"Interrupt_{itype.name}",
        ntype=NodeType.TERMINAL,
        predicate=predicate,
        meta={"interrupt_type": itype.name, "fan_in_allowed": True},
    )


def compute_plan_competition(
    nodes: Dict[str, Node],
    layer: str = "strategic",
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute soft-max competition weights for active plans.
    
    This allows multiple plans to be "active" but with varying degrees
    of influence, rather than a hard winner-take-all.
    
    Args:
        nodes: Dict of node ID to Node
        layer: Which layer to filter
        temperature: Softmax temperature (higher = more even, lower = more peaked)
        
    Returns:
        Dict of node_id to competition weight (sums to 1.0)
    """
    import math
    
    activations = {}
    for nid, node in nodes.items():
        if node.meta.get("layer") != layer:
            continue
        state = get_persistence_state(node)
        if state.accumulated > 0.01:
            activations[nid] = state.accumulated
    
    if not activations:
        return {}
    
    # Softmax
    max_act = max(activations.values())
    exp_vals = {nid: math.exp((act - max_act) / temperature) for nid, act in activations.items()}
    total = sum(exp_vals.values())
    
    if total == 0:
        return {nid: 1.0 / len(exp_vals) for nid in exp_vals}
    
    return {nid: exp_v / total for nid, exp_v in exp_vals.items()}

