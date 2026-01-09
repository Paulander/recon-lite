"""
Goal Delegation Pack - spawnable hierarchical structure for M5 evolution.

Implements POR-based "Stem Cell Packs" that follow pure ReCoN philosophy:
- Top-down requests via SUB/POR
- Bottom-up confirmations via SUR (declarative wait)
- Native multi-tick depth without SubgraphLock hack

Structure:
    pack_root (SCRIPT)
        ├── detect ──POR──► execute ──POR──► finish ──POR──► wait
        │   (sensor)        (SCRIPT)         (sensor)        (wait)
        │                      │
        │                   stem cells attach here
"""

from __future__ import annotations

import random
from functools import partial
from typing import Callable, Dict, List, Optional, Any

from recon_lite.graph import Graph, Node, NodeType, LinkType


# Configuration
MAX_BRANCH_DEPTH = 5  # Prevent O(n²) propagation
DECAY_DEPTH_MULTIPLIER = 1.2  # Deeper packs prune faster


def spawn_goal_delegation_pack(
    goal_name: str,
    parent_id: str,
    graph: Graph,
    condition_sensor_fn: Callable[[Dict[str, Any]], bool],
    sentinel_fn: Callable[[Dict[str, Any]], bool],
    actuator_fn: Callable[[Dict[str, Any]], Optional[str]],
    depth: int = 0,
    is_trial: bool = True,
    parent_signature: Optional[List[float]] = None,
    attach_stem_cells: int = 1,
    mutate_edges: bool = True,
) -> Dict[str, str]:
    """
    Spawn a complete Goal Delegation Pack with POR sequence.
    
    Args:
        goal_name: Unique identifier for this pack's goal
        parent_id: Node ID to attach pack root as SUB child
        graph: ReCoN graph to add nodes to
        condition_sensor_fn: Predicate for detect phase (should this pack activate?)
        sentinel_fn: Predicate for finish phase (was the goal achieved?)
        actuator_fn: Action function for execute phase
        depth: Current recursion depth (for MAX_BRANCH_DEPTH check)
        is_trial: Whether pack is in TRIAL state (affects actuator weight)
        parent_signature: Parent's pattern signature for inheritance (80% + 20% mutation)
        attach_stem_cells: Number of stem cell slots to create under execute (1-2)
        mutate_edges: Whether to add weight variation to edges
    
    Returns:
        Dict of created node IDs: {"root", "detect", "execute", "finish", "wait", "actuator"}
        Empty dict if depth limit exceeded.
    """
    if depth > MAX_BRANCH_DEPTH:
        return {}  # Depth limit fallback
    
    prefix = f"pack_{goal_name}_{depth}"
    created_ids: Dict[str, str] = {}
    
    # =========================================================================
    # Create root SCRIPT node
    # =========================================================================
    root_id = f"{prefix}_root"
    root_node = Node(nid=root_id, ntype=NodeType.SCRIPT)
    root_node.meta["origin"] = "goal_delegation_pack"
    root_node.meta["depth"] = depth
    root_node.meta["goal"] = goal_name
    root_node.meta["transient"] = is_trial
    root_node.meta["tier"] = "trial" if is_trial else "mature"
    root_node.meta["decay_multiplier"] = DECAY_DEPTH_MULTIPLIER ** depth  # Deeper = faster decay
    
    # Inherit pattern signature with mutation (80% parent + 20% random)
    if parent_signature:
        signature = _inherit_signature(parent_signature, mutation_rate=0.2)
        root_node.meta["pattern_signature"] = signature
    
    graph.add_node(root_node)
    created_ids["root"] = root_id
    
    # Create 4-phase children: detect → execute → finish → wait
    # NOTE: All phases must be SCRIPT nodes for POR edges to work!
    # Sensors are added as TERMINAL children under each SCRIPT phase.
    # =========================================================================
    detect_id = f"{prefix}_detect"
    execute_id = f"{prefix}_execute"
    finish_id = f"{prefix}_finish"
    wait_id = f"{prefix}_wait"
    
    # detect (SCRIPT with TERMINAL sensor child)
    detect = Node(nid=detect_id, ntype=NodeType.SCRIPT)
    detect.meta["role"] = "detect_phase"
    detect.meta["pack_phase"] = "detect"
    graph.add_node(detect)
    created_ids["detect"] = detect_id
    
    # Add condition sensor as TERMINAL child of detect
    detect_sensor_id = f"{prefix}_detect_sensor"
    detect_sensor = Node(nid=detect_sensor_id, ntype=NodeType.TERMINAL, predicate=condition_sensor_fn)
    detect_sensor.meta["role"] = "condition_sensor"
    graph.add_node(detect_sensor)
    graph.add_edge(detect_id, detect_sensor_id, LinkType.SUB)
    
    # execute (SCRIPT - can have children for recursion)
    execute = Node(nid=execute_id, ntype=NodeType.SCRIPT)
    execute.meta["role"] = "execute_phase"
    execute.meta["pack_phase"] = "execute"
    execute.meta["aggregation"] = "avg"  # OR-like for sibling actuators
    graph.add_node(execute)
    created_ids["execute"] = execute_id
    
    # finish (SCRIPT with TERMINAL sentinel child)
    finish = Node(nid=finish_id, ntype=NodeType.SCRIPT)
    finish.meta["role"] = "finish_phase"
    finish.meta["pack_phase"] = "finish"
    graph.add_node(finish)
    created_ids["finish"] = finish_id
    
    # Add sentinel sensor as TERMINAL child of finish
    finish_sensor_id = f"{prefix}_finish_sensor"
    finish_sensor = Node(nid=finish_sensor_id, ntype=NodeType.TERMINAL, predicate=sentinel_fn)
    finish_sensor.meta["role"] = "success_sentinel"
    graph.add_node(finish_sensor)
    graph.add_edge(finish_id, finish_sensor_id, LinkType.SUB)
    
    # wait (SCRIPT with TERMINAL wait child)
    wait = Node(nid=wait_id, ntype=NodeType.SCRIPT)
    wait.meta["role"] = "wait_phase"
    wait.meta["pack_phase"] = "wait"
    graph.add_node(wait)
    created_ids["wait"] = wait_id
    
    # Add wait sensor as TERMINAL child
    wait_sensor_id = f"{prefix}_wait_sensor"
    wait_sensor = Node(nid=wait_sensor_id, ntype=NodeType.TERMINAL, predicate=lambda env: True)
    wait_sensor.meta["role"] = "wait_for_confirmation"
    graph.add_node(wait_sensor)
    graph.add_edge(wait_id, wait_sensor_id, LinkType.SUB)
    
    # =========================================================================
    # Wire SUB links: root → children (top-down requests)
    # =========================================================================
    base_weight = 1.0
    if mutate_edges:
        base_weight = random.uniform(0.85, 1.15)  # Weight variation (stored in meta, not add_edge)
    
    graph.add_edge(root_id, detect_id, LinkType.SUB)
    graph.add_edge(root_id, execute_id, LinkType.SUB)
    graph.add_edge(root_id, finish_id, LinkType.SUB)
    graph.add_edge(root_id, wait_id, LinkType.SUB)
    
    # =========================================================================
    # Wire POR sequence: detect → execute → finish → wait (temporal ordering)
    # All are SCRIPT nodes now, so POR edges are valid!
    # =========================================================================
    graph.add_edge(detect_id, execute_id, LinkType.POR)
    graph.add_edge(execute_id, finish_id, LinkType.POR)
    graph.add_edge(finish_id, wait_id, LinkType.POR)
    
    # =========================================================================
    # Wire to parent (attach pack under parent's SUB)
    # =========================================================================
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, root_id, LinkType.SUB)
    
    # =========================================================================
    # Add actuator to execute phase
    # =========================================================================
    actuator_id = f"{prefix}_actuator"
    actuator = Node(nid=actuator_id, ntype=NodeType.TERMINAL, predicate=actuator_fn)
    actuator.meta["role"] = "actuator"
    actuator.meta["pack_phase"] = "execute"
    
    # TRIAL actuators get low weight (0.3) to prevent hallucinated moves
    actuator.meta["actuator_weight"] = 0.3 if is_trial else 1.0
    actuator.meta["transient"] = is_trial
    
    graph.add_node(actuator)
    graph.add_edge(execute_id, actuator_id, LinkType.SUB)
    created_ids["actuator"] = actuator_id
    
    # =========================================================================
    # Create stem cell attachment slots under execute (for recursion)
    # =========================================================================
    attach_stem_cells = min(attach_stem_cells, 2)  # Cap at 2
    created_ids["stem_slots"] = []
    
    for i in range(attach_stem_cells):
        slot_id = f"{prefix}_stem_slot_{i}"
        # Slots are SCRIPT nodes that stem cells can attach to later
        slot = Node(nid=slot_id, ntype=NodeType.SCRIPT)
        slot.meta["role"] = "stem_cell_slot"
        slot.meta["slot_index"] = i
        slot.meta["attached"] = False
        slot.meta["transient"] = True
        
        # Inherit signature for future children
        if parent_signature:
            slot.meta["pattern_signature"] = _inherit_signature(parent_signature, 0.2)
        
        graph.add_node(slot)
        graph.add_edge(execute_id, slot_id, LinkType.SUB)
        created_ids["stem_slots"].append(slot_id)
    
    return created_ids


def _inherit_signature(
    parent_signature: List[float], 
    mutation_rate: float = 0.2
) -> List[float]:
    """
    Create child signature: 80% parent + 20% mutation.
    
    This enables pattern exploration while preserving proven features.
    """
    if not parent_signature:
        return []
    
    child_sig = []
    for val in parent_signature:
        if random.random() < mutation_rate:
            # Mutate: add noise centered on parent value
            mutated = val + random.gauss(0, 0.1)
            child_sig.append(max(0.0, min(1.0, mutated)))  # Clamp [0,1]
        else:
            # Inherit directly
            child_sig.append(val)
    
    return child_sig


def get_actuator_weight(node: Node, current_xp: Optional[int] = None) -> float:
    """
    Get confidence-weighted actuator strength with ramping.
    
    TRIAL → 0.3 weight (safe exploration)
    MATURE → 1.0 weight (full confidence)
    XP-based ramp: Linear from 0.3 (XP=50) to 1.0 (XP=100)
    """
    base_weight = node.meta.get("actuator_weight", 1.0)
    
    # If XP provided, use linear ramp
    if current_xp is not None:
        # XP=50 → 0.3, XP=100 → 1.0
        progress = min(1.0, max(0.0, (current_xp - 50) / 50))
        return 0.3 + progress * 0.7
    
    # Otherwise use tier-based weight
    if node.meta.get("transient", False):
        tier = node.meta.get("tier", "trial")
        if tier == "trial":
            return base_weight * 0.3
    
    return base_weight


def boost_actuator_for_affordance(
    node: Node, 
    affordance_delta: float,
    boost_threshold: float = 0.2,
    boost_amount: float = 0.1
) -> float:
    """
    Temporarily boost actuator weight if positive affordance delta.
    
    If delta > threshold, add boost_amount to weight for this tick.
    """
    base = get_actuator_weight(node)
    
    if affordance_delta > boost_threshold:
        return min(1.0, base + boost_amount)
    
    return base


def prune_pack(graph: Graph, pack_root_id: str) -> List[str]:
    """
    Remove entire pack from graph (root + all children).
    
    Returns list of pruned node IDs.
    """
    if pack_root_id not in graph.nodes:
        return []
    
    pruned = []
    
    # BFS to find all descendants
    to_visit = [pack_root_id]
    while to_visit:
        node_id = to_visit.pop(0)
        if node_id in graph.nodes:
            # Find SUB children
            for edge_key, edge in list(graph.edges.items()):
                if edge.src == node_id and edge.ltype == LinkType.SUB:
                    to_visit.append(edge.dst)
            
            # Remove node and its edges
            graph.remove_node(node_id)
            pruned.append(node_id)
    
    return pruned


# =============================================================================
# STEM CELL TEMPLATE PACKS
# Reusable patterns for M5 evolution - like "NAND gates" for neural circuits
# =============================================================================

def spawn_and_gate_pack(
    gate_name: str,
    parent_id: str,
    graph: Graph,
    conditions: List[Callable[[Dict[str, Any]], bool]],
    then_action: Callable[[Dict[str, Any]], Optional[str]],
    is_trial: bool = True,
) -> Dict[str, str]:
    """
    AND-Gate Pack: Triggers action only if ALL conditions fire.
    
    Uses "min" aggregation - the gate node activates only when all
    condition sensors report True. Essential for co-conditions like
    "fence_established AND king_close" for KRK shrinking.
    
    Args:
        gate_name: Unique identifier for this gate
        parent_id: Node ID to attach gate under (via SUB)
        graph: ReCoN graph to add nodes to
        conditions: List of predicate functions (all must return True)
        then_action: Action function to execute when all conditions fire
        is_trial: Whether gate is in TRIAL state
        
    Returns:
        Dict of created node IDs: {"gate", "actuator", "conditions": [...]}
    """
    if not conditions:
        return {}
    
    created_ids: Dict[str, Any] = {}
    prefix = f"and_{gate_name}"
    
    # Create gate SCRIPT node with min aggregation
    gate_id = f"{prefix}_gate"
    gate_node = Node(nid=gate_id, ntype=NodeType.SCRIPT)
    gate_node.meta["origin"] = "and_gate_pack"
    gate_node.meta["aggregation"] = "min"  # All must fire
    gate_node.meta["transient"] = is_trial
    gate_node.meta["tier"] = "trial" if is_trial else "mature"
    graph.add_node(gate_node)
    created_ids["gate"] = gate_id
    
    # Attach to parent
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, gate_id, LinkType.SUB)
    
    # Create condition sensors and attach via SUB
    cond_ids = []
    for i, cond_fn in enumerate(conditions):
        cond_id = f"{prefix}_cond_{i}"
        cond_node = Node(nid=cond_id, ntype=NodeType.TERMINAL, predicate=cond_fn)
        cond_node.meta["role"] = "and_condition"
        cond_node.meta["condition_index"] = i
        graph.add_node(cond_node)
        graph.add_edge(gate_id, cond_id, LinkType.SUB)
        cond_ids.append(cond_id)
    created_ids["conditions"] = cond_ids
    
    # Create actuator and attach via POR (triggers after gate confirms)
    actuator_id = f"{prefix}_action"
    actuator_node = Node(nid=actuator_id, ntype=NodeType.TERMINAL, predicate=then_action)
    actuator_node.meta["role"] = "gate_actuator"
    actuator_node.meta["actuator_weight"] = 0.3 if is_trial else 1.0
    graph.add_node(actuator_node)
    graph.add_edge(gate_id, actuator_id, LinkType.SUB)  # Changed: TERMINAL can only be targeted by SUB
    created_ids["actuator"] = actuator_id
    
    return created_ids


def spawn_or_gate_pack(
    gate_name: str,
    parent_id: str,
    graph: Graph,
    conditions: List[Callable[[Dict[str, Any]], bool]],
    then_action: Callable[[Dict[str, Any]], Optional[str]],
    is_trial: bool = True,
) -> Dict[str, str]:
    """
    OR-Gate Pack: Triggers action if ANY condition fires.
    
    Uses "max" aggregation - the gate node activates when at least one
    condition sensor reports True. Useful for alternatives like
    "shrink_box OR take_opposition" in exploration.
    
    Args:
        gate_name: Unique identifier for this gate
        parent_id: Node ID to attach gate under (via SUB)
        graph: ReCoN graph to add nodes to
        conditions: List of predicate functions (any can return True)
        then_action: Action function to execute when any condition fires
        is_trial: Whether gate is in TRIAL state
        
    Returns:
        Dict of created node IDs: {"gate", "actuator", "conditions": [...]}
    """
    if not conditions:
        return {}
    
    created_ids: Dict[str, Any] = {}
    prefix = f"or_{gate_name}"
    
    # Create gate SCRIPT node with max aggregation
    gate_id = f"{prefix}_gate"
    gate_node = Node(nid=gate_id, ntype=NodeType.SCRIPT)
    gate_node.meta["origin"] = "or_gate_pack"
    gate_node.meta["aggregation"] = "max"  # Any fires = gate fires
    gate_node.meta["transient"] = is_trial
    gate_node.meta["tier"] = "trial" if is_trial else "mature"
    graph.add_node(gate_node)
    created_ids["gate"] = gate_id
    
    # Attach to parent
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, gate_id, LinkType.SUB)
    
    # Create condition sensors and attach via SUB
    cond_ids = []
    for i, cond_fn in enumerate(conditions):
        cond_id = f"{prefix}_cond_{i}"
        cond_node = Node(nid=cond_id, ntype=NodeType.TERMINAL, predicate=cond_fn)
        cond_node.meta["role"] = "or_condition"
        cond_node.meta["condition_index"] = i
        graph.add_node(cond_node)
        graph.add_edge(gate_id, cond_id, LinkType.SUB)
        cond_ids.append(cond_id)
    created_ids["conditions"] = cond_ids
    
    # Create actuator and attach via POR
    actuator_id = f"{prefix}_action"
    actuator_node = Node(nid=actuator_id, ntype=NodeType.TERMINAL, predicate=then_action)
    actuator_node.meta["role"] = "gate_actuator"
    actuator_node.meta["actuator_weight"] = 0.3 if is_trial else 1.0
    graph.add_node(actuator_node)
    graph.add_edge(gate_id, actuator_id, LinkType.SUB)  # Changed: TERMINAL can only be targeted by SUB
    created_ids["actuator"] = actuator_id
    
    return created_ids


def spawn_not_gate_pack(
    gate_name: str,
    parent_id: str,
    graph: Graph,
    condition: Callable[[Dict[str, Any]], bool],
    then_action: Callable[[Dict[str, Any]], Optional[str]],
    is_trial: bool = True,
) -> Dict[str, str]:
    """
    NOT-Gate Pack: Triggers action when condition is OFF.
    
    Inverts a condition - useful for safety checks like
    "move if NOT stalemate_danger" or "approach if NOT rook_threatened".
    
    Args:
        gate_name: Unique identifier for this gate
        parent_id: Node ID to attach gate under (via SUB)
        graph: ReCoN graph to add nodes to
        condition: Predicate function to invert
        then_action: Action function to execute when condition is False
        is_trial: Whether gate is in TRIAL state
        
    Returns:
        Dict of created node IDs: {"gate", "actuator", "inverted_condition"}
    """
    created_ids: Dict[str, str] = {}
    prefix = f"not_{gate_name}"
    
    # Create inverted condition sensor
    def inverted_predicate(node: Node, env: Dict[str, Any]) -> bool:
        result = condition(node, env) if callable(condition) else condition(env)
        return not result
    
    gate_id = f"{prefix}_gate"
    gate_node = Node(nid=gate_id, ntype=NodeType.TERMINAL, predicate=inverted_predicate)
    gate_node.meta["origin"] = "not_gate_pack"
    gate_node.meta["inverted"] = True
    gate_node.meta["transient"] = is_trial
    graph.add_node(gate_node)
    created_ids["gate"] = gate_id
    
    # Attach to parent
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, gate_id, LinkType.SUB)
    
    # Create actuator and attach via POR
    actuator_id = f"{prefix}_action"
    actuator_node = Node(nid=actuator_id, ntype=NodeType.TERMINAL, predicate=then_action)
    actuator_node.meta["role"] = "gate_actuator"
    actuator_node.meta["actuator_weight"] = 0.3 if is_trial else 1.0
    graph.add_node(actuator_node)
    graph.add_edge(gate_id, actuator_id, LinkType.POR)
    created_ids["actuator"] = actuator_id
    
    return created_ids


def spawn_sequence_pack(
    seq_name: str,
    parent_id: str,
    graph: Graph,
    steps: List[Dict[str, Callable]],
    final_sentinel: Optional[Callable[[Dict[str, Any]], bool]] = None,
    is_trial: bool = True,
) -> Dict[str, Any]:
    """
    Sequence Pack: POR-chained steps for multi-move tactics.
    
    Creates a chain of SCRIPT nodes linked by POR, enabling multi-tick
    execution of tactics like "drive → tempo → approach → mate".
    Each step can have optional sensor (precondition) and actuator (action).
    
    Args:
        seq_name: Unique identifier for this sequence
        parent_id: Node ID to attach sequence under (via SUB)
        graph: ReCoN graph to add nodes to
        steps: List of dicts with optional 'sensor' and 'actuator' callables
               [{"sensor": fn, "actuator": fn}, ...]
        final_sentinel: Optional final condition to confirm sequence completion
        is_trial: Whether sequence is in TRIAL state
        
    Returns:
        Dict with "root", "steps": [...], "final_sentinel" (if provided)
    
    Example:
        steps = [
            {"sensor": cut_established, "actuator": approach_king},
            {"sensor": king_close, "actuator": shrink_box},
            {"actuator": deliver_mate},
        ]
    """
    if not steps:
        return {}
    
    created_ids: Dict[str, Any] = {"steps": []}
    prefix = f"seq_{seq_name}"
    
    # Create root SCRIPT for sequence
    root_id = f"{prefix}_root"
    root_node = Node(nid=root_id, ntype=NodeType.SCRIPT)
    root_node.meta["origin"] = "sequence_pack"
    root_node.meta["step_count"] = len(steps)
    root_node.meta["transient"] = is_trial
    root_node.meta["tier"] = "trial" if is_trial else "mature"
    graph.add_node(root_node)
    created_ids["root"] = root_id
    
    # Attach to parent
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, root_id, LinkType.SUB)
    
    # Create POR-linked steps
    prev_id = root_id
    for i, step in enumerate(steps):
        step_info: Dict[str, str] = {"index": i}
        step_id = f"{prefix}_step_{i}"
        
        # Create step SCRIPT node
        step_node = Node(nid=step_id, ntype=NodeType.SCRIPT)
        step_node.meta["role"] = "sequence_step"
        step_node.meta["step_index"] = i
        graph.add_node(step_node)
        step_info["script"] = step_id
        
        # Link from previous via POR (temporal ordering)
        graph.add_edge(prev_id, step_id, LinkType.POR)
        
        # Add optional sensor (precondition for this step)
        if step.get("sensor"):
            sensor_id = f"{prefix}_step_{i}_sensor"
            sensor_node = Node(nid=sensor_id, ntype=NodeType.TERMINAL, predicate=step["sensor"])
            sensor_node.meta["role"] = "step_precondition"
            graph.add_node(sensor_node)
            graph.add_edge(step_id, sensor_id, LinkType.SUB)
            step_info["sensor"] = sensor_id
        
        # Add optional actuator (action for this step)
        if step.get("actuator"):
            actuator_id = f"{prefix}_step_{i}_actuator"
            actuator_node = Node(nid=actuator_id, ntype=NodeType.TERMINAL, predicate=step["actuator"])
            actuator_node.meta["role"] = "step_actuator"
            actuator_node.meta["actuator_weight"] = 0.3 if is_trial else 1.0
            graph.add_node(actuator_node)
            graph.add_edge(step_id, actuator_id, LinkType.SUB)
            step_info["actuator"] = actuator_id
        
        created_ids["steps"].append(step_info)
        prev_id = step_id
    
    # Add optional final sentinel
    if final_sentinel:
        sentinel_id = f"{prefix}_final_sentinel"
        sentinel_node = Node(nid=sentinel_id, ntype=NodeType.TERMINAL, predicate=final_sentinel)
        sentinel_node.meta["role"] = "sequence_sentinel"
        graph.add_node(sentinel_node)
        graph.add_edge(prev_id, sentinel_id, LinkType.POR)
        created_ids["final_sentinel"] = sentinel_id
    
    return created_ids


def spawn_phase_triad_pack(
    triad_name: str,
    parent_id: str,
    graph: Graph,
    check_fn: Callable[[Dict[str, Any]], bool],
    move_fn: Callable[[Dict[str, Any]], Optional[str]],
    wait_fn: Callable[[Dict[str, Any]], bool],
    is_trial: bool = True,
) -> Dict[str, str]:
    """
    Phase Triad Pack: check → move → wait cycle.
    
    Matches the existing topology's pN_check-pN_move-pN_wait pattern.
    Useful for phased execution: "check fence → move rook → wait confirm".
    
    Args:
        triad_name: Unique identifier for this triad
        parent_id: Node ID to attach triad under (via SUB)
        graph: ReCoN graph to add nodes to
        check_fn: Condition sensor for check phase
        move_fn: Action function for move phase
        wait_fn: Confirmation sensor for wait phase
        is_trial: Whether triad is in TRIAL state
        
    Returns:
        Dict of created node IDs: {"check", "move", "wait"}
    """
    created_ids: Dict[str, str] = {}
    prefix = f"triad_{triad_name}"
    
    # Check phase (SCRIPT with condition sensor)
    check_id = f"{prefix}_check"
    check_node = Node(nid=check_id, ntype=NodeType.SCRIPT)
    check_node.meta["origin"] = "phase_triad_pack"
    check_node.meta["phase"] = "check"
    check_node.meta["transient"] = is_trial
    graph.add_node(check_node)
    created_ids["check"] = check_id
    
    # Attach to parent
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, check_id, LinkType.SUB)
    
    # Check sensor
    check_sensor_id = f"{prefix}_check_sensor"
    check_sensor = Node(nid=check_sensor_id, ntype=NodeType.TERMINAL, predicate=check_fn)
    check_sensor.meta["role"] = "check_condition"
    graph.add_node(check_sensor)
    graph.add_edge(check_id, check_sensor_id, LinkType.SUB)
    created_ids["check_sensor"] = check_sensor_id
    
    # Move phase (SCRIPT with actuator)
    move_id = f"{prefix}_move"
    move_node = Node(nid=move_id, ntype=NodeType.SCRIPT)
    move_node.meta["phase"] = "move"
    graph.add_node(move_node)
    graph.add_edge(check_id, move_id, LinkType.POR)  # After check confirms
    created_ids["move"] = move_id
    
    # Move actuator
    move_actuator_id = f"{prefix}_move_actuator"
    move_actuator = Node(nid=move_actuator_id, ntype=NodeType.TERMINAL, predicate=move_fn)
    move_actuator.meta["role"] = "move_actuator"
    move_actuator.meta["actuator_weight"] = 0.3 if is_trial else 1.0
    graph.add_node(move_actuator)
    graph.add_edge(move_id, move_actuator_id, LinkType.SUB)
    created_ids["move_actuator"] = move_actuator_id
    
    # Wait phase (TERMINAL sensor for confirmation)
    wait_id = f"{prefix}_wait"
    wait_node = Node(nid=wait_id, ntype=NodeType.TERMINAL, predicate=wait_fn)
    wait_node.meta["role"] = "wait_confirmation"
    wait_node.meta["phase"] = "wait"
    graph.add_node(wait_node)
    graph.add_edge(move_id, wait_id, LinkType.POR)  # After move completes
    created_ids["wait"] = wait_id
    
    return created_ids


def spawn_branch_pack(
    branch_name: str,
    parent_id: str,
    graph: Graph,
    children: List[Dict[str, Any]],
    arbiter_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    is_trial: bool = True,
) -> Dict[str, Any]:
    """
    Branch Pack: Parallel SUB branches from root.
    
    Creates a root SCRIPT with multiple children executing in parallel.
    Optional arbiter for merging results. Like rook_leg/king_leg in krk_legs.
    
    Args:
        branch_name: Unique identifier for this branch
        parent_id: Node ID to attach branch under (via SUB)
        graph: ReCoN graph to add nodes to
        children: List of dicts with 'sensor' and/or 'actuator' callables
        arbiter_fn: Optional arbiter for merging parallel results
        is_trial: Whether branch is in TRIAL state
        
    Returns:
        Dict with "root", "children": [...], "arbiter" (if provided)
    """
    if not children:
        return {}
    
    created_ids: Dict[str, Any] = {"children": []}
    prefix = f"branch_{branch_name}"
    
    # Create root SCRIPT
    root_id = f"{prefix}_root"
    root_node = Node(nid=root_id, ntype=NodeType.SCRIPT)
    root_node.meta["origin"] = "branch_pack"
    root_node.meta["child_count"] = len(children)
    root_node.meta["transient"] = is_trial
    root_node.meta["tier"] = "trial" if is_trial else "mature"
    graph.add_node(root_node)
    created_ids["root"] = root_id
    
    # Attach to parent
    if parent_id in graph.nodes:
        graph.add_edge(parent_id, root_id, LinkType.SUB)
    
    # Create parallel children
    for i, child in enumerate(children):
        child_info: Dict[str, str] = {"index": i}
        child_id = f"{prefix}_child_{i}"
        
        # Create child SCRIPT
        child_node = Node(nid=child_id, ntype=NodeType.SCRIPT)
        child_node.meta["role"] = "branch_child"
        child_node.meta["child_index"] = i
        graph.add_node(child_node)
        graph.add_edge(root_id, child_id, LinkType.SUB)  # Parallel via SUB
        child_info["script"] = child_id
        
        # Add optional sensor
        if child.get("sensor"):
            sensor_id = f"{prefix}_child_{i}_sensor"
            sensor_node = Node(nid=sensor_id, ntype=NodeType.TERMINAL, predicate=child["sensor"])
            sensor_node.meta["role"] = "branch_sensor"
            graph.add_node(sensor_node)
            graph.add_edge(child_id, sensor_id, LinkType.SUB)
            child_info["sensor"] = sensor_id
        
        # Add optional actuator
        if child.get("actuator"):
            actuator_id = f"{prefix}_child_{i}_actuator"
            actuator_node = Node(nid=actuator_id, ntype=NodeType.TERMINAL, predicate=child["actuator"])
            actuator_node.meta["role"] = "branch_actuator"
            actuator_node.meta["actuator_weight"] = 0.3 if is_trial else 1.0
            graph.add_node(actuator_node)
            graph.add_edge(child_id, actuator_id, LinkType.SUB)
            child_info["actuator"] = actuator_id
        
        created_ids["children"].append(child_info)
    
    # Add optional arbiter
    if arbiter_fn:
        arbiter_id = f"{prefix}_arbiter"
        arbiter_node = Node(nid=arbiter_id, ntype=NodeType.SCRIPT)
        arbiter_node.meta["role"] = "branch_arbiter"
        graph.add_node(arbiter_node)
        graph.add_edge(root_id, arbiter_id, LinkType.POR)  # After children
        created_ids["arbiter"] = arbiter_id
        
        # Arbiter actuator
        arbiter_actuator_id = f"{prefix}_arbiter_actuator"
        arbiter_actuator = Node(nid=arbiter_actuator_id, ntype=NodeType.TERMINAL, predicate=arbiter_fn)
        arbiter_actuator.meta["role"] = "arbiter_actuator"
        graph.add_node(arbiter_actuator)
        graph.add_edge(arbiter_id, arbiter_actuator_id, LinkType.SUB)
        created_ids["arbiter_actuator"] = arbiter_actuator_id
    
    return created_ids

