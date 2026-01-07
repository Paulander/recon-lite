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
    
    # =========================================================================
    # Create 4-phase children: detect → execute → finish → wait
    # =========================================================================
    detect_id = f"{prefix}_detect"
    execute_id = f"{prefix}_execute"
    finish_id = f"{prefix}_finish"
    wait_id = f"{prefix}_wait"
    
    # detect (TERMINAL sensor - condition check)
    detect = Node(nid=detect_id, ntype=NodeType.TERMINAL, predicate=condition_sensor_fn)
    detect.meta["role"] = "condition_sensor"
    detect.meta["pack_phase"] = "detect"
    graph.add_node(detect)
    created_ids["detect"] = detect_id
    
    # execute (SCRIPT - can have children for recursion)
    execute = Node(nid=execute_id, ntype=NodeType.SCRIPT)
    execute.meta["role"] = "execute_phase"
    execute.meta["pack_phase"] = "execute"
    execute.meta["aggregation"] = "avg"  # OR-like for sibling actuators
    graph.add_node(execute)
    created_ids["execute"] = execute_id
    
    # finish (TERMINAL sensor - success detector/sentinel)
    finish = Node(nid=finish_id, ntype=NodeType.TERMINAL, predicate=sentinel_fn)
    finish.meta["role"] = "success_sentinel"
    finish.meta["pack_phase"] = "finish"
    graph.add_node(finish)
    created_ids["finish"] = finish_id
    
    # wait (TERMINAL sensor - wait for SUR confirmation)
    wait = Node(nid=wait_id, ntype=NodeType.TERMINAL, predicate=lambda env: True)
    wait.meta["role"] = "wait_for_confirmation"
    wait.meta["pack_phase"] = "wait"
    graph.add_node(wait)
    created_ids["wait"] = wait_id
    
    # =========================================================================
    # Wire SUB links: root → children (top-down requests)
    # =========================================================================
    base_weight = 1.0
    if mutate_edges:
        base_weight = random.uniform(0.85, 1.15)  # Weight variation
    
    graph.add_edge(root_id, detect_id, LinkType.SUB, weight=base_weight)
    graph.add_edge(root_id, execute_id, LinkType.SUB, weight=base_weight)
    graph.add_edge(root_id, finish_id, LinkType.SUB, weight=base_weight)
    graph.add_edge(root_id, wait_id, LinkType.SUB, weight=base_weight)
    
    # =========================================================================
    # Wire POR sequence: detect → execute → finish → wait (temporal ordering)
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
