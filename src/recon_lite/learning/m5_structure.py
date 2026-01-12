"""M5 Structure Learning Module.

Implements the "Dreamer" component that analyzes traces for affordance spikes,
identifies high-impact stem cells, and promotes them to permanent nodes.

Usage:
    from recon_lite.learning.m5_structure import StructureLearner
    
    learner = StructureLearner(registry, trace_db)
    
    # After a batch of games
    stats = learner.apply_structural_phase(
        stem_manager,
        episodes,
        max_promotions=2
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from ..nodes.stem_cell import StemCellTerminal, StemCellManager, StemCellState
    HAS_STEM_CELL = True
except ImportError:
    HAS_STEM_CELL = False

try:
    from ..models.registry import TopologyRegistry
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False


# =============================================================================
# M5 RECURSIVE BRANCHING - Depth Limiting Constants
# =============================================================================

# Backbone nodes that form the "trunk" of the network
BACKBONE_NODES: Set[str] = {
    "kpk_root", "kpk_detect", "kpk_execute", "kpk_finish", "kpk_wait",
    "krk_root", "krk_detect", "krk_execute", "krk_finish",
    "kqk_root", "kqk_detect", "kqk_execute", "kqk_finish",
}

# Maximum depth for vertical branches (beyond this, hoist to new manager)
MAX_BRANCH_DEPTH = 5


@dataclass
class AffordanceSpike:
    """Record of a high-impact activation moment."""
    tick: int
    episode_id: str
    node_id: str
    reward: float
    fen: str
    features: Optional[List[float]] = None
    move: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "episode_id": self.episode_id,
            "node_id": self.node_id,
            "reward": self.reward,
            "fen": self.fen,
            "features": self.features,
            "move": self.move,
        }


@dataclass
class PromotionResult:
    """Result of a stem cell promotion."""
    cell_id: str
    new_node_id: str
    parent_id: str
    tick: int
    success: bool
    signature_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class PruningResult:
    """Result of edge pruning analysis."""
    edge_key: str
    old_weight: float
    new_weight: float
    pruned: bool
    games_at_zero: int


# =============================================================================
# M5 RECURSIVE BRANCHING - Depth Computation
# =============================================================================

def compute_node_depth(graph: "Graph", node_id: str, backbone_nodes: Set[str] = None) -> int:
    """
    Compute depth from backbone to a node.
    
    Depth 0 = backbone node
    Depth 1 = direct child of backbone
    Depth 2+ = nested in hierarchy
    
    Args:
        graph: The Graph to traverse
        node_id: Node to compute depth for
        backbone_nodes: Set of backbone node IDs (uses BACKBONE_NODES if None)
        
    Returns:
        Depth from nearest backbone ancestor (0 if node is backbone)
    """
    if backbone_nodes is None:
        backbone_nodes = BACKBONE_NODES
    
    if node_id in backbone_nodes:
        return 0
    
    depth = 0
    current = node_id
    visited = set()
    
    while current and current not in backbone_nodes and current not in visited:
        visited.add(current)
        parent = graph.parent_of(current)
        if parent:
            depth += 1
            current = parent
        else:
            break
    
    return depth


def is_at_max_depth(graph: "Graph", node_id: str, max_depth: int = MAX_BRANCH_DEPTH) -> bool:
    """Check if a node is at maximum allowed branch depth."""
    return compute_node_depth(graph, node_id) >= max_depth


# =============================================================================
# M5 RECURSIVE BRANCHING - SCRIPT Wrappers for POR Links
# =============================================================================

def wrap_terminal_as_script(
    graph: "Graph",
    registry: "TopologyRegistry",
    terminal_id: str,
    current_tick: int = 0,
) -> Optional[str]:
    """
    Wrap a TERMINAL sensor in a SCRIPT node to enable POR links.
    
    This transforms a "Sensor" into a "Tactical Goal" that can participate
    in sequential reasoning chains. POR links can only connect SCRIPT nodes
    (per graph.py validation), so we need this wrapper to enable sequential
    patterns like Opposition -> Protect -> Promote.
    
    Args:
        graph: The Graph to modify
        registry: TopologyRegistry for persisting changes
        terminal_id: ID of the terminal to wrap
        current_tick: Current tick for metadata
        
    Returns:
        ID of the new wrapper SCRIPT node, or None if failed
    """
    from ..graph import Node, NodeType, LinkType
    
    if terminal_id not in graph.nodes:
        return None
    
    terminal_node = graph.nodes[terminal_id]
    if terminal_node.ntype != NodeType.TERMINAL:
        return None  # Only wrap terminals
    
    # Check if already wrapped
    for edge in graph.edges:
        if edge.dst == terminal_id and edge.ltype == LinkType.SUB:
            parent = graph.nodes.get(edge.src)
            if parent and parent.meta.get("origin") == "por_wrapper":
                # Already wrapped
                return edge.src
    
    wrapper_id = f"goal_{terminal_id}"
    
    # Create SCRIPT wrapper node
    wrapper_node = Node(
        nid=wrapper_id,
        ntype=NodeType.SCRIPT,
    )
    wrapper_node.meta["wrapped_terminal"] = terminal_id
    wrapper_node.meta["origin"] = "por_wrapper"
    wrapper_node.meta["created_tick"] = current_tick
    
    # Find terminal's current parent
    old_parent = graph.parent_of(terminal_id)
    
    try:
        # Add wrapper node to graph
        graph.add_node(wrapper_node)
        
        if old_parent:
            # Remove old SUB edge from parent to terminal
            graph.edges = [
                e for e in graph.edges
                if not (e.src == old_parent and e.dst == terminal_id and e.ltype == LinkType.SUB)
            ]
            # Update parent tracking
            graph.parent[terminal_id] = None
            if terminal_id in graph.parents_fanin:
                graph.parents_fanin[terminal_id] = [
                    p for p in graph.parents_fanin[terminal_id] if p != old_parent
                ]
            
            # Wire wrapper under old parent
            graph.add_edge(old_parent, wrapper_id, LinkType.SUB)
        
        # Wire terminal under wrapper
        graph.add_edge(wrapper_id, terminal_id, LinkType.SUB)
        
        # Persist to registry
        if registry:
            node_spec = {
                "id": wrapper_id,
                "type": "SCRIPT",
                "group": "por_wrapper",
                "factory": None,
                "meta": wrapper_node.meta,
            }
            registry.add_node(node_spec, tick=current_tick)
            if old_parent:
                registry.add_edge(old_parent, wrapper_id, "SUB", weight=1.0, tick=current_tick)
            registry.add_edge(wrapper_id, terminal_id, "SUB", weight=1.0, tick=current_tick)
        
        return wrapper_id
        
    except Exception as e:
        # Rollback on failure
        if wrapper_id in graph.nodes:
            graph.remove_node(wrapper_id)
        return None


# =============================================================================
# M5.1 MICRO-TEMPORAL POR DISCOVERY - Short Game Sequence Analysis
# =============================================================================

def analyze_micro_temporal_sequences(
    episodes: List[Any],
    stem_manager: "StemCellManager",
    max_game_length: int = 10,
    min_games: int = 5,
) -> Dict[Tuple[str, str], float]:
    """
    Analyze tick-by-tick sequences for short games (<10 ticks) to find
    strong temporal dependencies like King -> Pawn.
    
    For very short games, every tick matters. This function tracks the
    exact firing order of nodes within each game and calculates the
    probability that node A fires before node B in winning games.
    
    Args:
        episodes: List of EpisodeRecord objects
        stem_manager: Manager with cell tracking info
        max_game_length: Maximum game length to consider "short"
        min_games: Minimum games to calculate confidence
        
    Returns:
        Dict mapping (predecessor_cell_id, successor_cell_id) to confidence score
    """
    # Track firing order in short winning games
    # Key: (cell_a, cell_b), Value: {"a_before_b": count, "b_before_a": count}
    order_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    
    for episode in episodes:
        # Check if this was a win
        outcome = getattr(episode, 'outcome', None)
        if outcome != 'win':
            continue
        
        ticks = getattr(episode, 'ticks', [])
        if len(ticks) > max_game_length:
            continue  # Skip long games
        
        # Track first activation tick for each cell
        first_activation: Dict[str, int] = {}
        
        for tick in ticks:
            tick_id = getattr(tick, 'tick_id', 0)
            active_nodes = getattr(tick, 'active_nodes', [])
            
            for node_id in active_nodes:
                # Find corresponding cell
                for cell_id, cell in stem_manager.cells.items():
                    if cell.trial_node_id == node_id:
                        if cell_id not in first_activation:
                            first_activation[cell_id] = tick_id
                        break
        
        # Compare all pairs of activated cells
        activated_cells = list(first_activation.keys())
        for i, cell_a in enumerate(activated_cells):
            for cell_b in activated_cells[i+1:]:
                tick_a = first_activation[cell_a]
                tick_b = first_activation[cell_b]
                
                # Create canonical key (sorted)
                key = tuple(sorted([cell_a, cell_b]))
                if key not in order_counts:
                    order_counts[key] = {"a_before_b": 0, "b_before_a": 0, "same_tick": 0}
                
                # Determine order
                if tick_a < tick_b:
                    if cell_a < cell_b:
                        order_counts[key]["a_before_b"] += 1
                    else:
                        order_counts[key]["b_before_a"] += 1
                elif tick_b < tick_a:
                    if cell_b < cell_a:
                        order_counts[key]["a_before_b"] += 1
                    else:
                        order_counts[key]["b_before_a"] += 1
                else:
                    order_counts[key]["same_tick"] += 1
    
    # Calculate confidence scores for each pair
    result: Dict[Tuple[str, str], float] = {}
    
    for (cell_a, cell_b), counts in order_counts.items():
        total = counts["a_before_b"] + counts["b_before_a"]
        if total < min_games:
            continue
        
        # Confidence = how consistently one fires before the other
        if counts["a_before_b"] > counts["b_before_a"]:
            confidence = counts["a_before_b"] / total
            result[(cell_a, cell_b)] = confidence
        else:
            confidence = counts["b_before_a"] / total
            result[(cell_b, cell_a)] = confidence
    
    return result


# =============================================================================
# M5 RECURSIVE BRANCHING - POR Chain Discovery
# =============================================================================

def discover_por_chains(
    graph: "Graph",
    stem_manager: "StemCellManager",
    registry: "TopologyRegistry",
    min_sequence_confidence: float = 0.7,
    current_tick: int = 0,
    episodes: Optional[List[Any]] = None,  # For micro-temporal analysis
) -> List[Tuple[str, str, float]]:
    """
    Discover sequential patterns like Opposition -> Protect -> Promote.
    
    Analyzes activation history to find pairs where A consistently
    fires BEFORE B during winning games. Creates POR links between
    SCRIPT nodes (wrapping terminals if needed).
    
    MICRO-TEMPORAL MODE: For short games (<10 ticks), uses tick-by-tick
    analysis to find strong King -> Pawn dependencies.
    
    Args:
        graph: The Graph to modify
        stem_manager: Manager with win-coactivation data
        registry: TopologyRegistry for persisting changes
        min_sequence_confidence: Minimum confidence to create POR link (default 0.7)
        current_tick: Current tick for metadata
        episodes: Episode list for micro-temporal analysis
        
    Returns:
        List of (predecessor_id, successor_id, confidence) tuples for created POR links
    """
    from ..graph import Node, NodeType, LinkType
    
    discovered_chains: List[Tuple[str, str, float]] = []
    
    # MICRO-TEMPORAL ANALYSIS: Prioritize short game sequences
    micro_temporal_pairs: Dict[Tuple[str, str], float] = {}
    if episodes:
        micro_temporal_pairs = analyze_micro_temporal_sequences(
            episodes=episodes,
            stem_manager=stem_manager,
            max_game_length=10,  # Focus on games < 10 ticks
            min_games=5,
        )
    
    # Find highly correlated cell pairs from win-coactivation tracking
    correlated_pairs = stem_manager.find_win_correlated_pairs(
        min_coactivations=30,  # Lower threshold for sequence detection
        min_ratio=min_sequence_confidence,
    )
    
    # Merge micro-temporal pairs with regular correlated pairs
    # Micro-temporal has higher priority for short games
    all_pairs_to_process: List[Tuple[str, str, float, int, bool]] = []
    
    # Add micro-temporal pairs (priority)
    for (cell_a, cell_b), confidence in micro_temporal_pairs.items():
        if confidence >= min_sequence_confidence:
            all_pairs_to_process.append((cell_a, cell_b, confidence, 0, True))  # True = micro-temporal
    
    # Add regular correlated pairs
    for cell_a, cell_b, ratio, co_count in correlated_pairs:
        # Skip if already in micro-temporal
        if (cell_a, cell_b) not in micro_temporal_pairs and (cell_b, cell_a) not in micro_temporal_pairs:
            all_pairs_to_process.append((cell_a, cell_b, ratio, co_count, False))
    
    if not all_pairs_to_process:
        return discovered_chains
    
    for cell_a, cell_b, ratio, co_count, is_micro_temporal in all_pairs_to_process:
        # Get the corresponding graph nodes for these cells
        cell_obj_a = stem_manager.cells.get(cell_a)
        cell_obj_b = stem_manager.cells.get(cell_b)
        
        if not cell_obj_a or not cell_obj_b:
            continue
        
        # Only consider TRIAL or MATURE cells with graph nodes
        node_a = cell_obj_a.trial_node_id
        node_b = cell_obj_b.trial_node_id
        
        if not node_a or not node_b:
            continue
        
        if node_a not in graph.nodes or node_b not in graph.nodes:
            continue
        
        # Determine temporal order (which fires first)
        # Use sample timestamps if available
        samples_a = cell_obj_a.samples
        samples_b = cell_obj_b.samples
        
        # Determine temporal order
        if is_micro_temporal:
            # Micro-temporal analysis already determined order
            # cell_a is the predecessor, cell_b is the successor
            pred_node_id, succ_node_id = node_a, node_b
        else:
            # Use sample timestamps for regular pairs
            if not samples_a or not samples_b:
                continue
            
            # Calculate average tick for each cell
            avg_tick_a = sum(s.tick for s in samples_a) / len(samples_a)
            avg_tick_b = sum(s.tick for s in samples_b) / len(samples_b)
            
            # Predecessor fires first (lower average tick)
            if avg_tick_a < avg_tick_b:
                pred_node_id, succ_node_id = node_a, node_b
            else:
                pred_node_id, succ_node_id = node_b, node_a
        
        # Get actual nodes
        pred_node = graph.nodes.get(pred_node_id)
        succ_node = graph.nodes.get(succ_node_id)
        
        if not pred_node or not succ_node:
            continue
        
        # POR links can only connect SCRIPT nodes - wrap terminals if needed
        if pred_node.ntype == NodeType.TERMINAL:
            wrapped_pred = wrap_terminal_as_script(graph, registry, pred_node_id, current_tick)
            if wrapped_pred:
                pred_node_id = wrapped_pred
                pred_node = graph.nodes.get(pred_node_id)
            else:
                continue  # Skip if wrapping failed
        
        if succ_node.ntype == NodeType.TERMINAL:
            wrapped_succ = wrap_terminal_as_script(graph, registry, succ_node_id, current_tick)
            if wrapped_succ:
                succ_node_id = wrapped_succ
                succ_node = graph.nodes.get(succ_node_id)
            else:
                continue  # Skip if wrapping failed
        
        # Check if POR link already exists
        existing_por = False
        for edge in graph.edges:
            if edge.src == pred_node_id and edge.dst == succ_node_id and edge.ltype == LinkType.POR:
                existing_por = True
                break
        
        if existing_por:
            continue  # Already connected
        
        # Create POR link
        try:
            graph.add_edge(pred_node_id, succ_node_id, LinkType.POR)
            
            # Configure Soft-POR on successor for weighted gating
            succ_node.meta["por_policy"] = "weighted"
            succ_node.meta["por_theta"] = 0.4  # Lowered threshold for easier triggering
            
            # Mark edge metadata
            for edge in graph.edges:
                if edge.src == pred_node_id and edge.dst == succ_node_id and edge.ltype == LinkType.POR:
                    edge.meta["origin"] = "micro_temporal" if is_micro_temporal else "por_discovery"
                    edge.meta["confidence"] = ratio
                    edge.meta["co_activations"] = co_count
                    edge.meta["created_tick"] = current_tick
                    edge.meta["is_micro_temporal"] = is_micro_temporal
                    break
            
            # Persist to registry
            if registry:
                registry.add_edge(
                    pred_node_id, succ_node_id, "POR",
                    weight=ratio,  # Use correlation as weight
                    tick=current_tick,
                )
            
            discovered_chains.append((pred_node_id, succ_node_id, ratio))
            
        except ValueError as e:
            # POR link validation failed (e.g., not both SCRIPT nodes)
            continue
    
    return discovered_chains


# =============================================================================
# M5 RECURSIVE BRANCHING - KRK Box Method POR Discovery
# =============================================================================

def discover_krk_box_method_por(
    graph: "Graph",
    stem_manager: "StemCellManager",
    registry: "TopologyRegistry",
    episodes: List[Any],
    min_sequence_confidence: float = 0.70,
    current_tick: int = 0,
) -> Dict[str, Any]:
    """
    Discover the "Box Method" sequential pattern for KRK endgames.
    
    The Box Method is a specific checkmating technique:
    1. Rook_Cuts_Rank (establishes fence) → 
    2. King_Approaches (closes the net) → 
    3. Rook_Shrinks_Box (tightens confinement)
    
    This sequence CANNOT be "vibed" - if you do step 3 before step 2,
    the enemy king escapes. This is where POR-gating proves its value.
    
    Args:
        graph: The Graph to modify
        stem_manager: Manager with stem cells
        registry: TopologyRegistry for persisting changes  
        episodes: Episode list for sequence analysis
        min_sequence_confidence: Minimum confidence to create POR (default 0.70)
        current_tick: Current tick for metadata
        
    Returns:
        Dict with discovered patterns and stats
    """
    from ..graph import Node, NodeType, LinkType
    import os
    
    results = {
        "box_method_sequences_found": 0,
        "rook_cuts_then_king": 0,
        "king_then_shrink": 0,
        "por_links_created": [],
        "tactical_manager_created": False,
    }
    
    # Only process KRK episodes
    krk_episodes = [
        ep for ep in episodes
        if hasattr(ep, 'notes') and ep.notes and ep.notes.get('domain') == 'krk'
    ]
    
    if not krk_episodes:
        # Check if any episodes have krk-related node activations
        krk_episodes = [
            ep for ep in episodes
            if any('krk' in str(tick.active_nodes).lower() for tick in ep.ticks if hasattr(tick, 'active_nodes'))
        ]
    
    if not krk_episodes:
        return results
    
    # Track sequence patterns
    rook_before_king_count = 0
    king_before_shrink_count = 0
    total_wins = 0
    
    # Analyze winning episodes for sequence patterns
    for ep in krk_episodes:
        if not hasattr(ep, 'result') or ep.result not in ('1-0', 'win'):
            continue
        
        total_wins += 1
        
        # Track when key actions happen
        rook_cut_tick = None
        king_approach_tick = None
        box_shrink_tick = None
        
        for tick in ep.ticks:
            if not hasattr(tick, 'active_nodes'):
                continue
            
            active = tick.active_nodes or []
            tick_id = getattr(tick, 'tick_id', 0)
            
            # Detect Rook cut action
            if any('rook_leg' in n.lower() or 'cut' in n.lower() or 'fence' in n.lower() for n in active):
                if rook_cut_tick is None:
                    rook_cut_tick = tick_id
            
            # Detect King approach action
            if any('king_leg' in n.lower() or 'king_drive' in n.lower() or 'approach' in n.lower() for n in active):
                if king_approach_tick is None:
                    king_approach_tick = tick_id
            
            # Detect Box shrink action
            if any('shrink' in n.lower() or 'box' in n.lower() or 'confinement' in n.lower() for n in active):
                if box_shrink_tick is None:
                    box_shrink_tick = tick_id
        
        # Check for Box Method sequence: Rook → King → Shrink
        if rook_cut_tick is not None and king_approach_tick is not None:
            if rook_cut_tick < king_approach_tick:
                rook_before_king_count += 1
        
        if king_approach_tick is not None and box_shrink_tick is not None:
            if king_approach_tick < box_shrink_tick:
                king_before_shrink_count += 1
    
    if total_wins == 0:
        return results
    
    # Calculate confidence for each sequence
    rook_king_confidence = rook_before_king_count / total_wins if total_wins > 0 else 0
    king_shrink_confidence = king_before_shrink_count / total_wins if total_wins > 0 else 0
    
    results["rook_cuts_then_king"] = rook_before_king_count
    results["king_then_shrink"] = king_before_shrink_count
    
    # If both sequences detected in >70% of wins, create Tactical Box Manager
    if rook_king_confidence >= min_sequence_confidence and king_shrink_confidence >= min_sequence_confidence:
        results["box_method_sequences_found"] = total_wins
        
        # Force-hoist a SCRIPT manager called Tactical_Box_Manager
        try:
            tactical_manager_id = f"tactical_box_manager_{current_tick}"
            
            tactical_node = Node(
                nid=tactical_manager_id,
                ntype=NodeType.SCRIPT,
            )
            tactical_node.meta["origin"] = "box_method_discovery"
            tactical_node.meta["aggregation"] = "sequence"  # Sequential execution
            tactical_node.meta["rook_king_confidence"] = rook_king_confidence
            tactical_node.meta["king_shrink_confidence"] = king_shrink_confidence
            tactical_node.meta["total_wins_analyzed"] = total_wins
            tactical_node.meta["created_tick"] = current_tick
            
            graph.add_node(tactical_node)
            
            # Wire to KRK execute phase
            if "krk_execute" in graph.nodes:
                graph.add_edge("krk_execute", tactical_manager_id, LinkType.SUB)
            elif "krk_root" in graph.nodes:
                graph.add_edge("krk_root", tactical_manager_id, LinkType.SUB)
            
            # Create POR-linked child nodes for the sequence
            # Step 1: Rook cuts
            rook_cut_node_id = f"rook_cuts_step_{current_tick}"
            rook_cut_node = Node(nid=rook_cut_node_id, ntype=NodeType.SCRIPT)
            rook_cut_node.meta["box_method_step"] = 1
            rook_cut_node.meta["action"] = "establish_fence"
            graph.add_node(rook_cut_node)
            graph.add_edge(tactical_manager_id, rook_cut_node_id, LinkType.SUB)
            
            # Step 2: King approaches
            king_approach_node_id = f"king_approaches_step_{current_tick}"
            king_approach_node = Node(nid=king_approach_node_id, ntype=NodeType.SCRIPT)
            king_approach_node.meta["box_method_step"] = 2
            king_approach_node.meta["action"] = "close_net"
            graph.add_node(king_approach_node)
            graph.add_edge(tactical_manager_id, king_approach_node_id, LinkType.SUB)
            
            # Step 3: Rook shrinks box
            box_shrink_node_id = f"rook_shrinks_step_{current_tick}"
            box_shrink_node = Node(nid=box_shrink_node_id, ntype=NodeType.SCRIPT)
            box_shrink_node.meta["box_method_step"] = 3
            box_shrink_node.meta["action"] = "tighten_confinement"
            graph.add_node(box_shrink_node)
            graph.add_edge(tactical_manager_id, box_shrink_node_id, LinkType.SUB)
            
            # Create POR links for sequence: Step1 → Step2 → Step3
            graph.add_edge(rook_cut_node_id, king_approach_node_id, LinkType.POR)
            graph.add_edge(king_approach_node_id, box_shrink_node_id, LinkType.POR)
            
            # Set Soft-POR policy
            king_approach_node.meta["por_policy"] = "weighted"
            king_approach_node.meta["por_theta"] = 0.5
            box_shrink_node.meta["por_policy"] = "weighted"
            box_shrink_node.meta["por_theta"] = 0.5
            
            results["tactical_manager_created"] = True
            results["por_links_created"] = [
                (rook_cut_node_id, king_approach_node_id, rook_king_confidence),
                (king_approach_node_id, box_shrink_node_id, king_shrink_confidence),
            ]
            
            # Persist to registry
            if registry:
                registry.add_node({
                    "id": tactical_manager_id,
                    "type": "SCRIPT",
                    "group": "tactical",
                    "meta": dict(tactical_node.meta),
                }, tick=current_tick)
                
                registry.add_edge("krk_execute" if "krk_execute" in graph.nodes else "krk_root",
                                  tactical_manager_id, "SUB", weight=1.0, tick=current_tick)
                
                for src, dst, conf in results["por_links_created"]:
                    registry.add_edge(src, dst, "POR", weight=conf, tick=current_tick)
            
            print(f"    📦 KRK BOX METHOD: Created Tactical_Box_Manager")
            print(f"       Rook→King confidence: {rook_king_confidence:.1%}")
            print(f"       King→Shrink confidence: {king_shrink_confidence:.1%}")
            
        except Exception as e:
            results["error"] = str(e)
    
    return results


# =============================================================================
# M5 RECURSIVE BRANCHING - Maturation Metrics
# =============================================================================

def compute_branching_metrics(
    graph: "Graph",
    backbone_nodes: Set[str] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for the 'Budding' process (M5 Recursive Branching).
    
    These metrics help monitor the transition from flat to hierarchical topology.
    
    Args:
        graph: The Graph to analyze
        backbone_nodes: Set of backbone node IDs (uses BACKBONE_NODES if None)
        
    Returns:
        Dict with branching metrics:
        - speculative_ands: Count of min()-aggregated nodes
        - branching_factor: Avg children per non-backbone SCRIPT
        - non_backbone_scripts: Count of SCRIPT nodes not in backbone
        - max_depth: Maximum depth from backbone
        - por_count: Number of POR edges
        - sub_count: Number of SUB edges
    """
    from ..graph import NodeType, LinkType
    
    if backbone_nodes is None:
        backbone_nodes = BACKBONE_NODES
    
    # Find non-backbone SCRIPT nodes
    non_backbone_scripts = [
        nid for nid, n in graph.nodes.items()
        if n.ntype == NodeType.SCRIPT and nid not in backbone_nodes
    ]
    
    # Count children per non-backbone SCRIPT
    children_counts = [len(graph.children(nid)) for nid in non_backbone_scripts]
    branching_factor = (
        sum(children_counts) / len(children_counts) 
        if children_counts else 0.0
    )
    
    # Count speculative AND nodes (min-aggregated)
    speculative_ands = sum(
        1 for nid in non_backbone_scripts
        if graph.nodes[nid].meta.get("aggregation") == "and"
        and graph.nodes[nid].meta.get("speculative")
    )
    
    # Count all AND nodes (not just speculative)
    total_and_nodes = sum(
        1 for nid in non_backbone_scripts
        if graph.nodes[nid].meta.get("aggregation") == "and"
    )
    
    # Compute max depth
    max_depth = 0
    for nid in graph.nodes:
        depth = compute_node_depth(graph, nid, backbone_nodes)
        max_depth = max(max_depth, depth)
    
    # Count edge types
    por_count = sum(1 for e in graph.edges if e.ltype == LinkType.POR)
    sub_count = sum(1 for e in graph.edges if e.ltype == LinkType.SUB)
    total_edges = len(graph.edges)
    
    # Calculate edge type ratios
    por_ratio = por_count / total_edges if total_edges > 0 else 0.0
    sub_ratio = sub_count / total_edges if total_edges > 0 else 0.0
    
    # Count POR wrappers (terminals wrapped for sequential gating)
    por_wrappers = sum(
        1 for nid, n in graph.nodes.items()
        if n.meta.get("origin") == "por_wrapper"
    )
    
    return {
        "speculative_ands": speculative_ands,
        "total_and_nodes": total_and_nodes,
        "branching_factor": round(branching_factor, 2),
        "non_backbone_scripts": len(non_backbone_scripts),
        "max_depth": max_depth,
        "por_count": por_count,
        "sub_count": sub_count,
        "por_ratio": round(por_ratio, 3),
        "sub_ratio": round(sub_ratio, 3),
        "por_wrappers": por_wrappers,
    }


# =============================================================================
# LINK-XP: Neural Darwinism for Edge Pruning
# =============================================================================

def edge_key(src: str, dst: str, ltype: str) -> str:
    """Generate canonical edge key for tracking."""
    return f"{src}->{dst}:{ltype}"


def update_link_xp_for_game(
    graph: "Graph",  # type: ignore
    active_edges: List[str],  # Edge keys that were active
    game_won: bool,
) -> Dict[str, int]:
    """
    Update link_xp for all active edges based on game outcome.
    
    NEURAL DARWINISM: Links that were active during wins get +1 XP.
    Links that were active during losses get -1 XP.
    
    Args:
        graph: The Graph containing edges
        active_edges: List of edge keys that fired during the game
        game_won: Whether the game was won
        
    Returns:
        Dict mapping edge_key to new xp value
    """
    xp_changes = {}
    delta = 1 if game_won else -1
    
    for edge in graph.edges:
        key = edge_key(edge.src, edge.dst, edge.ltype.name)
        if key in active_edges:
            current_xp = edge.meta.get("link_xp", 0)
            new_xp = current_xp + delta
            edge.meta["link_xp"] = new_xp
            edge.meta["link_games"] = edge.meta.get("link_games", 0) + 1
            xp_changes[key] = new_xp
    
    return xp_changes


def prune_weak_links(
    graph: "Graph",  # type: ignore
    fast_threshold_games: int = 25,
    audit_threshold_games: int = 100,
    min_contribution: float = 0.05,
) -> List[str]:
    """
    Two-tier aggressive link pruning.
    
    Tier 1 (25-game fast kill): If xp <= 0 after 25 games, kill immediately.
    Tier 2 (100-game audit): If contribution < 5% after 100 games, kill.
    
    Args:
        graph: The Graph to prune
        fast_threshold_games: Games for tier 1 fast kill
        audit_threshold_games: Games for tier 2 audit
        min_contribution: Minimum contribution ratio for tier 2
        
    Returns:
        List of pruned edge keys
    """
    pruned = []
    edges_to_remove = []
    
    for edge in graph.edges:
        key = edge_key(edge.src, edge.dst, edge.ltype.name)
        games = edge.meta.get("link_games", 0)
        xp = edge.meta.get("link_xp", 0)
        
        # Skip non-hypothesis edges (no tracking data)
        if games == 0:
            continue
        
        # Tier 1: Fast kill (25 games, xp <= 0)
        if games >= fast_threshold_games and xp <= 0:
            edges_to_remove.append(edge)
            pruned.append(key)
            continue
        
        # Tier 2: Audit (100 games, contribution < 5%)
        if games >= audit_threshold_games:
            # Contribution = (xp / games) normalized to 0-1
            # xp can be -games to +games, so contribution = (xp + games) / (2 * games)
            contribution = (xp + games) / (2 * games) if games > 0 else 0.5
            if contribution < min_contribution:
                edges_to_remove.append(edge)
                pruned.append(key)
    
    # Remove pruned edges
    for edge in edges_to_remove:
        if edge in graph.edges:
            graph.edges.remove(edge)
    
    return pruned


def spawn_hypothesis_links(
    graph: "Graph",  # type: ignore
    new_node_id: str,
    stem_manager: "StemCellManager",  # type: ignore
    most_active_sensor_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Relational Spawning: Create 3 hypothesis links for a new terminal.
    
    Instead of just connecting to backbone, create lateral/upward connections:
    1. SUB to a random leg (kpk_pawn_leg or kpk_king_leg)
    2. SUR (confirmation) to an existing SOLID node with Soft-POR config
    3. Trigger hoist if correlated with most active sensor
    
    Args:
        graph: The Graph to modify
        new_node_id: ID of the newly created node
        stem_manager: Manager with SOLID cells info
        most_active_sensor_id: ID of most active sensor for hoisting
        
    Returns:
        Dict with spawned link info
    """
    from ..graph import LinkType, Node
    from random import choice
    
    result = {
        "leg_link": None,
        "sur_link": None,
        "hoisted": None,
    }
    
    # Link 1: SUB to random leg
    legs = ["kpk_pawn_leg", "kpk_king_leg"]
    target_leg = choice(legs)
    if target_leg in graph.nodes:
        graph.add_edge(target_leg, new_node_id, LinkType.SUB)
        # Track this as hypothesis link
        for edge in graph.edges:
            if edge.src == target_leg and edge.dst == new_node_id:
                edge.meta["hypothesis"] = True
                edge.meta["link_xp"] = 0
                edge.meta["link_games"] = 0
                break
        result["leg_link"] = target_leg
    
    # Link 2: SUR (confirmation) to existing SOLID node with Soft-POR
    solid_cells = [
        c for c in stem_manager.cells.values()
        if c.state == StemCellState.MATURE and c.trial_node_id
    ]
    if solid_cells:
        target_solid = choice(solid_cells)
        target_id = target_solid.trial_node_id
        if target_id in graph.nodes:
            graph.add_edge(new_node_id, target_id, LinkType.SUR)
            # Configure Soft-POR on the target node
            target_node = graph.nodes[target_id]
            target_node.meta["por_policy"] = "weighted"
            target_node.meta["por_theta"] = 0.4  # EXPANSION: lowered from 0.5 for easier triggering
            # Track as hypothesis link
            for edge in graph.edges:
                if edge.src == new_node_id and edge.dst == target_id and edge.ltype == LinkType.SUR:
                    edge.meta["hypothesis"] = True
                    edge.meta["link_xp"] = 0
                    edge.meta["link_games"] = 0
                    break
            result["sur_link"] = target_id
    
    # Link 3: Trigger hoist if correlated with active sensor
    if most_active_sensor_id and most_active_sensor_id in graph.nodes:
        # Try to hoist these two into an AND cluster
        new_cell = next(
            (c for c in stem_manager.cells.values() if c.trial_node_id == new_node_id),
            None
        )
        active_cell = next(
            (c for c in stem_manager.cells.values() if c.trial_node_id == most_active_sensor_id),
            None
        )
        if new_cell and active_cell:
            # Compute similarity
            similarity = stem_manager.compute_pattern_similarity(new_cell, active_cell)
            if similarity > 0.8:
                cluster_id = stem_manager.hoist_cluster(
                    [new_cell.cell_id, active_cell.cell_id],
                    graph,
                    parent_node_id="kpk_execute",  # Managers connect to execute for move influence
                    aggregation_mode="and",  # TRUE AND gate!
                )
                result["hoisted"] = cluster_id
    
    return result


class StructureLearner:
    """
    M5 Dreamer: Analyzes traces and proposes structural changes.
    
    The structural phase runs periodically (e.g., every 500 games) to:
    1. Scan traces for affordance spikes (high reward moments)
    2. Identify stem cells that fired right before spikes
    3. Promote promising stem cells to permanent nodes
    4. Prune edges with consistently negative confirmations
    """
    
    def __init__(
        self,
        registry: "TopologyRegistry",
        cooldown_ticks: int = 1000,
        min_spike_reward: float = 0.5,
        decay_rate: float = 0.95,
        prune_threshold_games: int = 100,
        signature_dir: Optional[Path] = None,
    ):
        """
        Args:
            registry: Topology registry for structural changes
            cooldown_ticks: Minimum ticks between promotions for same parent
            min_spike_reward: Minimum reward to count as a spike
            decay_rate: Weight decay rate for negative confirmations
            prune_threshold_games: Games at zero weight before pruning
            signature_dir: Directory for signature PNG files
        """
        self.registry = registry
        self.cooldown_ticks = cooldown_ticks
        self.min_spike_reward = min_spike_reward
        self.decay_rate = decay_rate
        self.prune_threshold_games = prune_threshold_games
        self.signature_dir = signature_dir or Path("signatures")
        
        # Track promotion cooldowns: parent_id -> last_promotion_tick
        self.cooldowns: Dict[str, int] = {}
        
        # Track edge confirmation history: edge_key -> (weight, games_at_zero)
        self.edge_history: Dict[str, Tuple[float, int]] = {}
    
    def scan_for_affordance_spikes(
        self,
        episodes: List[Any],  # List[EpisodeRecord]
        threshold: Optional[float] = None,
    ) -> List[AffordanceSpike]:
        """
        Find high-impact moments in traces.
        
        Scans episode tick records for moments where:
        - reward_tick >= threshold
        - A significant state transition occurred
        """
        threshold = threshold or self.min_spike_reward
        spikes: List[AffordanceSpike] = []
        
        for episode in episodes:
            episode_id = getattr(episode, 'episode_id', str(id(episode)))
            ticks = getattr(episode, 'ticks', [])
            
            for tick in ticks:
                reward = getattr(tick, 'reward_tick', None)
                if reward is None:
                    reward = tick.meta.get('reward_tick', 0) if hasattr(tick, 'meta') else 0
                
                if abs(reward) >= threshold:
                    # Found a spike
                    active_nodes = getattr(tick, 'active_nodes', [])
                    fen = getattr(tick, 'board_fen', '')
                    action = getattr(tick, 'action', None)
                    tick_id = getattr(tick, 'tick_id', 0)
                    
                    # Create spike for each active node
                    for node_id in active_nodes:
                        spike = AffordanceSpike(
                            tick=tick_id,
                            episode_id=episode_id,
                            node_id=node_id,
                            reward=float(reward),
                            fen=fen,
                            move=action,
                        )
                        spikes.append(spike)
        
        return spikes
    
    def find_high_impact_stem_cells(
        self,
        stem_manager: "StemCellManager",
        spikes: List[AffordanceSpike],
        lookback_ticks: int = 5,
    ) -> List["StemCellTerminal"]:
        """
        Identify stem cells that were active near affordance spikes.
        
        A stem cell is "high-impact" if it collected samples within
        `lookback_ticks` of a spike.
        """
        if not HAS_STEM_CELL:
            return []
        
        high_impact: List[StemCellTerminal] = []
        
        # Build a set of (episode_id, tick_range) for faster lookup
        spike_ranges: Dict[str, List[Tuple[int, int]]] = {}
        for spike in spikes:
            if spike.episode_id not in spike_ranges:
                spike_ranges[spike.episode_id] = []
            spike_ranges[spike.episode_id].append(
                (spike.tick - lookback_ticks, spike.tick)
            )
        
        # Check each candidate stem cell
        for cell in stem_manager.cells.values():
            if cell.state not in (StemCellState.CANDIDATE, StemCellState.EXPLORING):
                continue
            
            # Check if any of its samples are near spikes
            impact_score = 0
            for sample in cell.samples:
                # Check sample tick against spike ranges
                sample_tick = sample.tick
                for ep_id, ranges in spike_ranges.items():
                    for start_tick, end_tick in ranges:
                        if start_tick <= sample_tick <= end_tick:
                            impact_score += abs(sample.reward)
            
            if impact_score > 0:
                cell.metadata["impact_score"] = impact_score
                high_impact.append(cell)
        
        # Sort by impact score descending
        high_impact.sort(
            key=lambda c: c.metadata.get("impact_score", 0),
            reverse=True
        )
        
        return high_impact
    
    def promote_stem_cell(
        self,
        cell: "StemCellTerminal",
        parent_id: str,
        current_tick: int,
    ) -> Optional[PromotionResult]:
        """
        Promote a stem cell to a permanent node.
        
        1. Create node entry in topology.json
        2. Inherit weights/filters from the stem cell
        3. Wire to parent with SUB edge
        4. Generate signature.png heatmap
        
        Returns:
            PromotionResult if successful/failed
        """
        if not HAS_STEM_CELL or not HAS_REGISTRY:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id="",
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error="Missing required modules"
            )
        
        # Check cooldown
        last_promo = self.cooldowns.get(parent_id, 0)
        if current_tick - last_promo < self.cooldown_ticks:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id="",
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error=f"Parent {parent_id} on cooldown"
            )
        
        # Analyze pattern to get signature
        consistency, signature = cell.analyze_pattern()
        if consistency < 0.5:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id="",
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error=f"Pattern consistency too low: {consistency:.2f}"
            )
        
        # Generate new node ID
        new_node_id = f"SC_{cell.cell_id}_{current_tick}"
        
        try:
            # Create node spec
            node_spec = {
                "id": new_node_id,
                "type": "TERMINAL",
                "group": "stem_promoted",
                "factory": "recon_lite.nodes.stem_cell:create_pattern_sensor",
                "pattern_signature": signature,
                "weight_source": cell.cell_id,
                "meta": {
                    "promoted_tick": current_tick,
                    "consistency": consistency,
                    "sample_count": len(cell.samples),
                    "avg_reward": sum(s.reward for s in cell.samples) / len(cell.samples) if cell.samples else 0,
                }
            }
            
            # Add to registry
            self.registry.add_node(node_spec, tick=current_tick)
            
            # Wire to parent
            self.registry.add_edge(
                parent_id, new_node_id, "SUB",
                weight=1.0,
                tick=current_tick
            )
            
            # Record promotion
            self.registry.record_promotion(
                cell_id=cell.cell_id,
                new_node_id=new_node_id,
                parent_id=parent_id,
                tick=current_tick,
                pattern_signature=signature,
            )
            
            # Update cooldown
            self.cooldowns[parent_id] = current_tick
            
            # Mark stem cell as specialized
            cell.state = StemCellState.SPECIALIZED
            
            # Generate signature visualization
            signature_path = None
            try:
                signature_path = self._generate_signature(cell, new_node_id)
            except Exception as e:
                print(f"Warning: Could not generate signature: {e}")
            
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id=new_node_id,
                parent_id=parent_id,
                tick=current_tick,
                success=True,
                signature_path=signature_path,
            )
            
        except Exception as e:
            return PromotionResult(
                cell_id=cell.cell_id,
                new_node_id=new_node_id,
                parent_id=parent_id,
                tick=current_tick,
                success=False,
                error=str(e),
            )
    
    def _generate_signature(
        self,
        cell: "StemCellTerminal",
        node_id: str,
    ) -> Optional[Path]:
        """Generate signature heatmap for a promoted node."""
        try:
            from recon_lite.viz.signature_viz import generate_signature_heatmap
            
            self.signature_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.signature_dir / f"{node_id}.png"
            
            return generate_signature_heatmap(
                samples=cell.samples,
                output_path=output_path,
            )
        except ImportError:
            return None
    
    def check_edge_for_pruning(
        self,
        src: str,
        dst: str,
        ltype: str,
        confirmation_value: float,
    ) -> PruningResult:
        """
        Apply SUR-based pruning logic.
        
        If confirmation consistently negative: decay weight.
        If weight stays at 0 for N games: mark for removal.
        """
        edge_key = f"{src}->{dst}:{ltype}"
        
        # Get current state
        old_weight, games_at_zero = self.edge_history.get(edge_key, (1.0, 0))
        
        # Update based on confirmation
        if confirmation_value < 0:
            new_weight = old_weight * self.decay_rate
        elif confirmation_value > 0:
            # Positive confirmation: slight recovery
            new_weight = min(1.0, old_weight * 1.05)
        else:
            new_weight = old_weight
        
        # Track games at zero
        if new_weight < 0.01:
            new_weight = 0.0
            games_at_zero += 1
        else:
            games_at_zero = 0
        
        # Check if should prune
        should_prune = games_at_zero >= self.prune_threshold_games
        
        # Update history
        self.edge_history[edge_key] = (new_weight, games_at_zero)
        
        # Update registry
        self.registry.update_edge_weight(src, dst, ltype, new_weight)
        
        return PruningResult(
            edge_key=edge_key,
            old_weight=old_weight,
            new_weight=new_weight,
            pruned=should_prune,
            games_at_zero=games_at_zero,
        )
    
    def apply_structural_phase(
        self,
        stem_manager: "StemCellManager",
        episodes: List[Any],
        max_promotions: int = 2,
        parent_candidates: Optional[List[str]] = None,
        current_win_rate: float = 0.0,  # For Perfect Success bypass
        mastered_sensors: Optional[Dict[int, Any]] = None,  # BACKWARD CHAINING: stage_id -> sensor info
        current_stage_id: int = 0,  # Current curriculum stage for chain lookup
    ) -> Dict[str, Any]:
        """
        Run a full structural phase:
        1. Analyze traces for spikes
        2. Find promising stem cells
        3. Promote top candidates
        4. Check edges for pruning
        
        Args:
            stem_manager: Manager for stem cells
            episodes: List of EpisodeRecord from recent games
            max_promotions: Max nodes to promote this cycle
            parent_candidates: Optional list of eligible parent node IDs
            current_win_rate: Current cycle win rate for Perfect Success bypass
            mastered_sensors: Dict of mastered stage sensors for backward chaining
            current_stage_id: Current curriculum stage ID
            
        Returns:
            Stats dict with counts and results
        """
        current_tick = 0
        if episodes:
            last_ep = episodes[-1]
            ticks = getattr(last_ep, 'ticks', [])
            if ticks:
                current_tick = getattr(ticks[-1], 'tick_id', 0)
        
        # CRITICAL: Initialize graph ONCE at the start and reuse throughout
        # This ensures spawned packs/structures are preserved across steps
        graph = None
        try:
            from recon_lite_chess.graph.builder import build_graph_from_topology
            graph = build_graph_from_topology(self.registry.path, self.registry)
        except Exception:
            pass  # Will try again later if needed
        
        # Step 1: Find spikes
        spikes = self.scan_for_affordance_spikes(episodes)
        
        # Step 2: Find high-impact stem cells
        high_impact = self.find_high_impact_stem_cells(stem_manager, spikes)
        
        # =====================================================================
        # Step 2b: EXTREME FAILURE BYPASS - Direct CANDIDATE promotion
        # When win_rate < 5% for 1000+ games and high_impact is empty,
        # directly select active CANDIDATE cells for promotion. This breaks
        # the "no spikes → no high_impact → no promotions" deadlock.
        # =====================================================================
        games_total = self._games_at_current_stage if hasattr(self, '_games_at_current_stage') else 0
        extreme_failure_mode = (
            current_win_rate < 0.05 and 
            games_total >= 1000 and 
            len(high_impact) == 0
        )
        
        if extreme_failure_mode:
            # Select CANDIDATE cells with enough samples for forced promotion
            # M5_MIN_SAMPLES env var allows tuning (default: 15, lowered from 30)
            import os
            min_samples_for_promotion = int(os.environ.get("M5_MIN_SAMPLES", "15"))
            
            for cell in stem_manager.cells.values():
                if (cell.state == StemCellState.CANDIDATE and 
                    len(cell.samples) >= min_samples_for_promotion):
                    high_impact.append(cell)
                    cell.metadata["force_promoted"] = True
                    if len(high_impact) >= max_promotions * 2:  # Double the quota
                        break
        
        # Step 3: Promote CANDIDATE cells to TRIAL tier
        trial_promotions: List[str] = []
        trial_errors: List[str] = []
        
        # Default parent candidates if not specified
        if parent_candidates is None:
            parent_candidates = ["kpk_detect", "kpk_execute"]
        
        for cell in high_impact:
            if len(trial_promotions) >= max_promotions:
                break
            
            # Only promote CANDIDATE cells (not already in TRIAL)
            if cell.state != StemCellState.CANDIDATE:
                continue
            
            # =====================================================================
            # M5.1 SUCCESS-BASED PROMOTION ("Bypass")
            # If reward_average > 0.90 over 50+ samples, force-promote even if
            # consistency math is undefined/zero (Zero-Variance Trap escape)
            # =====================================================================
            sample_count = len(cell.samples) if hasattr(cell, 'samples') else 0
            
            # Calculate reward average from samples
            if sample_count >= 50:
                avg_reward = sum(s.reward for s in cell.samples) / sample_count
            else:
                avg_reward = 0.0
            
            # SUCCESS BYPASS: High average reward bypasses consistency check
            success_bypass = (avg_reward > 0.90 and sample_count >= 50)
            
            # PERFECT SUCCESS BYPASS: 100% win rate is ultimate consistency
            perfect_success = (current_win_rate >= 1.0 and sample_count >= 50)
            
            # =====================================================================
            # M5.1 EXTREME FAILURE BYPASS: Force promotion in cold-start scenarios
            # When win_rate < 5% for 200+ games, the consistency check is too strict
            # and prevents ANY structure from forming. Force promote to TRIAL to
            # allow failure-driven exploration and engagement XP to kick in.
            # =====================================================================
            games_total = self._games_at_current_stage if hasattr(self, '_games_at_current_stage') else 0
            extreme_failure_bypass = (
                current_win_rate < 0.05 and 
                games_total >= 200 and  # Lowered from 1000 - faster intervention
                sample_count >= 30  # Lower bar for extreme failure
            )
            
            if extreme_failure_bypass:
                cell.metadata["promotion_reason"] = "extreme_failure_bypass"
            
            # =====================================================================
            # M5.2 SURVIVOR BYPASS: Auto-promote CANDIDATEs that survived 2+ cycles
            # If a CANDIDATE has survived through 2 or more structural phases,
            # it's stable enough to be trialed. This prevents deadlock where all
            # cells get pruned during easy stages leaving none for hard stages.
            # =====================================================================
            cycles_survived = cell.metadata.get("cycles_survived", 0)
            survivor_bypass = (
                cell.state == StemCellState.CANDIDATE and
                cycles_survived >= 2 and
                sample_count >= 15  # Minimal data requirement
            )
            
            if survivor_bypass:
                cell.metadata["promotion_reason"] = f"survivor_bypass_{cycles_survived}_cycles"
            
            # Either bypass allows promotion without consistency check
            bypass_enabled = success_bypass or perfect_success or extreme_failure_bypass or survivor_bypass
            
            # Check if ready for trial - EXPANSION: lowered to 0.40
            consistency, _ = cell.analyze_pattern()
            
            if not bypass_enabled and consistency < 0.40:
                trial_errors.append(f"{cell.cell_id}: consistency {consistency:.2f} < 0.40")
                continue
            
            # =====================================================================
            # M5 VERTICAL PARENTING: Use local_root_id for spawned children
            # This enables hierarchical growth: Sensor -> Sub-goal -> Leg -> Backbone
            # =====================================================================
            local_root_id = cell.metadata.get("local_root_id")
            
            if local_root_id:
                # This cell was spawned from a SOLID parent - use vertical parenting
                # Check depth limit to keep propagate_activation() efficient
                try:
                    from recon_lite_chess.graph.builder import build_graph_from_topology
                    graph = build_graph_from_topology(self.registry.topology_path, self.registry)
                    
                    current_depth = compute_node_depth(graph, local_root_id, BACKBONE_NODES)
                    if current_depth >= MAX_BRANCH_DEPTH:
                        # At max depth - fall back to backbone
                        parent_id = parent_candidates[0] if parent_candidates else "kpk_detect"
                        cell.metadata["depth_limited"] = True
                        cell.metadata["original_local_root"] = local_root_id
                    else:
                        # Use vertical parent (local_root)
                        parent_id = local_root_id
                        cell.metadata["vertical_promotion"] = True
                except Exception:
                    # Fall back to backbone on error
                    parent_id = parent_candidates[0] if parent_candidates else "kpk_detect"
            else:
                # Regular cell - use backbone parent
                # DUAL-WIRING: Sensors connect to kpk_detect (for sensing/activation)
                parent_id = parent_candidates[0] if parent_candidates else "kpk_detect"
            
            # Promote to TRIAL (not MATURE yet)
            if cell.promote_to_trial(self.registry, parent_id, current_tick):
                trial_promotions.append(cell.cell_id)
                # Track promotion reason for debugging
                if success_bypass and not perfect_success:
                    cell.metadata["promotion_reason"] = f"success_bypass_avg_{avg_reward:.2f}"
                elif perfect_success:
                    cell.metadata["promotion_reason"] = "perfect_success"
                elif cell.metadata.get("vertical_promotion"):
                    cell.metadata["promotion_reason"] = "vertical_to_" + parent_id
                else:
                    cell.metadata["promotion_reason"] = f"consistency_{consistency:.2f}"
            else:
                trial_errors.append(f"{cell.cell_id}: trial promotion failed")
        
        # Step 4: Apply XP decay to all TRIAL cells (-1 XP per cycle)
        xp_decays: List[Tuple[str, int]] = []
        for cell in stem_manager.cells.values():
            if cell.state == StemCellState.TRIAL:
                new_xp = cell.decay_xp()
                xp_decays.append((cell.cell_id, new_xp))
        
        # Step 4a: Increment cycles_survived for CANDIDATE cells
        # This enables SURVIVOR BYPASS: auto-promote after 2+ cycles
        for cell in stem_manager.cells.values():
            if cell.state == StemCellState.CANDIDATE:
                cycles_survived = cell.metadata.get("cycles_survived", 0) + 1
                cell.metadata["cycles_survived"] = cycles_survived
        
        # Step 4b: MATURITY BOOST - Fast-track high-quality cells
        # If a TRIAL node has consistency > 0.70 and survived 500+ ticks, promote immediately
        maturity_boosted: List[str] = []
        for cell in list(stem_manager.cells.values()):
            if cell.state != StemCellState.TRIAL:
                continue
            
            consistency = cell.trial_consistency or 0.0
            ticks_survived = current_tick - (cell.trial_tick or current_tick)
            
            if consistency > 0.70 and ticks_survived >= 500:
                # Immediately boost to MATURE
                if cell.solidify_to_mature(self.registry, current_tick):
                    maturity_boosted.append(cell.cell_id)
        
        # Step 5: Check for normal solidification (XP >= 100) or demotion (XP <= 0)
        solidified: List[str] = []
        demoted: List[str] = []
        
        for cell in list(stem_manager.cells.values()):
            if cell.state != StemCellState.TRIAL:
                continue
            
            should_change, new_state = cell.check_solidification()
            if should_change:
                if new_state == "mature":
                    if cell.solidify_to_mature(self.registry, current_tick):
                        solidified.append(cell.cell_id)
                elif new_state == "demoted":
                    if cell.demote_to_exploring(self.registry):
                        demoted.append(cell.cell_id)
        
        # Step 6: Collection confirmation stats for pruning
        pruning_results: List[PruningResult] = []
        
        # Step 7: ACTIVE HOISTING - Trigger for TRIAL nodes at 0.85 correlation
        # EXPANSION: Don't wait for solidification - hoist as soon as TRIAL nodes correlate
        hoisted_clusters: List[str] = []
        trial_cells = [c for c in stem_manager.cells.values() if c.state == StemCellState.TRIAL]
        if len(trial_cells) >= 2:  # Need at least 2 to hoist
            try:
                from recon_lite_chess.graph.builder import build_graph_from_topology
                if graph is None:
                    graph = build_graph_from_topology(self.registry.topology_path, self.registry)
                hoisted_clusters = stem_manager.auto_hoist(
                    graph,
                    parent_node_id="kpk_execute",  # Managers connect to execute for move influence
                    min_similarity=0.85,  # ACTIVE: 85% correlation for TRIAL hoisting
                )
                if hoisted_clusters:
                    # PERSIST: Add hoisted clusters to registry
                    for cluster_id in hoisted_clusters:
                        if cluster_id in graph.nodes:
                            node = graph.nodes[cluster_id]
                            node_spec = {
                                "id": cluster_id,
                                "type": node.ntype.name if hasattr(node.ntype, "name") else str(node.ntype),
                                "group": "hoisted",
                                "factory": None,
                                "meta": node.meta,
                            }
                            try:
                                self.registry.add_node(node_spec, tick=current_tick)
                                # Managers connect to execute for move influence
                                self.registry.add_edge("kpk_execute", cluster_id, "SUB", 0.5, tick=current_tick)
                            except ValueError:
                                pass  # Already exists
                    # Save the updated registry
                    self.registry.save()
            except Exception as e:
                hoisted_clusters = [f"error: {e}"]
        
        # Step 7b: SPECULATIVE HOISTING - Hoist CANDIDATE cells if 85%+ win-coactivation
        # M5.1 Unblock: Don't wait for TRIAL - hoist promising CANDIDATE pairs early
        speculative_hoists: List[str] = []
        try:
            # Get win-correlated pairs including candidates
            win_correlated = stem_manager.find_win_correlated_pairs(
                min_coactivations=20,  # Lower bar for speculation
                min_ratio=0.85,
            )
            
            for cell_a_id, cell_b_id, ratio, co_count in win_correlated:
                cell_a = stem_manager.cells.get(cell_a_id)
                cell_b = stem_manager.cells.get(cell_b_id)
                
                # Include CANDIDATE cells (not just TRIAL)
                if not cell_a or not cell_b:
                    continue
                
                # At least one must be CANDIDATE for "speculative"
                is_speculative = (
                    cell_a.state == StemCellState.CANDIDATE or 
                    cell_b.state == StemCellState.CANDIDATE
                )
                
                if not is_speculative:
                    continue  # Already handled by TRIAL hoisting
                
                # Both must be at least CANDIDATE state
                valid_states = {StemCellState.CANDIDATE, StemCellState.TRIAL, StemCellState.MATURE}
                if cell_a.state not in valid_states or cell_b.state not in valid_states:
                    continue
                
                # Trigger speculative hoist
                from recon_lite_chess.graph.builder import build_graph_from_topology
                if graph is None:
                    graph = build_graph_from_topology(self.registry.topology_path, self.registry)
                
                cluster_id = stem_manager.hoist_cluster(
                    [cell_a_id, cell_b_id],
                    graph,
                    parent_node_id="kpk_execute",  # Managers connect to execute for move influence
                    aggregation_mode="and",  # True AND gate for tactical patterns
                )
                
                if cluster_id:
                    speculative_hoists.append(cluster_id)
                    # Mark as speculative
                    if cluster_id in graph.nodes:
                        graph.nodes[cluster_id].meta["speculative"] = True
                        graph.nodes[cluster_id].meta["win_correlation"] = ratio
                        
                        # PERSIST: Add cluster node to registry
                        node = graph.nodes[cluster_id]
                        node_spec = {
                            "id": cluster_id,
                            "type": node.ntype.name if hasattr(node.ntype, "name") else str(node.ntype),
                            "group": "hoisted",
                            "factory": None,
                            "meta": node.meta,
                        }
                        try:
                            self.registry.add_node(node_spec, tick=current_tick)
                            # Managers connect to execute for move influence
                            self.registry.add_edge("kpk_execute", cluster_id, "SUB", 0.5, tick=current_tick)
                        except ValueError:
                            pass  # Already exists
                    self.registry.save()
                    
        except Exception as e:
            speculative_hoists.append(f"error: {e}")
        
        # Step 7c: FORCED HOISTING - "Structural Crisis" or "Success Trap" Mode
        # 
        # Two triggers:
        # 1. Crisis Mode: win_rate < 20% - struggling, need structural help
        # 2. Success Trap Mode: win_rate > 95% - doing too well, force hierarchy growth
        #
        # The Success Trap prevents "vibe ceiling" where flat networks win without
        # building any hierarchical structure. By forcing AND-gates even at high
        # win rates, we seed the hierarchy before weight inertia sets in.
        import os
        forced_hoists: List[str] = []
        force_hoist_enabled = os.environ.get("M5_ENABLE_FORCED_HOISTING", "0") == "1"
        force_hoist_threshold_low = float(os.environ.get("M5_FORCED_HOIST_THRESHOLD_WIN_RATE", "0.20"))
        force_hoist_threshold_high = float(os.environ.get("M5_FORCED_HOIST_THRESHOLD_HIGH", "0.95"))
        force_hoist_interval = int(os.environ.get("M5_FORCED_HOIST_INTERVAL_CYCLES", "5"))
        
        # Check both triggers: struggling OR succeeding too well
        crisis_mode = current_win_rate < force_hoist_threshold_low
        success_trap = current_win_rate > force_hoist_threshold_high
        
        if force_hoist_enabled and (crisis_mode or success_trap):
            # Check if we're at the right cycle interval (use current_tick as proxy)
            cycle_number = current_tick // 100  # Approximate cycle from tick
            if cycle_number % force_hoist_interval == 0 and stem_manager.win_active_counts:
                # Find the two most active sensors in winning games
                win_active = sorted(
                    stem_manager.win_active_counts.items(),
                    key=lambda x: -x[1]
                )[:2]
                
                if len(win_active) >= 2:
                    cell_a_id, count_a = win_active[0]
                    cell_b_id, count_b = win_active[1]
                    
                    cell_a = stem_manager.cells.get(cell_a_id)
                    cell_b = stem_manager.cells.get(cell_b_id)
                    
                    if cell_a and cell_b:
                        try:
                            from recon_lite_chess.graph.builder import build_graph_from_topology
                            if graph is None:
                                graph = build_graph_from_topology(self.registry.topology_path, self.registry)
                            
                            cluster_id = stem_manager.hoist_cluster(
                                [cell_a_id, cell_b_id],
                                graph,
                                parent_node_id="kpk_execute",  # Managers connect to execute for move influence
                                aggregation_mode="and",
                            )
                            
                            if cluster_id:
                                forced_hoists.append(cluster_id)
                                if cluster_id in graph.nodes:
                                    graph.nodes[cluster_id].meta["forced"] = True
                                    graph.nodes[cluster_id].meta["crisis_mode"] = crisis_mode
                                    graph.nodes[cluster_id].meta["success_trap"] = success_trap
                                    graph.nodes[cluster_id].meta["source_cells"] = [cell_a_id, cell_b_id]
                                    graph.nodes[cluster_id].meta["source_win_counts"] = [count_a, count_b]
                                    graph.nodes[cluster_id].meta["trigger_win_rate"] = current_win_rate
                                    
                                    # PERSIST: Add cluster node to registry
                                    node = graph.nodes[cluster_id]
                                    node_spec = {
                                        "id": cluster_id,
                                        "type": node.ntype.name if hasattr(node.ntype, "name") else str(node.ntype),
                                        "group": "hoisted",
                                        "factory": None,
                                        "meta": node.meta,
                                    }
                                    try:
                                        self.registry.add_node(node_spec, tick=current_tick)
                                    except ValueError:
                                        pass  # Node already exists
                                    
                                    # PERSIST: Add edges from parent to cluster and cluster to children
                                    # Managers connect to execute for move influence
                                    try:
                                        self.registry.add_edge(
                                            src="kpk_execute",
                                            dst=cluster_id,
                                            ltype="SUB",
                                            weight=0.5,
                                            tick=current_tick,
                                        )
                                    except ValueError:
                                        pass  # Edge already exists
                                    
                                    # Add edges from cluster to source cells (if they have TRIAL nodes)
                                    for cell_id in [cell_a_id, cell_b_id]:
                                        trial_node_id = f"TRIAL_{cell_id}_{stem_manager._next_id - 1}"
                                        cell = stem_manager.cells.get(cell_id)
                                        if cell and cell.trial_node_id:
                                            trial_node_id = cell.trial_node_id
                                        try:
                                            self.registry.add_edge(
                                                src=cluster_id,
                                                dst=trial_node_id,
                                                ltype="SUB",
                                                weight=0.5,
                                                tick=current_tick,
                                            )
                                        except ValueError:
                                            pass  # Edge already exists
                                    
                                self.registry.save()
                        except Exception as e:
                            forced_hoists.append(f"error: {e}")
        
        # Step 8: SPAWN_NEIGHBORS - Recursive outgrowth for solidified cells
        spawned_neighbors: List[str] = []
        for cell_id in solidified:
            cell = stem_manager.cells.get(cell_id)
            if cell and cell.state == StemCellState.MATURE:
                # Spawn 2-3 children targeting specific legs
                from random import choice
                target_leg = choice(["kpk_pawn_leg", "kpk_king_leg"])
                new_ids = cell.spawn_neighbors(stem_manager, target_leg=target_leg, spawn_count=2)
                spawned_neighbors.extend(new_ids)
        
        # =====================================================================
        # Step 8b: FAILURE-DRIVEN SPAWNING - Exploration during failure states
        # =====================================================================
        # HYBRID FIX: When win_rate < 20% for 20+ games, spawn from TRIAL nodes
        # to break the "no wins → no XP → no growth" deadlock.
        # 
        # GROWTH MODE: Also spawn when winning (30% chance) to build structure depth
        exploration_spawned: List[str] = []
        
        FAILURE_WIN_RATE_THRESHOLD = 0.20  # Raised from 0.10 to 0.20 for earlier intervention
        FAILURE_GAMES_THRESHOLD = 20  # Lowered from 50 to 20 for faster response
        GROWTH_MODE_CHANCE = 0.30  # 30% chance to spawn packs when winning
        
        # Track failure state (use instance variable since registry.metadata doesn't exist)
        if not hasattr(self, '_games_at_current_stage'):
            self._games_at_current_stage = 0
        self._games_at_current_stage += len(episodes)
        
        # Debug logging for pack spawning
        if current_win_rate < FAILURE_WIN_RATE_THRESHOLD:
            trial_count = len([c for c in stem_manager.cells.values() if c.state == StemCellState.TRIAL])
            print(f"    [M5] Failure check: win={current_win_rate:.1%}, games={self._games_at_current_stage}, trials={trial_count}")
        
        # GROWTH MODE: Spawn packs even when winning to build topology depth
        import random as rnd
        growth_mode_active = current_win_rate >= 0.80 and rnd.random() < GROWTH_MODE_CHANCE
        
        if growth_mode_active:
            print(f"    [M5] 🌱 GROWTH MODE: Spawning packs to build depth (win={current_win_rate:.1%})")
        
        if (current_win_rate < FAILURE_WIN_RATE_THRESHOLD and self._games_at_current_stage >= FAILURE_GAMES_THRESHOLD) or growth_mode_active:
            # FAILURE STATE or GROWTH MODE: Trigger exploration spawning with LOTTERY
            # Uses adaptive balance based on win_rate
            packs_spawned = []
            print(f"    [M5] ⚡ FAILURE MODE TRIGGERED: Spawning packs/singles from {len([c for c in stem_manager.cells.values() if c.state == StemCellState.TRIAL])} TRIALs")
            
            # Get graph for pack injection - reuse existing if available
            if graph is None:
                try:
                    from recon_lite_chess.graph.builder import build_graph_from_topology
                    graph = build_graph_from_topology(self.registry.path, self.registry)
                    print(f"      Graph loaded: {len(graph.nodes)} nodes")
                except Exception as e:
                    print(f"      ⚠ Graph load failed: {e}")
            else:
                print(f"      Graph reused: {len(graph.nodes)} nodes (preserving earlier changes)")
            
            for cell in stem_manager.cells.values():
                if cell.state == StemCellState.TRIAL:
                    # BACKWARD CHAINING: Look for previous mastered stage sensor
                    # This becomes the sentinel for the spawned pack
                    previous_sentinel = None
                    if mastered_sensors and current_stage_id > 0:
                        # Find the highest mastered stage below current
                        for stage_id in sorted(mastered_sensors.keys(), reverse=True):
                            if stage_id < current_stage_id:
                                sensor_info = mastered_sensors[stage_id]
                                previous_sentinel = {
                                    "stage_id": stage_id,
                                    "stage_name": sensor_info.get("stage_name"),
                                    "pattern_signature": sensor_info.get("pattern_signature"),
                                    "trust_score": sensor_info.get("trust_score", 0.9),
                                }
                                break
                    
                    # Use lottery for balanced spawning with dynamic win_rate bias
                    result = cell.spawn_with_lottery(
                        manager=stem_manager, 
                        graph=graph,
                        current_tick=current_tick,
                        win_rate=current_win_rate,  # Dynamic bias based on stage performance
                        backward_chain_sentinel=previous_sentinel,  # For reverse linking
                    )
                    
                    if result.get("type") == "pack":
                        packs_spawned.append(result.get("ids", {}).get("root"))
                        if previous_sentinel:
                            print(f"      ↩ Pack chained to {previous_sentinel['stage_name']} sensor")
                    elif result.get("ids"):
                        exploration_spawned.extend(result["ids"])
        
        # Step 9: POR CHAIN DISCOVERY - Sequential Gating
        # Discover temporal patterns like Opposition -> Protect -> Promote
        discovered_por_chains: List[Tuple[str, str, float]] = []
        try:
            from recon_lite_chess.graph.builder import build_graph_from_topology
            # Only rebuild graph if we don't already have one with spawned packs
            if 'graph' not in locals() or graph is None:
                graph = build_graph_from_topology(self.registry.topology_path, self.registry)
            
            discovered_por_chains = discover_por_chains(
                graph=graph,
                stem_manager=stem_manager,
                registry=self.registry,
                min_sequence_confidence=0.7,
                current_tick=current_tick,
                episodes=episodes,  # Enable micro-temporal analysis
            )
            
            if discovered_por_chains:
                # Save updated graph with POR links
                self.registry.save()
        except Exception as e:
            discovered_por_chains = []
        
        # Step 9b: KRK-SPECIFIC BOX METHOD POR DISCOVERY
        # Discover the "Box Method" sequential pattern for KRK endgames
        # Sequence: Rook_Cuts_Rank → King_Approaches → Rook_Shrinks_Box
        krk_box_method_results: Dict[str, Any] = {}
        try:
            # Check if we're in a KRK domain (look for krk nodes)
            is_krk_domain = any('krk' in node_id.lower() for node_id in self.registry.list_nodes())
            
            if is_krk_domain and episodes:
                from recon_lite_chess.graph.builder import build_graph_from_topology
                if graph is None:
                    graph = build_graph_from_topology(self.registry.topology_path, self.registry)
                
                krk_box_method_results = discover_krk_box_method_por(
                    graph=graph,
                    stem_manager=stem_manager,
                    registry=self.registry,
                    episodes=episodes,
                    min_sequence_confidence=0.70,
                    current_tick=current_tick,
                )
                
                if krk_box_method_results.get("tactical_manager_created"):
                    self.registry.save()
        except Exception as e:
            krk_box_method_results = {"error": str(e)}
        
        # Save changes
        self.registry.save()
        
        # Count vertical promotions (M5 Recursive Branching metric)
        vertical_promotions = sum(
            1 for cid in trial_promotions
            if stem_manager.cells.get(cid) and 
               stem_manager.cells[cid].metadata.get("vertical_promotion")
        )
        
        # Compute sequence confidence (average correlation for discovered POR links)
        sequence_confidence = (
            sum(conf for _, _, conf in discovered_por_chains) / len(discovered_por_chains)
            if discovered_por_chains else 0.0
        )
        
        return {
            "spikes_found": len(spikes),
            "high_impact_cells": len(high_impact),
            # Trial lifecycle stats
            "trial_promotions": len(trial_promotions),
            "trial_promoted": trial_promotions,
            "trial_errors": trial_errors,
            "xp_decays": len(xp_decays),
            "solidified": len(solidified) + len(maturity_boosted),
            "solidified_cells": solidified + maturity_boosted,
            "maturity_boosted": maturity_boosted,
            "demoted": len(demoted),
            "demoted_cells": demoted,
            # NEW: Vertical growth stats (M5 Recursive Branching)
            "hoisted_clusters": hoisted_clusters,
            "speculative_hoists": speculative_hoists,  # M5.1: CANDIDATE-level hoisting
            "forced_hoists": forced_hoists,  # M5.1: Crisis mode forced AND-gates
            "spawned_neighbors": spawned_neighbors,
            "vertical_promotions": vertical_promotions,  # Nodes parented to SOLID, not backbone
            # POR Chain Discovery (Sequential Gating)
            "discovered_por_chains": len(discovered_por_chains),
            "por_chain_details": [(p, s, round(c, 3)) for p, s, c in discovered_por_chains],
            "sequence_confidence": round(sequence_confidence, 3),
            # KRK Box Method Discovery
            "krk_box_method": krk_box_method_results,
            # Legacy compat
            "promotions_attempted": len(trial_promotions),
            "promotions_succeeded": len(trial_promotions),
            "promotions": trial_promotions,
            "promotion_errors": trial_errors,
            "pruning_results": [r.edge_key for r in pruning_results if r.pruned],
            "current_tick": current_tick,
            # CRITICAL: Return the final graph with all spawned packs
            # This allows the snapshot to capture dynamically created topology
            "graph": graph,  # Graph is initialized at function start, always exists
        }


def create_pattern_sensor(node_id: str):
    """
    Factory function for creating pattern-matching sensors from promoted stem cells.
    
    This is called when loading a graph from topology.json for nodes with
    factory="recon_lite.nodes.stem_cell:create_pattern_sensor"
    """
    from recon_lite.graph import Node, NodeType
    
    def _pattern_predicate(node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Pattern matching predicate for promoted stem cells.
        
        Compares current board features against the stored pattern signature.
        """
        signature = node.meta.get("pattern_signature")
        if signature is None:
            return True, True  # No signature, always match
        
        # Get current features
        features = env.get("features")
        if features is None:
            return True, True  # No features to compare, pass through
        
        # Simple cosine similarity match
        try:
            import numpy as np
            sig_arr = np.array(signature)
            feat_arr = np.array(features)
            
            # Normalize
            sig_norm = sig_arr / (np.linalg.norm(sig_arr) + 1e-8)
            feat_norm = feat_arr / (np.linalg.norm(feat_arr) + 1e-8)
            
            similarity = float(np.dot(sig_norm, feat_norm))
            node.meta["last_similarity"] = similarity
            
            # Match if similarity exceeds threshold
            threshold = node.meta.get("threshold", 0.7)
            matched = similarity >= threshold
            
            return matched, matched
            
        except Exception:
            return True, True
    
    return Node(nid=node_id, ntype=NodeType.TERMINAL, predicate=_pattern_predicate)
