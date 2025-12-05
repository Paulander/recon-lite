"""
Unified ReCoN Graph Builder for Chess.

Integrates all subgraphs (KRK, KPK, tactics) into a single unified graph
for full game play with proper gating and parallel tactics detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

from recon_lite import Graph, LinkType, Node, NodeType, NodeState


# All supported tactic types
TACTIC_TYPES = [
    "fork", "pin", "skewer", "hangingPiece", "backRankMate",
    "discoveredAttack", "doubleCheck", "smotheredMate", "attraction",
    "deflection", "interference", "sacrifice", "quietMove",
    "exposedKing", "trappedPiece", "zugzwang",
]


def build_unified_graph(
    include_endgames: bool = True,
    include_tactics: bool = True,
    include_sensors: bool = True,
) -> Graph:
    """
    Build the complete unified ReCoN graph for full game play.
    
    Integrates:
    - Main game structure (sensors, strategic layer)
    - Endgame subgraphs (KRK, KPK) - gated by material
    - Tactics subgraphs (16 patterns) - parallel, context-weighted
    
    Args:
        include_endgames: Include KRK/KPK endgame subgraphs
        include_tactics: Include tactical pattern subgraphs
        include_sensors: Include sensor nodes
        
    Returns:
        Unified Graph with all subgraphs integrated
    """
    g = Graph()
    
    # 1. Add main game structure
    _add_root_and_strategic_layer(g)
    
    if include_sensors:
        _add_sensors(g)
    
    # 2. Integrate endgame subgraphs
    if include_endgames:
        _integrate_krk_subgraph(g)
        _integrate_kpk_subgraph(g)
        _integrate_kqk_subgraph(g)
    
    # 3. Integrate tactics subgraphs
    if include_tactics:
        _integrate_tactics_subgraphs(g)
    
    # 4. Wire strategic connections
    _wire_strategic_connections(g)
    
    # 5. Mark all edges for consolidation tracking
    _mark_edges_for_consolidation(g)
    
    return g


def _add_root_and_strategic_layer(g: Graph) -> None:
    """Add root node and strategic layer."""
    # Root
    game_root = Node("GameRoot", NodeType.SCRIPT, meta={
        "layer": "root",
        "subgraph": "main",
    })
    g.add_node(game_root)
    
    # Strategy branches
    win_strategy = Node("WinStrategy", NodeType.SCRIPT, meta={
        "layer": "ultimate",
        "goal": "WIN",
        "alt": True,
        "subgraph": "main",
    })
    draw_strategy = Node("DrawStrategy", NodeType.SCRIPT, meta={
        "layer": "ultimate",
        "goal": "DRAW",
        "alt": True,
        "subgraph": "main",
    })
    survive_strategy = Node("SurviveStrategy", NodeType.SCRIPT, meta={
        "layer": "ultimate",
        "goal": "SURVIVE",
        "alt": True,
        "subgraph": "main",
    })
    
    g.add_node(win_strategy)
    g.add_node(draw_strategy)
    g.add_node(survive_strategy)
    
    # Strategic plans
    plans = [
        ("Develop", "POSITIONAL"),
        ("Castle", "POSITIONAL"),
        ("CenterControl", "POSITIONAL"),
        ("AttackKing", "ATTACKING"),
        ("Simplify", "DEFENSIVE"),
        ("ImproveWorstPiece", "POSITIONAL"),
        ("CreateWeakness", "ATTACKING"),
        ("ConvertAdvantage", "ATTACKING"),
        ("WinMaterial", "ATTACKING"),
        ("DefendWeakness", "DEFENSIVE"),
    ]
    
    for plan_id, category in plans:
        node = Node(plan_id, NodeType.SCRIPT, meta={
            "layer": "strategic",
            "category": category,
            "subgraph": "main",
        })
        g.add_node(node)
        
        # Each plan gets a terminal placeholder
        terminal = Node(f"{plan_id}_Terminal", NodeType.TERMINAL, meta={
            "subgraph": "main",
        })
        g.add_node(terminal)
        g.add_edge(plan_id, f"{plan_id}_Terminal", LinkType.SUB)
    
    # Wire root to strategies
    g.add_edge("GameRoot", "WinStrategy", LinkType.SUB)
    g.add_edge("GameRoot", "DrawStrategy", LinkType.SUB)
    g.add_edge("GameRoot", "SurviveStrategy", LinkType.SUB)
    
    # Wire strategies to plans
    for plan_id in ["AttackKing", "WinMaterial", "ConvertAdvantage", "CreateWeakness"]:
        if plan_id in g.nodes:
            g.add_edge("WinStrategy", plan_id, LinkType.SUB)
    
    for plan_id in ["Develop", "Castle", "CenterControl", "ImproveWorstPiece"]:
        if plan_id in g.nodes:
            g.add_edge("DrawStrategy", plan_id, LinkType.SUB)
    
    for plan_id in ["DefendWeakness", "Simplify"]:
        if plan_id in g.nodes:
            g.add_edge("SurviveStrategy", plan_id, LinkType.SUB)


def _add_sensors(g: Graph) -> None:
    """Add sensor nodes (fan-in allowed)."""
    sensors = [
        ("MaterialSensor", "sensor"),
        ("PhaseSensor", "sensor"),
        ("UltimateGoal", "ultimate"),
        ("DevelopmentSensor", "sensor"),
        ("CenterControlSensor", "sensor"),
        ("KingSafetySensor", "sensor"),
        ("PieceActivitySensor", "sensor"),
        ("TacticsSensor", "sensor"),  # New: triggers tactics scanning
        ("EndgameDetector", "sensor"),  # New: triggers endgame subgraphs
    ]
    
    for nid, layer in sensors:
        node = Node(nid, NodeType.TERMINAL, meta={
            "layer": layer,
            "fan_in_allowed": True,
            "subgraph": "main",
        })
        g.add_node(node)
    
    # Wire sensors to root
    g.add_edge("GameRoot", "UltimateGoal", LinkType.SUB)
    g.add_edge("GameRoot", "PhaseSensor", LinkType.SUB)
    g.add_edge("GameRoot", "MaterialSensor", LinkType.SUB)
    g.add_edge("GameRoot", "TacticsSensor", LinkType.SUB)
    g.add_edge("GameRoot", "EndgameDetector", LinkType.SUB)


def _integrate_krk_subgraph(g: Graph) -> None:
    """
    Integrate KRK endgame subgraph with prefixed node IDs.
    
    All KRK nodes get 'krk_' prefix and are marked with subgraph="krk".
    """
    prefix = "krk_"
    
    # KRK Root (gated by MaterialSensor)
    krk_root = Node(f"{prefix}root", NodeType.SCRIPT, meta={
        "layer": "endgame",
        "subgraph": "krk",
        "gate_source": "EndgameDetector",
    })
    g.add_node(krk_root)
    
    # Connect to main graph
    g.add_edge("GameRoot", f"{prefix}root", LinkType.SUB)
    
    # KRK Phase nodes
    phases = [
        ("phase0_establish_cut", "Establish first cut"),
        ("phase1_drive_to_edge", "Drive king to edge"),
        ("phase2_shrink_box", "Shrink the box"),
        ("phase3_take_opposition", "Take opposition"),
        ("phase4_deliver_mate", "Deliver checkmate"),
    ]
    
    for phase_id, desc in phases:
        node = Node(f"{prefix}{phase_id}", NodeType.SCRIPT, meta={
            "layer": "endgame_phase",
            "subgraph": "krk",
            "description": desc,
        })
        g.add_node(node)
        g.add_edge(f"{prefix}root", f"{prefix}{phase_id}", LinkType.SUB)
    
    # Phase sequencing (POR)
    g.add_edge(f"{prefix}phase0_establish_cut", f"{prefix}phase1_drive_to_edge", LinkType.POR)
    g.add_edge(f"{prefix}phase1_drive_to_edge", f"{prefix}phase2_shrink_box", LinkType.POR)
    g.add_edge(f"{prefix}phase2_shrink_box", f"{prefix}phase3_take_opposition", LinkType.POR)
    g.add_edge(f"{prefix}phase3_take_opposition", f"{prefix}phase4_deliver_mate", LinkType.POR)
    
    # KRK Evaluators (terminals)
    evaluators = [
        "king_at_edge",
        "king_confined",
        "barrier_ready",
        "box_can_shrink",
        "can_take_opposition",
        "can_deliver_mate",
        "is_stalemate",
        "cut_established",
        "rook_lost",
    ]
    
    for eval_id in evaluators:
        node = Node(f"{prefix}{eval_id}", NodeType.TERMINAL, meta={
            "layer": "endgame_terminal",
            "subgraph": "krk",
        })
        g.add_node(node)
    
    # KRK Move generators
    move_gens = [
        "choose_phase0",
        "king_drive_moves",
        "confinement_moves",
        "barrier_placement_moves",
        "box_shrink_moves",
        "opposition_moves",
        "mate_moves",
    ]
    
    for mg_id in move_gens:
        node = Node(f"{prefix}{mg_id}", NodeType.TERMINAL, meta={
            "layer": "endgame_actuator",
            "subgraph": "krk",
        })
        g.add_node(node)
    
    # Internal phase wiring (simplified)
    g.add_edge(f"{prefix}phase0_establish_cut", f"{prefix}cut_established", LinkType.SUB)
    g.add_edge(f"{prefix}phase0_establish_cut", f"{prefix}choose_phase0", LinkType.SUB)
    
    g.add_edge(f"{prefix}phase1_drive_to_edge", f"{prefix}king_at_edge", LinkType.SUB)
    g.add_edge(f"{prefix}phase1_drive_to_edge", f"{prefix}king_drive_moves", LinkType.SUB)
    g.add_edge(f"{prefix}phase1_drive_to_edge", f"{prefix}confinement_moves", LinkType.SUB)
    
    g.add_edge(f"{prefix}phase2_shrink_box", f"{prefix}box_can_shrink", LinkType.SUB)
    g.add_edge(f"{prefix}phase2_shrink_box", f"{prefix}box_shrink_moves", LinkType.SUB)
    
    g.add_edge(f"{prefix}phase3_take_opposition", f"{prefix}can_take_opposition", LinkType.SUB)
    g.add_edge(f"{prefix}phase3_take_opposition", f"{prefix}opposition_moves", LinkType.SUB)
    
    g.add_edge(f"{prefix}phase4_deliver_mate", f"{prefix}can_deliver_mate", LinkType.SUB)
    g.add_edge(f"{prefix}phase4_deliver_mate", f"{prefix}mate_moves", LinkType.SUB)
    
    # Sentinels
    g.add_edge(f"{prefix}root", f"{prefix}is_stalemate", LinkType.SUB)
    g.add_edge(f"{prefix}root", f"{prefix}rook_lost", LinkType.SUB)
    
    # Add wait node
    wait_node = Node(f"{prefix}wait", NodeType.TERMINAL, meta={
        "layer": "endgame_terminal",
        "subgraph": "krk",
    })
    g.add_node(wait_node)


def _integrate_kpk_subgraph(g: Graph) -> None:
    """
    Integrate KPK endgame subgraph with prefixed node IDs.
    """
    prefix = "kpk_"
    
    # KPK Root
    kpk_root = Node(f"{prefix}root", NodeType.SCRIPT, meta={
        "layer": "endgame",
        "subgraph": "kpk",
        "gate_source": "EndgameDetector",
    })
    g.add_node(kpk_root)
    
    # Connect to main graph
    g.add_edge("GameRoot", f"{prefix}root", LinkType.SUB)
    
    # KPK Script nodes
    scripts = ["detect", "execute", "finish", "wait"]
    for script_id in scripts:
        node = Node(f"{prefix}{script_id}", NodeType.SCRIPT, meta={
            "layer": "endgame_script",
            "subgraph": "kpk",
        })
        g.add_node(node)
        g.add_edge(f"{prefix}root", f"{prefix}{script_id}", LinkType.SUB)
    
    # Sequencing
    g.add_edge(f"{prefix}detect", f"{prefix}execute", LinkType.POR)
    g.add_edge(f"{prefix}execute", f"{prefix}finish", LinkType.POR)
    g.add_edge(f"{prefix}finish", f"{prefix}wait", LinkType.POR)
    
    # KPK Terminals
    terminals = [
        "material_check",
        "push_window",
        "opposition_probe",
        "promotion_probe",
        "move_selector",
        "wait_for_change",
    ]
    
    for term_id in terminals:
        node = Node(f"{prefix}{term_id}", NodeType.TERMINAL, meta={
            "layer": "endgame_terminal",
            "subgraph": "kpk",
        })
        g.add_node(node)
    
    # Internal wiring
    g.add_edge(f"{prefix}detect", f"{prefix}material_check", LinkType.SUB)
    g.add_edge(f"{prefix}detect", f"{prefix}push_window", LinkType.SUB)
    g.add_edge(f"{prefix}execute", f"{prefix}move_selector", LinkType.SUB)
    g.add_edge(f"{prefix}execute", f"{prefix}opposition_probe", LinkType.SUB)
    g.add_edge(f"{prefix}finish", f"{prefix}promotion_probe", LinkType.SUB)
    g.add_edge(f"{prefix}wait", f"{prefix}wait_for_change", LinkType.SUB)


def _integrate_kqk_subgraph(g: Graph) -> None:
    """
    Integrate KQK endgame subgraph with prefixed node IDs.
    
    King + Queen vs King endgame.
    """
    prefix = "kqk_"
    
    # KQK Root (gated by MaterialSensor)
    kqk_root = Node(f"{prefix}root", NodeType.SCRIPT, meta={
        "layer": "endgame",
        "subgraph": "kqk",
        "gate_source": "EndgameDetector",
    })
    g.add_node(kqk_root)
    
    # Connect to main graph
    g.add_edge("GameRoot", f"{prefix}root", LinkType.SUB)
    
    # KQK Phase nodes
    phases = [
        ("phase1_drive", "Drive enemy king to edge"),
        ("phase2_corner", "Push king to corner"),
        ("phase3_mate", "Deliver checkmate"),
    ]
    
    for phase_id, desc in phases:
        node = Node(f"{prefix}{phase_id}", NodeType.SCRIPT, meta={
            "layer": "endgame_phase",
            "subgraph": "kqk",
            "description": desc,
        })
        g.add_node(node)
        g.add_edge(f"{prefix}root", f"{prefix}{phase_id}", LinkType.SUB)
    
    # Phase sequencing (POR)
    g.add_edge(f"{prefix}phase1_drive", f"{prefix}phase2_corner", LinkType.POR)
    g.add_edge(f"{prefix}phase2_corner", f"{prefix}phase3_mate", LinkType.POR)
    
    # KQK Evaluators (terminals)
    evaluators = [
        "material_check",
        "edge_detector",
        "corner_detector",
        "mate_detector",
        "restriction_eval",
    ]
    
    for eval_id in evaluators:
        node = Node(f"{prefix}{eval_id}", NodeType.TERMINAL, meta={
            "layer": "endgame_terminal",
            "subgraph": "kqk",
        })
        g.add_node(node)
    
    # KQK Move generators
    move_gens = [
        "drive_moves",
        "approach_moves",
        "mate_moves",
        "move_selector",
    ]
    
    for mg_id in move_gens:
        node = Node(f"{prefix}{mg_id}", NodeType.TERMINAL, meta={
            "layer": "endgame_actuator",
            "subgraph": "kqk",
        })
        g.add_node(node)
    
    # Internal phase wiring
    g.add_edge(f"{prefix}phase1_drive", f"{prefix}edge_detector", LinkType.SUB)
    g.add_edge(f"{prefix}phase1_drive", f"{prefix}restriction_eval", LinkType.SUB)
    g.add_edge(f"{prefix}phase1_drive", f"{prefix}drive_moves", LinkType.SUB)
    
    g.add_edge(f"{prefix}phase2_corner", f"{prefix}corner_detector", LinkType.SUB)
    g.add_edge(f"{prefix}phase2_corner", f"{prefix}approach_moves", LinkType.SUB)
    
    g.add_edge(f"{prefix}phase3_mate", f"{prefix}mate_detector", LinkType.SUB)
    g.add_edge(f"{prefix}phase3_mate", f"{prefix}mate_moves", LinkType.SUB)
    
    # Global move selector (fallback)
    g.add_edge(f"{prefix}root", f"{prefix}move_selector", LinkType.SUB)
    g.add_edge(f"{prefix}root", f"{prefix}material_check", LinkType.SUB)
    
    # Add wait node
    wait_node = Node(f"{prefix}wait", NodeType.TERMINAL, meta={
        "layer": "endgame_terminal",
        "subgraph": "kqk",
    })
    g.add_node(wait_node)
    g.add_edge(f"{prefix}root", f"{prefix}wait", LinkType.SUB)


def _integrate_tactics_subgraphs(g: Graph) -> None:
    """
    Integrate all tactical pattern subgraphs.
    
    Each tactic gets:
    - detect_{tactic} node (sensor)
    - exploit_{tactic} node (actuator)
    
    Connected under a tactics_root with learnable edge weights.
    Note: tactics_root connects to GameRoot directly (not strategies)
    to avoid single-parent constraint violation.
    """
    # Tactics root connects directly to GameRoot (parallel to strategies)
    tactics_root = Node("tactics_root", NodeType.SCRIPT, meta={
        "layer": "tactics",
        "subgraph": "tactics",
        "gate_source": "TacticsSensor",
    })
    g.add_node(tactics_root)
    
    # Connect tactics directly to GameRoot (parallel execution with strategies)
    g.add_edge("GameRoot", "tactics_root", LinkType.SUB)
    
    # Add each tactic type
    # Structure: tactics_root -> tactic_script -> detector (terminal)
    #                                          -> exploiter (terminal)
    for tactic in TACTIC_TYPES:
        # Script node for this tactic type (can have SUB children)
        tactic_script = Node(f"tactic_{tactic}", NodeType.SCRIPT, meta={
            "layer": "tactics_script",
            "subgraph": f"tactics_{tactic}",
            "tactic_type": tactic,
        })
        g.add_node(tactic_script)
        
        # Detector terminal node
        detector = Node(f"detect_{tactic}", NodeType.TERMINAL, meta={
            "layer": "tactics_detector",
            "subgraph": f"tactics_{tactic}",
            "tactic_type": tactic,
        })
        g.add_node(detector)
        
        # Exploiter terminal node
        exploiter = Node(f"exploit_{tactic}", NodeType.TERMINAL, meta={
            "layer": "tactics_actuator",
            "subgraph": f"tactics_{tactic}",
            "tactic_type": tactic,
        })
        g.add_node(exploiter)
        
        # Wire: root -> script -> detector, script -> exploiter
        g.add_edge("tactics_root", f"tactic_{tactic}", LinkType.SUB)
        g.add_edge(f"tactic_{tactic}", f"detect_{tactic}", LinkType.SUB)
        g.add_edge(f"tactic_{tactic}", f"exploit_{tactic}", LinkType.SUB)
    
    # Special handling for hangingPiece (two response options)
    if "hangingPiece" in TACTIC_TYPES:
        # Add protect option as alternative under the hangingPiece script
        protect_node = Node("protect_hangingPiece", NodeType.TERMINAL, meta={
            "layer": "tactics_actuator",
            "subgraph": "tactics_hangingPiece",
            "tactic_type": "hangingPiece",
        })
        g.add_node(protect_node)
        g.add_edge("tactic_hangingPiece", "protect_hangingPiece", LinkType.SUB)


def _wire_strategic_connections(g: Graph) -> None:
    """Wire strategic layer connections with sensors."""
    sensor_links = {
        "Develop": ["DevelopmentSensor", "CenterControlSensor"],
        "Castle": ["DevelopmentSensor"],
        "CenterControl": ["CenterControlSensor"],
        "AttackKing": ["KingSafetySensor", "MaterialSensor"],
        "Simplify": ["MaterialSensor", "PhaseSensor"],
        "ImproveWorstPiece": ["PieceActivitySensor"],
        "ConvertAdvantage": ["MaterialSensor", "PhaseSensor"],
        "WinMaterial": ["MaterialSensor", "TacticsSensor"],
        "DefendWeakness": ["KingSafetySensor"],
        "CreateWeakness": ["PieceActivitySensor", "TacticsSensor"],
    }
    
    for plan_id, sensors in sensor_links.items():
        if plan_id not in g.nodes:
            continue
        for sensor_id in sensors:
            if sensor_id in g.nodes:
                g.add_edge(plan_id, sensor_id, LinkType.SUB)


def _get_edge_subgraph(g: Graph, src: str, dst: str) -> str:
    """Determine which subgraph an edge belongs to."""
    src_node = g.nodes.get(src)
    dst_node = g.nodes.get(dst)
    
    # Both nodes must exist
    if not src_node or not dst_node:
        return "main"
    
    # Check node subgraphs
    src_sg = src_node.meta.get("subgraph", "main")
    dst_sg = dst_node.meta.get("subgraph", "main")
    
    # If both are same subgraph, use that
    if src_sg == dst_sg:
        return src_sg
    
    # Cross-subgraph edges belong to the parent subgraph
    # (e.g., GameRoot->krk_root is a "main" edge)
    if src_sg == "main":
        return "main"
    
    return src_sg


def _mark_edges_for_consolidation(g: Graph) -> None:
    """
    Mark all edges in the graph for consolidation tracking.
    
    This enables the two-speed learning system to train the entire network:
    - Fast plasticity: within-game, per-tick eligibility traces
    - Slow consolidation: cross-game, baseline weight updates
    
    Each edge gets metadata:
    - consolidate: bool - whether to track this edge
    - subgraph: str - which subgraph this edge belongs to (for grouped save/load)
    - edge_key: str - canonical key for edge identification
    """
    for edge in g.edges:
        # Generate canonical edge key
        edge_key = f"{edge.src}->{edge.dst}:{edge.ltype.name}"
        
        # Determine subgraph
        subgraph = _get_edge_subgraph(g, edge.src, edge.dst)
        
        # Mark for consolidation
        edge.meta = getattr(edge, 'meta', {}) or {}
        edge.meta["consolidate"] = True
        edge.meta["subgraph"] = subgraph
        edge.meta["edge_key"] = edge_key
        
        # Initialize trace if not present
        if not hasattr(edge, 'trace'):
            edge.trace = 0.0


def get_edges_for_consolidation(g: Graph, subgraph: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Get all edges marked for consolidation with their current weights.
    
    Args:
        g: The graph
        subgraph: Optional filter by subgraph name (None = all edges)
        
    Returns:
        List of (edge_key, weight) tuples
    """
    edges = []
    
    for edge in g.edges:
        meta = getattr(edge, 'meta', {})
        
        if not meta.get("consolidate", False):
            continue
        
        if subgraph and meta.get("subgraph") != subgraph:
            continue
        
        edge_key = meta.get("edge_key", f"{edge.src}->{edge.dst}:{edge.ltype.name}")
        edges.append((edge_key, float(edge.w)))
    
    return edges


def get_active_edge_traces(g: Graph, threshold: float = 0.01) -> Dict[str, float]:
    """
    Get edges with active traces above threshold.
    
    This is used for accumulating episode data during consolidation.
    
    Args:
        g: The graph
        threshold: Minimum trace value to include
        
    Returns:
        Dict mapping edge_key to trace value
    """
    active = {}
    
    for edge in g.edges:
        trace = getattr(edge, 'trace', 0.0)
        if trace > threshold:
            meta = getattr(edge, 'meta', {})
            edge_key = meta.get("edge_key", f"{edge.src}->{edge.dst}:{edge.ltype.name}")
            active[edge_key] = trace
    
    return active


def reset_edge_traces(g: Graph) -> None:
    """Reset all edge traces to 0 (called between games)."""
    for edge in g.edges:
        edge.trace = 0.0


def load_all_weights(g: Graph, weights_dir: Path) -> Dict[str, int]:
    """
    Load weights for all subgraphs from a directory.
    
    Expected structure:
        weights_dir/
        ├── fullgame_consol.json
        ├── krk_consol.json
        ├── kpk_consol.json
        ├── kqk_consol.json
        └── tactics/
            ├── fork_consol.json
            ├── pin_consol.json
            └── ...
    
    Args:
        g: The unified graph
        weights_dir: Directory containing weight files
        
    Returns:
        Dict with counts of edges updated per subgraph
    """
    stats = {
        "fullgame": 0,
        "krk": 0,
        "kpk": 0,
        "kqk": 0,
        "tactics": 0,
    }
    
    weights_dir = Path(weights_dir)
    
    # Main graph weights
    fullgame_path = weights_dir / "fullgame_consol.json"
    if fullgame_path.exists():
        stats["fullgame"] = _apply_weight_file(g, fullgame_path, prefix=None)
    
    # KRK subgraph weights
    krk_path = weights_dir / "krk_consol.json"
    if krk_path.exists():
        stats["krk"] = _apply_weight_file(g, krk_path, prefix="krk_")
    
    # KPK subgraph weights
    kpk_path = weights_dir / "kpk_consol.json"
    if kpk_path.exists():
        stats["kpk"] = _apply_weight_file(g, kpk_path, prefix="kpk_")
    
    # KQK subgraph weights
    kqk_path = weights_dir / "kqk_consol.json"
    if kqk_path.exists():
        stats["kqk"] = _apply_weight_file(g, kqk_path, prefix="kqk_")
    
    # Tactics weights (16 files)
    tactics_dir = weights_dir / "tactics"
    if tactics_dir.exists():
        for tactic_file in tactics_dir.glob("*_consol.json"):
            count = _apply_weight_file(g, tactic_file, prefix=None)
            stats["tactics"] += count
    
    return stats


def _apply_weight_file(g: Graph, path: Path, prefix: Optional[str]) -> int:
    """Apply weights from a consolidation JSON file to the graph."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0
    
    w_base = data.get("w_base", {})
    count = 0
    
    for edge_key, weight in w_base.items():
        # Parse edge key: "src->dst:type"
        try:
            if "->" not in edge_key:
                continue
            parts = edge_key.split("->")
            src = parts[0]
            dst_type = parts[1].split(":")
            dst = dst_type[0]
            
            # Apply prefix if specified
            if prefix:
                src = f"{prefix}{src}" if not src.startswith(prefix) else src
                dst = f"{prefix}{dst}" if not dst.startswith(prefix) else dst
            
            # Find and update edge
            for edge in g.edges:
                if edge.src == src and edge.dst == dst:
                    edge.w = float(weight)
                    count += 1
                    break
        except Exception:
            continue
    
    return count


def get_subgraph_summary(g: Graph) -> Dict[str, Dict[str, Any]]:
    """
    Get a summary of all subgraphs in the unified graph.
    
    Returns:
        Dict mapping subgraph name to summary info
    """
    subgraphs: Dict[str, Dict[str, Any]] = {}
    
    for nid, node in g.nodes.items():
        sg = node.meta.get("subgraph", "main")
        if sg not in subgraphs:
            subgraphs[sg] = {
                "node_count": 0,
                "edge_count": 0,
                "nodes": [],
                "root": None,
            }
        
        subgraphs[sg]["node_count"] += 1
        subgraphs[sg]["nodes"].append(nid)
        
        # Identify root
        if "root" in nid.lower() and subgraphs[sg]["root"] is None:
            subgraphs[sg]["root"] = nid
    
    # Count edges per subgraph
    for edge in g.edges:
        src_sg = g.nodes.get(edge.src, Node("", NodeType.SCRIPT)).meta.get("subgraph", "main")
        subgraphs.get(src_sg, {}).setdefault("edge_count", 0)
        if src_sg in subgraphs:
            subgraphs[src_sg]["edge_count"] += 1
    
    return subgraphs

