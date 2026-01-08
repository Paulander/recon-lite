#!/usr/bin/env python3
"""
Export the programmatic KRK network to a JSON topology file.

This converts the build_krk_network() Graph into a TopologyRegistry-compatible
JSON file that can be used by the M5 evolution system.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from demos.shared.krk_network import build_krk_network
from recon_lite.graph import LinkType, NodeType


def get_factory_for_node(node_id: str) -> str | None:
    """
    Map node IDs to their factory functions.
    Returns the fully qualified factory path or None for generic scripts.
    """
    # Terminal evaluators
    terminal_factories = {
        "king_at_edge": "recon_lite_chess.krk_nodes:create_king_edge_detector",
        "king_confined": "recon_lite_chess.krk_nodes:create_confinement_evaluator",
        "barrier_ready": "recon_lite_chess.krk_nodes:create_barrier_ready_evaluator",
        "box_can_shrink": "recon_lite_chess.krk_nodes:create_box_shrink_evaluator",
        "can_take_opposition": "recon_lite_chess.krk_nodes:create_opposition_evaluator",
        "can_deliver_mate": "recon_lite_chess.krk_nodes:create_mate_deliver_evaluator",
        "is_stalemate": "recon_lite_chess.krk_nodes:create_stalemate_detector",
        "cut_established": "recon_lite_chess.krk_nodes:create_cut_established_detector",
        "rook_lost": "recon_lite_chess.krk_nodes:create_rook_lost_detector",
        # Wait terminals
        "wait_after_p0": "recon_lite_chess.krk_nodes:create_wait_for_board_change",
        "wait_after_p1": "recon_lite_chess.krk_nodes:create_wait_for_board_change",
        "wait_after_p2": "recon_lite_chess.krk_nodes:create_wait_for_board_change",
        "wait_after_p3": "recon_lite_chess.krk_nodes:create_wait_for_board_change",
        "wait_after_p4": "recon_lite_chess.krk_nodes:create_wait_for_board_change",
    }
    
    # Actuators (move generators)
    actuator_factories = {
        "choose_phase0": "recon_lite_chess.krk_nodes:create_phase0_choose_moves",
        "king_drive_moves": "recon_lite_chess.krk_nodes:create_king_drive_moves",
        "confinement_moves": "recon_lite_chess.krk_nodes:create_confinement_moves",
        "barrier_placement_moves": "recon_lite_chess.krk_nodes:create_barrier_placement_moves",
        "box_shrink_moves": "recon_lite_chess.krk_nodes:create_box_shrink_moves",
        "opposition_moves": "recon_lite_chess.krk_nodes:create_opposition_moves",
        "mate_moves": "recon_lite_chess.krk_nodes:create_mate_moves",
        "random_legal_moves": "recon_lite_chess.krk_nodes:create_random_legal_moves",
        "no_progress_watch": "recon_lite_chess.krk_nodes:create_no_progress_watch",
    }
    
    # Script phases (no factory - generic SCRIPT)
    script_factories = {
        "krk_root": "recon_lite_chess.krk_nodes:create_krk_root",
        "phase0_establish_cut": "recon_lite_chess.krk_nodes:create_phase0_establish_cut",
        "phase1_drive_to_edge": "recon_lite_chess.krk_nodes:create_phase1_drive_to_edge",
        "phase2_shrink_box": "recon_lite_chess.krk_nodes:create_phase2_shrink_box",
        "phase3_take_opposition": "recon_lite_chess.krk_nodes:create_phase3_take_opposition",
        "phase4_deliver_mate": "recon_lite_chess.krk_nodes:create_phase4_deliver_mate",
    }
    
    if node_id in terminal_factories:
        return terminal_factories[node_id]
    if node_id in actuator_factories:
        return actuator_factories[node_id]
    if node_id in script_factories:
        return script_factories[node_id]
    
    # Generic phase script subnodes (p0_check, p1_move, etc.)
    return None


def get_node_group(node_id: str, ntype: NodeType) -> str:
    """Classify nodes into logical groups."""
    if ntype == NodeType.SCRIPT:
        if "phase" in node_id or node_id == "krk_root":
            return "backbone"
        if node_id.startswith("p") and "_" in node_id:  # p0_check, p1_move, etc.
            return "backbone"
        return "backbone"
    
    if ntype == NodeType.TERMINAL:
        # Actuators (move generators)
        if any(x in node_id for x in ["moves", "choose", "watch"]):
            return "actuator"
        # Sensors/evaluators
        return "sensor"
    
    return "other"


def export_krk_to_json(output_path: Path) -> dict:
    """
    Export the KRK network to JSON topology format.
    
    Returns:
        dict: The topology as a dictionary
    """
    g = build_krk_network()
    
    topology = {
        "version": "2.0",
        "network": "krk_legs",
        "description": "KRK (King+Rook vs King) checkmate network with phase-based architecture",
        "created": datetime.now().isoformat(),
        "nodes": [],
        "edges": [],
    }
    
    # Export nodes
    for node_id, node in g.nodes.items():
        node_type = node.ntype.name if hasattr(node.ntype, 'name') else str(node.ntype)
        group = get_node_group(node_id, node.ntype)
        factory = get_factory_for_node(node_id)
        
        node_entry = {
            "id": node_id,
            "type": node_type,
            "group": group,
            "factory": factory,
            "meta": dict(node.meta) if node.meta else {},
        }
        topology["nodes"].append(node_entry)
    
    # Export edges
    for edge in g.edges:
        link_type = edge.ltype
        if hasattr(link_type, 'name'):
            link_type_str = link_type.name
        else:
            link_type_str = str(link_type)
        
        edge_entry = {
            "src": edge.src,
            "dst": edge.dst,
            "type": link_type_str,
            "weight": edge.meta.get("weight", 1.0),
            "consolidate": edge.meta.get("consolidate", True),
            "confirmation_count": edge.meta.get("confirmation_count", 0),
        }
        topology["edges"].append(edge_entry)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(topology, f, indent=2)
    
    print(f"✅ Exported KRK topology to {output_path}")
    print(f"   Nodes: {len(topology['nodes'])}")
    print(f"   Edges: {len(topology['edges'])}")
    
    # Summarize by group
    groups = {}
    for n in topology["nodes"]:
        grp = n["group"]
        groups[grp] = groups.get(grp, 0) + 1
    print(f"   Groups: {groups}")
    
    return topology


def create_krk_legs_topology(output_path: Path) -> dict:
    """
    Create a simplified KRK "legs" topology for M5 evolution.
    
    This is analogous to kpk_legs_topology.json - a flattened structure 
    with detect/execute/finish/wait phases and leg actuators.
    """
    topology = {
        "version": "2.0",
        "network": "krk_legs",
        "description": "KRK with legs architecture (rook_leg/king_leg/arbiter) for M5 evolution",
        "created": datetime.now().isoformat(),
        "nodes": [
            # Root
            {"id": "krk_root", "type": "SCRIPT", "group": "backbone", "factory": None, "meta": {}},
            # Backbone phases (detect -> execute -> finish -> wait)
            {"id": "krk_detect", "type": "SCRIPT", "group": "backbone", "factory": None, "meta": {}},
            {"id": "krk_execute", "type": "SCRIPT", "group": "backbone", "factory": None, "meta": {}},
            {"id": "krk_finish", "type": "SCRIPT", "group": "backbone", "factory": None, "meta": {}},
            {"id": "krk_wait", "type": "SCRIPT", "group": "backbone", "factory": None, "meta": {}},
            
            # Sensors (detect phase) - UNIVERSAL features that transfer from KPK
            {"id": "krk_king_distance", "type": "TERMINAL", "group": "sensor", 
             "factory": "recon_lite_chess.krk_nodes:create_king_edge_detector",
             "meta": {"universal": True, "transfer_from": "king_distance"}},
            {"id": "krk_opposition_probe", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_opposition_evaluator",
             "meta": {"universal": True, "transfer_from": "opposition"}},
            
            # Sensors (detect phase) - KRK-SPECIFIC features
            {"id": "krk_box_area", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_box_shrink_evaluator",
             "meta": {"krk_specific": True}},
            {"id": "krk_cut_established", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_cut_established_detector",
             "meta": {"krk_specific": True}},
            {"id": "krk_mate_ready", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_mate_deliver_evaluator",
             "meta": {"krk_specific": True}},
            {"id": "krk_stalemate_danger", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_stalemate_detector",
             "meta": {"krk_specific": True}},
            
            # Actuator legs (execute phase)
            {"id": "krk_rook_leg", "type": "SCRIPT", "group": "actuator",
             "factory": "recon_lite_chess.krk_nodes:create_box_shrink_moves",
             "meta": {"layer": "actuator", "piece": "rook"}},
            {"id": "krk_king_leg", "type": "SCRIPT", "group": "actuator",
             "factory": "recon_lite_chess.krk_nodes:create_king_drive_moves",
             "meta": {"layer": "actuator", "piece": "king", "universal": True}},
            {"id": "krk_arbiter", "type": "TERMINAL", "group": "actuator",
             "factory": "recon_lite_chess.krk_nodes:create_phase0_choose_moves",
             "meta": {"layer": "actuator"}},
            
            # Finish phase sensor
            {"id": "krk_checkmate_probe", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_mate_deliver_evaluator", "meta": {}},
            
            # Wait terminal
            {"id": "krk_wait_for_change", "type": "TERMINAL", "group": "sensor",
             "factory": "recon_lite_chess.krk_nodes:create_wait_for_board_change", "meta": {}},
        ],
        "edges": [
            # Root SUB connections
            {"src": "krk_root", "dst": "krk_detect", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_root", "dst": "krk_execute", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_root", "dst": "krk_finish", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_root", "dst": "krk_wait", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            
            # Detect phase sensors
            {"src": "krk_detect", "dst": "krk_king_distance", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_detect", "dst": "krk_opposition_probe", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_detect", "dst": "krk_box_area", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_detect", "dst": "krk_cut_established", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_detect", "dst": "krk_mate_ready", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_detect", "dst": "krk_stalemate_danger", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            
            # Execute phase legs
            {"src": "krk_execute", "dst": "krk_rook_leg", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_execute", "dst": "krk_king_leg", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_execute", "dst": "krk_arbiter", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            
            # Finish phase
            {"src": "krk_finish", "dst": "krk_checkmate_probe", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            
            # Wait phase
            {"src": "krk_wait", "dst": "krk_wait_for_change", "type": "SUB", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            
            # POR sequencing: detect -> execute -> finish -> wait
            {"src": "krk_detect", "dst": "krk_execute", "type": "POR", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_execute", "dst": "krk_finish", "type": "POR", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_finish", "dst": "krk_wait", "type": "POR", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            
            # Leg POR sequencing (rook first, then king, arbiter resolves)
            {"src": "krk_rook_leg", "dst": "krk_king_leg", "type": "POR", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_rook_leg", "dst": "krk_arbiter", "type": "POR", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
            {"src": "krk_king_leg", "dst": "krk_arbiter", "type": "POR", "weight": 1.0, "consolidate": True, "confirmation_count": 0},
        ],
        "last_modified": datetime.now().isoformat(),
    }
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(topology, f, indent=2)
    
    print(f"✅ Created KRK legs topology at {output_path}")
    print(f"   Nodes: {len(topology['nodes'])}")
    print(f"   Edges: {len(topology['edges'])}")
    
    return topology


if __name__ == "__main__":
    # Export both topologies
    project_root = Path(__file__).parent.parent
    
    # Full topology (all phases, detailed)
    full_path = project_root / "topologies" / "krk_topology.json"
    export_krk_to_json(full_path)
    
    print()
    
    # Legs topology (M5-compatible, for evolution)
    legs_path = project_root / "topologies" / "krk_legs_topology.json"
    create_krk_legs_topology(legs_path)

