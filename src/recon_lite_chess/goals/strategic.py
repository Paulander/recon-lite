"""Strategic Plan Layer for M6 Goal Hierarchy.

Strategic plans are medium-term goals (5-15 moves) that serve
ultimate goals. Each plan:
- Has a base_weight (inherent priority)
- Has phase_boost (multiplier per game phase)
- Uses persistence (inertia, decay) stored in activation

Plans query shared sensor terminals (fan-in) to make decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Tuple, List, Optional

import chess

from recon_lite.graph import Graph, Node, NodeType, LinkType


class PlanCategory(Enum):
    """Categories of strategic plans."""
    OPENING = auto()      # Early game development
    MIDDLEGAME = auto()   # Complex play with many pieces
    ENDGAME = auto()      # Technical conversion
    UNIVERSAL = auto()    # Applies to all phases


@dataclass
class StrategicPlan:
    """Definition of a strategic plan."""
    id: str
    name: str
    description: str
    category: PlanCategory
    base_weight: float = 1.0
    phase_boost: Dict[str, float] = field(default_factory=dict)
    # Persistence parameters
    default_inertia: float = 0.7  # How sticky the plan is
    default_decay: float = 0.1    # How fast it fades without reinforcement
    # Which sensors this plan queries (for building graph connections)
    sensors: List[str] = field(default_factory=list)
    
    def phase_adjusted_weight(self, phase_weights: Dict[str, float]) -> float:
        """Calculate weight adjusted for current game phase."""
        weight = self.base_weight
        for phase, phase_w in phase_weights.items():
            boost = self.phase_boost.get(phase, 1.0)
            weight += (boost - 1.0) * phase_w
        return max(0.0, weight)


# === Strategic Plan Definitions ===

STRATEGIC_PLANS: Dict[str, StrategicPlan] = {
    # --- Opening Plans ---
    "Develop": StrategicPlan(
        id="Develop",
        name="Develop Pieces",
        description="Get minor pieces off starting squares",
        category=PlanCategory.OPENING,
        base_weight=1.5,
        phase_boost={"opening": 2.0, "middlegame": 0.5, "endgame": 0.1},
        sensors=["PhaseSensor", "MaterialSensor"],
    ),
    "Castle": StrategicPlan(
        id="Castle",
        name="Castle Early",
        description="Secure the king through castling",
        category=PlanCategory.OPENING,
        base_weight=1.2,
        phase_boost={"opening": 2.0, "middlegame": 0.3, "endgame": 0.0},
        sensors=["PhaseSensor"],
    ),
    "CenterControl": StrategicPlan(
        id="CenterControl",
        name="Control Center",
        description="Occupy or control central squares",
        category=PlanCategory.OPENING,
        base_weight=1.3,
        phase_boost={"opening": 1.5, "middlegame": 1.0, "endgame": 0.3},
        sensors=["PhaseSensor"],
    ),
    
    # --- Middlegame Plans ---
    "AttackKing": StrategicPlan(
        id="AttackKing",
        name="Attack Enemy King",
        description="Launch an attack on the enemy king",
        category=PlanCategory.MIDDLEGAME,
        base_weight=1.0,
        phase_boost={"opening": 0.2, "middlegame": 2.0, "endgame": 0.5},
        default_inertia=0.8,  # Once attacking, keep attacking
        sensors=["PhaseSensor", "MaterialSensor"],
    ),
    "CreateWeakness": StrategicPlan(
        id="CreateWeakness",
        name="Create Weaknesses",
        description="Target isolated or doubled pawns, weak squares",
        category=PlanCategory.MIDDLEGAME,
        base_weight=0.8,
        phase_boost={"opening": 0.3, "middlegame": 1.5, "endgame": 1.2},
        sensors=["PhaseSensor"],
    ),
    "ImproveWorstPiece": StrategicPlan(
        id="ImproveWorstPiece",
        name="Improve Worst Piece",
        description="Find and activate passive pieces",
        category=PlanCategory.MIDDLEGAME,
        base_weight=0.9,
        phase_boost={"opening": 0.5, "middlegame": 1.5, "endgame": 0.8},
        sensors=["PhaseSensor"],
    ),
    "Simplify": StrategicPlan(
        id="Simplify",
        name="Simplify When Ahead",
        description="Trade pieces to convert material advantage",
        category=PlanCategory.MIDDLEGAME,
        base_weight=0.7,
        phase_boost={"opening": 0.1, "middlegame": 1.2, "endgame": 1.5},
        sensors=["PhaseSensor", "MaterialSensor"],
    ),
    
    # --- Endgame Plans ---
    "ConvertAdvantage": StrategicPlan(
        id="ConvertAdvantage",
        name="Convert Advantage",
        description="Use material/positional edge to win",
        category=PlanCategory.ENDGAME,
        base_weight=1.0,
        phase_boost={"opening": 0.1, "middlegame": 0.5, "endgame": 2.0},
        sensors=["PhaseSensor", "MaterialSensor"],
    ),
    "CreatePassedPawn": StrategicPlan(
        id="CreatePassedPawn",
        name="Create Passed Pawn",
        description="Advance a pawn to become passed",
        category=PlanCategory.ENDGAME,
        base_weight=0.9,
        phase_boost={"opening": 0.0, "middlegame": 0.3, "endgame": 2.0},
        sensors=["PhaseSensor"],
    ),
    "KingActivation": StrategicPlan(
        id="KingActivation",
        name="Activate King",
        description="Move king to active central position",
        category=PlanCategory.ENDGAME,
        base_weight=0.8,
        phase_boost={"opening": 0.0, "middlegame": 0.2, "endgame": 2.0},
        sensors=["PhaseSensor"],
    ),
    
    # --- Universal/Tactical Plans ---
    "WinMaterial": StrategicPlan(
        id="WinMaterial",
        name="Win Material",
        description="Capture undefended or insufficiently defended pieces",
        category=PlanCategory.UNIVERSAL,
        base_weight=1.5,
        phase_boost={"opening": 1.0, "middlegame": 1.0, "endgame": 1.0},
        sensors=["MaterialSensor"],
    ),
    "DefendWeakness": StrategicPlan(
        id="DefendWeakness",
        name="Defend Weaknesses",
        description="Protect attacked or weak points",
        category=PlanCategory.UNIVERSAL,
        base_weight=1.2,
        phase_boost={"opening": 1.0, "middlegame": 1.0, "endgame": 1.0},
        sensors=[],
    ),
}


def create_strategic_plan_node(plan: StrategicPlan) -> Node:
    """Create a script node for a strategic plan."""
    return Node(
        nid=plan.id,
        ntype=NodeType.SCRIPT,
        meta={
            "layer": "strategic",
            "plan": plan.id,
            "category": plan.category.name,
            "base_weight": plan.base_weight,
            "phase_boost": plan.phase_boost,
            "inertia": plan.default_inertia,
            "decay": plan.default_decay,
        },
    )


def build_strategic_layer(
    g: Graph,
    parent_id: str,
    plans: List[str],
    *,
    connect_sensors: bool = True,
) -> List[str]:
    """
    Build strategic plan nodes and connect them to a parent.
    
    Args:
        g: The graph to modify
        parent_id: ID of the parent node (e.g., "WinStrategy")
        plans: List of plan IDs to add
        connect_sensors: If True, connect plans to shared sensor terminals
        
    Returns:
        List of created plan node IDs
    """
    created = []
    first_plan = True
    prev_plan_id = None
    
    for plan_id in plans:
        if plan_id not in STRATEGIC_PLANS:
            continue
            
        plan = STRATEGIC_PLANS[plan_id]
        node = create_strategic_plan_node(plan)
        g.add_node(node)
        created.append(plan_id)
        
        # Each plan needs at least one terminal child
        # Create a placeholder if sensors not connected
        placeholder = Node(f"{plan_id}_Placeholder", NodeType.TERMINAL)
        g.add_node(placeholder)
        g.add_edge(plan_id, f"{plan_id}_Placeholder", LinkType.SUB)
        
        # Connect to parent
        g.add_edge(parent_id, plan_id, LinkType.SUB)
        
        # Connect plans as alternatives (POR/RET)
        # This allows the parent to confirm when ANY plan confirms
        if not first_plan and prev_plan_id:
            # Plans are alternatives, not sequences
            # Mark them as alternatives using meta
            g.nodes[plan_id].meta["alt"] = True
        
        # Connect to shared sensors (fan-in)
        if connect_sensors:
            for sensor_id in plan.sensors:
                if sensor_id in g.nodes:
                    # Use fan-in: plan SUB-links to existing sensor
                    g.add_edge(plan_id, sensor_id, LinkType.SUB)
        
        first_plan = False
        prev_plan_id = plan_id
    
    # Set parent to use "or" confirmation policy
    if created:
        g.set_confirm_policy(parent_id, policy="or")
    
    return created


def get_active_plans_for_goal(
    goal: str,
    phase_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """
    Get plans appropriate for an ultimate goal, weighted by phase.
    
    Args:
        goal: "WIN", "DRAW", or "SURVIVE"
        phase_weights: Current phase weights (opening/middlegame/endgame)
        
    Returns:
        List of (plan_id, adjusted_weight) tuples, sorted by weight
    """
    phase_weights = phase_weights or {"opening": 0.0, "middlegame": 1.0, "endgame": 0.0}
    
    # Goal to plan category mapping
    goal_plans = {
        "WIN": [
            "AttackKing", "WinMaterial", "CreatePassedPawn", 
            "ConvertAdvantage", "Simplify",
        ],
        "DRAW": [
            "CenterControl", "DefendWeakness", "ImproveWorstPiece",
            "Develop", "Castle",
        ],
        "SURVIVE": [
            "DefendWeakness", "Simplify", "CenterControl",
        ],
    }
    
    plan_ids = goal_plans.get(goal, [])
    results = []
    
    for pid in plan_ids:
        if pid in STRATEGIC_PLANS:
            plan = STRATEGIC_PLANS[pid]
            weight = plan.phase_adjusted_weight(phase_weights)
            results.append((pid, weight))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

