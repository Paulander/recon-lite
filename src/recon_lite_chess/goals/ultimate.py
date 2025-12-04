"""Ultimate Goal Layer for M6 Goal Hierarchy.

Ultimate goals represent the highest level of the goal hierarchy,
determining the overall direction of play based on position assessment.

Goals:
- WIN: We have an advantage and should convert it
- DRAW: Position is roughly equal or we're slightly worse
- SURVIVE: We're losing and need to minimize damage

The ultimate goal influences which strategic plans are relevant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Tuple, List

import chess

from recon_lite.graph import Graph, Node, NodeType, LinkType
from ..sensors.material import assess_material, MaterialCategory


class UltimateGoal(Enum):
    """Top-level game objectives."""
    WIN = auto()      # Convert advantage to victory
    DRAW = auto()     # Maintain equality, seek draw if needed
    SURVIVE = auto()  # Minimize losses, avoid checkmate


@dataclass
class UltimateAssessment:
    """Assessment of which ultimate goal applies."""
    goal: UltimateGoal
    confidence: float  # 0.0 to 1.0
    reason: str
    material_balance: float
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal.name,
            "confidence": self.confidence,
            "reason": self.reason,
            "material_balance": self.material_balance,
        }


def assess_ultimate_goal(board: chess.Board, side_to_move: bool) -> UltimateAssessment:
    """
    Assess which ultimate goal applies for the side to move.
    
    Args:
        board: Current chess position
        side_to_move: True for White, False for Black
        
    Returns:
        UltimateAssessment with goal, confidence, and reasoning
    """
    material = assess_material(board)
    
    # Adjust balance for side to move (positive = we're ahead)
    balance = material.balance if side_to_move else -material.balance
    
    # Check for checkmate/stalemate
    if board.is_checkmate():
        # We're checkmated (it's our turn and we're in checkmate)
        return UltimateAssessment(
            goal=UltimateGoal.SURVIVE,
            confidence=1.0,
            reason="Checkmated",
            material_balance=balance,
        )
    
    if board.is_stalemate():
        return UltimateAssessment(
            goal=UltimateGoal.DRAW,
            confidence=1.0,
            reason="Stalemate",
            material_balance=balance,
        )
    
    # Significant winning position
    if balance >= 3.0:
        confidence = min(1.0, 0.6 + (balance - 3.0) * 0.1)
        return UltimateAssessment(
            goal=UltimateGoal.WIN,
            confidence=confidence,
            reason=f"Material advantage (+{balance:.1f})",
            material_balance=balance,
        )
    
    # Significant losing position
    if balance <= -3.0:
        confidence = min(1.0, 0.6 + (-balance - 3.0) * 0.1)
        return UltimateAssessment(
            goal=UltimateGoal.SURVIVE,
            confidence=confidence,
            reason=f"Material deficit ({balance:.1f})",
            material_balance=balance,
        )
    
    # Known winning endgame patterns
    winning_patterns = [MaterialCategory.KRK, MaterialCategory.KQK]
    if material.category in winning_patterns:
        is_attacker = (balance > 0)
        if is_attacker:
            return UltimateAssessment(
                goal=UltimateGoal.WIN,
                confidence=0.95,
                reason=f"Winning endgame pattern ({material.category.name})",
                material_balance=balance,
            )
        else:
            return UltimateAssessment(
                goal=UltimateGoal.SURVIVE,
                confidence=0.95,
                reason=f"Defending vs {material.category.name}",
                material_balance=balance,
            )
    
    # Equal-ish position
    if abs(balance) < 1.0:
        return UltimateAssessment(
            goal=UltimateGoal.DRAW,
            confidence=0.7,
            reason="Roughly equal position",
            material_balance=balance,
        )
    
    # Slight advantage (1-3 pawns)
    if balance > 0:
        return UltimateAssessment(
            goal=UltimateGoal.WIN,
            confidence=0.4 + balance * 0.1,
            reason=f"Slight advantage (+{balance:.1f})",
            material_balance=balance,
        )
    else:
        return UltimateAssessment(
            goal=UltimateGoal.DRAW,
            confidence=0.5,
            reason=f"Slightly worse ({balance:.1f}), seeking equality",
            material_balance=balance,
        )


def ultimate_goal_predicate(node: Any, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Terminal predicate that assesses ultimate goal.
    
    Stores assessment in node.activation.meta and env.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    side_to_move = board.turn
    assessment = assess_ultimate_goal(board, side_to_move)
    
    # Store in node
    node.activation.meta["ultimate_goal"] = assessment.as_dict()
    node.activation.meta["goal"] = assessment.goal.name
    
    # Store in env
    env["ultimate_goal"] = assessment.as_dict()
    
    # Activation encodes goal: 0=SURVIVE, 0.5=DRAW, 1.0=WIN
    goal_map = {UltimateGoal.SURVIVE: 0.0, UltimateGoal.DRAW: 0.5, UltimateGoal.WIN: 1.0}
    node.activation.value = goal_map[assessment.goal] * assessment.confidence
    
    return True, True


def create_ultimate_goal_node() -> Node:
    """Create the ultimate goal assessment terminal."""
    return Node(
        nid="UltimateGoal",
        ntype=NodeType.TERMINAL,
        predicate=ultimate_goal_predicate,
        meta={"layer": "ultimate", "fan_in_allowed": True},
    )


def build_ultimate_goal_hierarchy(g: Graph) -> None:
    """
    Build the ultimate goal hierarchy in the graph.
    
    Structure:
        GameRoot (script)
        ├── sub → UltimateGoal (terminal, fan-in)
        ├── sub → WinStrategy (script) - activated when WIN goal
        ├── sub → DrawStrategy (script) - activated when DRAW goal
        └── sub → SurviveStrategy (script) - activated when SURVIVE goal
    
    The terminal provides the assessment, and the script children
    are alternative strategies that get activated based on the goal.
    """
    # Create ultimate goal terminal
    ultimate = create_ultimate_goal_node()
    g.add_node(ultimate)
    
    # Create the root game script
    root = Node(
        nid="GameRoot",
        ntype=NodeType.SCRIPT,
        meta={"layer": "root"},
    )
    g.add_node(root)
    
    # Create placeholder strategy scripts
    # These will have their children (strategic plans) added later
    win_strategy = Node(
        nid="WinStrategy", 
        ntype=NodeType.SCRIPT,
        meta={"layer": "ultimate", "goal": "WIN", "alt": True},
    )
    draw_strategy = Node(
        nid="DrawStrategy",
        ntype=NodeType.SCRIPT,
        meta={"layer": "ultimate", "goal": "DRAW", "alt": True},
    )
    survive_strategy = Node(
        nid="SurviveStrategy",
        ntype=NodeType.SCRIPT,
        meta={"layer": "ultimate", "goal": "SURVIVE", "alt": True},
    )
    
    g.add_node(win_strategy)
    g.add_node(draw_strategy)
    g.add_node(survive_strategy)
    
    # Link root to ultimate goal terminal (for assessment)
    g.add_edge("GameRoot", "UltimateGoal", LinkType.SUB)
    
    # Add placeholder terminals to make strategies valid
    # These will be replaced with actual strategic plans
    win_placeholder = Node("WinPlaceholder", NodeType.TERMINAL)
    draw_placeholder = Node("DrawPlaceholder", NodeType.TERMINAL)
    survive_placeholder = Node("SurvivePlaceholder", NodeType.TERMINAL)
    
    g.add_node(win_placeholder)
    g.add_node(draw_placeholder)
    g.add_node(survive_placeholder)
    
    g.add_edge("WinStrategy", "WinPlaceholder", LinkType.SUB)
    g.add_edge("DrawStrategy", "DrawPlaceholder", LinkType.SUB)
    g.add_edge("SurviveStrategy", "SurvivePlaceholder", LinkType.SUB)
    
    # Link root to strategy alternatives (will be replaced with real plans)
    g.add_edge("GameRoot", "WinStrategy", LinkType.SUB)
    g.add_edge("GameRoot", "DrawStrategy", LinkType.SUB)
    g.add_edge("GameRoot", "SurviveStrategy", LinkType.SUB)
    
    # Set up confirmation policy - "or" means any one strategy confirming is enough
    g.set_confirm_policy("GameRoot", policy="or")

