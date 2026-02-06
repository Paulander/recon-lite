"""
KRK Entry Point Integration

Integrates the KRK mate-in-1 capability as a proper Leg in the full ReCoN architecture.
Includes:
- Affordance detection (is this a KRK endgame with mate-in-1?)
- Bandit selection at Hub (UCB for Leg selection)
- Integration with spawn points for runtime growth
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import math

import chess
import numpy as np

from recon_lite.graph import Graph, Node, NodeType
from recon_lite_chess.baseline_teacher import KRKTeacher
from recon_lite_chess.krk_checkmate_actuator import create_checkmate_actuator
from recon_lite_chess.spawn_point import SpawnPointManager, SpawnPointConfig


_teacher = KRKTeacher()


# =============================================================================
# Bandit Selection (UCB1)
# =============================================================================

@dataclass
class BanditArm:
    """Statistics for one Leg in the bandit"""
    arm_id: str
    wins: int = 0
    plays: int = 0
    ucb_score: float = float('inf')
    
    def update(self, reward: float):
        """Update arm statistics with reward (0 or 1)"""
        self.plays += 1
        self.wins += reward
    
    def compute_ucb(self, total_plays: int, exploration_constant: float = 1.41):
        """Compute UCB1 score"""
        if self.plays == 0:
            self.ucb_score = float('inf')
        else:
            exploitation = self.wins / self.plays
            exploration = exploration_constant * math.sqrt(math.log(total_plays) / self.plays)
            self.ucb_score = exploitation + exploration
        return self.ucb_score


class HubBandit:
    """
    UCB1 Bandit for selecting between Legs at the Hub.
    
    Each Leg is an arm. The bandit learns which Leg is most effective
    for different board states.
    """
    
    def __init__(self, exploration_constant: float = 1.41):
        self.arms: Dict[str, BanditArm] = {}
        self.exploration_constant = exploration_constant
        self.total_plays = 0
    
    def add_arm(self, arm_id: str):
        """Add a new Leg as an arm"""
        if arm_id not in self.arms:
            self.arms[arm_id] = BanditArm(arm_id=arm_id)
    
    def select_arm(self) -> str:
        """Select best arm using UCB1"""
        if not self.arms:
            return None
        
        # Compute UCB scores
        for arm in self.arms.values():
            arm.compute_ucb(max(1, self.total_plays), self.exploration_constant)
        
        # Select arm with highest UCB
        best_arm = max(self.arms.values(), key=lambda a: a.ucb_score)
        return best_arm.arm_id
    
    def update(self, arm_id: str, reward: float):
        """Update arm with reward"""
        if arm_id in self.arms:
            self.arms[arm_id].update(reward)
            self.total_plays += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics"""
        return {
            "total_plays": self.total_plays,
            "arms": {
                arm_id: {
                    "wins": arm.wins,
                    "plays": arm.plays,
                    "win_rate": arm.wins / arm.plays if arm.plays > 0 else 0,
                    "ucb": arm.ucb_score
                }
                for arm_id, arm in self.arms.items()
            }
        }


# =============================================================================
# KRK Affordance Detection
# =============================================================================

def is_krk_position(board: chess.Board) -> bool:
    """
    Check if the position is a KRK endgame (King + Rook vs King).
    
    Returns True if:
    - White has exactly King + Rook
    - Black has exactly King
    - It's White's turn
    """
    if board.turn != chess.WHITE:
        return False
    
    # Count white pieces
    white_pieces = board.occupied_co[chess.WHITE]
    white_king = board.pieces(chess.KING, chess.WHITE)
    white_rooks = board.pieces(chess.ROOK, chess.WHITE)
    
    # White should have only King + Rook
    if len(white_king) != 1 or len(white_rooks) != 1:
        return False
    if bin(white_pieces).count('1') != 2:
        return False
    
    # Black should have only King
    black_pieces = board.occupied_co[chess.BLACK]
    black_king = board.pieces(chess.KING, chess.BLACK)
    if len(black_king) != 1:
        return False
    if bin(black_pieces).count('1') != 1:
        return False
    
    return True


def get_krk_affordance(board: chess.Board) -> float:
    """
    Get affordance score for KRK Leg activation.
    
    Returns:
        1.0 if this is a KRK position with mate-in-1 possible
        0.5 if this is a KRK position without immediate mate
        0.0 if this is not a KRK position
    """
    if not is_krk_position(board):
        return 0.0
    
    features = _teacher.features(board)
    can_deliver_mate = features[12] > 0.5  # Feature 12
    
    if can_deliver_mate:
        return 1.0
    else:
        return 0.5


# =============================================================================
# Integrated KRK Entry Node
# =============================================================================

class KRKEntryLeg:
    """
    Full KRK Entry Leg with:
    - Affordance detection
    - Checkmate actuator
    - Spawn point integration
    """
    
    def __init__(self, leg_id: str = "krk_entry_leg"):
        self.leg_id = leg_id
        self.graph = Graph()
        
        # Create checkmate actuator as the main action node
        self.actuator = create_checkmate_actuator("krk_checkmate")
        self.graph.add_node(self.actuator)
        
        # Create spawn point manager (for runtime growth)
        self.spawn_manager = SpawnPointManager(SpawnPointConfig(
            spawn_probability=0.3,
            max_trials=5,
            trial_lifetime=50
        ))
        
        # Add leg node for spawn points
        leg_node = Node(nid=self.leg_id, ntype=NodeType.SCRIPT)
        self.graph.add_node(leg_node)
        self.spawn_manager.attach_to_legs(self.graph, leg_prefix=self.leg_id)
        
        # Statistics
        self.activations = 0
        self.successful_mates = 0
    
    def get_affordance(self, board: chess.Board) -> float:
        """Get affordance score for this Leg"""
        return get_krk_affordance(board)
    
    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Select the best move for this position.
        
        Returns:
            Best move or None if this Leg can't handle the position
        """
        if not is_krk_position(board):
            return None
        
        env = {"board": board}
        success, _ = self.actuator.predicate(self.actuator, self.graph, env)
        
        if success and "suggested_move" in env:
            self.activations += 1
            return env["suggested_move"]
        
        return None
    
    def update_reward(self, move: chess.Move, board: chess.Board, achieved_mate: bool):
        """
        Update Leg with reward signal.
        
        Also updates spawn points.
        """
        if achieved_mate:
            self.successful_mates += 1
        
        # Update spawn points
        self.spawn_manager.process_position(board, move, achieved_mate)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Leg statistics"""
        return {
            "leg_id": self.leg_id,
            "activations": self.activations,
            "successful_mates": self.successful_mates,
            "spawn_stats": self.spawn_manager.get_stats()
        }


# =============================================================================
# Full Orchestrator
# =============================================================================

class KRKOrchestrator:
    """
    Full orchestrator that combines:
    - Hub with bandit selection
    - KRK Entry Leg
    - (Future: other Legs like KQK, KPK)
    """
    
    def __init__(self):
        self.hub_bandit = HubBandit()
        self.legs: Dict[str, KRKEntryLeg] = {}
        
        # Add KRK entry leg
        self.add_leg(KRKEntryLeg("krk_entry"))
    
    def add_leg(self, leg: KRKEntryLeg):
        """Add a Leg to the orchestrator"""
        self.legs[leg.leg_id] = leg
        self.hub_bandit.add_arm(leg.leg_id)
    
    def select_move(self, board: chess.Board) -> Tuple[Optional[chess.Move], str]:
        """
        Select the best move using the orchestrator.
        
        Process:
        1. Compute affordance for each Leg
        2. Use bandit to select among viable Legs
        3. Get move from selected Leg
        
        Returns:
            (selected_move, selected_leg_id)
        """
        # Compute affordances
        affordances = {
            leg_id: leg.get_affordance(board)
            for leg_id, leg in self.legs.items()
        }
        
        # Filter to viable legs (affordance > 0)
        viable_legs = [lid for lid, aff in affordances.items() if aff > 0]
        
        if not viable_legs:
            return None, None
        
        # If only one viable leg, use it
        if len(viable_legs) == 1:
            selected_leg_id = viable_legs[0]
        else:
            # Use bandit to select
            selected_leg_id = self.hub_bandit.select_arm()
            if selected_leg_id not in viable_legs:
                selected_leg_id = viable_legs[0]
        
        # Get move from selected leg
        move = self.legs[selected_leg_id].select_move(board)
        
        return move, selected_leg_id
    
    def update_reward(self, leg_id: str, move: chess.Move, board: chess.Board, achieved_mate: bool):
        """Update Leg and bandit with reward"""
        reward = 1.0 if achieved_mate else 0.0
        
        # Update bandit
        self.hub_bandit.update(leg_id, reward)
        
        # Update leg
        if leg_id in self.legs:
            self.legs[leg_id].update_reward(move, board, achieved_mate)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "bandit": self.hub_bandit.get_stats(),
            "legs": {
                leg_id: leg.get_stats()
                for leg_id, leg in self.legs.items()
            }
        }


# =============================================================================
# Convenience function
# =============================================================================

def create_krk_orchestrator() -> KRKOrchestrator:
    """Create a KRK orchestrator ready for use"""
    return KRKOrchestrator()
