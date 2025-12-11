"""
Reverse Curriculum Training Manager.

Implements the "train backwards from anchors" philosophy:
- Phase 1 (Anchor): Perfect endgame conversion
- Phase 2 (Bridge): Learn transitions to Phase 1
- Phase 3 (Wilderness): Tactical survival
- Phase 4 (Integration): Full game play

The system advances phases when exit criteria are met.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess


# Phase identifiers
PHASE_ANCHOR = "anchor"
PHASE_BRIDGE = "bridge"
PHASE_WILDERNESS = "wilderness"
PHASE_INTEGRATION = "integration"


class PhaseType(Enum):
    """Curriculum phase types."""
    ANCHOR = auto()        # Perfect endgame conversion
    BRIDGE = auto()        # Learn transitions
    WILDERNESS = auto()    # Complex tactical positions
    INTEGRATION = auto()   # Full game from opening


@dataclass
class PhaseStats:
    """
    Statistics for a curriculum phase.
    
    Tracks wins, losses, draws, move counts, and affordance metrics.
    """
    total_episodes: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_moves: int = 0
    theoretical_min_moves: int = 0
    
    # Affordance tracking for Bridge phase
    endgame_activations: int = 0  # Times endgame subgraph activated
    affordance_crossings: int = 0  # Times affordance crossed threshold
    
    # Validation against Stockfish (Wilderness phase)
    elo_estimate: float = 0.0
    stockfish_wins: int = 0
    stockfish_losses: int = 0
    stockfish_draws: int = 0
    
    @property
    def win_rate(self) -> float:
        """Win rate as a fraction."""
        if self.total_episodes == 0:
            return 0.0
        return self.wins / self.total_episodes
    
    @property
    def avg_moves(self) -> float:
        """Average moves per episode."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_moves / self.total_episodes
    
    @property
    def move_efficiency(self) -> float:
        """How close to theoretical minimum moves."""
        if self.theoretical_min_moves == 0 or self.total_episodes == 0:
            return 0.0
        return self.theoretical_min_moves / self.avg_moves if self.avg_moves > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "total_moves": self.total_moves,
            "avg_moves": round(self.avg_moves, 2),
            "theoretical_min_moves": self.theoretical_min_moves,
            "move_efficiency": round(self.move_efficiency, 4),
            "endgame_activations": self.endgame_activations,
            "affordance_crossings": self.affordance_crossings,
            "elo_estimate": round(self.elo_estimate, 1),
            "stockfish_wins": self.stockfish_wins,
            "stockfish_losses": self.stockfish_losses,
            "stockfish_draws": self.stockfish_draws,
        }


@dataclass
class CurriculumPhase:
    """
    Definition of a curriculum phase.
    
    Attributes:
        name: Phase identifier
        phase_type: Type of phase
        description: Human-readable description
        position_generator: Function that generates training positions
        exit_criteria: Function that checks if phase is complete
        config: Phase-specific configuration
    """
    name: str
    phase_type: PhaseType
    description: str
    position_generator: Callable[[], chess.Board]
    exit_criteria: Callable[[PhaseStats], bool]
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters for this phase
    plasticity_eta: float = 0.3      # Learning rate
    exploration_c: float = 1.0       # UCB exploration coefficient
    consolidation_weight: float = 0.5  # How much to consolidate
    
    def __hash__(self):
        return hash(self.name)


class CurriculumManager:
    """
    Orchestrates training across curriculum phases.
    
    Manages phase transitions, statistics tracking, and position generation.
    """
    
    def __init__(self, phases: List[CurriculumPhase]):
        if not phases:
            raise ValueError("Must provide at least one phase")
        
        self.phases = phases
        self.current_phase_idx = 0
        self.phase_stats: Dict[str, PhaseStats] = {
            p.name: PhaseStats() for p in phases
        }
        self.start_time = datetime.now().isoformat()
        self.total_episodes = 0
        self.phase_history: List[Dict[str, Any]] = []
    
    @property
    def current_phase(self) -> CurriculumPhase:
        """Get the current phase."""
        return self.phases[self.current_phase_idx]
    
    @property
    def current_stats(self) -> PhaseStats:
        """Get stats for current phase."""
        return self.phase_stats[self.current_phase.name]
    
    def is_complete(self) -> bool:
        """Check if all phases are complete."""
        return self.current_phase_idx >= len(self.phases)
    
    def get_training_position(self) -> chess.Board:
        """
        Get a training position for the current phase.
        
        Returns:
            A chess.Board set up for training
        """
        if self.is_complete():
            # Default to starting position if all phases complete
            return chess.Board()
        
        return self.current_phase.position_generator()
    
    def record_episode_result(
        self,
        outcome: str,  # "win", "loss", "draw"
        moves: int,
        theoretical_min: int = 0,
        *,
        endgame_activated: bool = False,
        affordance_crossed: bool = False,
        stockfish_result: Optional[str] = None,
    ) -> bool:
        """
        Record the result of a training episode.
        
        Args:
            outcome: "win", "loss", or "draw"
            moves: Number of moves in the episode
            theoretical_min: Theoretical minimum moves (for Anchor phase)
            endgame_activated: Whether endgame subgraph activated (Bridge)
            affordance_crossed: Whether affordance crossed threshold (Bridge)
            stockfish_result: Result against Stockfish (Wilderness)
            
        Returns:
            True if phase advanced, False otherwise
        """
        if self.is_complete():
            return False
        
        stats = self.current_stats
        stats.total_episodes += 1
        stats.total_moves += moves
        self.total_episodes += 1
        
        if outcome == "win":
            stats.wins += 1
        elif outcome == "loss":
            stats.losses += 1
        else:
            stats.draws += 1
        
        if theoretical_min > 0:
            stats.theoretical_min_moves += theoretical_min
        
        if endgame_activated:
            stats.endgame_activations += 1
        
        if affordance_crossed:
            stats.affordance_crossings += 1
        
        if stockfish_result:
            if stockfish_result == "win":
                stats.stockfish_wins += 1
            elif stockfish_result == "loss":
                stats.stockfish_losses += 1
            else:
                stats.stockfish_draws += 1
        
        # Check for phase advancement
        return self._check_phase_advancement()
    
    def _check_phase_advancement(self) -> bool:
        """Check if current phase exit criteria are met."""
        if self.is_complete():
            return False
        
        if self.current_phase.exit_criteria(self.current_stats):
            self._advance_phase()
            return True
        
        return False
    
    def _advance_phase(self) -> None:
        """Advance to the next phase."""
        # Record phase completion
        self.phase_history.append({
            "phase": self.current_phase.name,
            "stats": self.current_stats.to_dict(),
            "completed_at": datetime.now().isoformat(),
        })
        
        self.current_phase_idx += 1
    
    def force_advance(self) -> bool:
        """Force advancement to next phase (for debugging/testing)."""
        if self.is_complete():
            return False
        self._advance_phase()
        return True
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training parameters for current phase."""
        if self.is_complete():
            return {}
        
        phase = self.current_phase
        return {
            "phase_name": phase.name,
            "phase_type": phase.phase_type.name,
            "plasticity_eta": phase.plasticity_eta,
            "exploration_c": phase.exploration_c,
            "consolidation_weight": phase.consolidation_weight,
            **phase.config,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current curriculum status."""
        return {
            "current_phase": self.current_phase.name if not self.is_complete() else "COMPLETE",
            "current_phase_idx": self.current_phase_idx,
            "total_phases": len(self.phases),
            "total_episodes": self.total_episodes,
            "start_time": self.start_time,
            "phase_stats": {
                name: stats.to_dict() for name, stats in self.phase_stats.items()
            },
            "phase_history": self.phase_history,
        }
    
    def save_state(self, path: Path) -> None:
        """Save curriculum state to file."""
        state = self.get_status()
        state["version"] = "1.0"
        path.write_text(json.dumps(state, indent=2))
    
    def load_state(self, path: Path) -> None:
        """Load curriculum state from file."""
        data = json.loads(path.read_text())
        
        self.current_phase_idx = data.get("current_phase_idx", 0)
        self.total_episodes = data.get("total_episodes", 0)
        self.start_time = data.get("start_time", datetime.now().isoformat())
        self.phase_history = data.get("phase_history", [])
        
        # Restore phase stats
        for name, stats_dict in data.get("phase_stats", {}).items():
            if name in self.phase_stats:
                stats = self.phase_stats[name]
                stats.total_episodes = stats_dict.get("total_episodes", 0)
                stats.wins = stats_dict.get("wins", 0)
                stats.draws = stats_dict.get("draws", 0)
                stats.losses = stats_dict.get("losses", 0)
                stats.total_moves = stats_dict.get("total_moves", 0)
                stats.theoretical_min_moves = stats_dict.get("theoretical_min_moves", 0)
                stats.endgame_activations = stats_dict.get("endgame_activations", 0)
                stats.affordance_crossings = stats_dict.get("affordance_crossings", 0)
                stats.elo_estimate = stats_dict.get("elo_estimate", 0.0)
                stats.stockfish_wins = stats_dict.get("stockfish_wins", 0)
                stats.stockfish_losses = stats_dict.get("stockfish_losses", 0)
                stats.stockfish_draws = stats_dict.get("stockfish_draws", 0)


# ============================================================================
# Default Phase Exit Criteria
# ============================================================================

def _anchor_exit_criteria(stats: PhaseStats) -> bool:
    """
    Anchor phase exit: >99% win rate, avg moves < theoretical + 10.
    
    Requires at least 100 episodes for statistical significance.
    """
    if stats.total_episodes < 100:
        return False
    
    # Check win rate
    if stats.win_rate < 0.99:
        return False
    
    # Check move efficiency (at least 50% efficient)
    if stats.move_efficiency < 0.5:
        return False
    
    return True


def _bridge_exit_criteria(stats: PhaseStats) -> bool:
    """
    Bridge phase exit: Consistent endgame activation from middlegame.
    
    Requires endgame subgraphs to activate in >80% of episodes.
    """
    if stats.total_episodes < 50:
        return False
    
    activation_rate = stats.endgame_activations / stats.total_episodes
    return activation_rate > 0.8


def _wilderness_exit_criteria(stats: PhaseStats) -> bool:
    """
    Wilderness phase exit: Positive performance vs Stockfish (levels 1-3).
    
    Need >50% win+draw rate against weak Stockfish.
    """
    if stats.total_episodes < 100:
        return False
    
    sf_total = stats.stockfish_wins + stats.stockfish_losses + stats.stockfish_draws
    if sf_total < 50:
        return False
    
    # Win or draw rate > 50%
    non_loss_rate = (stats.stockfish_wins + stats.stockfish_draws) / sf_total
    return non_loss_rate > 0.5


def _integration_exit_criteria(stats: PhaseStats) -> bool:
    """
    Integration phase exit: Stable performance in full games.
    
    Runs indefinitely by default (never exits automatically).
    """
    # Integration phase doesn't auto-exit
    # It's meant for ongoing training
    return False


# ============================================================================
# Position Generators (import from generators module)
# ============================================================================

def _generate_anchor_position() -> chess.Board:
    """Generate a random KRK or KPK position."""
    from .generators import generate_anchor_position
    return generate_anchor_position()


def _generate_bridge_position() -> chess.Board:
    """Generate a simplified middlegame position."""
    from .generators import generate_bridge_position
    return generate_bridge_position()


def _generate_wilderness_position() -> chess.Board:
    """Generate a complex tactical position."""
    from .generators import generate_wilderness_position
    return generate_wilderness_position(randomize_moves=2)


def _generate_integration_position() -> chess.Board:
    """Generate starting position."""
    from .generators import generate_integration_position
    return generate_integration_position()


# ============================================================================
# Factory Functions
# ============================================================================

def create_default_curriculum() -> CurriculumManager:
    """
    Create a CurriculumManager with the default 4-phase curriculum.
    
    Returns:
        Configured CurriculumManager
    """
    phases = [
        CurriculumPhase(
            name=PHASE_ANCHOR,
            phase_type=PhaseType.ANCHOR,
            description="Perfect endgame conversion (KRK, KPK, KQK)",
            position_generator=_generate_anchor_position,
            exit_criteria=_anchor_exit_criteria,
            config={
                "endgame_types": ["KRK", "KPK", "KQK"],
                "max_moves": 50,
            },
            plasticity_eta=0.5,  # High learning rate for anchors
            exploration_c=0.7,   # Lower exploration (known strategies)
            consolidation_weight=0.7,  # High consolidation
        ),
        CurriculumPhase(
            name=PHASE_BRIDGE,
            phase_type=PhaseType.BRIDGE,
            description="Learn transitions from simplified middlegame to endgame",
            position_generator=_generate_bridge_position,
            exit_criteria=_bridge_exit_criteria,
            config={
                "material_advantage_required": True,
                "piece_count_range": (8, 12),
            },
            plasticity_eta=0.3,
            exploration_c=1.2,   # Higher exploration for discovery
            consolidation_weight=0.5,
        ),
        CurriculumPhase(
            name=PHASE_WILDERNESS,
            phase_type=PhaseType.WILDERNESS,
            description="Tactical survival in complex positions",
            position_generator=_generate_wilderness_position,
            exit_criteria=_wilderness_exit_criteria,
            config={
                "stockfish_levels": [1, 2, 3],
                "time_control": "bullet",
            },
            plasticity_eta=0.2,
            exploration_c=1.5,   # High exploration in unknown territory
            consolidation_weight=0.3,
        ),
        CurriculumPhase(
            name=PHASE_INTEGRATION,
            phase_type=PhaseType.INTEGRATION,
            description="Full game play from opening",
            position_generator=_generate_integration_position,
            exit_criteria=_integration_exit_criteria,
            config={
                "include_opening_book": True,
                "full_game": True,
            },
            plasticity_eta=0.1,  # Low learning rate for stability
            exploration_c=1.0,
            consolidation_weight=0.5,
        ),
    ]
    
    return CurriculumManager(phases)


def create_anchor_only_curriculum() -> CurriculumManager:
    """Create a curriculum with only the Anchor phase (for testing)."""
    phases = [
        CurriculumPhase(
            name=PHASE_ANCHOR,
            phase_type=PhaseType.ANCHOR,
            description="Perfect endgame conversion",
            position_generator=_generate_anchor_position,
            exit_criteria=_anchor_exit_criteria,
            config={"endgame_types": ["KRK", "KPK"]},
        ),
    ]
    return CurriculumManager(phases)

