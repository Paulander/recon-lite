"""
KRK Curriculum: Backward-Chained Training for King + Rook vs King.

Implements a 10-stage curriculum that teaches the Box Method through
proximity-based learning (backward chaining from mate positions).

Core Principles:
1. Backward Chaining: Learn mate first, then positions N-moves away
2. Multiple Positions Per Stage: 3-5 positions to learn relative patterns
3. Move Penalty: Hard penalty for suboptimal move counts
4. Box Escape Detection: Heavy penalty for letting enemy king escape

Stages:
    0. Mate_In_1: Back-rank mate in 1 move
    1. Mate_In_2: 2 moves to mate (tempo variants)
    2. Edge_Trapped_Tempo: Opposition/waiting move at edge
    3. Edge_Cut_Hold: 1x8 box, maintain cut
    4. King_Close_1: King 1 square from ideal
    5. King_Close_2: King 2 squares from ideal
    6. King_Far_Cut_Held: Rook guarding, king must approach
    7. Box_Small: 3x3 box shrinking
    8. Box_Medium: 4x4 box with opposition
    9. Full_KRK: Random winning positions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import chess


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class KRKStagePosition:
    """A specific position template for a curriculum stage."""
    fen: str
    optimal_moves: int
    description: str
    failure_condition: Optional[str] = None  # e.g., "box_grew", "stalemate"
    
    def to_board(self) -> chess.Board:
        """Convert to chess.Board."""
        return chess.Board(self.fen)


@dataclass
class KRKStage:
    """Configuration for a KRK curriculum stage."""
    stage_id: int
    name: str
    description: str
    positions: List[KRKStagePosition]
    target_win_rate: float = 0.8
    max_moves_multiplier: float = 2.0  # Allow 2x optimal before hard fail
    distance_to_mate: str = ""  # e.g., "1 move", "2-3 moves"
    key_lesson: str = ""
    
    def select_position(self) -> chess.Board:
        """Randomly select a position from this stage."""
        template = random.choice(self.positions)
        return template.to_board()
    
    def get_optimal_moves(self, board: chess.Board) -> int:
        """Get optimal moves for a given board position."""
        fen = board.fen().split(" ")[0]  # Just piece placement
        for pos in self.positions:
            if pos.fen.split(" ")[0] == fen:
                return pos.optimal_moves
        # Default estimate based on stage
        return self.positions[0].optimal_moves if self.positions else 10


# ============================================================================
# Stage 0: Mate_In_1 (5 Positions)
# Back-rank mate patterns - trivial starting point
# ============================================================================

STAGE_0_MATE_IN_1 = KRKStage(
    stage_id=0,
    name="Mate_In_1",
    description="Back-rank mate in 1 move",
    distance_to_mate="1 move",
    key_lesson="Basic rook mating pattern",
    target_win_rate=0.95,
    positions=[
        # a) King at a8 corner, rook delivers Ra8#
        KRKStagePosition(
            fen="k7/8/1K6/8/8/8/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King a8, Ra8# mate",
        ),
        # b) King at h1 corner, rook delivers Rh1#
        KRKStagePosition(
            fen="8/8/8/8/8/6K1/8/R6k w - - 0 1",
            optimal_moves=1,
            description="King h1, Rh1# or Ra1# mate",
        ),
        # c) King at a4 (edge), rook mates Ra1# 
        KRKStagePosition(
            fen="8/8/8/8/k7/2K5/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King a4 edge, Ra4# mate",
        ),
        # d) King at h6 (edge), rook mates Rh8#
        KRKStagePosition(
            fen="7R/8/6Kk/8/8/8/8/8 w - - 0 1",
            optimal_moves=1,
            description="King h6 edge, Rh8# mate",
        ),
        # e) King at e8 (back rank), rook mates Re1# or Ra8#
        KRKStagePosition(
            fen="4k3/8/4K3/8/8/8/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King e8, Ra8# mate",
        ),
    ]
)


# ============================================================================
# Stage 1: Mate_In_2 (5 Positions)
# Varied tempo requirements - approach + mate
# ============================================================================

STAGE_1_MATE_IN_2 = KRKStage(
    stage_id=1,
    name="Mate_In_2",
    description="Mate in 2 moves with varied patterns",
    distance_to_mate="2 moves",
    key_lesson="King-rook coordination",
    target_win_rate=0.90,
    positions=[
        # a) King approach needed, then mate
        KRKStagePosition(
            fen="k7/8/K7/8/8/8/8/7R w - - 0 1",
            optimal_moves=2,
            description="Approach Kb6, then Ra1#",
        ),
        # b) Rook reposition, then mate
        KRKStagePosition(
            fen="8/8/8/8/8/k7/8/R3K3 w - - 0 1",
            optimal_moves=2,
            description="Ra3+ forces king, then mate",
        ),
        # c) Check forcing king to corner, then mate
        KRKStagePosition(
            fen="8/8/8/8/8/1k6/8/R3K3 w - - 0 1",
            optimal_moves=2,
            description="Ra3+ Kb2, Ra2# or Kb1 Ra1#",
        ),
        # d) Waiting move required (opposition), then mate
        KRKStagePosition(
            fen="7k/5K2/8/8/8/8/8/7R w - - 0 1",
            optimal_moves=2,
            description="Rook waits Rh2, then Rh8#",
        ),
        # e) Cut maintained, approach, mate
        KRKStagePosition(
            fen="k7/R7/8/8/K7/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Kb5, Ra8#",
        ),
    ]
)


# ============================================================================
# Stage 2: Edge_Trapped_Tempo (4 Positions)
# Critical: Teach waiting moves for opposition
# ============================================================================

STAGE_2_EDGE_TRAPPED_TEMPO = KRKStage(
    stage_id=2,
    name="Edge_Trapped_Tempo",
    description="Opposition timing at edge",
    distance_to_mate="1-2 moves",
    key_lesson="Waiting moves / tempo",
    target_win_rate=0.85,
    positions=[
        # a) Our turn, direct approach wins
        KRKStagePosition(
            fen="1k6/8/1K6/8/8/8/8/R7 w - - 0 1",
            optimal_moves=2,
            description="Direct: Kc6 then Ra8#",
        ),
        # b) Our turn, waiting move needed first (rook not on a-file)
        KRKStagePosition(
            fen="1k6/8/1K6/8/8/8/8/7R w - - 0 1",
            optimal_moves=2,
            description="Rh8+ Kc7, then Ra8 or approach",
        ),
        # c) King on 8th rank, need tempo for opposition
        KRKStagePosition(
            fen="2k5/8/2K5/8/8/8/R7/8 w - - 0 1",
            optimal_moves=2,
            description="Ra8+ Kb7, then Rb8 or approach",
        ),
        # d) Near-corner, must avoid stalemate
        KRKStagePosition(
            fen="k7/2K5/8/8/8/8/8/R7 w - - 0 1",
            optimal_moves=2,
            description="AVOID stalemate! Kb6 then Ra8#",
            failure_condition="stalemate",
        ),
    ]
)


# ============================================================================
# Stage 3: Edge_Cut_Hold (4 Positions)
# 1x8 box scenarios - DON'T LET KING ESCAPE
# ============================================================================

STAGE_3_EDGE_CUT_HOLD = KRKStage(
    stage_id=3,
    name="Edge_Cut_Hold",
    description="Maintain cut while approaching (1x8 box)",
    distance_to_mate="2-3 moves",
    key_lesson="Don't let king escape! Smallest side must not grow.",
    target_win_rate=0.80,
    positions=[
        # a) King e8, Rook a7, our King a6 (user's example)
        KRKStagePosition(
            fen="4k3/R7/K7/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Maintain 7th rank cut, approach with king",
            failure_condition="box_grew",
        ),
        # b) King a3 (left edge), Rook b3, our King d5
        KRKStagePosition(
            fen="8/8/8/3K4/8/kR6/8/8 w - - 0 1",
            optimal_moves=3,
            description="Maintain b-file cut, push to corner",
            failure_condition="box_grew",
        ),
        # c) King h5 (right edge), Rook g5, our King e4
        KRKStagePosition(
            fen="8/8/8/6Rk/4K3/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Maintain g-file cut, approach",
            failure_condition="box_grew",
        ),
        # d) King d1 (bottom edge), Rook d2, our King f3
        KRKStagePosition(
            fen="8/8/8/8/8/5K2/3R4/3k4 w - - 0 1",
            optimal_moves=3,
            description="Maintain 2nd rank cut",
            failure_condition="box_grew",
        ),
    ]
)


# ============================================================================
# Stage 4: King_Close_1 (3 Positions)
# King 1 square away from Stage 3 ideal positions
# ============================================================================

STAGE_4_KING_CLOSE_1 = KRKStage(
    stage_id=4,
    name="King_Close_1",
    description="King 1 square from ideal position",
    distance_to_mate="3-4 moves",
    key_lesson="Approach while maintaining cut",
    target_win_rate=0.75,
    positions=[
        # a) King needs to move 1 square closer to e8 cut
        KRKStagePosition(
            fen="4k3/R7/8/K7/8/8/8/8 w - - 0 1",
            optimal_moves=4,
            description="Kb6 approaches, maintain cut",
            failure_condition="box_grew",
        ),
        # b) King 1 square from controlling corner
        KRKStagePosition(
            fen="8/8/8/4K3/8/kR6/8/8 w - - 0 1",
            optimal_moves=4,
            description="Kd4 approaches trapped king",
            failure_condition="box_grew",
        ),
        # c) King 1 step from opposition
        KRKStagePosition(
            fen="8/8/8/6Rk/5K2/8/8/8 w - - 0 1",
            optimal_moves=4,
            description="Kg4 or Ke4 approaches",
            failure_condition="box_grew",
        ),
    ]
)


# ============================================================================
# Stage 5: King_Close_2 (3 Positions)
# King 2 squares away from ideal
# ============================================================================

STAGE_5_KING_CLOSE_2 = KRKStage(
    stage_id=5,
    name="King_Close_2",
    description="King 2 squares from ideal position",
    distance_to_mate="4-6 moves",
    key_lesson="Multi-step approach planning",
    target_win_rate=0.70,
    positions=[
        # a) King 2 squares back from edge cut
        KRKStagePosition(
            fen="4k3/R7/8/8/K7/8/8/8 w - - 0 1",
            optimal_moves=5,
            description="Kb5 then Kb6, maintain cut",
            failure_condition="box_grew",
        ),
        # b) King 2 squares from corner control
        KRKStagePosition(
            fen="8/8/8/8/5K2/kR6/8/8 w - - 0 1",
            optimal_moves=5,
            description="Ke3 then Kd3 approach",
            failure_condition="box_grew",
        ),
        # c) King 2 steps from mating net
        KRKStagePosition(
            fen="8/8/8/6Rk/8/4K3/8/8 w - - 0 1",
            optimal_moves=5,
            description="Kf4 then Kg4 approach",
            failure_condition="box_grew",
        ),
    ]
)


# ============================================================================
# Stage 6: King_Far_Cut_Held (3 Positions)
# Rook guarding, king must make long march
# ============================================================================

STAGE_6_KING_FAR_CUT_HELD = KRKStage(
    stage_id=6,
    name="King_Far_Cut_Held",
    description="King far away, rook maintains cut",
    distance_to_mate="6-10 moves",
    key_lesson="Long march while rook holds",
    target_win_rate=0.65,
    positions=[
        # a) King at opposite corner
        KRKStagePosition(
            fen="4k3/R7/8/8/8/8/8/K7 w - - 0 1",
            optimal_moves=8,
            description="King must march from a1 to help",
            failure_condition="box_grew",
        ),
        # b) King in center, enemy at edge
        KRKStagePosition(
            fen="8/8/8/8/3K4/kR6/8/8 w - - 0 1",
            optimal_moves=6,
            description="King approaches from center",
            failure_condition="box_grew",
        ),
        # c) Long diagonal march
        KRKStagePosition(
            fen="8/8/8/6Rk/8/8/8/K7 w - - 0 1",
            optimal_moves=9,
            description="King must cross board diagonally",
            failure_condition="box_grew",
        ),
    ]
)


# ============================================================================
# Stage 7: Box_Small (3 Positions)
# 3x3 box, systematic shrinking
# ============================================================================

STAGE_7_BOX_SMALL = KRKStage(
    stage_id=7,
    name="Box_Small",
    description="3x3 confinement box shrinking",
    distance_to_mate="5-8 moves",
    key_lesson="Systematic box reduction",
    target_win_rate=0.60,
    positions=[
        # a) 3x3 box in corner
        KRKStagePosition(
            fen="8/8/8/8/8/5K2/5R2/6k1 w - - 0 1",
            optimal_moves=6,
            description="Shrink box to corner",
        ),
        # b) 3x3 box on edge
        KRKStagePosition(
            fen="8/8/8/8/R7/8/2k1K3/8 w - - 0 1",
            optimal_moves=7,
            description="Shrink box on bottom edge",
        ),
        # c) 3x3 box top edge
        KRKStagePosition(
            fen="2k5/8/2K1R3/8/8/8/8/8 w - - 0 1",
            optimal_moves=6,
            description="Shrink to corner",
        ),
    ]
)


# ============================================================================
# Stage 8: Box_Medium (3 Positions)
# 4x4 box with opposition needed
# ============================================================================

STAGE_8_BOX_MEDIUM = KRKStage(
    stage_id=8,
    name="Box_Medium",
    description="4x4 box requiring opposition",
    distance_to_mate="8-15 moves",
    key_lesson="Opposition + systematic shrinking",
    target_win_rate=0.55,
    positions=[
        # a) 4x4 box, opposition matters
        KRKStagePosition(
            fen="8/8/8/8/4K3/R7/4k3/8 w - - 0 1",
            optimal_moves=10,
            description="Opposition key for progress",
        ),
        # b) 4x4 box, tempo play
        KRKStagePosition(
            fen="8/8/8/8/3k4/8/3K4/R7 w - - 0 1",
            optimal_moves=12,
            description="Tempo needed for box shrink",
        ),
        # c) Larger initial box
        KRKStagePosition(
            fen="8/8/8/3k4/8/R7/8/3K4 w - - 0 1",
            optimal_moves=14,
            description="Full Box Method execution",
        ),
    ]
)


# ============================================================================
# Stage 9: Full_KRK (Random)
# Complete random winning positions
# ============================================================================

STAGE_9_FULL_KRK = KRKStage(
    stage_id=9,
    name="Full_KRK",
    description="Random winning KRK positions",
    distance_to_mate="15-30 moves",
    key_lesson="Complete Box Method",
    target_win_rate=0.50,
    max_moves_multiplier=2.5,  # Allow more flexibility
    positions=[
        # These serve as fallbacks; random generation is preferred
        KRKStagePosition(
            fen="8/8/8/4k3/8/8/8/R3K3 w - - 0 1",
            optimal_moves=16,
            description="Standard KRK position",
        ),
        KRKStagePosition(
            fen="8/8/3k4/8/8/8/8/R6K w - - 0 1",
            optimal_moves=18,
            description="King in center",
        ),
        KRKStagePosition(
            fen="4k3/8/8/8/8/8/8/R2K4 w - - 0 1",
            optimal_moves=15,
            description="Back rank start",
        ),
    ]
)


# ============================================================================
# All Stages List
# ============================================================================

KRK_STAGES: List[KRKStage] = [
    STAGE_0_MATE_IN_1,
    STAGE_1_MATE_IN_2,
    STAGE_2_EDGE_TRAPPED_TEMPO,
    STAGE_3_EDGE_CUT_HOLD,
    STAGE_4_KING_CLOSE_1,
    STAGE_5_KING_CLOSE_2,
    STAGE_6_KING_FAR_CUT_HELD,
    STAGE_7_BOX_SMALL,
    STAGE_8_BOX_MEDIUM,
    STAGE_9_FULL_KRK,
]


def get_stage(stage_id: int) -> KRKStage:
    """Get a stage by ID."""
    if 0 <= stage_id < len(KRK_STAGES):
        return KRK_STAGES[stage_id]
    return KRK_STAGES[-1]  # Default to Full_KRK


def generate_krk_curriculum_position(stage_id: int) -> chess.Board:
    """Generate a position for a specific curriculum stage."""
    stage = get_stage(stage_id)
    return stage.select_position()


# ============================================================================
# Box Tracking Utilities
# ============================================================================

def compute_confinement_box(board: chess.Board) -> Tuple[int, int]:
    """
    Compute the confinement box dimensions for the enemy king.
    
    Returns:
        (width, height) of the box containing the enemy king
    """
    # Find pieces
    rook_sq = None
    enemy_king_sq = None
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            if piece.piece_type == chess.ROOK and piece.color == chess.WHITE:
                rook_sq = sq
            elif piece.piece_type == chess.KING and piece.color == chess.BLACK:
                enemy_king_sq = sq
    
    if rook_sq is None or enemy_king_sq is None:
        return (8, 8)  # No confinement
    
    rook_file = chess.square_file(rook_sq)
    rook_rank = chess.square_rank(rook_sq)
    king_file = chess.square_file(enemy_king_sq)
    king_rank = chess.square_rank(enemy_king_sq)
    
    # Determine which side of the rook the king is on
    # and compute box dimensions
    
    # Horizontal cut (rook on same rank as king or between king and edge)
    if rook_file < king_file:
        # King is right of rook - confined to right side
        width = 7 - rook_file
    elif rook_file > king_file:
        # King is left of rook - confined to left side
        width = rook_file
    else:
        width = 8  # Rook same file - no horizontal cut
    
    # Vertical cut (rook on same file as king or between king and edge)
    if rook_rank < king_rank:
        # King is above rook - confined to top
        height = 7 - rook_rank
    elif rook_rank > king_rank:
        # King is below rook - confined to bottom
        height = rook_rank
    else:
        height = 8  # Rook same rank - no vertical cut
    
    return (width, height)


def box_min_side(board: chess.Board) -> int:
    """Get the minimum dimension of the confinement box."""
    width, height = compute_confinement_box(board)
    return min(width, height)


def did_box_grow(board_before: chess.Board, board_after: chess.Board) -> bool:
    """
    Check if the confinement box grew (king escaped).
    
    This is a FAILURE condition - the smallest side should not increase.
    """
    min_before = box_min_side(board_before)
    min_after = box_min_side(board_after)
    return min_after > min_before


# ============================================================================
# KRK Reward Function
# ============================================================================

def krk_reward(
    won: bool,
    move_count: int,
    optimal_moves: int,
    box_grew: bool = False,
    stalemate: bool = False,
) -> float:
    """
    Compute KRK curriculum reward with hard move penalty.
    
    Args:
        won: True if game was won
        move_count: Number of moves taken
        optimal_moves: Theoretical optimal moves for position
        box_grew: True if confinement box grew (king escaped)
        stalemate: True if game ended in stalemate
    
    Returns:
        Reward value in range [-0.5, 1.0]
    """
    # Stalemate is very bad in KRK
    if stalemate:
        return -0.5
    
    # Loss
    if not won:
        return 0.0
    
    # Win! Calculate reward with move penalty
    excess = max(0, move_count - optimal_moves)
    base = max(0.1, 1.0 - 0.02 * excess)  # Floor at 0.1
    
    # Heavy penalty for letting king escape
    if box_grew:
        base *= 0.5
    
    return base


def compute_krk_game_reward(
    board_before: chess.Board,
    board_after: chess.Board,
    move_count: int,
    stage: KRKStage,
) -> float:
    """
    Compute reward for a KRK game given the curriculum stage.
    
    Args:
        board_before: Position before last move
        board_after: Final position
        move_count: Total moves in game
        stage: Current curriculum stage
    
    Returns:
        Reward value
    """
    # Check outcomes
    won = board_after.is_checkmate()  # We're attacking
    stalemate = board_after.is_stalemate()
    box_grew = did_box_grow(board_before, board_after)
    
    optimal = stage.get_optimal_moves(board_before)
    
    return krk_reward(
        won=won,
        move_count=move_count,
        optimal_moves=optimal,
        box_grew=box_grew,
        stalemate=stalemate,
    )


# ============================================================================
# Stage Progression Logic
# ============================================================================

@dataclass
class KRKStageStats:
    """Statistics for a KRK curriculum stage."""
    stage_id: int
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    stalemates: int = 0
    total_moves: int = 0
    box_escapes: int = 0  # Times king escaped
    
    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games
    
    @property
    def avg_moves(self) -> float:
        if self.wins == 0:
            return 0.0
        return self.total_moves / self.wins
    
    @property
    def escape_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.box_escapes / self.total_games


class KRKCurriculumManager:
    """
    Manages progression through the KRK curriculum.
    
    Tracks stats per stage and handles stage advancement.
    """
    
    def __init__(
        self,
        min_games_per_stage: int = 50,
        win_rate_threshold: float = 0.8,
    ):
        self.current_stage_id = 0
        self.min_games_per_stage = min_games_per_stage
        self.win_rate_threshold = win_rate_threshold
        self.stage_stats: Dict[int, KRKStageStats] = {
            i: KRKStageStats(stage_id=i) for i in range(len(KRK_STAGES))
        }
    
    @property
    def current_stage(self) -> KRKStage:
        return get_stage(self.current_stage_id)
    
    @property
    def current_stats(self) -> KRKStageStats:
        return self.stage_stats[self.current_stage_id]
    
    def get_position(self) -> chess.Board:
        """Get a training position for current stage."""
        return self.current_stage.select_position()
    
    def record_game(
        self,
        won: bool,
        move_count: int,
        stalemate: bool = False,
        box_escaped: bool = False,
    ) -> bool:
        """
        Record game result and check for stage advancement.
        
        Returns:
            True if stage advanced, False otherwise
        """
        stats = self.current_stats
        stats.total_games += 1
        
        if won:
            stats.wins += 1
            stats.total_moves += move_count
        elif stalemate:
            stats.stalemates += 1
        else:
            stats.losses += 1
        
        if box_escaped:
            stats.box_escapes += 1
        
        # Check for advancement
        return self._check_advancement()
    
    def _check_advancement(self) -> bool:
        """Check if current stage should advance."""
        stats = self.current_stats
        stage = self.current_stage
        
        # Need minimum games
        if stats.total_games < self.min_games_per_stage:
            return False
        
        # Check win rate threshold (use stage-specific if available)
        threshold = stage.target_win_rate
        if stats.win_rate >= threshold:
            return self._advance_stage()
        
        return False
    
    def _advance_stage(self) -> bool:
        """Advance to next stage if possible."""
        if self.current_stage_id >= len(KRK_STAGES) - 1:
            return False  # Already at final stage
        
        self.current_stage_id += 1
        return True
    
    def force_advance(self) -> bool:
        """Force advancement (for testing)."""
        return self._advance_stage()
    
    def reset_stage(self, stage_id: int = 0):
        """Reset to a specific stage."""
        self.current_stage_id = max(0, min(stage_id, len(KRK_STAGES) - 1))
        self.stage_stats[self.current_stage_id] = KRKStageStats(stage_id=self.current_stage_id)
    
    def get_summary(self) -> Dict:
        """Get curriculum summary."""
        return {
            "current_stage": self.current_stage_id,
            "stage_name": self.current_stage.name,
            "total_stages": len(KRK_STAGES),
            "stages": {
                i: {
                    "name": KRK_STAGES[i].name,
                    "games": self.stage_stats[i].total_games,
                    "win_rate": round(self.stage_stats[i].win_rate, 3),
                    "avg_moves": round(self.stage_stats[i].avg_moves, 1),
                    "escape_rate": round(self.stage_stats[i].escape_rate, 3),
                }
                for i in range(len(KRK_STAGES))
                if self.stage_stats[i].total_games > 0
            }
        }

