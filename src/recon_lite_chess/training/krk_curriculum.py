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
    target_win_rate=0.98,  # Raised from 0.95 to ensure mastery
    positions=[
        # a) King at a8 corner, rook delivers Rh8#
        KRKStagePosition(
            fen="k7/8/1K6/8/8/8/8/7R w - - 0 1",
            optimal_moves=1,
            description="King a8, Rh8# mate",
        ),
        # b) King at h8 corner, rook delivers Rh1#
        KRKStagePosition(
            fen="7k/5K2/8/8/8/8/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King h8, Rh1# mate",
        ),
        # c) King at h4 (side edge), rook mates Rh1#
        KRKStagePosition(
            fen="8/8/8/8/5K1k/8/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King h4 side, Rh1# mate",
        ),
        # d) King at h8 (corner), rook mates Ra8#
        KRKStagePosition(
            fen="7k/8/6K1/8/8/8/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King h8 corner, Ra8# mate",
        ),
        # e) King at e8 (back rank), rook mates Ra8#
        KRKStagePosition(
            fen="4k3/8/4K3/8/8/8/8/R7 w - - 0 1",
            optimal_moves=1,
            description="King e8, Ra8# mate",
        ),
    ]
)


# ============================================================================
# Stage 1: Mate_In_2 (5 Positions) - FIXED to require true 2-move solutions
# Varied tempo requirements - approach + mate
# ============================================================================

STAGE_1_MATE_IN_2 = KRKStage(
    stage_id=1,
    name="Mate_In_2",
    description="Mate in 2 moves with varied patterns - no shortcuts",
    distance_to_mate="2 moves",
    key_lesson="King-rook coordination",
    target_win_rate=0.95,  # Raised from 0.90 to ensure mastery
    positions=[
        # a) King approach needed, then mate (FIXED: king on b6 blocks Rh8#)
        # Old was: k7/8/K7 - Rh8# was instant mate!
        # Now: White king on c5, must approach to b6 then Ra1#
        KRKStagePosition(
            fen="k7/8/8/2K5/8/8/8/7R w - - 0 1",
            optimal_moves=2,
            description="1.Kb6 (approach) 2.Ra1#",
        ),
        # b) Rook check forces king to edge, then mate (TRUE MATE-IN-2)
        # 1.Rh8+ Ka7 2.Ra8# (rook cuts then delivers)
        KRKStagePosition(
            fen="8/k7/1K6/8/8/8/8/7R w - - 0 1",
            optimal_moves=2,
            description="1.Rh8+ Ka7 2.Ra8# (check forces edge, then mate)",
        ),
        # c) King approach gives opposition, then rook mates (TRUE MATE-IN-2)
        # 1.Kb6 (opposition) 2.Ra8# regardless of black's reply
        KRKStagePosition(
            fen="k7/8/1K6/8/8/8/8/R7 w - - 0 1",
            optimal_moves=2,
            description="1.Kb6 (any) 2.Ra8# (approach + mate)",
        ),
        # d) Tempo/waiting move required (TRUE MATE-IN-2)
        # 1.Kg6 (opposition) 2.Rh1# (king has no escape)
        KRKStagePosition(
            fen="7k/8/5K2/8/8/8/8/7R w - - 0 1",
            optimal_moves=2,
            description="1.Kg6 (opposition) 2.Rh1#",
        ),
        # e) Rook tempo then mate (TRUE MATE-IN-2)
        # 1.Ra6+ Kb8 2.Ra8# (rook cuts file, delivers mate)
        KRKStagePosition(
            fen="k7/8/8/8/8/K7/8/R7 w - - 0 1",
            optimal_moves=2,
            description="1.Ra6+ Kb8 2.Ra8# (rook drives + mates)",
        ),
    ]
)


# ============================================================================
# Stage 2 (Planned): Edge‑Trap Conversion (split into 2A/2B/2C)
# NOTE: Planned stages below are scaffolded and not yet wired into CURRICULUM_STAGES.
# Add validated FENs (5–10 per category, White to move) before activating.
# ============================================================================

STAGE_2A_EDGE_TRAP_CLOSE = KRKStage(
    stage_id=2,
    name="Edge_Trap_Close",
    description="Enemy king trapped at edge; our king close and between enemy and rook",
    distance_to_mate="1-2 moves",
    key_lesson="Convert edge‑trap to mate‑in‑2 with correct geometry",
    target_win_rate=0.95,
    positions=[
        # 2A‑1: W:Kc4 Rb8 | B:Ka2
        KRKStagePosition(
            fen="1R6/8/8/8/2K5/8/k7/8 w - - 0 1",
            optimal_moves=2,
            description="Edge‑trap close; king inside, no immediate mate.",
        ),
        # 2A‑2: W:Kf3 Rf2 | B:Kd1
        KRKStagePosition(
            fen="8/8/8/8/8/5K2/5R2/3k4 w - - 0 1",
            optimal_moves=2,
            description="Bottom edge trap with rook fence; approach conversion.",
        ),
        # 2A‑3: W:Kf3 Rg2 | B:Kd1
        KRKStagePosition(
            fen="8/8/8/8/8/5K2/6R1/3k4 w - - 0 1",
            optimal_moves=2,
            description="Bottom edge trap; rook on g2, king close.",
        ),
        # 2A‑4: W:Kc3 Rb2 | B:Kc1
        KRKStagePosition(
            fen="8/8/8/8/8/2K5/1R6/2k5 w - - 0 1",
            optimal_moves=2,
            description="Corner‑adjacent trap; king inside, rook fence.",
        ),
        # 2A‑5: W:Ke3 Re2 | B:Kc1
        KRKStagePosition(
            fen="8/8/8/8/8/4K3/4R3/2k5 w - - 0 1",
            optimal_moves=2,
            description="Edge trap with rook fence on 2nd rank.",
        ),
        # 2A‑6: W:Kc5 Rb4 | B:Ka7
        KRKStagePosition(
            fen="8/k7/8/2K5/1R6/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Edge trap (top edge), king inside, rook fence.",
        ),
        # 2A‑7: W:Kf5 Rg2 | B:Kh7
        KRKStagePosition(
            fen="8/7k/8/5K2/8/8/6R1/8 w - - 0 1",
            optimal_moves=2,
            description="Side edge trap; king inside, rook fence on g2.",
        ),
        # 2A‑8: W:Kc8 Rb5 | B:Ka6
        KRKStagePosition(
            fen="2K5/8/k7/1R6/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Edge trap with king near corner; rook fence on b5.",
        ),
        # 2A‑9: W:Kd6 Rc7 | B:Kb8
        KRKStagePosition(
            fen="1k6/2R5/3K4/8/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Classic edge trap; king close, rook on 7th.",
        ),
        # 2A‑10: W:Kd3 Rf2 | B:Kb1
        KRKStagePosition(
            fen="8/8/8/8/8/3K4/5R2/1k6 w - - 0 1",
            optimal_moves=2,
            description="Bottom edge trap; king inside, rook fence on f2.",
        ),
    ],
)

STAGE_2B_EDGE_TRAP_ENEMY_BETWEEN = KRKStage(
    stage_id=3,
    name="Edge_Trap_Enemy_Between",
    description="Enemy king trapped at edge; enemy between our king and rook",
    distance_to_mate="1-3 moves",
    key_lesson="Tempo/waiting moves to avoid rook capture",
    target_win_rate=0.90,
    positions=[
        # 2B‑1: W:Kf1 Rb2 | B:Kd1
        KRKStagePosition(
            fen="8/8/8/8/8/8/1R6/3k1K2 w - - 0 1",
            optimal_moves=3,
            description="Enemy between king/rook; tempo needed.",
        ),
        # 2B‑2: W:Kf8 Rg4 | B:Kh7
        KRKStagePosition(
            fen="5K2/7k/8/8/6R1/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Side edge trap; enemy between, rook fence on g4.",
        ),
        # 2B‑3: W:Kf6 Ra7 | B:Ke8
        KRKStagePosition(
            fen="4k3/R7/5K2/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Top edge trap; enemy between king and rook.",
        ),
        # 2B‑4: W:Kg5 Rg2 | B:Kh3
        KRKStagePosition(
            fen="8/8/8/6K1/8/7k/6R1/8 w - - 0 1",
            optimal_moves=3,
            description="Side edge trap; waiting move likely.",
        ),
        # 2B‑5: W:Kf6 Rg4 | B:Kh5
        KRKStagePosition(
            fen="8/8/5K2/7k/6R1/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Enemy between; avoid rook capture.",
        ),
        # 2B‑6: W:Kb2 Rf2 | B:Kd1
        KRKStagePosition(
            fen="8/8/8/8/8/8/1K3R2/3k4 w - - 0 1",
            optimal_moves=3,
            description="Bottom edge trap; enemy between.",
        ),
        # 2B‑7: W:Kc1 Rb5 | B:Ka2
        KRKStagePosition(
            fen="8/8/8/1R6/8/8/k7/2K5 w - - 0 1",
            optimal_moves=3,
            description="Left edge trap; enemy between king and rook.",
        ),
        # 2B‑8: W:Kc1 Rb3 | B:Ka2
        KRKStagePosition(
            fen="8/8/8/8/8/1R6/k7/2K5 w - - 0 1",
            optimal_moves=3,
            description="Left edge trap; waiting move likely.",
        ),
        # 2B‑9: W:Kd7 Rh7 | B:Kf8
        KRKStagePosition(
            fen="5k2/3K3R/8/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Top edge trap; enemy between.",
        ),
        # 2B‑10: W:Ka8 Re7 | B:Kc8
        KRKStagePosition(
            fen="K1k5/4R3/8/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Top edge trap; enemy between (rook on 7th).",
        ),
    ],
)

STAGE_2C_EDGE_TRAP_WRONG_TEMPO = KRKStage(
    stage_id=4,
    name="Edge_Trap_Wrong_Tempo",
    description="Edge‑trap with wrong tempo at knight distance",
    distance_to_mate="2-3 moves",
    key_lesson="Fix tempo before conversion",
    target_win_rate=0.90,
    positions=[
        # 2C‑1: W:Kf7 Rg7 | B:Kh8
        KRKStagePosition(
            fen="7k/5KR1/8/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Knight distance; wrong tempo at edge.",
        ),
        # 2C‑2: W:Ke3 Rd2 | B:Kf1
        KRKStagePosition(
            fen="8/8/8/8/8/4K3/3R4/5k2 w - - 0 1",
            optimal_moves=3,
            description="Bottom edge; tempo fix required.",
        ),
        # 2C‑3: W:Kb8 Rb7 | B:Ka6
        KRKStagePosition(
            fen="1K6/1R6/k7/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Corner edge; knight distance mismatch.",
        ),
        # 2C‑4: W:Kc7 Rb8 | B:Ka6
        KRKStagePosition(
            fen="1R6/2K5/k7/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Corner edge; wrong tempo at knight distance.",
        ),
        # 2C‑5: W:Kf6 Re7 | B:Kg8
        KRKStagePosition(
            fen="6k1/4R3/5K2/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Side edge; fix tempo before conversion.",
        ),
        # 2C‑6: W:Kb7 Rb8 | B:Ka5
        KRKStagePosition(
            fen="1R6/1K6/8/k7/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Edge trap with wrong tempo (left edge).",
        ),
        # 2C‑7: W:Kd2 Ra2 | B:Kf1
        KRKStagePosition(
            fen="8/8/8/8/8/8/R2K4/5k2 w - - 0 1",
            optimal_moves=3,
            description="Bottom edge; knight distance mismatch.",
        ),
        # 2C‑8: W:Kc7 Re7 | B:Ka8
        KRKStagePosition(
            fen="k7/2K1R3/8/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Top edge; wrong tempo at knight distance.",
        ),
        # 2C‑9: W:Kf6 Rg1 | B:Kh7
        KRKStagePosition(
            fen="8/7k/5K2/8/8/8/8/6R1 w - - 0 1",
            optimal_moves=3,
            description="Side edge; rook on g1, tempo fix required.",
        ),
        # 2C‑10: W:Kc5 Rb3 | B:Ka6
        KRKStagePosition(
            fen="8/8/k7/2K5/8/1R6/8/8 w - - 0 1",
            optimal_moves=3,
            description="Edge trap; knight distance mismatch.",
        ),
    ],
)

# Legacy Stage 2 (kept for backward compatibility until new Stage‑2A/B/C are populated)
STAGE_2_EDGE_TRAPPED_TEMPO = KRKStage(
    stage_id=2,
    name="Edge_Trapped_Tempo",
    description="Opposition timing at edge (legacy)",
    distance_to_mate="1-2 moves",
    key_lesson="Waiting moves / tempo",
    target_win_rate=0.95,
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
    ],
)


# ============================================================================
# Stage 2.1: Edge_Fence_Knight (Legacy)
# Opponent king on edge (rank 8), rook holding fence (rank 7),
# our king at knight distance on rank 6 - optimal finishing position
# ============================================================================

STAGE_2_1_EDGE_FENCE_KNIGHT = KRKStage(
    stage_id=3,  # Position after stages 0, 1, 2
    name="Edge_Fence_Knight",
    description="King at knight distance from trapped enemy - optimal finish",
    distance_to_mate="1-2 moves",
    key_lesson="Knight distance = checkmate threat with no escape",
    target_win_rate=0.95,  # High threshold - these are optimal positions
    positions=[
        # a) King d6 (knight distance from e8), Rook e7, enemy e8
        KRKStagePosition(
            fen="4k3/4R3/3K4/8/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Knight distance: Ke6 then Re8# or immediate threat",
        ),
        # b) King f6 (knight distance from e8), Rook e7, enemy e8
        KRKStagePosition(
            fen="4k3/4R3/5K2/8/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Knight distance right: Ke6 then Re8#",
        ),
        # c) King c6 (knight distance from b8), Rook b7, enemy b8
        KRKStagePosition(
            fen="1k6/1R6/2K5/8/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Knight distance corner: Kb6 then Ra8#",
        ),
        # d) King g3 (knight distance from h1), Rook h2, enemy h1 (bottom edge)
        KRKStagePosition(
            fen="8/8/8/8/8/6K1/7R/7k w - - 0 1",
            optimal_moves=2,
            description="Knight distance bottom: Kf2 then Rh1# or threats",
        ),
        # e) King b3 (knight distance from a1), Rook a2, enemy a1
        KRKStagePosition(
            fen="8/8/8/8/8/1K6/R7/k7 w - - 0 1",
            optimal_moves=2,
            description="Knight distance corner: Ka2 or Kc2 then Ra1#",
        ),
    ],
)


# ============================================================================
# Stage 2.2: Edge_Fence_Approach (Legacy)
# Same setup but our king NOT at knight distance - must reposition
# Harder because requires finding approach path
# ============================================================================

STAGE_2_2_EDGE_FENCE_APPROACH = KRKStage(
    stage_id=4,  # Position after Edge_Fence_Knight
    name="Edge_Fence_Approach",
    description="King must reposition to achieve knight distance",
    distance_to_mate="2-3 moves",
    key_lesson="Approach diagonally to reach knight distance",
    target_win_rate=0.90,
    positions=[
        # a) King a6 (not knight distance from e8), Rook e7, enemy e8
        KRKStagePosition(
            fen="4k3/4R3/K7/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Approach: Kb6-c6 or diagonal to reach knight distance",
        ),
        # b) King h6 (not knight distance from e8), Rook e7, enemy e8
        KRKStagePosition(
            fen="4k3/4R3/7K/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Approach: Kg6-f6 to reach knight distance",
        ),
        # c) King a6 (not knight distance from b8), Rook b7, enemy b8
        KRKStagePosition(
            fen="1k6/1R6/K7/8/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Approach corner: Kb6 directly achieves knight distance",
        ),
        # d) King e3 (not knight distance from h1), Rook h2, enemy h1
        KRKStagePosition(
            fen="8/8/8/8/8/4K3/7R/7k w - - 0 1",
            optimal_moves=3,
            description="Approach: Kf2-g2 to reach knight distance",
        ),
        # e) King d3 (not knight distance from a1), Rook a2, enemy a1
        KRKStagePosition(
            fen="8/8/8/8/8/3K4/R7/k7 w - - 0 1",
            optimal_moves=3,
            description="Approach corner: Kc2-b2 path or Kc3-b3",
        ),
    ],
)


# ============================================================================
# Stage 2.3: Edge_Fence_Deep (NEW - 2-Space Depth Drill)
# Opponent at edge (rank 8), rook at rank 6 (2 space gap), king supporting
# Drills tightening loose setups without interfering with box shrink
# ============================================================================

STAGE_2_3_EDGE_FENCE_DEEP = KRKStage(
    stage_id=5,  # Position after Edge_Fence_Approach
    name="Edge_Fence_Deep",
    description="Rook 2 ranks from edge - tighten loose confinement",
    distance_to_mate="3-4 moves",
    key_lesson="Shrink 2-space gap to 1-space by rook advancement",
    target_win_rate=0.85,
    positions=[
        # a) King d5, Rook e6 (2 ranks from e8), enemy e8
        KRKStagePosition(
            fen="4k3/8/4R3/3K4/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Advance rook: Re7 tightens, then approach",
            failure_condition="box_grew",
        ),
        # b) King c5, Rook b6 (2 ranks from b8), enemy b8
        KRKStagePosition(
            fen="1k6/8/1R6/2K5/8/8/8/8 w - - 0 1",
            optimal_moves=3,
            description="Advance rook: Rb7 then Kb6 or direct",
            failure_condition="box_grew",
        ),
        # c) King g4, Rook h3 (2 ranks from h1), enemy h1
        KRKStagePosition(
            fen="8/8/8/8/6K1/7R/8/7k w - - 0 1",
            optimal_moves=3,
            description="Advance rook: Rh2 tightens bottom",
            failure_condition="box_grew",
        ),
        # d) King b4, Rook a3 (2 files from a1), enemy a1
        KRKStagePosition(
            fen="8/8/8/8/1K6/R7/8/k7 w - - 0 1",
            optimal_moves=3,
            description="Advance rook: Ra2 then Ka2-b2",
            failure_condition="box_grew",
        ),
        # e) Loose setup: King f4, Rook d6, enemy d8
        KRKStagePosition(
            fen="3k4/8/3R4/8/5K2/8/8/8 w - - 0 1",
            optimal_moves=4,
            description="Tighten: Rd7 cuts, then approach",
            failure_condition="box_grew",
        ),
    ],
)
# ============================================================================
# Stage 2.5 (Bridge): Anchored_Cut (5 Positions)
# King already adjacent to rook - removes coordination difficulty
# Focus purely on the "Hold" mechanic
# ============================================================================

STAGE_2_5_ANCHORED_CUT = KRKStage(
    stage_id=3,  # Position in final list (after stages 0, 1, 2)
    name="Anchored_Cut",
    description="King already anchored next to rook - maintain cut",
    distance_to_mate="2 moves",
    key_lesson="Hold the cut with pre-positioned king (coordination already done)",
    target_win_rate=0.90,
    positions=[
        # a) King adjacent to rook, enemy at edge
        KRKStagePosition(
            fen="4k3/1R6/1K6/8/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Anchored: Kc6 then Rb8#",
        ),
        # b) King on b3, Rook on a3 (adjacent) - enemy on a1
        KRKStagePosition(
            fen="8/8/8/8/8/RK6/8/k7 w - - 0 1",
            optimal_moves=2,
            description="Anchored corner: Ka2 then Ra3#",
        ),
        # c) King on g5, Rook on h5 (adjacent) - enemy on h7
        KRKStagePosition(
            fen="8/7k/8/6KR/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Anchored edge: Kh6 then Rh1-h8#",
        ),
        # d) King on f7, Rook on e7 (adjacent) - enemy on h8
        KRKStagePosition(
            fen="7k/4RK2/8/8/8/8/8/8 w - - 0 1",
            optimal_moves=1,
            description="Anchored back rank: Re8#",
        ),
        # e) King on c6, Rook on c7 (adjacent) - enemy on a8
        KRKStagePosition(
            fen="k7/2R5/2K5/8/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Anchored: Kb6 then Ra7#",
        ),
    ],
)


# ============================================================================
# Stage 3: Edge_Cut_Hold (4 Positions)
# 1x8 box scenarios - DON'T LET KING ESCAPE
# ============================================================================

STAGE_3_EDGE_CUT_HOLD = KRKStage(
    stage_id=4,  # Shifted after bridge stage
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
    stage_id=5,  # Shifted after bridge stage
    name="King_Close_1",
    description="King 1 square from ideal position",
    distance_to_mate="3-4 moves",
    key_lesson="Approach while maintaining cut",
    target_win_rate=0.90,
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
    stage_id=6,  # Shifted after bridge stage
    name="King_Close_2",
    description="King 2 squares from ideal position",
    distance_to_mate="4-6 moves",
    key_lesson="Multi-step approach planning",
    target_win_rate=0.85,
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
    stage_id=7,  # Shifted after bridge stage
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
    stage_id=8,  # Shifted after bridge stage
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
    stage_id=9,  # Shifted after bridge stage
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
    stage_id=10,  # Shifted after bridge stage
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
# DRIVE METHOD STAGES (New - based on fence/tempo/opposition technique)
# Alternative to Box Method - drive king to edge using fence and opposition
# ============================================================================

# Drive Stage D1: Fence Established - rook cuts board in half
STAGE_D1_FENCE_ESTABLISHED = KRKStage(
    stage_id=20,
    name="Fence_Established",
    description="Rook fence divides board - enemy king confined",
    distance_to_mate="2-3 moves",
    key_lesson="Recognize and maintain the fence (rook cut)",
    target_win_rate=0.60,
    positions=[
        # EASY: King almost at edge, fence holds for quick mate
        KRKStagePosition(
            fen="k7/8/1K6/R7/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Ra8# or approach then Ra8#",
        ),
        # EASY: King at edge, fence holds
        KRKStagePosition(
            fen="7k/8/5K2/7R/8/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Kg7 then Rh8# or Rh8+ first",
        ),
        # MEDIUM: Rook on 4th rank, king confined to upper half
        KRKStagePosition(
            fen="4k3/8/8/8/R7/8/4K3/8 w - - 0 1",
            optimal_moves=4,
            description="Fence on 4th rank, drive king to 8th",
        ),
        # MEDIUM: King already near edge, fence holds
        KRKStagePosition(
            fen="7k/8/8/8/R7/8/5K2/8 w - - 0 1",
            optimal_moves=3,
            description="Enemy at edge, approach to finish",
        ),
    ]
)

# Drive Stage D2: Opposition Approach - knight distance between kings
STAGE_D2_OPPOSITION_APPROACH = KRKStage(
    stage_id=21,
    name="Opposition_Approach",
    description="Knight distance between kings, our king on 3rd row",
    distance_to_mate="2-3 moves",
    key_lesson="Approach with knight distance (L-shape) for opposition",
    target_win_rate=0.65,
    positions=[
        # Knight distance: king can approach in L-shape
        # Our king on 3rd row from enemy, between enemy and rook
        KRKStagePosition(
            fen="7k/8/5K2/8/8/8/8/R7 w - - 0 1",
            optimal_moves=3,
            description="Kg7 takes opposition, then Rh1#",
        ),
        # Enemy must go towards rook or into opposition
        KRKStagePosition(
            fen="k7/8/2K5/8/8/8/8/R7 w - - 0 1",
            optimal_moves=2,
            description="Kb6 forces Ka8, then Ra1#",
        ),
        # Wider spacing but same pattern
        KRKStagePosition(
            fen="4k3/8/3K4/8/8/8/8/R7 w - - 0 1",
            optimal_moves=3,
            description="Approach maintaining knight distance",
        ),
    ]
)

# Drive Stage D3: Tempo Wait - rook slides along fence
STAGE_D3_TEMPO_WAIT = KRKStage(
    stage_id=22,
    name="Tempo_Wait",
    description="Rook slides along fence for tempo when king 'stuck'",
    distance_to_mate="2-3 moves",
    key_lesson="Wait move with rook to gain opposition tempo",
    target_win_rate=0.60,
    positions=[
        # King has opposition but wrong king to move
        # Rook must wait to transfer tempo
        KRKStagePosition(
            fen="k7/8/1K6/8/8/8/8/R7 w - - 0 1",
            optimal_moves=2,
            description="Ra2 waits, then Ka6 Ra8#",
        ),
        # Tempo move on back rank
        KRKStagePosition(
            fen="1k6/8/2K5/8/8/8/8/R7 w - - 0 1",
            optimal_moves=2,
            description="Rb1 tempo, then approach",
        ),
        # Fence wait
        KRKStagePosition(
            fen="8/k7/8/1K6/R7/8/8/8 w - - 0 1",
            optimal_moves=2,
            description="Ra7+ forces Ka8, then Ra1#",
        ),
    ]
)


# ============================================================================
# All Stages List
# ============================================================================

KRK_STAGES: List[KRKStage] = [
    # PHASE 1: Endgame basics - recognize checkmate patterns
    STAGE_0_MATE_IN_1,           # Stage 0: Mate in 1 (98% win rate)
    STAGE_1_MATE_IN_2,           # Stage 1: Reach Stage-0

    # Stage 2A/2B/2C: Edge-trap conversion sub-stages (activated)
    STAGE_2A_EDGE_TRAP_CLOSE,
    STAGE_2B_EDGE_TRAP_ENEMY_BETWEEN,
    STAGE_2C_EDGE_TRAP_WRONG_TEMPO,
    
    # PHASE 1.5: Edge Fence - optimal finishing patterns (NEW)
    STAGE_2_1_EDGE_FENCE_KNIGHT,   # Stage 3: Knight distance (optimal finish)
    STAGE_2_2_EDGE_FENCE_APPROACH, # Stage 4: Approach to knight distance
    STAGE_2_3_EDGE_FENCE_DEEP,     # Stage 5: 2-space tightening
    
    # PHASE 2: Drive Method - opposition and fence technique
    # These teach "drive to edge" pattern before complex box maintenance
    STAGE_D2_OPPOSITION_APPROACH,  # Stage 6: Knight distance approach
    STAGE_D3_TEMPO_WAIT,           # Stage 7: Rook tempo/wait moves
    STAGE_D1_FENCE_ESTABLISHED,    # Stage 8: Fence (rook cut) maintenance
    
    # PHASE 3: Box Method - confine and shrink
    STAGE_7_BOX_SMALL,           # Stage 9: 3x3 box
    STAGE_8_BOX_MEDIUM,          # Stage 10: 4x4 box
    
    # PHASE 4: Cut and Approach - complex coordination
    STAGE_2_5_ANCHORED_CUT,      # Stage 11: Pre-coordinated king+rook
    STAGE_3_EDGE_CUT_HOLD,       # Stage 12: 1x8 box maintenance
    STAGE_4_KING_CLOSE_1,        # Stage 13: King approach 1
    STAGE_5_KING_CLOSE_2,        # Stage 14: King approach 2
    STAGE_6_KING_FAR_CUT_HELD,   # Stage 15: Long approach
    
    # PHASE 5: Full game
    STAGE_9_FULL_KRK,            # Stage 16: Full KRK
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

