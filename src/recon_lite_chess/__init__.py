# src/recon_lite_chess/__init__.py
"""
Chess domain module for ReCoN networks (KRK endgame).
"""

from .krk_nodes import (
    # Terminal evaluators
    KingAtEdgeDetector, BoxShrinkEvaluator, OppositionEvaluator,
    MateDeliverEvaluator, StalemateDetector, WaitForBoardChange,
    CutEstablishedDetector, RookLostDetector,

    # Move generators (actuators)
    Phase0ChooseMoves, KingDriveMoves, BoxShrinkMoves, OppositionMoves, MateMoves, RandomLegalMoves,

    # Script phase nodes
    Phase0EstablishCut, Phase1DriveToEdge, Phase2ShrinkBox,
    Phase3TakeOpposition, Phase4DeliverMate, KRKCheckmateRoot,

    # Factories (evaluators)
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_stalemate_detector, create_wait_for_board_change,
    create_cut_established_detector, create_rook_lost_detector,

    # Factories (actuators)
    create_phase0_choose_moves, create_king_drive_moves, create_box_shrink_moves,
    create_opposition_moves, create_mate_moves, create_random_legal_moves, create_no_progress_watch,

    # Factories (scripts)
    create_phase0_establish_cut, create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate, create_krk_root,

    # New confinement-aware nodes
    ConfinementEvaluator, BarrierReadyEvaluator,
    ConfinementMoves, BarrierPlacementMoves,
    create_confinement_evaluator, create_barrier_ready_evaluator,
    create_confinement_moves, create_barrier_placement_moves,
)

# Export wiring helper
from .krk_nodes import wire_default_krk

# Import predicates/actuators submodules so callers can use them if needed
from . import predicates
from . import actuators
from . import strategy
from . import actuators_blend

__all__ = [
    # Terminal evaluators
    "KingAtEdgeDetector", "BoxShrinkEvaluator", "OppositionEvaluator",
    "MateDeliverEvaluator", "StalemateDetector", "WaitForBoardChange",
    "CutEstablishedDetector", "RookLostDetector",

    # Move generators (actuators)
    "Phase0ChooseMoves", "KingDriveMoves", "BoxShrinkMoves", "OppositionMoves", "MateMoves", "RandomLegalMoves",

    # Script phases
    "Phase0EstablishCut", "Phase1DriveToEdge", "Phase2ShrinkBox",
    "Phase3TakeOpposition", "Phase4DeliverMate", "KRKCheckmateRoot",

    # Factories
    "create_king_edge_detector", "create_box_shrink_evaluator",
    "create_opposition_evaluator", "create_mate_deliver_evaluator",
    "create_stalemate_detector", "create_wait_for_board_change",
    "create_cut_established_detector", "create_rook_lost_detector",
    "create_phase0_choose_moves", "create_king_drive_moves", "create_box_shrink_moves",
    "create_opposition_moves", "create_mate_moves", "create_random_legal_moves", "create_no_progress_watch",
    "create_phase0_establish_cut", "create_phase1_drive_to_edge", "create_phase2_shrink_box",
    "create_phase3_take_opposition", "create_phase4_deliver_mate", "create_krk_root",

    # New confinement-aware nodes
    "ConfinementEvaluator", "BarrierReadyEvaluator",
    "ConfinementMoves", "BarrierPlacementMoves",
    "create_confinement_evaluator", "create_barrier_ready_evaluator",
    "create_confinement_moves", "create_barrier_placement_moves",

    # Helper
    "wire_default_krk",

    # Submodules
    "predicates", "actuators", "strategy", "actuators_blend",
]
