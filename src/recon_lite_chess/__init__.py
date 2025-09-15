# src/recon_lite_chess/__init__.py
"""
Chess domain module for ReCoN networks.

This module provides chess-specific node implementations for the KRK (King+Rook vs King)
checkmate challenge. It depends on the core recon_lite library and python-chess.
"""

from .krk_nodes import (
    # Terminal evaluators
    KingAtEdgeDetector, BoxShrinkEvaluator, OppositionEvaluator,
    MateDeliverEvaluator, StalemateDetector,

    # Move generators (actuators)
    KingDriveMoves, RandomLegalMoves,

    # Script phase nodes
    Phase1DriveToEdge, Phase2ShrinkBox, Phase3TakeOpposition,
    Phase4DeliverMate, KRKCheckmateRoot,

    # Factory functions for terminal nodes
    create_king_edge_detector, create_box_shrink_evaluator,
    create_opposition_evaluator, create_mate_deliver_evaluator,
    create_stalemate_detector,

    # Factory functions for move generators
    create_king_drive_moves, create_random_legal_moves,

    # Factory functions for script nodes
    create_phase1_drive_to_edge, create_phase2_shrink_box,
    create_phase3_take_opposition, create_phase4_deliver_mate,
    create_krk_root
)

__all__ = [
    # Terminal evaluators
    "KingAtEdgeDetector", "BoxShrinkEvaluator", "OppositionEvaluator",
    "MateDeliverEvaluator", "StalemateDetector",

    # Move generators (actuators)
    "KingDriveMoves", "RandomLegalMoves",

    # Script phase nodes
    "Phase1DriveToEdge", "Phase2ShrinkBox", "Phase3TakeOpposition",
    "Phase4DeliverMate", "KRKCheckmateRoot",

    # Factory functions
    "create_king_edge_detector", "create_box_shrink_evaluator",
    "create_opposition_evaluator", "create_mate_deliver_evaluator",
    "create_stalemate_detector",
    "create_king_drive_moves", "create_random_legal_moves",
    "create_phase1_drive_to_edge", "create_phase2_shrink_box",
    "create_phase3_take_opposition", "create_phase4_deliver_mate",
    "create_krk_root"
]
