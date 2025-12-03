"""Expose chess script builders (KPK, rook endings, opening, middlegame)."""

from .kpk import (
    build_kpk_network,
    create_kpk_material_detector,
    create_kpk_push_window,
    create_kpk_opposition_probe,
    create_kpk_promotion_probe,
    create_kpk_move_selector,
)
from .rook_endings import (
    build_rook_techniques_network,
    create_rook_cutoff_detector,
    create_rook_bridge_detector,
    create_rook_ladder_detector,
    create_rook_summary_probe,
)
from .opening import (
    build_opening_hierarchy,
    create_development_sensor,
    create_castling_sensor,
    create_center_control_sensor,
    get_opening_move_candidates,
)
from .middlegame import (
    build_middlegame_hierarchy,
    create_king_safety_sensor,
    create_piece_activity_sensor,
    create_structure_sensor,
    get_middlegame_move_candidates,
)

__all__ = [
    # KPK
    "build_kpk_network",
    "create_kpk_material_detector",
    "create_kpk_push_window",
    "create_kpk_opposition_probe",
    "create_kpk_promotion_probe",
    "create_kpk_move_selector",
    # Rook endings
    "build_rook_techniques_network",
    "create_rook_cutoff_detector",
    "create_rook_bridge_detector",
    "create_rook_ladder_detector",
    "create_rook_summary_probe",
    # M6: Opening
    "build_opening_hierarchy",
    "create_development_sensor",
    "create_castling_sensor",
    "create_center_control_sensor",
    "get_opening_move_candidates",
    # M6: Middlegame
    "build_middlegame_hierarchy",
    "create_king_safety_sensor",
    "create_piece_activity_sensor",
    "create_structure_sensor",
    "get_middlegame_move_candidates",
]
