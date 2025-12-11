"""Expose chess script builders (KPK, KQK, rook endings, opening, middlegame)."""

from .kpk import (
    build_kpk_network,
    create_kpk_material_detector,
    create_kpk_push_window,
    create_kpk_opposition_probe,
    create_kpk_promotion_probe,
    create_kpk_move_selector,
)
from .kqk import (
    build_kqk_network,
    create_random_kqk_board,
    is_kqk_position,
    create_kqk_material_detector,
    create_kqk_move_selector,
    analyze_stalemate_danger,
    StalemateDangerLevel,
)
from .stalemate_detector import (
    create_stalemate_danger_sensor,
    create_stalemate_gate,
    create_wait_move_selector,
    StalemateAnalysis,
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
    # KQK
    "build_kqk_network",
    "create_random_kqk_board",
    "is_kqk_position",
    "create_kqk_material_detector",
    "create_kqk_move_selector",
    "analyze_stalemate_danger",
    "StalemateDangerLevel",
    # Shared Stalemate Detector
    "create_stalemate_danger_sensor",
    "create_stalemate_gate",
    "create_wait_move_selector",
    "StalemateAnalysis",
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
