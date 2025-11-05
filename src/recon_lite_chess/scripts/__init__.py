"""Expose chess script builders (KPK, rook endings, ...)."""

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

__all__ = [
    "build_kpk_network",
    "create_kpk_material_detector",
    "create_kpk_push_window",
    "create_kpk_opposition_probe",
    "create_kpk_promotion_probe",
    "create_kpk_move_selector",
    "build_rook_techniques_network",
    "create_rook_cutoff_detector",
    "create_rook_bridge_detector",
    "create_rook_ladder_detector",
    "create_rook_summary_probe",
]
