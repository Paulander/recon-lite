"""Chess sensor helpers spanning structural and tactical cues."""

from .structure import (
    summarize_kpk_material,
    pawn_distance_to_promotion,
    pawn_has_clear_path,
)
from .tactics import (
    can_push_pawn_safely,
    has_opposition_alignment,
    opposition_after_push,
)

__all__ = [
    "summarize_kpk_material",
    "pawn_distance_to_promotion",
    "pawn_has_clear_path",
    "can_push_pawn_safely",
    "has_opposition_alignment",
    "opposition_after_push",
]
