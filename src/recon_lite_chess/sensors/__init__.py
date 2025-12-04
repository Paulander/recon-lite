"""Chess sensor helpers spanning structural and tactical cues.

M6 Extension: Material and Phase sensors as fan-in terminals.
"""

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
from .material import (
    MaterialCategory,
    MaterialAssessment,
    assess_material,
    material_sensor_predicate,
    create_material_sensor_node,
)
from .phase import (
    PhaseWeights,
    estimate_phase,
    phase_sensor_predicate,
    create_phase_sensor_node,
)

__all__ = [
    # Structure sensors
    "summarize_kpk_material",
    "pawn_distance_to_promotion",
    "pawn_has_clear_path",
    # Tactical sensors
    "can_push_pawn_safely",
    "has_opposition_alignment",
    "opposition_after_push",
    # M6: Material sensor
    "MaterialCategory",
    "MaterialAssessment",
    "assess_material",
    "material_sensor_predicate",
    "create_material_sensor_node",
    # M6: Phase sensor
    "PhaseWeights",
    "estimate_phase",
    "phase_sensor_predicate",
    "create_phase_sensor_node",
]
