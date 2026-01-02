"""Learning subpackage for structure learning and plasticity."""

from .m5_structure import (
    StructureLearner,
    AffordanceSpike,
    PromotionResult,
    PruningResult,
    create_pattern_sensor,
)

__all__ = [
    "StructureLearner",
    "AffordanceSpike",
    "PromotionResult",
    "PruningResult",
    "create_pattern_sensor",
]
