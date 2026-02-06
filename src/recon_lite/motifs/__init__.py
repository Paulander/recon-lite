"""M5: Motif extraction and descriptor module for structure discovery.

M10: Extended with pattern induction pipeline for stem cell integration.
"""

from .descriptors import (
    BindingDescriptor,
    MotifType,
    MotifDataset,
)
from .extractors import (
    extract_3x3_patch,
    extract_king_zone,
    extract_pawn_chain,
    extract_hanging_pieces,
    extract_tactical_features,
    extract_all_features,
)
from .induction import (
    PatternInduction,
    PromotionConfig,
    PromotionCandidate,
    PromotedSensor,
)

__all__ = [
    # M5: Descriptors and extractors
    "BindingDescriptor",
    "MotifType",
    "MotifDataset",
    "extract_3x3_patch",
    "extract_king_zone",
    "extract_pawn_chain",
    "extract_hanging_pieces",
    "extract_tactical_features",
    "extract_all_features",
    # M10: Induction
    "PatternInduction",
    "PromotionConfig",
    "PromotionCandidate",
    "PromotedSensor",
]

