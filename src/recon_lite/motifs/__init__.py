"""M5: Motif extraction and descriptor module for structure discovery."""

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

__all__ = [
    "BindingDescriptor",
    "MotifType",
    "MotifDataset",
    "extract_3x3_patch",
    "extract_king_zone",
    "extract_pawn_chain",
    "extract_hanging_pieces",
    "extract_tactical_features",
    "extract_all_features",
]

