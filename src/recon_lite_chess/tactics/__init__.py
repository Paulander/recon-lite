"""MLP-based tactics detection module."""

from .features import (
    extract_tactics_features,
    extract_back_rank_features,
    extract_double_check_features,
    extract_smothered_mate_features,
    FEATURE_COUNT,
)
from .mlp_detector import TacticMLPDetector, get_mlp_detector

__all__ = [
    "extract_tactics_features",
    "extract_back_rank_features",
    "extract_double_check_features",
    "extract_smothered_mate_features",
    "FEATURE_COUNT",
    "TacticMLPDetector",
    "get_mlp_detector",
]

