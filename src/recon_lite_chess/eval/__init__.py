"""
Evaluation functions for chess positions in ReCoN-lite.

This package provides:
- heuristic: Material + king safety + mobility + pawn structure + piece activity
- manager: Unified evaluation interface supporting multiple backends
- distill: ML-based distilled evaluation trained on Stockfish (M7)
- features: Feature extraction for ML models (M7)
"""

from .heuristic import (
    eval_position,
    eval_position_fast,
    eval_position_full,
    compute_reward_tick,
    eval_position_stockfish,
)
from .manager import (
    EvalMode,
    EvalConfig,
    EvalResult,
    EvalManager,
)
from .distill import (
    DistillationConfig,
    DistillationSample,
    DistillationDataset,
    DistilledEvaluator,
    train_distilled_eval,
)
from .features import (
    extract_features,
    FeatureVector,
    FEATURE_COUNT,
)

__all__ = [
    # heuristic
    "eval_position",
    "eval_position_fast",
    "eval_position_full",
    "compute_reward_tick",
    "eval_position_stockfish",
    # manager
    "EvalMode",
    "EvalConfig",
    "EvalResult",
    "EvalManager",
    # distill
    "DistillationConfig",
    "DistillationSample",
    "DistillationDataset",
    "DistilledEvaluator",
    "train_distilled_eval",
    # features
    "extract_features",
    "FeatureVector",
    "FEATURE_COUNT",
]

