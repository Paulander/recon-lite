"""
Evaluation functions for chess positions in ReCoN-lite.

This package provides:
- heuristic: Material + king safety + mobility + pawn structure + piece activity
- manager: Unified evaluation interface supporting multiple backends
- distill: Future ML-based distilled evaluation (stub)
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
]

