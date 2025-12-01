"""
Evaluation functions for chess positions in ReCoN-lite.

This package provides:
- heuristic: Simple material + king safety + mobility evaluation
"""

from .heuristic import (
    eval_position,
    compute_reward_tick,
    eval_position_stockfish,
)

__all__ = [
    "eval_position",
    "compute_reward_tick",
    "eval_position_stockfish",
]

