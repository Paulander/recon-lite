"""
Affordance Signal Module for ReCoN Chess.

Provides continuous [0.0, 1.0] signals indicating "distance to applicability"
for each subgraph. This enables implicit lookahead where the planning layer
can sense valid strategies before they are fully executable.

Example:
    from recon_lite_chess.affordance import compute_all_affordances, AffordanceSignal
    
    signals = compute_all_affordances(board)
    if signals["krk"].value > 0.5:
        # KRK endgame is becoming relevant
        pass
"""

from .sensors import (
    AffordanceSignal,
    compute_krk_affordance,
    compute_kpk_affordance,
    compute_kqk_affordance,
    compute_all_affordances,
    AffordanceConfig,
)

__all__ = [
    "AffordanceSignal",
    "compute_krk_affordance",
    "compute_kpk_affordance",
    "compute_kqk_affordance",
    "compute_all_affordances",
    "AffordanceConfig",
]

