"""
Micro-tick utilities for gently settling continuous activations between discrete
ReCoN ticks. A micro-tick loop repeatedly asks a callback for activation
targets, nudges the tracked states toward those targets, and optionally stores a
history for visualization/diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ..core.activations import ActivationState


@dataclass
class MicrotickConfig:
    """
    Configuration payload passed through the engine env.

    Attributes:
        states: mutable mapping of latent-id → ActivationState
        compute_targets: callable(states) -> mapping of latent-id -> target value
        steps: number of micro-ticks to run (non-negative integer)
        eta: smoothing factor applied on every settle step
        history: if True, retain per-step snapshots in `run_microticks` return
    """

    states: MutableMapping[str, ActivationState]
    compute_targets: Callable[[MutableMapping[str, ActivationState]], Mapping[str, float]]
    steps: int = 0
    eta: float = 0.3
    history: bool = False


def settle(y: float, target: float, eta: float, k: float = 1.0) -> float:
    """
    Single smoothing step toward target. Result is clamped to [0, 1] to keep
    activations within interpretable bounds.
    """
    new_val = y + eta * k * (target - y)
    if new_val < 0.0:
        return 0.0
    if new_val > 1.0:
        return 1.0
    return new_val


def run_microticks(config: MicrotickConfig) -> Optional[List[Dict[str, float]]]:
    """
    Execute a micro-tick loop. Returns a list of per-step activation snapshots if
    `config.history` is True; otherwise returns None.
    """
    steps = max(0, int(config.steps))
    if steps == 0:
        return [] if config.history else None

    history: List[Dict[str, float]] = []
    states = config.states
    eta = float(config.eta)

    for _ in range(steps):
        targets = config.compute_targets(states)
        for key, target in targets.items():
            if key not in states:
                states[key] = ActivationState()
            states[key].target = float(target)

        for state in states.values():
            state.value = settle(state.value, state.target, eta, state.k)

        if config.history:
            history.append({key: state.value for key, state in states.items()})

    return history if config.history else None
