"""
Utilities for managing continuous activations inside the ReCoN-lite core.

This module provides light-weight numerical helpers (sigmoid/softmax) and an
`ActivationState` data container that tracks the current and target values for a
node-level latent.  The class keeps the semantics simple: we only store scalar
activations, but the interface leaves room for future vector extensions by
exposing hooks for per-state gain (`k`) and optional metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, Mapping, MutableMapping, Optional


def sigmoid(x: float) -> float:
    """Classic logistic with minimal numerical safeguards."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(values: Iterable[float], temperature: float = 1.0) -> list[float]:
    """
    Numerically stable softmax. A mild temperature (>0) smooths the distribution.

    Returns a list so callers can iterate multiple times without recomputing the
    generator. Raises ValueError on non-positive temperature.
    """
    vals = list(values)
    if not vals:
        return []
    if temperature <= 0:
        raise ValueError("softmax temperature must be > 0")
    scaled = [v / temperature for v in vals]
    max_v = max(scaled)
    exp_vals = [math.exp(v - max_v) for v in scaled]
    denom = sum(exp_vals)
    if denom == 0:
        return [1.0 / len(exp_vals)] * len(exp_vals)
    return [v / denom for v in exp_vals]


@dataclass
class ActivationState:
    """
    Tracks a scalar activation alongside its latest target. The optional `k`
    factor lets micro-tick integration modulate responsiveness per state.
    """

    value: float = 0.0
    target: float = 0.0
    k: float = 1.0
    meta: Dict[str, float] = field(default_factory=dict)

    def nudge(self, target: float, eta: float) -> float:
        """
        Move the activation toward `target` using a simple exponential smoothing
        step: y ← y + η * k * (target − y). Result is clamped to [0, 1] for
        stability. Returns the updated value.
        """
        self.target = float(target)
        delta = self.target - self.value
        self.value += float(eta) * float(self.k) * delta
        if self.value < 0.0:
            self.value = 0.0
        elif self.value > 1.0:
            self.value = 1.0
        return self.value

    def reset(self, value: float = 0.0) -> None:
        """Reset both value and target to `value`."""
        val = float(value)
        self.value = val
        self.target = val
        self.meta.clear()


def ensure_states(
    mapping: MutableMapping[str, ActivationState],
    keys: Iterable[str],
    *,
    default: Optional[ActivationState] = None,
) -> MutableMapping[str, ActivationState]:
    """
    Guarantee that `mapping` contains an ActivationState per key. The optional
    `default` is cloned (by value/target/k) for missing keys.
    """
    template = default or ActivationState()
    for key in keys:
        if key not in mapping:
            mapping[key] = ActivationState(
                value=template.value,
                target=template.target,
                k=template.k,
                meta=dict(template.meta),
            )
    return mapping


def activation_snapshot(states: Mapping[str, ActivationState]) -> Dict[str, float]:
    """Utility to materialize current activation values for logging."""
    return {key: float(state.value) for key, state in states.items()}
