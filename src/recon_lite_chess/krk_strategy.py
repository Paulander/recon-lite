"""
Strategic scaffolding for the KRK persistent demo.

`compute_phase_logits` inspects the current board and produces logits for each
phase script. The logits are turned into soft activations so we can blend phase
intent inside the persistent loop and expose those latents to the logger/viz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import chess

from .predicates import (
    box_min_side,
    box_min_side_after,
    can_deliver_mate,
    enemy_at_edge,
    has_opposition_after,
    has_stable_cut,
    rook_distance_to_target_fence,
)
from recon_lite.core.activations import ActivationState, softmax

PHASE_IDS = ("phase0", "phase1", "phase2", "phase3", "phase4")
SCRIPT_BY_PHASE = {
    "phase0": "phase0_establish_cut",
    "phase1": "phase1_drive_to_edge",
    "phase2": "phase2_shrink_box",
    "phase3": "phase3_take_opposition",
    "phase4": "phase4_deliver_mate",
}


def compute_phase_logits(board: chess.Board) -> Dict[str, float]:
    """
    Produce heuristic logits per KRK phase. Higher logits indicate phases that
    should carry more weight during micro-tick settling.
    """
    logits = {phase: -2.0 for phase in PHASE_IDS}
    logits["phase0"] = 0.0  # default fallback

    try:
        if can_deliver_mate(board):
            logits["phase4"] += 6.0
        else:
            logits["phase4"] += -3.0
    except Exception:
        pass

    try:
        if has_stable_cut(board):
            logits["phase1"] += 3.0
        else:
            logits["phase1"] += -1.0
    except Exception:
        pass

    try:
        min_side = box_min_side(board)
        logits["phase0"] += max(0.0, 3.0 - min_side)
        logits["phase2"] += max(0.0, 4.0 - min_side)
        logits["phase3"] += max(0.0, 2.0 - min_side) * 0.5
    except Exception:
        pass

    try:
        if enemy_at_edge(board):
            logits["phase2"] += 2.5
            logits["phase3"] += 2.0
    except Exception:
        pass

    try:
        base_min_side = box_min_side(board)
        for move in board.legal_moves:
            if has_opposition_after(board, move):
                logits["phase3"] += 2.5
                break
            # Encourage phase2 moves that shrink the box
            try:
                shrink = base_min_side - box_min_side_after(board, move)
                if shrink > 0:
                    logits["phase2"] += min(2.0, shrink)
            except Exception:
                continue
    except Exception:
        pass

    try:
        fence_dist = rook_distance_to_target_fence(board)
        logits["phase1"] += max(0.0, 3.5 - fence_dist)
        logits["phase2"] += max(0.0, 2.5 - fence_dist) * 0.5
    except Exception:
        pass

    return logits


def phase_latents_from_logits(
    logits: Mapping[str, float],
    *,
    temperature: float = 1.4,
) -> Dict[str, float]:
    """Softmax helper that returns activations keyed by script ids."""
    ordered = [logits.get(phase, -5.0) for phase in PHASE_IDS]
    dist = softmax(ordered, temperature=temperature)
    return {
        SCRIPT_BY_PHASE[phase]: dist[idx]
        for idx, phase in enumerate(PHASE_IDS)
    }


@dataclass(frozen=True)
class OutcomeMode:
    weights: Tuple[float, float, float]

    def as_dict(self) -> Dict[str, float]:
        return {"play_to_win": self.weights[0], "neutral": self.weights[1], "play_to_draw": self.weights[2]}


@dataclass(frozen=True)
class StyleBias:
    weights: Tuple[float, float, float]

    def as_dict(self) -> Dict[str, float]:
        return {"tactical": self.weights[0], "positional": self.weights[1], "endgame": self.weights[2]}


def neutral_outcome_mode() -> OutcomeMode:
    return OutcomeMode((1 / 3, 1 / 3, 1 / 3))


def neutral_style_bias() -> StyleBias:
    return StyleBias((1 / 3, 1 / 3, 1 / 3))


def ensure_phase_states(
    states: Dict[str, ActivationState],
) -> Dict[str, ActivationState]:
    """
    Guarantee that a mapping contains ActivationState entries for every script
    id. Useful when the persistent loop needs to keep state across plies.
    """
    for script_id in SCRIPT_BY_PHASE.values():
        states.setdefault(script_id, ActivationState())
    return states
