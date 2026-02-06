"""
Optional blended move chooser that scores proposals from phase1/phase2/phase3
against the current phase latents. The goal is to keep behaviour compatible
with existing actuators while allowing a smooth interpolation between phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import chess

from .actuators import (
    choose_move_phase1,
    choose_move_phase2,
    choose_move_phase3,
)
from .predicates import (
    box_min_side,
    box_min_side_after,
    chebyshev,
    dist_to_edge,
    has_opposition_after,
)
from .krk_strategy import SCRIPT_BY_PHASE


PHASE_CHOOSERS = {
    "phase1": choose_move_phase1,
    "phase2": choose_move_phase2,
    "phase3": choose_move_phase3,
}

_PHASE_WEIGHT_ENV = "RECON_PHASE_WEIGHT_FILE"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PHASE_WEIGHT_PATH = _PROJECT_ROOT / "weights" / "krk_phase_weight_pack.swp"
_DEFAULT_PHASE_BIASES = {
    "phase1": 1.0,
    "phase2": 1.0,
    "phase3": 1.0,
}


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_weights(weights: Mapping[str, object]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for phase, value in weights.items():
        num = _to_float(value)
        if num is None or num < 0:
            continue
        cleaned[str(phase)] = num
    if not cleaned:
        return dict(_DEFAULT_PHASE_BIASES)
    avg = sum(cleaned.values()) / len(cleaned)
    if avg <= 0:
        return dict(_DEFAULT_PHASE_BIASES)
    return {phase: val / avg for phase, val in cleaned.items()}


@lru_cache(maxsize=1)
def _phase_weight_table() -> Dict[str, float]:
    path_str = os.environ.get(_PHASE_WEIGHT_ENV)
    path = Path(path_str) if path_str else _DEFAULT_PHASE_WEIGHT_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = payload.get("phase_weights") or payload.get("weights") or {}
    except (FileNotFoundError, json.JSONDecodeError):
        weights = {}
    table = _normalize_weights(weights)
    for phase, default in _DEFAULT_PHASE_BIASES.items():
        table.setdefault(phase, default)
    return table


def clear_phase_weight_cache() -> None:
    """Reset cached phase weights (primarily for tests)."""
    _phase_weight_table.cache_clear()


def _phase_bias(phase: str, env: Optional[Mapping[str, object]]) -> float:
    if env:
        override = env.get("phase_weight_override")
        if isinstance(override, Mapping):
            value = _to_float(override.get(phase))
            if value is not None:
                return max(0.0, value)
    return _phase_weight_table().get(phase, 1.0)


def _cheap_eval(board: chess.Board) -> float:
    """Very light evaluation encouraging edge pressure and king proximity."""
    enemy = board.king(not board.turn)
    if enemy is None:
        return 0.0
    score = (4 - dist_to_edge(enemy)) * 1.5
    our_king = board.king(board.turn)
    if our_king is not None:
        score += max(0, 4 - chebyshev(our_king, enemy)) * 0.4
    # Encourage rook support
    rook_sq = next(
        (sq for sq, piece in board.piece_map().items() if piece.color == board.turn and piece.piece_type == chess.ROOK),
        None,
    )
    if rook_sq is not None:
        score += max(0, 4 - chebyshev(rook_sq, enemy)) * 0.2
    return float(score)


def cheap_eval_after(board: chess.Board, move: chess.Move) -> float:
    board.push(move)
    try:
        return _cheap_eval(board)
    finally:
        board.pop()


def _phase_weight(phase_latents: Dict[str, float], phase: str) -> float:
    script_id = SCRIPT_BY_PHASE.get(phase)
    if script_id is None:
        return 0.0
    return float(phase_latents.get(script_id, 0.0))


def _phase_score(board: chess.Board, move: chess.Move, phase: str, base_min_side: Optional[int]) -> float:
    if phase == "phase3":
        return 1.5 if has_opposition_after(board, move) else 0.0
    if base_min_side is None:
        try:
            base_min_side = box_min_side(board)
        except Exception:
            base_min_side = None
    try:
        new_side = box_min_side_after(board, move)
    except Exception:
        new_side = None
    if new_side is None or base_min_side is None:
        return 0.0
    improvement = float(base_min_side - new_side)
    return max(0.0, improvement)


@dataclass
class BlendedCandidate:
    uci: str
    phase: str
    phase_weight: float
    phase_score: float
    cheap_eval: float
    phase_bias: float = 1.0

    @property
    def blended_score(self) -> float:
        return self.phase_weight * self.phase_bias * self.phase_score + self.cheap_eval

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "move": self.uci,
            "phase": self.phase,
            "phase_weight": self.phase_weight,
            "phase_bias": self.phase_bias,
            "phase_score": self.phase_score,
            "cheap_eval": self.cheap_eval,
            "score": self.blended_score,
        }


def gather_candidates(
    board: chess.Board,
    phase_latents: Dict[str, float],
    env: Optional[Dict[str, object]] = None,
) -> List[BlendedCandidate]:
    """Collect unique candidate moves from P1/P2/P3 choosers."""
    seen: set[str] = set()
    candidates: List[BlendedCandidate] = []
    try:
        base_min_side = box_min_side(board)
    except Exception:
        base_min_side = None

    for phase, chooser in PHASE_CHOOSERS.items():
        try:
            uci = chooser(board, env)
        except Exception:
            continue
        if not uci or uci in seen:
            continue
        seen.add(uci)
        move = chess.Move.from_uci(uci)
        weight = _phase_weight(phase_latents, phase)
        phase_score = _phase_score(board, move, phase, base_min_side)
        cheap = cheap_eval_after(board, move)
        bias = _phase_bias(phase, env)
        candidates.append(
            BlendedCandidate(
                uci=uci,
                phase=phase,
                phase_weight=weight,
                phase_score=phase_score,
                cheap_eval=cheap,
                phase_bias=bias,
            )
        )
    return candidates


def choose_blended_move(
    board: chess.Board,
    phase_latents: Dict[str, float],
    env: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[str], List[Dict[str, float | str]]]:
    """
    Return (best_move_uci, diagnostics). Diagnostics is a list of candidate
    dictionaries sorted by blended score.
    """
    candidates = gather_candidates(board, phase_latents, env)
    if not candidates:
        # Fall back to whichever chooser can still supply a move.
        for phase, chooser in PHASE_CHOOSERS.items():
            try:
                fallback = chooser(board, env)
            except Exception:
                continue
            if fallback:
                return fallback, []
        return None, []

    ordered = sorted(candidates, key=lambda c: c.blended_score, reverse=True)
    return ordered[0].uci, [cand.as_dict() for cand in ordered]
