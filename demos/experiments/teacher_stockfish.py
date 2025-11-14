#!/usr/bin/env python3
"""Minimal Stockfish-backed teacher to refresh macro sidecar weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import chess

try:
    import chess.engine
except Exception:  # pragma: no cover - python-chess optional extras may be missing
    chess = chess  # type: ignore[misc]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_lite.macro_engine import MacroEngine
from recon_lite_chess.actuators import (
    choose_move_phase1,
    choose_move_phase2,
    choose_move_phase3,
)
from recon_lite_chess.actuators_blend import cheap_eval_after
from recon_lite_chess.sensors.structure import summarize_kpk_material

DEFAULT_SIDE_CAR = Path("weights/macro_weights.json")
DEFAULT_PHASE_OUTPUT = Path("weights/phase_child_weights.json")
PHASE_WEIGHT_CHOOSERS = {
    "phase1": choose_move_phase1,
    "phase2": choose_move_phase2,
    "phase3": choose_move_phase3,
}


def _load_fens(path: Path) -> Iterable[str]:
    data = path.read_text().strip().splitlines()
    return [line.strip() for line in data if line.strip() and not line.strip().startswith("#")]


def _classify(board: chess.Board) -> str:
    if MacroEngine._is_krk_board(board):
        return "krk"
    if MacroEngine._is_kpk_board(board):
        return "kpk"
    summary = summarize_kpk_material(board)
    if summary.get("is_kpk"):
        return "kpk"
    pieces = board.piece_map().values()
    if any(p.piece_type == chess.ROOK for p in pieces if p.color == board.turn):
        return "rook"
    return "generic"


def _open_engine(engine_path: Optional[str]) -> Optional[chess.engine.SimpleEngine]:
    if not engine_path:
        return None
    return chess.engine.SimpleEngine.popen_uci(engine_path)


def _analyse(engine: Optional[chess.engine.SimpleEngine], board: chess.Board, depth: int) -> float:
    if engine is None:
        return 0.0
    try:
        info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score is None:
            return 0.0
        return float(score.white().score(mate_score=10000) or 0.0)
    except Exception:
        return 0.0


def _score_phase_candidates(
    board: chess.Board,
    engine: Optional[chess.engine.SimpleEngine],
    depth: int,
) -> Dict[str, float]:
    """Return positive scores for each phase chooser that produced a move."""
    scores: Dict[str, float] = {}
    for phase, chooser in PHASE_WEIGHT_CHOOSERS.items():
        try:
            uci = chooser(board, None)
        except Exception:
            continue
        if not uci:
            continue
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            continue
        if engine is None:
            score = cheap_eval_after(board.copy(stack=False), move)
        else:
            trial = board.copy(stack=False)
            try:
                trial.push(move)
            except ValueError:
                continue
            score = _analyse(engine, trial, depth)
        if score <= 0.0:
            continue
        scores[phase] = float(score)
    return scores


def _derive_phase_totals(
    fens: Sequence[str],
    engine_path: Optional[str],
    depth: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    totals = {phase: 0.0 for phase in PHASE_WEIGHT_CHOOSERS}
    counts = {phase: 0 for phase in PHASE_WEIGHT_CHOOSERS}
    engine = _open_engine(engine_path)
    try:
        for fen in fens:
            if not fen:
                continue
            board = chess.Board(fen)
            if not MacroEngine._is_krk_board(board):
                continue
            scores = _score_phase_candidates(board, engine, depth)
            for phase, score in scores.items():
                totals[phase] += score
                counts[phase] += 1
    finally:
        if engine is not None:
            engine.quit()
    return totals, counts


def _normalize_phase_totals(totals: Mapping[str, float]) -> Dict[str, float]:
    filtered = {phase: max(0.0, totals.get(phase, 0.0)) for phase in PHASE_WEIGHT_CHOOSERS}
    total = sum(filtered.values())
    if total <= 0.0:
        return {phase: 1.0 for phase in PHASE_WEIGHT_CHOOSERS}
    avg = total / len(filtered)
    if avg <= 0.0:
        avg = 1.0
    return {phase: round(filtered.get(phase, 0.0) / avg, 3) for phase in PHASE_WEIGHT_CHOOSERS}


def label_fens(fens: Iterable[str], engine_path: Optional[str], depth: int) -> Tuple[Dict[str, int], float]:
    """Label an iterable of FENs and return (counts, avg_score)."""
    fens = [fen for fen in fens if fen]
    if not fens:
        raise ValueError("No FEN positions supplied for labeling.")

    counts: Dict[str, int] = {"krk": 0, "kpk": 0, "rook": 0, "generic": 0}
    score_total = 0.0
    total_positions = 0

    engine = _open_engine(engine_path)
    try:
        for fen in fens:
            board = chess.Board(fen)
            label = _classify(board)
            counts[label] = counts.get(label, 0) + 1
            score_total += _analyse(engine, board, depth)
            total_positions += 1
    finally:
        if engine is not None:
            engine.quit()

    avg_score = score_total / max(1, total_positions)
    return counts, avg_score


def apply_weight_update(payload: Dict[str, object], counts: Dict[str, int], avg_score: float) -> None:
    """Mutate the weight payload using aggregated teacher counts."""
    total = max(1, sum(counts.values()))
    por_edges = payload.setdefault("por_edges", {})
    base_plan = 0.6 + 0.05 * (counts.get("kpk", 0) + counts.get("krk", 0))
    base_move = 0.4 + 0.05 * counts.get("rook", 0)
    por_edges["LearningSupervisor->PlanHub"] = round(min(1.8, base_plan), 3)
    por_edges["LearningSupervisor->MoveSynth"] = round(min(1.6, base_move), 3)

    por_policies = payload.setdefault("por_policies", {})
    plan_policy = por_policies.setdefault("PlanHub", {})
    plan_policy.setdefault("policy", "weighted")
    plan_policy["theta"] = round(0.5 + total * 0.02, 3)

    notes = payload.setdefault("notes", {})
    if not isinstance(notes, dict):
        notes = {"legacy": notes}
        payload["notes"] = notes
    teacher_meta = notes.setdefault("teacher", {})
    teacher_meta.update(
        {
            "labels": counts,
            "avg_score": round(avg_score, 2),
        }
    )


def run_teacher(
    fen_path: Path,
    output_path: Path,
    engine_path: Optional[str],
    depth: int,
    *,
    fens_override: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    fens = list(fens_override) if fens_override is not None else list(_load_fens(fen_path))
    if not fens:
        raise SystemExit(f"No FEN positions found in {fen_path}")

    counts, avg_score = label_fens(fens, engine_path, depth)

    payload = json.loads(output_path.read_text()) if output_path.exists() else {"version": "0.1"}
    apply_weight_update(payload, counts, avg_score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_phase_teacher(
    fens: Sequence[str],
    output_path: Path,
    engine_path: Optional[str],
    depth: int,
) -> Dict[str, object]:
    if not fens:
        raise SystemExit("No FEN positions supplied for phase weight training.")
    totals, counts = _derive_phase_totals(fens, engine_path, depth)
    weights = _normalize_phase_totals(totals)
    payload: Dict[str, object] = {
        "version": "0.1",
        "phase_weights": weights,
        "notes": {
            "teacher": {
                "phase_counts": counts,
                "positions_considered": int(sum(counts.values())),
                "engine": bool(engine_path),
            }
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Label macrograph preferences using Stockfish and update sidecar weights.")
    parser.add_argument("fen_file", type=Path, help="Path to a file containing FEN positions (one per line).")
    parser.add_argument("--output", type=Path, default=DEFAULT_SIDE_CAR, help="Sidecar JSON to update (default: weights/macro_weights.json)")
    parser.add_argument("--phase-output", type=Path, default=DEFAULT_PHASE_OUTPUT, help="Path for blended phase weight sidecar (default: weights/phase_child_weights.json)")
    parser.add_argument("--skip-phase-output", action="store_true", help="Skip blended phase weight computation")
    parser.add_argument("--engine", type=str, default=None, help="Path to Stockfish binary (optional – heuristics used if omitted)")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for Stockfish analysis")
    args = parser.parse_args()

    fens = list(_load_fens(args.fen_file))
    if not fens:
        raise SystemExit(f"No FEN positions found in {args.fen_file}")

    payload = run_teacher(args.fen_file, args.output, args.engine, args.depth, fens_override=fens)
    print("Updated", args.output)
    teacher_meta = payload.get("notes", {}).get("teacher", {})
    if teacher_meta:
        print("Counts:", teacher_meta.get("labels", {}))
        print("Average score:", teacher_meta.get("avg_score"))

    if not args.skip_phase_output and args.phase_output:
        phase_payload = run_phase_teacher(fens, args.phase_output, args.engine, args.depth)
        print("Updated", args.phase_output)
        phase_meta = phase_payload.get("notes", {}).get("teacher", {})
        if phase_meta:
            print("Phase counts:", phase_meta.get("phase_counts", {}))
            print("Positions considered:", phase_meta.get("positions_considered"))


if __name__ == "__main__":
    main()
