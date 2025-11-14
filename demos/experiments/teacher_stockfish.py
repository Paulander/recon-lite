#!/usr/bin/env python3
"""Minimal Stockfish-backed teacher to refresh macro sidecar weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import chess

try:
    import chess.engine
except Exception:  # pragma: no cover - python-chess optional extras may be missing
    chess = chess  # type: ignore[misc]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_lite.macro_engine import MacroEngine
from recon_lite_chess.sensors.structure import summarize_kpk_material

DEFAULT_SIDE_CAR = Path("weights/macro_weights.json")


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
) -> Dict[str, object]:
    fens = list(_load_fens(fen_path))
    if not fens:
        raise SystemExit(f"No FEN positions found in {fen_path}")

    counts, avg_score = label_fens(fens, engine_path, depth)

    payload = json.loads(output_path.read_text()) if output_path.exists() else {"version": "0.1"}
    apply_weight_update(payload, counts, avg_score)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Label macrograph preferences using Stockfish and update sidecar weights.")
    parser.add_argument("fen_file", type=Path, help="Path to a file containing FEN positions (one per line).")
    parser.add_argument("--output", type=Path, default=DEFAULT_SIDE_CAR, help="Sidecar JSON to update (default: weights/macro_weights.json)")
    parser.add_argument("--engine", type=str, default=None, help="Path to Stockfish binary (optional – heuristics used if omitted)")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for Stockfish analysis")
    args = parser.parse_args()

    payload = run_teacher(args.fen_file, args.output, args.engine, args.depth)
    print("Updated", args.output)
    teacher_meta = payload.get("notes", {}).get("teacher", {})
    if teacher_meta:
        print("Counts:", teacher_meta.get("labels", {}))
        print("Average score:", teacher_meta.get("avg_score"))


if __name__ == "__main__":
    main()
