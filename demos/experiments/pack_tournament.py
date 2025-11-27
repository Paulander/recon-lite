#!/usr/bin/env python3
"""
Simple pack tournament: evaluate multiple SWPs over a FEN suite and rank by metrics.

For now: runs batch_eval for each pack, reports wins/draws/stalls/expected_win_rate.
Stockfish labeling is optional but recommended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as a script without -m by adding project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.experiments.batch_eval import evaluate_batch, _load_fens


def run_tournament(
    packs: List[Path],
    fen_file: Path,
    *,
    mode: str,
    runs: int,
    max_plies: int,
    max_ticks: int,
    engine: Path | None,
    depth: int,
) -> Dict[str, object]:
    fens = _load_fens(fen_file)
    if not fens:
        raise SystemExit(f"No FENs found in {fen_file}")

    results = []
    for pack in packs:
        stats = evaluate_batch(
            mode,
            fens,
            runs,
            max_plies=max_plies,
            max_ticks_per_move=max_ticks,
            pack_paths=[pack],
            trace_path=None,
            engine_path=engine,
            depth=depth,
            block_size=0,
            checkpoint_dir=None,
        )
        ew_rate = stats.get("expected_win_success", 0) / stats.get("expected_win_total", 1) if stats.get("expected_win_total") else None
        results.append({
            "pack": str(pack),
            "wins": stats.get("wins"),
            "draws": stats.get("draws"),
            "stall": stats.get("stall"),
            "losses": stats.get("losses"),
            "expected_win_rate": ew_rate,
        })

    # Sort by expected_win_rate then wins
    results_sorted = sorted(
        results,
        key=lambda r: (r.get("expected_win_rate") or 0, r.get("wins") or 0),
        reverse=True,
    )
    return {"mode": mode, "fen_file": str(fen_file), "results": results_sorted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple packs and rank them.")
    parser.add_argument("--mode", choices=["krk", "kpk"], default="krk")
    parser.add_argument("--fen-file", type=Path, required=True)
    parser.add_argument("--pack", action="append", type=Path, required=True, help="Pack(s) to evaluate")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--engine", type=Path, default=None, help="Stockfish path (optional)")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = run_tournament(
        packs=args.pack,
        fen_file=args.fen_file,
        mode=args.mode,
        runs=args.runs,
        max_plies=args.max_plies,
        max_ticks=args.max_ticks,
        engine=args.engine,
        depth=args.depth,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
