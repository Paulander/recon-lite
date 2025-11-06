#!/usr/bin/env python3
"""
Generate a compact macrograph timeline for the visualization.

This script runs the MacroEngine over a curated set of chess positions that
exercise the opening, middlegame, and endgame (KRK/KPK/Rook techniques) views.
The resulting frames are written to `demos/outputs/macrograph_demo.json` and can
be consumed by `demos/visualization/macrograph_view.html`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import chess

from recon_lite.macro_engine import MacroEngine

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "macrograph_demo.json"

# FEN positions that highlight distinct macrograph regimes.
SCENARIOS: Sequence[Tuple[str, str]] = (
    (
        "Opening initiative: develop and fight for center.",
        chess.STARTING_BOARD_FEN,
    ),
    (
        "Middlegame tension: semi-open center with both kings castled.",
        "r2qk2r/pp1bbppp/2n1pn2/2pp4/2P1P3/1PNPBN2/PB1Q1PPP/R4RK1 w kq - 2 10",
    ),
    (
        "Simplified: queenless, minor pieces remain.",
        "2r2rk1/1p2qppp/p1n1pn2/3p4/2PP4/1P1NPN2/P3BPPP/2RQ1RK1 w - - 4 19",
    ),
    (
        "KRK endgame (classic).",
        "4k3/6K1/8/8/8/8/R7/8 w - - 0 1",
    ),
    (
        "KPK conversion attempt.",
        "8/8/2k5/8/8/6K1/4P3/8 w - - 0 1",
    ),
    (
        "Rook technique: cut-off and bridge motifs.",
        "4k3/6K1/8/8/3R4/8/8/8 w - - 0 1",
    ),
)


def capture_macro_frame(engine: MacroEngine, board: chess.Board, label: str, tick: int) -> Dict[str, object]:
    """Run a single engine step for the provided board and capture the macro frame."""
    env: Dict[str, object] = {"board": board}
    engine.step(env)
    macro = env.get("macro_frame")
    if not isinstance(macro, dict):
        raise RuntimeError("MacroEngine did not populate macro_frame.")

    return {
        "tick": tick,
        "label": label,
        "board_fen": board.fen(),
        "macro_frame": macro,
    }


def build_timeline() -> List[Dict[str, object]]:
    """Build the macrograph timeline using fresh engine instances for each scenario."""
    frames: List[Dict[str, object]] = []
    for tick, (label, fen) in enumerate(SCENARIOS):
        engine = MacroEngine()
        board = chess.Board(fen)
        frames.append(capture_macro_frame(engine, board, label, tick))
    return frames


def main() -> None:
    frames = build_timeline()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(frames, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frames)} macro frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
