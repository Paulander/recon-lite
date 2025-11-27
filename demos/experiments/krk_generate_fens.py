#!/usr/bin/env python3
"""
Generate random legal KRK FEN positions for training/evaluation.

Rules enforced:
- Kings not adjacent.
- Position is legal (python-chess validity check).
- Side to move is white.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Set

import chess


def generate_krk_positions(count: int) -> Set[str]:
    fens: Set[str] = set()
    squares = list(chess.SQUARES)
    while len(fens) < count:
        wk, bk, wr = random.sample(squares, 3)
        board = chess.Board(None)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
        board.turn = chess.WHITE
        if chess.square_distance(wk, bk) <= 1:
            continue
        if not board.is_valid():
            continue
        if board.is_check():
            continue
        fens.add(board.fen())
    return fens


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random KRK FENs.")
    parser.add_argument("--count", type=int, default=100, help="Number of positions to generate.")
    parser.add_argument("--output", type=Path, default=Path("data/endgames/krk/random.fen"), help="Where to write the FENs.")
    args = parser.parse_args()

    fens = generate_krk_positions(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(fens)) + "\n", encoding="utf-8")
    print(f"Wrote {len(fens)} positions to {args.output}")


if __name__ == "__main__":
    main()
