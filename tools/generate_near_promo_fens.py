#!/usr/bin/env python3
"""
Generate near-promotion KPK FENs for bridging KPK -> KQK training.

Usage:
    uv run python tools/generate_near_promo_fens.py --count 50 -o data/bridge/near_promo.fens
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recon_lite_chess.training.generators import generate_kpk_near_promotion


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate near-promotion KPK FENs")
    parser.add_argument("--count", type=int, default=20, help="Number of FENs to generate")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output FEN file")
    parser.add_argument("--include-rook-pawns", action="store_true", help="Allow rook pawns on a/h files")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for _ in range(args.count):
            board = generate_kpk_near_promotion(allow_rook_pawn=args.include_rook_pawns)
            f.write(board.fen() + "\n")

    print(f"Wrote {args.count} FENs to {args.output}")


if __name__ == "__main__":
    main()
