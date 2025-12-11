#!/usr/bin/env python3
"""
Generate endgame FEN positions for curriculum training.

Uses the position generators from training/generators.py to create
random valid endgame positions.

Usage:
    # Generate 500 KPK positions
    python scripts/generate_endgame_fens.py --type kpk --count 500 -o data/endgames/kpk/random.fen
    
    # Generate 500 KQK positions
    python scripts/generate_endgame_fens.py --type kqk --count 500 -o data/endgames/kqk/random.fen
    
    # Generate KRK positions
    python scripts/generate_endgame_fens.py --type krk --count 500 -o data/endgames/krk/random.fen
    
    # Generate all endgame types
    python scripts/generate_endgame_fens.py --type all --count 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess

from recon_lite_chess.training.generators import (
    generate_krk_position,
    generate_kpk_position,
    generate_kqk_position,
)


def generate_positions(
    endgame_type: str,
    count: int,
    ensure_unique: bool = True,
) -> list[str]:
    """
    Generate a list of unique FEN positions.
    
    Args:
        endgame_type: "krk", "kpk", or "kqk"
        count: Number of positions to generate
        ensure_unique: If True, ensure no duplicate FENs
        
    Returns:
        List of FEN strings
    """
    generators = {
        "krk": generate_krk_position,
        "kpk": generate_kpk_position,
        "kqk": generate_kqk_position,
    }
    
    if endgame_type not in generators:
        raise ValueError(f"Unknown endgame type: {endgame_type}. Use: {list(generators.keys())}")
    
    generator = generators[endgame_type]
    positions: list[str] = []
    seen: set[str] = set()
    
    attempts = 0
    max_attempts = count * 10  # Allow for retries due to duplicates
    
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        try:
            board = generator()
            fen = board.fen()
            
            # Extract just the position part (without move counters) for uniqueness check
            fen_position = " ".join(fen.split()[:4])
            
            if ensure_unique and fen_position in seen:
                continue
            
            seen.add(fen_position)
            positions.append(fen)
            
        except Exception as e:
            print(f"Warning: Failed to generate position: {e}", file=sys.stderr)
            continue
    
    if len(positions) < count:
        print(f"Warning: Only generated {len(positions)}/{count} unique positions", file=sys.stderr)
    
    return positions


def write_fen_file(fens: list[str], output_path: Path) -> None:
    """Write FEN positions to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for fen in fens:
            f.write(fen + "\n")
    
    print(f"Wrote {len(fens)} positions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate endgame FEN positions for curriculum training."
    )
    parser.add_argument(
        "--type", "-t",
        choices=["krk", "kpk", "kqk", "all"],
        required=True,
        help="Endgame type to generate",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=500,
        help="Number of positions to generate (default: 500)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (default: data/endgames/<type>/random.fen)",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate positions",
    )
    
    args = parser.parse_args()
    
    if args.type == "all":
        # Generate all types
        types = ["krk", "kpk", "kqk"]
        for endgame_type in types:
            output_path = Path(f"data/endgames/{endgame_type}/random.fen")
            print(f"\nGenerating {args.count} {endgame_type.upper()} positions...")
            
            positions = generate_positions(
                endgame_type,
                args.count,
                ensure_unique=not args.allow_duplicates,
            )
            write_fen_file(positions, output_path)
    else:
        # Generate single type
        output_path = args.output or Path(f"data/endgames/{args.type}/random.fen")
        print(f"Generating {args.count} {args.type.upper()} positions...")
        
        positions = generate_positions(
            args.type,
            args.count,
            ensure_unique=not args.allow_duplicates,
        )
        write_fen_file(positions, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

