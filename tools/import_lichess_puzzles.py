#!/usr/bin/env python3
"""Import Lichess puzzles and organize by tactic type.

Downloads the Lichess puzzle database and extracts positions by theme.
Creates FEN files for training specific tactical patterns.

Usage:
    # Download and extract all themes
    uv run python tools/import_lichess_puzzles.py --download --extract-all
    
    # Extract specific themes
    uv run python tools/import_lichess_puzzles.py --themes fork pin backRankMate
    
    # Use local CSV file
    uv run python tools/import_lichess_puzzles.py --csv puzzles.csv --themes fork
    
    # Limit puzzles per theme
    uv run python tools/import_lichess_puzzles.py --themes fork --limit 1000

Lichess puzzle CSV format:
    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

Output FEN format:
    FEN ; best_move ; description
"""

import argparse
import csv
import gzip
import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
from urllib.request import urlretrieve
from urllib.error import URLError

# Lichess puzzle database URL
LICHESS_PUZZLE_DB_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_PUZZLE_CSV_URL = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"

# Themes we care about for tactical training
TACTICAL_THEMES = {
    "fork": "Fork - attack two pieces at once",
    "pin": "Pin - piece cannot move without exposing a more valuable piece",
    "skewer": "Skewer - attack through one piece to another",
    "discoveredAttack": "Discovered attack - moving one piece reveals attack from another",
    "backRankMate": "Back rank mate - checkmate on the back rank",
    "doubleCheck": "Double check - check from two pieces simultaneously",
    "smotheredMate": "Smothered mate - king trapped by own pieces",
    "deflection": "Deflection - lure defender away from duty",
    "attraction": "Attraction - lure piece to bad square",
    "interference": "Interference - block defender's line",
    "sacrifice": "Sacrifice - give up material for advantage",
    "hangingPiece": "Hanging piece - undefended piece capture",
    "trappedPiece": "Trapped piece - piece with no escape",
    "exposedKing": "Exposed king - unsafe king position",
    "quietMove": "Quiet move - strong non-forcing move",
    "zugzwang": "Zugzwang - any move worsens position",
}

# Map our internal names to Lichess theme names
THEME_ALIASES = {
    "back_rank": "backRankMate",
    "backrank": "backRankMate",
    "discovered": "discoveredAttack",
    "discovered_attack": "discoveredAttack",
    "double_check": "doubleCheck",
    "hanging": "hangingPiece",
    "trapped": "trappedPiece",
}


def download_puzzle_db(output_path: Path, verbose: bool = True) -> bool:
    """Download the Lichess puzzle database."""
    if verbose:
        print(f"Downloading puzzle database...")
        print(f"URL: {LICHESS_PUZZLE_CSV_URL}")
    
    try:
        # Try bz2 version first (smaller)
        def progress_hook(count, block_size, total_size):
            if verbose and total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\rDownloading: {pct}%", end="", flush=True)
        
        urlretrieve(LICHESS_PUZZLE_CSV_URL, str(output_path), progress_hook)
        if verbose:
            print("\nDownload complete!")
        return True
    except URLError as e:
        if verbose:
            print(f"\nDownload failed: {e}")
        return False


def parse_puzzle_line(line: str) -> Optional[Dict]:
    """Parse a single puzzle CSV line."""
    try:
        # CSV format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
        parts = line.strip().split(",")
        if len(parts) < 8:
            return None
        
        puzzle_id = parts[0]
        fen = parts[1]
        moves = parts[2].split()
        rating = int(parts[3]) if parts[3].isdigit() else 0
        themes = parts[7].split() if len(parts) > 7 else []
        
        return {
            "id": puzzle_id,
            "fen": fen,
            "moves": moves,
            "rating": rating,
            "themes": themes,
        }
    except Exception:
        return None


def extract_puzzles_by_theme(
    csv_path: Path,
    themes: Set[str],
    limit_per_theme: int = 5000,
    min_rating: int = 1000,
    max_rating: int = 2500,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Extract puzzles from CSV file grouped by theme.
    
    Returns dict mapping theme name to list of puzzle dicts.
    """
    results: Dict[str, List[Dict]] = {theme: [] for theme in themes}
    
    # Normalize theme names
    normalized_themes = set()
    theme_map = {}
    for theme in themes:
        normalized = THEME_ALIASES.get(theme.lower(), theme)
        normalized_themes.add(normalized)
        theme_map[normalized] = theme
    
    if verbose:
        print(f"Extracting themes: {', '.join(themes)}")
        print(f"Rating range: {min_rating}-{max_rating}")
        print(f"Limit per theme: {limit_per_theme}")
    
    line_count = 0
    match_count = 0
    
    # Handle different compression formats
    open_func = open
    mode = "r"
    
    if str(csv_path).endswith(".bz2"):
        import bz2
        open_func = bz2.open
        mode = "rt"
    elif str(csv_path).endswith(".gz"):
        open_func = gzip.open
        mode = "rt"
    elif str(csv_path).endswith(".zst"):
        try:
            import zstandard as zstd
            # For zstd, we need a different approach
            with open(csv_path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                reader = io.TextIOWrapper(dctx.stream_reader(fh))
                for line in reader:
                    line_count += 1
                    if line_count == 1:  # Skip header
                        continue
                    
                    puzzle = parse_puzzle_line(line)
                    if not puzzle:
                        continue
                    
                    if not (min_rating <= puzzle["rating"] <= max_rating):
                        continue
                    
                    for puzzle_theme in puzzle["themes"]:
                        if puzzle_theme in normalized_themes:
                            original_theme = theme_map[puzzle_theme]
                            if len(results[original_theme]) < limit_per_theme:
                                results[original_theme].append(puzzle)
                                match_count += 1
                    
                    if verbose and line_count % 100000 == 0:
                        print(f"\rProcessed {line_count} lines, found {match_count} matches...", end="")
                    
                    # Check if all themes are full
                    if all(len(results[t]) >= limit_per_theme for t in themes):
                        break
                
                if verbose:
                    print()
                return results
        except ImportError:
            print("Warning: zstandard not installed. Install with: pip install zstandard")
            return results
    
    with open_func(csv_path, mode) as f:
        for i, line in enumerate(f):
            line_count += 1
            
            if i == 0:  # Skip header
                continue
            
            puzzle = parse_puzzle_line(line)
            if not puzzle:
                continue
            
            if not (min_rating <= puzzle["rating"] <= max_rating):
                continue
            
            for puzzle_theme in puzzle["themes"]:
                if puzzle_theme in normalized_themes:
                    original_theme = theme_map[puzzle_theme]
                    if len(results[original_theme]) < limit_per_theme:
                        results[original_theme].append(puzzle)
                        match_count += 1
            
            if verbose and line_count % 100000 == 0:
                print(f"\rProcessed {line_count} lines, found {match_count} matches...", end="")
            
            # Check if all themes are full
            if all(len(results[t]) >= limit_per_theme for t in themes):
                break
    
    if verbose:
        print()
    
    return results


def write_fen_file(puzzles: List[Dict], output_path: Path, theme: str, verbose: bool = True):
    """Write puzzles to FEN file format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"# {theme} puzzles from Lichess\n")
        f.write(f"# Format: FEN ; best_move ; description\n")
        f.write(f"# Total puzzles: {len(puzzles)}\n\n")
        
        for puzzle in puzzles:
            fen = puzzle["fen"]
            best_move = puzzle["moves"][0] if puzzle["moves"] else "?"
            description = f"{theme} (rating: {puzzle['rating']}, id: {puzzle['id']})"
            f.write(f"{fen} ; {best_move} ; {description}\n")
    
    if verbose:
        print(f"  Wrote {len(puzzles)} puzzles to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Import Lichess puzzles and organize by tactic type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available themes:
  fork              - Attack two pieces at once
  pin               - Pin piece to more valuable piece
  skewer            - Attack through one piece to another
  discoveredAttack  - Moving piece reveals attack
  backRankMate      - Back rank checkmate
  doubleCheck       - Check from two pieces
  hangingPiece      - Capture undefended piece
  sacrifice         - Give up material for advantage
  deflection        - Lure defender away
  
Use aliases: back_rank, discovered, hanging, trapped
""",
    )
    parser.add_argument("--download", action="store_true", help="Download puzzle database")
    parser.add_argument("--csv", type=str, help="Path to local CSV file")
    parser.add_argument("--themes", nargs="+", default=[], help="Themes to extract")
    parser.add_argument("--extract-all", action="store_true", help="Extract all tactical themes")
    parser.add_argument("--limit", type=int, default=2000, help="Max puzzles per theme")
    parser.add_argument("--min-rating", type=int, default=1200, help="Minimum puzzle rating")
    parser.add_argument("--max-rating", type=int, default=2200, help="Maximum puzzle rating")
    parser.add_argument("--output-dir", type=str, default="data/puzzles", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    verbose = not args.quiet
    output_dir = Path(args.output_dir)
    
    # Determine CSV path
    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
    elif args.download:
        csv_path = output_dir / "lichess_puzzles.csv.bz2"
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not download_puzzle_db(csv_path, verbose):
                sys.exit(1)
    else:
        # Check for existing download
        for ext in [".csv", ".csv.bz2", ".csv.gz", ".csv.zst"]:
            candidate = output_dir / f"lichess_puzzles{ext}"
            if candidate.exists():
                csv_path = candidate
                break
        
        if csv_path is None:
            print("No puzzle database found. Use --download or --csv to specify input.")
            print("Example: uv run python tools/import_lichess_puzzles.py --download --extract-all")
            sys.exit(1)
    
    # Determine themes to extract
    themes: Set[str] = set()
    
    if args.extract_all:
        themes = set(TACTICAL_THEMES.keys())
    elif args.themes:
        themes = set(args.themes)
    else:
        print("No themes specified. Use --themes or --extract-all")
        print("Available themes:", ", ".join(sorted(TACTICAL_THEMES.keys())))
        sys.exit(1)
    
    if verbose:
        print(f"\n=== Lichess Puzzle Import ===")
        print(f"CSV file: {csv_path}")
        print(f"Output dir: {output_dir}")
    
    # Extract puzzles
    puzzles_by_theme = extract_puzzles_by_theme(
        csv_path,
        themes,
        limit_per_theme=args.limit,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        verbose=verbose,
    )
    
    # Write FEN files
    if verbose:
        print("\n=== Writing FEN Files ===")
    
    total_puzzles = 0
    for theme, puzzles in puzzles_by_theme.items():
        if puzzles:
            # Normalize theme name for filename
            filename = theme.lower().replace("mate", "_mate").replace("attack", "_attack")
            filename = filename.replace("piece", "_piece").replace("check", "_check")
            
            fen_path = output_dir / theme / f"lichess_{theme}.fen"
            write_fen_file(puzzles, fen_path, theme, verbose)
            total_puzzles += len(puzzles)
    
    # Create combined file
    all_puzzles = []
    for puzzles in puzzles_by_theme.values():
        all_puzzles.extend(puzzles)
    
    if all_puzzles:
        combined_path = output_dir / "combined_tactics.fen"
        write_fen_file(all_puzzles, combined_path, "combined", verbose)
    
    if verbose:
        print(f"\n=== Summary ===")
        print(f"Total puzzles extracted: {total_puzzles}")
        for theme, puzzles in sorted(puzzles_by_theme.items()):
            print(f"  {theme}: {len(puzzles)}")
        print(f"\nFiles written to: {output_dir}/")


if __name__ == "__main__":
    main()

