#!/usr/bin/env python3
"""
Collect training features for MLP tactics detectors.

This script extracts features from Lichess puzzle FENs and generates
training data for the MLP detectors.

Usage:
    python tools/collect_tactics_features.py --tactic backRankMate --limit 1000
    python tools/collect_tactics_features.py --all --limit 500

Output:
    data/tactics_training/{tactic}_features.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import chess

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite_chess.tactics.features import (
    extract_tactics_features,
    TacticsFeatureVector,
)


# Mapping of tactic names to puzzle directories
TACTIC_DIRS = {
    "backRankMate": "data/puzzles/backRankMate",
    "doubleCheck": "data/puzzles/doubleCheck",
    "smotheredMate": "data/puzzles/smotheredMate",
}

# For negative samples, use these directories
NEGATIVE_DIRS = [
    "data/puzzles/fork",
    "data/puzzles/pin",
    "data/puzzles/quietMove",
]

OUTPUT_DIR = Path("data/tactics_training")


def load_fen_file(path: Path, limit: Optional[int] = None) -> List[str]:
    """Load FEN strings from a file."""
    fens = []
    if not path.exists():
        print(f"Warning: {path} not found")
        return fens
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # FEN might have extra columns, take first part
                fen = line.split(",")[0].strip()
                if fen:
                    fens.append(fen)
                    if limit and len(fens) >= limit:
                        break
    
    return fens


def collect_positive_samples(
    tactic_type: str,
    limit: int = 1000,
) -> List[Tuple[TacticsFeatureVector, int]]:
    """
    Collect positive training samples for a tactic.
    
    Args:
        tactic_type: Type of tactic
        limit: Maximum number of samples
        
    Returns:
        List of (feature_vector, label=1) tuples
    """
    samples = []
    
    tactic_dir = Path(TACTIC_DIRS.get(tactic_type, f"data/puzzles/{tactic_type}"))
    fen_files = list(tactic_dir.glob("*.fen"))
    
    if not fen_files:
        print(f"No FEN files found in {tactic_dir}")
        return samples
    
    for fen_file in fen_files:
        fens = load_fen_file(fen_file, limit=limit)
        print(f"  Loaded {len(fens)} FENs from {fen_file.name}")
        
        for fen in fens:
            if len(samples) >= limit:
                break
            
            try:
                board = chess.Board(fen)
                fv = extract_tactics_features(board, tactic_type)
                samples.append((fv, 1))
            except Exception as e:
                print(f"  Warning: Failed to process FEN: {e}")
                continue
    
    return samples[:limit]


def collect_negative_samples(
    tactic_type: str,
    limit: int = 1000,
) -> List[Tuple[TacticsFeatureVector, int]]:
    """
    Collect negative training samples (positions where tactic doesn't exist).
    
    Uses positions from other tactic types as negatives.
    """
    samples = []
    
    # Use positions from other tactics as negatives
    for neg_dir in NEGATIVE_DIRS:
        neg_path = Path(neg_dir)
        if not neg_path.exists():
            continue
        
        for fen_file in neg_path.glob("*.fen"):
            fens = load_fen_file(fen_file, limit=limit // len(NEGATIVE_DIRS))
            
            for fen in fens:
                if len(samples) >= limit:
                    break
                
                try:
                    board = chess.Board(fen)
                    fv = extract_tactics_features(board, tactic_type)
                    samples.append((fv, 0))
                except Exception:
                    continue
    
    # Also add some random positions
    random_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ]
    
    for fen in random_fens:
        if len(samples) >= limit:
            break
        try:
            board = chess.Board(fen)
            fv = extract_tactics_features(board, tactic_type)
            samples.append((fv, 0))
        except Exception:
            continue
    
    return samples[:limit]


def collect_features(
    tactic_type: str,
    limit: int = 1000,
    balance_ratio: float = 1.0,
) -> Tuple[List[List[float]], List[int]]:
    """
    Collect balanced training features for a tactic.
    
    Args:
        tactic_type: Type of tactic
        limit: Maximum positive samples
        balance_ratio: Ratio of negative to positive samples
        
    Returns:
        Tuple of (features, labels)
    """
    print(f"\nCollecting features for {tactic_type}...")
    
    # Collect positive samples
    print(f"  Collecting positive samples...")
    positive = collect_positive_samples(tactic_type, limit)
    print(f"  Got {len(positive)} positive samples")
    
    # Collect negative samples
    neg_limit = int(len(positive) * balance_ratio)
    print(f"  Collecting {neg_limit} negative samples...")
    negative = collect_negative_samples(tactic_type, neg_limit)
    print(f"  Got {len(negative)} negative samples")
    
    # Combine and shuffle
    all_samples = positive + negative
    random.shuffle(all_samples)
    
    features = [sample[0].features for sample in all_samples]
    labels = [sample[1] for sample in all_samples]
    
    return features, labels


def save_features(
    tactic_type: str,
    features: List[List[float]],
    labels: List[int],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save collected features to JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{tactic_type}_features.jsonl"
    
    with open(output_path, "w") as f:
        for feat, label in zip(features, labels):
            f.write(json.dumps({
                "features": feat,
                "label": label,
                "tactic_type": tactic_type,
            }) + "\n")
    
    print(f"  Saved {len(features)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Collect tactics training features")
    parser.add_argument(
        "--tactic",
        type=str,
        help="Specific tactic type to collect",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect for all supported tactics (backRankMate, doubleCheck, smotheredMate)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum positive samples per tactic (default: 1000)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for feature files",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    # Determine which tactics to process
    if args.all:
        tactics = list(TACTIC_DIRS.keys())
    elif args.tactic:
        tactics = [args.tactic]
    else:
        parser.print_help()
        return
    
    print(f"Collecting features for tactics: {tactics}")
    print(f"Limit per tactic: {args.limit}")
    print(f"Balance ratio: {args.balance}")
    
    for tactic in tactics:
        features, labels = collect_features(
            tactic,
            limit=args.limit,
            balance_ratio=args.balance,
        )
        
        if features:
            save_features(tactic, features, labels, output_dir)
        else:
            print(f"  No features collected for {tactic}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

