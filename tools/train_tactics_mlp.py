#!/usr/bin/env python3
"""
Train MLP detectors for tactical patterns.

This script trains lightweight neural networks on the features collected
by collect_tactics_features.py.

Usage:
    python tools/train_tactics_mlp.py --tactic backRankMate
    python tools/train_tactics_mlp.py --all
    python tools/train_tactics_mlp.py --tactic smotheredMate --backend pytorch

Output:
    weights/tactics_mlp/{tactic}_detector.pkl (sklearn)
    weights/tactics_mlp/{tactic}_detector.pt (pytorch)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite_chess.tactics.mlp_detector import TacticMLPDetector


# Directories
FEATURES_DIR = Path("data/tactics_training")
WEIGHTS_DIR = Path("weights/tactics_mlp")

# Supported tactics
SUPPORTED_TACTICS = ["backRankMate", "doubleCheck", "smotheredMate"]


def load_features(tactic_type: str) -> Tuple[List[List[float]], List[int]]:
    """Load features from JSONL file."""
    features_path = FEATURES_DIR / f"{tactic_type}_features.jsonl"
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            f"Run: python tools/collect_tactics_features.py --tactic {tactic_type}"
        )
    
    features = []
    labels = []
    
    with open(features_path) as f:
        for line in f:
            data = json.loads(line)
            features.append(data["features"])
            labels.append(data["label"])
    
    print(f"Loaded {len(features)} samples for {tactic_type}")
    print(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return features, labels


def train_detector(
    tactic_type: str,
    backend: str = "sklearn",
    hidden_dims: List[int] = None,
    **kwargs,
) -> Tuple[TacticMLPDetector, Dict[str, float]]:
    """
    Train a detector for a tactic type.
    
    Args:
        tactic_type: Type of tactic
        backend: "sklearn" or "pytorch"
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional training parameters
        
    Returns:
        Tuple of (trained detector, training metrics)
    """
    hidden_dims = hidden_dims or [64, 32]
    
    # Load features
    features, labels = load_features(tactic_type)
    
    if len(features) < 100:
        print(f"Warning: Only {len(features)} samples available")
    
    # Create and train detector
    detector = TacticMLPDetector(
        tactic_type=tactic_type,
        hidden_dims=hidden_dims,
    )
    
    print(f"Training {tactic_type} detector with {backend} backend...")
    print(f"  Hidden dims: {hidden_dims}")
    
    metrics = detector.train(
        features,
        labels,
        backend=backend,
        **kwargs,
    )
    
    return detector, metrics


def save_detector(
    detector: TacticMLPDetector,
    backend: str,
) -> Path:
    """Save trained detector."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    suffix = ".pkl" if backend == "sklearn" else ".pt"
    output_path = WEIGHTS_DIR / f"{detector.tactic_type}_detector{suffix}"
    
    detector.save(output_path)
    print(f"Saved detector to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train MLP tactics detectors")
    parser.add_argument(
        "--tactic",
        type=str,
        help="Specific tactic type to train",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all supported tactics",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sklearn",
        choices=["sklearn", "pytorch"],
        help="Training backend (default: sklearn)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="64,32",
        help="Hidden layer dimensions, comma-separated (default: 64,32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for pytorch (default: 100)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Max iterations for sklearn (default: 500)",
    )
    
    args = parser.parse_args()
    
    # Parse hidden dims
    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]
    
    # Determine which tactics to train
    if args.all:
        tactics = SUPPORTED_TACTICS
    elif args.tactic:
        tactics = [args.tactic]
    else:
        parser.print_help()
        return
    
    print(f"Training detectors for: {tactics}")
    print(f"Backend: {args.backend}")
    print(f"Hidden dims: {hidden_dims}")
    print()
    
    results = {}
    
    for tactic in tactics:
        try:
            detector, metrics = train_detector(
                tactic,
                backend=args.backend,
                hidden_dims=hidden_dims,
                epochs=args.epochs,
                max_iter=args.max_iter,
            )
            
            save_detector(detector, args.backend)
            results[tactic] = metrics
            
            print(f"\n{tactic} results:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print()
            
        except FileNotFoundError as e:
            print(f"Skipping {tactic}: {e}")
            continue
        except Exception as e:
            print(f"Error training {tactic}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    
    for tactic, metrics in results.items():
        acc = metrics.get("accuracy", 0)
        prec = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        print(f"{tactic}: accuracy={acc:.2%}, precision={prec:.2%}, recall={rec:.2%}")
    
    print("\nModels saved to:", WEIGHTS_DIR)


if __name__ == "__main__":
    main()

