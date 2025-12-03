#!/usr/bin/env python3
"""
M5.6: Benchmark evaluation for tactical and endgame suites.

Usage:
    uv run python demos/experiments/benchmark_eval.py \
        --suite data/benchmarks/tactics_suite.fen \
        --out reports/benchmarks/tactics_results.json

    # Compare two weight packs
    uv run python demos/experiments/benchmark_eval.py \
        --suite data/benchmarks/tactics_suite.fen \
        --pack-a weights/baseline.swp \
        --pack-b weights/trained.swp \
        --out reports/benchmarks/comparison.json

Metrics:
- Success rate: % of positions where best move was found
- Average rank: Where the best move appeared in move ordering
- Conversion rate: % of winning positions converted (for endgames)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess


@dataclass
class BenchmarkPosition:
    """A single benchmark position."""
    fen: str
    best_move: str
    description: str
    category: str = ""


@dataclass
class PositionResult:
    """Result of evaluating a single position."""
    fen: str
    best_move: str
    description: str
    found_best: bool
    chosen_move: Optional[str] = None
    best_move_rank: int = -1  # Where best move appeared in ordering (-1 = not found)
    eval_score: float = 0.0
    notes: str = ""


@dataclass
class BenchmarkResults:
    """Aggregate results for a benchmark suite."""
    suite_name: str
    total_positions: int
    success_count: int
    success_rate: float
    avg_best_move_rank: float
    positions: List[PositionResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_positions": self.total_positions,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_best_move_rank": self.avg_best_move_rank,
            "positions": [asdict(p) for p in self.positions],
            "metadata": self.metadata,
        }


def load_benchmark_suite(path: Path) -> List[BenchmarkPosition]:
    """
    Load benchmark positions from FEN file.
    
    Format: FEN ; best_move ; description
    Lines starting with # are comments.
    """
    positions = []
    current_category = ""
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Extract category from comment
                if ":" not in line and len(line) > 2:
                    current_category = line[1:].strip()
                continue
            
            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 3:
                positions.append(BenchmarkPosition(
                    fen=parts[0],
                    best_move=parts[1],
                    description=parts[2] if len(parts) > 2 else "",
                    category=current_category,
                ))
    
    return positions


def evaluate_position_simple(
    board: chess.Board,
    best_move_uci: str,
) -> Tuple[bool, Optional[str], int]:
    """
    Simple evaluation: check if best move is legal and find its rank.
    
    Returns:
        (found_best, chosen_move, best_move_rank)
    """
    try:
        best_move = chess.Move.from_uci(best_move_uci)
    except ValueError:
        # Try SAN notation
        try:
            best_move = board.parse_san(best_move_uci)
        except ValueError:
            return False, None, -1
    
    if best_move not in board.legal_moves:
        return False, None, -1
    
    # Simple heuristic: rank moves by capture value + check bonus
    def move_score(m: chess.Move) -> float:
        score = 0.0
        
        # Capture value
        captured = board.piece_at(m.to_square)
        if captured:
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
            score += piece_values.get(captured.piece_type, 0) * 10
        
        # Check bonus
        board.push(m)
        if board.is_check():
            score += 5
        if board.is_checkmate():
            score += 100
        board.pop()
        
        # Promotion bonus
        if m.promotion:
            score += 8
        
        return score
    
    # Rank moves
    moves_with_scores = [(m, move_score(m)) for m in board.legal_moves]
    moves_with_scores.sort(key=lambda x: -x[1])
    
    # Find best move rank
    for rank, (m, _) in enumerate(moves_with_scores):
        if m == best_move:
            # The "chosen" move would be the top-ranked
            chosen = moves_with_scores[0][0].uci() if moves_with_scores else None
            found_best = (rank == 0)
            return found_best, chosen, rank
    
    return False, None, -1


def run_benchmark(
    suite_path: Path,
    pack_path: Optional[Path] = None,
) -> BenchmarkResults:
    """
    Run a benchmark suite and return results.
    
    Args:
        suite_path: Path to benchmark FEN file
        pack_path: Optional weight pack (not used in simple eval)
    
    Returns:
        BenchmarkResults with aggregate and per-position results
    """
    positions = load_benchmark_suite(suite_path)
    
    results = []
    success_count = 0
    rank_sum = 0
    rank_count = 0
    
    for pos in positions:
        try:
            board = chess.Board(pos.fen)
        except ValueError:
            results.append(PositionResult(
                fen=pos.fen,
                best_move=pos.best_move,
                description=pos.description,
                found_best=False,
                notes="Invalid FEN",
            ))
            continue
        
        found_best, chosen, rank = evaluate_position_simple(board, pos.best_move)
        
        results.append(PositionResult(
            fen=pos.fen,
            best_move=pos.best_move,
            description=pos.description,
            found_best=found_best,
            chosen_move=chosen,
            best_move_rank=rank,
        ))
        
        if found_best:
            success_count += 1
        if rank >= 0:
            rank_sum += rank
            rank_count += 1
    
    total = len(positions)
    
    return BenchmarkResults(
        suite_name=suite_path.stem,
        total_positions=total,
        success_count=success_count,
        success_rate=success_count / total if total > 0 else 0.0,
        avg_best_move_rank=rank_sum / rank_count if rank_count > 0 else -1,
        positions=results,
        metadata={
            "suite_path": str(suite_path),
            "pack_path": str(pack_path) if pack_path else None,
        },
    )


def compare_benchmarks(results_a: BenchmarkResults, results_b: BenchmarkResults) -> Dict[str, Any]:
    """Compare two benchmark runs."""
    return {
        "suite": results_a.suite_name,
        "pack_a": results_a.metadata.get("pack_path"),
        "pack_b": results_b.metadata.get("pack_path"),
        "success_rate_a": results_a.success_rate,
        "success_rate_b": results_b.success_rate,
        "success_rate_diff": results_b.success_rate - results_a.success_rate,
        "avg_rank_a": results_a.avg_best_move_rank,
        "avg_rank_b": results_b.avg_best_move_rank,
        "avg_rank_diff": results_b.avg_best_move_rank - results_a.avg_best_move_rank,
        "improved_positions": sum(
            1 for pa, pb in zip(results_a.positions, results_b.positions)
            if not pa.found_best and pb.found_best
        ),
        "regressed_positions": sum(
            1 for pa, pb in zip(results_a.positions, results_b.positions)
            if pa.found_best and not pb.found_best
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on tactical/endgame suites."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Path to benchmark FEN file",
    )
    parser.add_argument(
        "--pack-a",
        type=Path,
        default=None,
        help="First weight pack for comparison",
    )
    parser.add_argument(
        "--pack-b",
        type=Path,
        default=None,
        help="Second weight pack for comparison",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("reports/benchmarks/results.json"),
        help="Output path for results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed results",
    )
    
    args = parser.parse_args()
    
    if not args.suite.exists():
        print(f"Error: Suite file not found: {args.suite}")
        sys.exit(1)
    
    print(f"Loading benchmark suite: {args.suite}")
    
    # Run benchmark
    if args.pack_a and args.pack_b:
        # Comparison mode
        print(f"Running comparison: {args.pack_a.name} vs {args.pack_b.name}")
        results_a = run_benchmark(args.suite, args.pack_a)
        results_b = run_benchmark(args.suite, args.pack_b)
        comparison = compare_benchmarks(results_a, results_b)
        
        output = {
            "mode": "comparison",
            "comparison": comparison,
            "results_a": results_a.to_dict(),
            "results_b": results_b.to_dict(),
        }
        
        print(f"\n=== Comparison Results ===")
        print(f"Pack A success rate: {comparison['success_rate_a']:.1%}")
        print(f"Pack B success rate: {comparison['success_rate_b']:.1%}")
        print(f"Improvement: {comparison['success_rate_diff']:+.1%}")
        print(f"Improved positions: {comparison['improved_positions']}")
        print(f"Regressed positions: {comparison['regressed_positions']}")
    else:
        # Single evaluation mode
        results = run_benchmark(args.suite, args.pack_a)
        output = {
            "mode": "single",
            "results": results.to_dict(),
        }
        
        print(f"\n=== Benchmark Results ===")
        print(f"Suite: {results.suite_name}")
        print(f"Positions: {results.total_positions}")
        print(f"Success rate: {results.success_rate:.1%} ({results.success_count}/{results.total_positions})")
        print(f"Avg best move rank: {results.avg_best_move_rank:.2f}")
        
        if args.verbose:
            print(f"\n--- Per-position results ---")
            for pos in results.positions:
                status = "✓" if pos.found_best else "✗"
                print(f"{status} {pos.description[:40]}: best={pos.best_move}, rank={pos.best_move_rank}")
    
    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

