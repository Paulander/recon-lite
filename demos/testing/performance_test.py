#!/usr/bin/env python3
"""
KRK ReCoN Performance Testing Script

Runs the KRK ReCoN network multiple times to measure:
- Mate success rate
- Average game length
- Average evaluation ticks
- Common failure patterns

Usage:
    uv run python demos/testing/performance_test.py --runs 100 --max-plies 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.shared.krk_network import play_krk_game, create_random_krk_board


def run_performance_test(num_runs: int = 100, max_plies: int = 50,
                        output_file: str = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Run performance test for KRK ReCoN network.

    Args:
        num_runs: Number of games to play
        max_plies: Maximum plies per game
        output_file: File to save detailed results (optional)
        verbose: Whether to print progress

    Returns:
        Dict with performance statistics
    """

    print(f"🎯 Running KRK ReCoN Performance Test")
    print(f"   Games: {num_runs}")
    print(f"   Max plies per game: {max_plies}")
    print(f"   Verbose: {verbose}")
    print()

    results = []
    start_time = time.time()

    mate_games = 0
    stalemate_games = 0
    timeout_games = 0
    rook_lost_games = 0
    stall_games = 0

    game_lengths = []
    evaluation_ticks = []

    for run in range(num_runs):
        if verbose and (run + 1) % 10 == 0:
            print(f"   Progress: {run + 1}/{num_runs} games completed...")

        # Play one game
        game_result = play_krk_game(max_plies=max_plies)

        results.append(game_result)

        # Track outcomes
        if game_result["outcome"]["checkmate"]:
            mate_games += 1
        elif game_result["outcome"]["stalemate"]:
            stalemate_games += 1
        elif game_result["plies"] >= max_plies:
            timeout_games += 1

        if game_result["rook_lost"]:
            rook_lost_games += 1

        if game_result["stalls"] > 0:
            stall_games += 1

        game_lengths.append(game_result["plies"])

    total_time = time.time() - start_time

    # Calculate statistics
    mate_rate = mate_games / num_runs * 100
    avg_game_length = sum(game_lengths) / len(game_lengths)

    stats = {
        "total_runs": num_runs,
        "mate_games": mate_games,
        "stalemate_games": stalemate_games,
        "timeout_games": timeout_games,
        "rook_lost_games": rook_lost_games,
        "stall_games": stall_games,
        "mate_rate_percent": mate_rate,
        "avg_game_length": avg_game_length,
        "total_time_seconds": total_time,
        "avg_time_per_game": total_time / num_runs,
        "results": results
    }

    return stats


def print_performance_report(stats: Dict[str, Any]):
    """Print a formatted performance report."""

    print("📊 KRK RECON PERFORMANCE REPORT")
    print("=" * 50)

    print("🎯 OVERALL PERFORMANCE:")
    print(".1f")
    print(".1f")

    print("\n📈 DETAILED BREAKDOWN:")
    print(f"   ✅ Checkmates: {stats['mate_games']} ({stats['mate_games']/stats['total_runs']*100:.1f}%)")
    print(f"   🤝 Stalemates: {stats['stalemate_games']} ({stats['stalemate_games']/stats['total_runs']*100:.1f}%)")
    print(f"   ⏰ Timeouts: {stats['timeout_games']} ({stats['timeout_games']/stats['total_runs']*100:.1f}%)")
    print(f"   🏰 Rook Lost: {stats['rook_lost_games']} ({stats['rook_lost_games']/stats['total_runs']*100:.1f}%)")
    print(f"   🚫 Stall Games: {stats['stall_games']} ({stats['stall_games']/stats['total_runs']*100:.1f}%)")

    print("\n⏱️  TIMING:")
    print(".2f")
    print(".3f")

    print("\n🔍 ANALYSIS:")
    if stats['mate_rate_percent'] >= 80:
        print("   🏆 EXCELLENT: High mate success rate!")
    elif stats['mate_rate_percent'] >= 60:
        print("   👍 GOOD: Reasonable mate success rate")
    elif stats['mate_rate_percent'] >= 40:
        print("   🤔 OK: Moderate mate success rate")
    else:
        print("   📉 NEEDS IMPROVEMENT: Low mate success rate")

    if stats['stall_games'] > stats['total_runs'] * 0.1:
        print("   ⚠️  WARNING: High stall rate detected")
    if stats['rook_lost_games'] > 0:
        print(f"   🏰 Rook safety issue: {stats['rook_lost_games']} games lost the rook")


def save_results_to_file(stats: Dict[str, Any], filename: str):
    """Save detailed results to JSON file."""
    output_path = Path("demos/outputs") / filename

    # Create outputs directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="KRK ReCoN Performance Testing")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of games to play (default: 100)")
    parser.add_argument("--max-plies", type=int, default=50,
                       help="Maximum plies per game (default: 50)")
    parser.add_argument("--output", type=str,
                       help="Output filename for detailed results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print progress updates")

    args = parser.parse_args()

    # Generate default output filename if not specified
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"krk_performance_{args.runs}runs_{timestamp}.json"

    # Run the performance test
    stats = run_performance_test(
        num_runs=args.runs,
        max_plies=args.max_plies,
        output_file=args.output,
        verbose=args.verbose
    )

    # Print report
    print_performance_report(stats)

    # Save detailed results
    save_results_to_file(stats, args.output)

    print(f"\n✨ Performance test completed! 🎯")


if __name__ == "__main__":
    main()
