#!/usr/bin/env python3
"""
Train KQK until target win rate is achieved.

This script runs batches of KQK training games and continues until
the target win rate (default 90%) is reached.

Usage:
    uv run python scripts/train_kqk_until_target.py
    uv run python scripts/train_kqk_until_target.py --target 0.95 --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_progress_bar(iteration: int, total: int, prefix: str = "", suffix: str = "", length: int = 50):
    """Print a simple progress bar."""
    if total == 0:
        return
    
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "░" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="", flush=True)
    if iteration == total:
        print()  # New line when complete


def extract_win_rate(output: str) -> float | None:
    """Extract win_rate from training output."""
    # Try multiple patterns
    patterns = [
        r"'win_rate':\s*([\d.]+)",
        r'"win_rate":\s*([\d.]+)',
        r"win_rate[^0-9]*([\d.]+)",
        r"Win rate:\s*([\d.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    # Try to parse as JSON dict
    try:
        # Look for dict-like structure
        dict_match = re.search(r"\{[^}]*['\"]win_rate['\"][^}]*\}", output)
        if dict_match:
            stats_str = dict_match.group(0).replace("'", '"')
            stats = json.loads(stats_str)
            if "win_rate" in stats:
                return float(stats["win_rate"])
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    
    return None


def run_batch(
    iteration: int,
    batch_size: int,
    consolidation_pack: Path,
    trace_dir: Path,
    show_progress: bool = True,
) -> tuple[float, dict]:
    """
    Run a batch of KQK training games.
    
    Returns:
        (win_rate, stats_dict)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = trace_dir / f"kqk_iter_{iteration}_{timestamp}.jsonl"
    trace_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "uv", "run", "python", "demos/persistent/kqk_persistent_demo.py",
        "--batch", str(batch_size),
        "--plasticity",
        "--consolidate",
        "--consolidate-pack", str(consolidation_pack),
        "--max-plies", "100",
        "--trace-out", str(trace_file),
    ]
    
    print(f"\n{'='*60}")
    print(f"Iteration {iteration}: Running {batch_size} games...")
    print(f"Command: {' '.join(cmd[:6])} ...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Run with real-time output capture
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    
    output_lines = []
    last_progress_update = time.time()
    
    # Read output line by line
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        
        if line:
            line = line.strip()
            output_lines.append(line)
            
            # Show progress indicators
            if show_progress:
                now = time.time()
                if now - last_progress_update > 2.0:  # Update every 2 seconds
                    # Look for game completion indicators
                    if "game" in line.lower() or "plies" in line.lower():
                        print(f"  Progress: {line[:80]}")
                    last_progress_update = now
    
    # Wait for process to complete
    return_code = process.wait()
    elapsed = time.time() - start_time
    
    full_output = "\n".join(output_lines)
    
    # Print final output
    print("\n" + "="*60)
    print("Training Output:")
    print("="*60)
    # Show last 30 lines
    for line in output_lines[-30:]:
        print(line)
    print("="*60)
    
    if return_code != 0:
        print(f"\n⚠️  Warning: Process exited with code {return_code}")
    
    # Extract win rate
    win_rate = extract_win_rate(full_output)
    
    # Try to extract full stats
    stats = {}
    try:
        # Look for the final stats dict
        stats_match = re.search(r"\{[^}]*'wins'[^}]*\}", full_output, re.DOTALL)
        if stats_match:
            stats_str = stats_match.group(0).replace("'", '"')
            stats = json.loads(stats_str)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    print(f"\n⏱️  Elapsed time: {elapsed:.1f}s")
    
    if win_rate is not None:
        print(f"📊 Win rate: {win_rate:.1%}")
    else:
        print("⚠️  Could not extract win rate from output")
    
    return win_rate or 0.0, stats


def main():
    parser = argparse.ArgumentParser(
        description="Train KQK until target win rate is achieved"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.90,
        help="Target win rate (default: 0.90 = 90%%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of games per batch (default: 200)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of iterations (default: 20)",
    )
    parser.add_argument(
        "--consolidation-pack",
        type=Path,
        default=Path("weights/nightly/kqk_consol.json"),
        help="Path to consolidation pack file",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("reports/kqk_training"),
        help="Directory to save trace files",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.consolidation_pack.parent.exists():
        args.consolidation_pack.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {args.consolidation_pack.parent}")
    
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("KQK Training Until Target Win Rate")
    print("="*60)
    print(f"Target win rate: {args.target:.1%}")
    print(f"Batch size: {args.batch_size} games")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Consolidation pack: {args.consolidation_pack}")
    print(f"Trace directory: {args.trace_dir}")
    print("="*60)
    print()
    
    start_time = time.time()
    total_games = 0
    win_rates = []
    
    for iteration in range(1, args.max_iterations + 1):
        win_rate, stats = run_batch(
            iteration,
            args.batch_size,
            args.consolidation_pack,
            args.trace_dir,
            show_progress=not args.no_progress,
        )
        
        total_games += args.batch_size
        win_rates.append(win_rate)
        
        # Calculate average win rate
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
        
        print(f"\n📈 Progress Summary:")
        print(f"   Iteration: {iteration}/{args.max_iterations}")
        print(f"   Current win rate: {win_rate:.1%}")
        print(f"   Average win rate: {avg_win_rate:.1%}")
        print(f"   Total games: {total_games}")
        print(f"   Target: {args.target:.1%}")
        
        if win_rate >= args.target:
            elapsed = time.time() - start_time
            print("\n" + "="*60)
            print("✅ TARGET ACHIEVED!")
            print("="*60)
            print(f"Final win rate: {win_rate:.1%}")
            print(f"Target: {args.target:.1%}")
            print(f"Total iterations: {iteration}")
            print(f"Total games: {total_games}")
            print(f"Total time: {elapsed/60:.1f} minutes")
            print(f"Consolidation pack: {args.consolidation_pack}")
            print("="*60)
            return 0
        
        if iteration < args.max_iterations:
            print(f"\n⏳ Not yet at target. Continuing to iteration {iteration + 1}...")
            time.sleep(1)  # Brief pause
    
    # Reached max iterations
    elapsed = time.time() - start_time
    final_win_rate = win_rates[-1] if win_rates else 0.0
    
    print("\n" + "="*60)
    print("⚠️  MAX ITERATIONS REACHED")
    print("="*60)
    print(f"Final win rate: {final_win_rate:.1%}")
    print(f"Target: {args.target:.1%}")
    print(f"Total iterations: {args.max_iterations}")
    print(f"Total games: {total_games}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average win rate: {avg_win_rate:.1%}")
    print("="*60)
    
    return 1


if __name__ == "__main__":
    sys.exit(main())

