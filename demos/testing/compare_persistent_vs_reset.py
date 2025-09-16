#!/usr/bin/env python3
"""
Compare KRK ReCoN strategies: reset-per-move vs persistent state.

Runs both modes from the same starting positions and reports mate rate,
timeouts, rook losses, avg plies.
"""

import argparse
import time
import random
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.shared.krk_network import create_random_krk_board
from demos.gameplay.krk_play_demo import play_single_game
from demos.persistent.krk_persistent_demo import play_persistent_game


def run_compare(runs: int, max_plies: int) -> dict:
    seeds = [random.randint(0, 10_000_000) for _ in range(runs)]

    def make_random_fen(seed: int) -> str:
        random.seed(seed)
        return create_random_krk_board(white_to_move=True)

    def play_reset(fen: str) -> dict:
        return play_single_game(initial_fen=fen, max_plies=max_plies)

    def play_persist(fen: str) -> dict:
        return play_persistent_game(initial_fen=fen, max_plies=max_plies)

    stats = {
        "reset": {"mates": 0, "stalemates": 0, "timeouts": 0, "rook_lost": 0, "plies": []},
        "persistent": {"mates": 0, "stalemates": 0, "timeouts": 0, "rook_lost": 0, "plies": []},
    }

    t0 = time.time()
    for s in seeds:
        fen = make_random_fen(s)
        r = play_reset(fen)
        p = play_persist(fen)

        # Reset stats
        if r["checkmate"]:
            stats["reset"]["mates"] += 1
        elif r["plies"] >= max_plies:
            stats["reset"]["timeouts"] += 1
        if r["rook_lost"]:
            stats["reset"]["rook_lost"] += 1
        stats["reset"]["plies"].append(r["plies"])

        # Persistent stats
        if p["checkmate"]:
            stats["persistent"]["mates"] += 1
        elif p.get("plies", 0) >= max_plies:
            stats["persistent"]["timeouts"] += 1
        if p["rook_lost"]:
            stats["persistent"]["rook_lost"] += 1
        stats["persistent"]["plies"].append(p["plies"])

    stats["elapsed_sec"] = time.time() - t0
    stats["reset"]["avg_plies"] = sum(stats["reset"]["plies"]) / max(1, len(stats["reset"]["plies"]))
    stats["persistent"]["avg_plies"] = sum(stats["persistent"]["plies"]) / max(1, len(stats["persistent"]["plies"]))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=70)
    args = parser.parse_args()

    stats = run_compare(args.runs, args.max_plies)

    def report(label: str, s: dict):
        mates = s["mates"]
        timeouts = s["timeouts"]
        rook_lost = s["rook_lost"]
        avg_plies = s["avg_plies"]
        print(f"{label:>12}: mates={mates}/{args.runs} ({mates/args.runs*100:.1f}%), timeouts={timeouts}, rook_lost={rook_lost}, avg_plies={avg_plies:.1f}")

    print("\nKRK ReCoN Strategy Comparison")
    print("=" * 40)
    report("reset", stats["reset"])
    report("persistent", stats["persistent"])
    print(f"Elapsed: {stats['elapsed_sec']:.2f}s\n")


if __name__ == "__main__":
    main()
