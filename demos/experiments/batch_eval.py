#!/usr/bin/env python3
"""
Batch evaluator for KRK/KPK subgraphs.

CPU-friendly: uses the existing scripted KRK network against a random opponent.
Stockfish integration and KPK support are earmarked for follow-ups; the harness
already accepts pack metadata and writes JSONL traces for later analysis.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chess

from recon_lite import ReConEngine
from recon_lite.graph import NodeState
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from demos.shared.krk_network import build_krk_network


def _load_fens(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _random_opponent_move(board: chess.Board) -> Optional[chess.Move]:
    moves = list(board.legal_moves)
    if not moves:
        return None
    return random.choice(moves)


def run_krk_episode(
    fen: str,
    *,
    max_plies: int = 100,
    max_ticks_per_move: int = 200,
    pack_paths: Iterable[Path] = (),
) -> Tuple[str, int, EpisodeRecord]:
    board = chess.Board(fen)
    g = build_krk_network()
    eng = ReConEngine(g)
    g.nodes["krk_root"].state = NodeState.REQUESTED
    tick_counter = 0
    ticks: List[TickRecord] = []

    result = "unknown"
    plies = 0

    while not board.is_game_over() and plies < max_plies:
        env = {"board": board, "chosen_move": None}
        move_ticks = 0

        while move_ticks < max_ticks_per_move and env.get("chosen_move") is None:
            tick_counter += 1
            move_ticks += 1
            eng.step(env)
            ticks.append(
                TickRecord(
                    tick_id=tick_counter,
                    board_fen=board.fen(),
                    action=env.get("chosen_move"),
                    meta={"move_ticks": move_ticks},
                )
            )

        move_uci = env.get("chosen_move")
        if not move_uci:
            result = "stall"
            break

        try:
            board.push_uci(move_uci)
        except Exception:
            result = "illegal"
            break

        plies += 1
        if board.is_checkmate():
            result = "win"
            break
        if board.is_game_over():
            break

        # Opponent (random) move
        opp = _random_opponent_move(board)
        if opp is None:
            result = "win"
            break
        board.push(opp)
        plies += 1
        if board.is_checkmate():
            result = "loss"
            break

    if result == "unknown":
        if board.is_checkmate():
            result = "win"
        elif board.is_stalemate():
            result = "draw"
        else:
            result = "max_plies"

    episode = EpisodeRecord(
        episode_id=f"krk-{random.randint(0, 1_000_000)}",
        result=result,
        ticks=ticks,
        pack_meta=pack_fingerprint(pack_paths),
        notes={"plies": plies, "max_ticks_per_move": max_ticks_per_move},
    )
    return result, tick_counter, episode


def evaluate_krk(
    fens: List[str],
    runs: int,
    *,
    max_plies: int,
    max_ticks_per_move: int,
    pack_paths: Iterable[Path],
    trace_path: Optional[Path],
) -> dict:
    stats = {"wins": 0, "losses": 0, "draws": 0, "stall": 0, "illegal": 0, "max_plies": 0, "games": 0, "ticks": 0}
    trace_db: Optional[TraceDB] = TraceDB(trace_path) if trace_path else None

    for i in range(runs):
        fen = fens[i % len(fens)]
        result, ticks, episode = run_krk_episode(
            fen,
            max_plies=max_plies,
            max_ticks_per_move=max_ticks_per_move,
            pack_paths=pack_paths,
        )
        stats["games"] += 1
        stats["ticks"] += ticks
        if result in stats:
            stats[result] += 1
        if trace_db:
            trace_db.add_episode(episode)

    if trace_db:
        trace_db.flush()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluator for KRK (random opponent).")
    parser.add_argument("--fen-file", type=Path, required=True, help="Path to FEN file (one per line).")
    parser.add_argument("--runs", type=int, default=50, help="Number of games to run.")
    parser.add_argument("--max-plies", type=int, default=100, help="Max plies per game.")
    parser.add_argument("--max-ticks", type=int, default=200, help="Max ticks per move.")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="Weight pack path(s) to hash and record.")
    parser.add_argument("--trace-out", type=Path, default=None, help="Optional JSONL trace output.")
    args = parser.parse_args()

    fens = _load_fens(args.fen_file)
    if not fens:
        raise SystemExit(f"No FENs found in {args.fen_file}")

    stats = evaluate_krk(
        fens,
        args.runs,
        max_plies=args.max_plies,
        max_ticks_per_move=args.max_ticks,
        pack_paths=args.pack,
        trace_path=args.trace_out,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
