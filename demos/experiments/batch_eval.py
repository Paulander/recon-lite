#!/usr/bin/env python3
"""
Batch evaluator for KRK/KPK subgraphs.

CPU-friendly by default (random opponent). Optional shallow Stockfish labeling
lets us score only positions that should be wins (e.g., KPK winning cases) and
track success against that set. Writes stats plus optional JSONL traces.

Fixed in M8: Added timeout handling per episode to prevent stalling.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import chess
try:
    import chess.engine
except Exception:  # pragma: no cover - stockfish optional
    chess = chess  # type: ignore

# Allow running as a script without -m by adding project root to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon_lite import ReConEngine
from recon_lite.graph import NodeState
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from demos.shared.krk_network import build_krk_network
from recon_lite_chess.scripts.kpk import build_kpk_network


class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    pass


@contextmanager
def timeout_handler(seconds: int):
    """Context manager for timeout handling (Unix only, graceful fallback on Windows)."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Only use signals on Unix
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: no signal-based timeout, just run without timeout
        yield


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


def run_kpk_episode(
    fen: str,
    *,
    max_plies: int = 100,
    max_ticks_per_move: int = 200,
    pack_paths: Iterable[Path] = (),
) -> Tuple[str, int, EpisodeRecord]:
    board = chess.Board(fen)
    g = build_kpk_network()
    eng = ReConEngine(g)
    g.nodes["kpk_root"].state = NodeState.REQUESTED

    tick_counter = 0
    ticks: List[TickRecord] = []
    plies = 0
    result = "unknown"

    while not board.is_game_over() and plies < max_plies:
        env = {"board": board}
        move_ticks = 0
        suggested = None

        while move_ticks < max_ticks_per_move and suggested is None:
            tick_counter += 1
            move_ticks += 1
            eng.step(env)
            suggested = env.get("kpk", {}).get("policy", {}).get("suggested_move") if isinstance(env.get("kpk"), dict) else None
            ticks.append(
                TickRecord(
                    tick_id=tick_counter,
                    board_fen=board.fen(),
                    action=suggested,
                    meta={"move_ticks": move_ticks},
                )
            )

        move_uci = suggested
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
        episode_id=f"kpk-{random.randint(0, 1_000_000)}",
        result=result,
        ticks=ticks,
        pack_meta=pack_fingerprint(pack_paths),
        notes={"plies": plies, "max_ticks_per_move": max_ticks_per_move},
    )
    return result, tick_counter, episode


def _open_engine(engine_path: Optional[Path]) -> Optional[chess.engine.SimpleEngine]:
    if engine_path is None:
        return None
    try:
        return chess.engine.SimpleEngine.popen_uci(str(engine_path))
    except Exception:
        return None


def _expected_win(engine: Optional[chess.engine.SimpleEngine], board: chess.Board, depth: int) -> bool:
    if engine is None:
        # Default assumptions: KRK and KPK are winning for side to move in our data
        return True
    try:
        info = engine.analyse(board, limit=chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score is None:
            return True
        return (score.white().score(mate_score=10000) or 0.0) > 0
    except Exception:
        return True


def evaluate_batch(
    mode: str,
    fens: List[str],
    runs: int,
    *,
    max_plies: int,
    max_ticks_per_move: int,
    pack_paths: Iterable[Path],
    trace_path: Optional[Path],
    engine_path: Optional[Path],
    depth: int,
    block_size: int,
    checkpoint_dir: Optional[Path],
    episode_timeout: int = 60,
    verbose: bool = False,
) -> dict:
    stats = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "stall": 0,
        "illegal": 0,
        "max_plies": 0,
        "timeout": 0,
        "games": 0,
        "ticks": 0,
        "expected_win_total": 0,
        "expected_win_success": 0,
        "blocks": [],
    }
    trace_db: Optional[TraceDB] = TraceDB(trace_path) if trace_path else None
    engine = _open_engine(engine_path)

    for i in range(runs):
        fen = fens[i % len(fens)]
        episode_runner = run_krk_episode if mode == "krk" else run_kpk_episode
        board = chess.Board(fen)
        expected_win = _expected_win(engine, board, depth)
        
        # Run episode with timeout protection
        try:
            with timeout_handler(episode_timeout):
                result, ticks, episode = episode_runner(
                    fen,
                    max_plies=max_plies,
                    max_ticks_per_move=max_ticks_per_move,
                    pack_paths=pack_paths,
                )
        except TimeoutError:
            result = "timeout"
            ticks = 0
            episode = EpisodeRecord(
                episode_id=f"{mode}-timeout-{random.randint(0, 1_000_000)}",
                result="timeout",
                ticks=[],
                pack_meta=pack_fingerprint(pack_paths),
                notes={"timeout": episode_timeout, "fen": fen},
            )
            if verbose:
                print(f"  Episode {i+1} timed out on FEN: {fen[:50]}...")
        
        stats["games"] += 1
        stats["ticks"] += ticks
        if result in stats:
            stats[result] += 1
        if expected_win:
            stats["expected_win_total"] += 1
            if result == "win":
                stats["expected_win_success"] += 1
        if trace_db:
            trace_db.add_episode(episode)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{runs} games...")

        if block_size > 0 and (i + 1) % block_size == 0:
            block_idx = (i + 1) // block_size
            block_entry = {
                "block": block_idx,
                "games": stats["games"],
                "wins": stats["wins"],
                "expected_win_rate": (stats["expected_win_success"] / stats["expected_win_total"]) if stats["expected_win_total"] else None,
            }
            stats["blocks"].append(block_entry)
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                for p in pack_paths:
                    if p.exists():
                        shutil.copy2(p, checkpoint_dir / f"{p.stem}_block{block_idx}{p.suffix}")

    if trace_db:
        trace_db.flush()
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluator for KRK/KPK (random opponent; optional Stockfish labeling).")
    parser.add_argument("--mode", choices=["krk", "kpk"], default="krk", help="Which subgraph to run.")
    parser.add_argument("--fen-file", type=Path, required=True, help="Path to FEN file (one per line).")
    parser.add_argument("--runs", type=int, default=50, help="Number of games to run.")
    parser.add_argument("--max-plies", type=int, default=100, help="Max plies per game.")
    parser.add_argument("--max-ticks", type=int, default=200, help="Max ticks per move.")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="Weight pack path(s) to hash and record.")
    parser.add_argument("--trace-out", type=Path, default=None, help="Optional JSONL trace output.")
    parser.add_argument("--engine", type=Path, default=None, help="Path to Stockfish binary (optional).")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth when labeling expected wins.")
    parser.add_argument("--block-size", type=int, default=0, help="Emit block summaries/checkpoints every N games (0 disables).")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to copy pack snapshots per block.")
    parser.add_argument("--episode-timeout", type=int, default=60, help="Timeout in seconds per episode (default 60).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print progress during evaluation.")
    args = parser.parse_args()

    fens = _load_fens(args.fen_file)
    if not fens:
        raise SystemExit(f"No FENs found in {args.fen_file}")

    if args.verbose:
        print(f"Starting batch evaluation: {args.runs} games, mode={args.mode}")
        print(f"Episode timeout: {args.episode_timeout}s")

    stats = evaluate_batch(
        args.mode,
        fens,
        args.runs,
        max_plies=args.max_plies,
        max_ticks_per_move=args.max_ticks,
        pack_paths=args.pack,
        trace_path=args.trace_out,
        engine_path=args.engine,
        depth=args.depth,
        block_size=args.block_size,
        checkpoint_dir=args.checkpoint_dir,
        episode_timeout=args.episode_timeout,
        verbose=args.verbose,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
