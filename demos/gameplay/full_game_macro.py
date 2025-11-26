#!/usr/bin/env python3
"""
Full-game driver using the MacroEngine with optional Stockfish assistance.

Workflow per ply:
- MacroEngine runs until env["chosen_move"] is set or a watchdog triggers.
- If no move is chosen and Stockfish is provided, fall back to the engine.
- Opponent can be Stockfish or random.

Traces can be emitted via TraceDB (EpisodeRecord/TickRecord with pack fingerprints).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import chess
import chess.engine

from recon_lite.macro_engine import MacroEngine
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint


def _fallback_stockfish(board: chess.Board, engine: chess.engine.SimpleEngine, depth: int) -> Optional[chess.Move]:
    try:
        result = engine.play(board, limit=chess.engine.Limit(depth=depth))
        return result.move
    except Exception:
        return None


def play_full_game(
    *,
    initial_fen: Optional[str],
    max_plies: int,
    stockfish_path: Optional[Path],
    stockfish_depth: int,
    trace_db: Optional[TraceDB],
    pack_paths: Optional[list[Path]],
    episode_id: str,
) -> dict:
    board = chess.Board(initial_fen) if initial_fen else chess.Board()
    engine = MacroEngine("specs/macrograph_v0.json")
    macro_env = {
        "board": board,
        "fallback_move": True,
        "stockfish_path": str(stockfish_path) if stockfish_path else None,
    }
    sf_engine = chess.engine.SimpleEngine.popen_uci(str(stockfish_path)) if stockfish_path else None

    tick_records: list[TickRecord] = []
    pack_meta = pack_fingerprint(pack_paths or [])
    plies = 0

    try:
        while not board.is_game_over() and plies < max_plies:
            # Macro step until a move is chosen or watchdog hits
            macro_env["chosen_move"] = None
            ticks_this_ply = 0
            while ticks_this_ply < 64 and macro_env.get("chosen_move") is None:
                ticks_this_ply += 1
                now_req = engine.step(macro_env)
                tick_records.append(
                    TickRecord(
                        tick_id=len(tick_records) + 1,
                        goal_vector=macro_env.get("goal_vector"),
                        board_fen=board.fen(),
                        active_nodes=[nid for nid, node in engine.g.nodes.items() if node.state.name != "INACTIVE"],
                        fired_edges=[],
                        action=macro_env.get("chosen_move"),
                        meta={"new_requests": list(now_req.keys()), "features": macro_env.get("features")},
                    )
                )
                if macro_env.get("chosen_move"):
                    break
            move_uci = macro_env.get("chosen_move")
            if move_uci is None and sf_engine:
                mv = _fallback_stockfish(board, sf_engine, stockfish_depth)
                move_uci = mv.uci() if mv else None
            if move_uci is None:
                break
            try:
                board.push_uci(move_uci)
            except Exception:
                break
            plies += 1
            if board.is_game_over() or plies >= max_plies:
                break

            # Opponent move
            opp_move: Optional[chess.Move] = None
            if sf_engine:
                opp_move = _fallback_stockfish(board, sf_engine, stockfish_depth)
            if opp_move is None:
                moves = list(board.legal_moves)
                if moves:
                    opp_move = random.choice(moves)
            if opp_move is None:
                break
            board.push(opp_move)
            plies += 1

    finally:
        if sf_engine is not None:
            try:
                sf_engine.quit()
            except Exception:
                pass

    result = board.result() if board.is_game_over() else "*"
    if trace_db is not None:
        ep = EpisodeRecord(
            episode_id=episode_id,
            result=result,
            ticks=tick_records,
            pack_meta=pack_meta,
            notes={"plies": plies},
        )
        trace_db.add_episode(ep)

    return {"plies": plies, "result": result, "final_fen": board.fen()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-game driver using MacroEngine + optional Stockfish")
    parser.add_argument("--fen", type=str, default=None, help="Starting FEN (default: chess.STARTING_BOARD_FEN)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--engine", type=Path, default=None, help="Path to Stockfish (optional)")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth")
    parser.add_argument("--trace-out", type=Path, default=None, help="JSONL trace output (EpisodeRecord/TickRecord)")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="Weight pack paths to fingerprint")
    args = parser.parse_args()

    trace_db = TraceDB(args.trace_out) if args.trace_out else None
    res = play_full_game(
        initial_fen=args.fen,
        max_plies=args.max_plies,
        stockfish_path=args.engine,
        stockfish_depth=args.depth,
        trace_db=trace_db,
        pack_paths=args.pack,
        episode_id="macro-full-game",
    )
    if trace_db:
        trace_db.flush()
    print(res)


if __name__ == "__main__":
    main()

