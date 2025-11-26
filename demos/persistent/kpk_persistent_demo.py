#!/usr/bin/env python3
"""
Minimal persistent loop for the KPK subgraph with optional tracing.
Reuses the existing KPK network and per-move selector; no opponent policy beyond random.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite import ReConEngine  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.graph import NodeState  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.logger import RunLogger  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess.scripts.kpk import build_kpk_network  # type: ignore  # pylint: disable=wrong-import-position


def play_persistent_game(
    initial_fen: str | None = None,
    *,
    max_plies: int = 100,
    max_ticks_per_move: int = 200,
    split_logs: bool = True,
    output_basename: str = "kpk_persistent",
    stockfish_path: Optional[str] = None,
    stockfish_depth: int = 2,
    trace_db: Optional[TraceDB] = None,
    trace_episode_id: Optional[str] = None,
    pack_paths: Optional[list[Path]] = None,
) -> dict:
    viz_logger = RunLogger()
    debug_logger = RunLogger() if split_logs else viz_logger

    board = chess.Board(initial_fen) if initial_fen else chess.Board()
    g = build_kpk_network()
    eng = ReConEngine(g)
    g.nodes["kpk_root"].state = NodeState.REQUESTED

    viz_logger.attach_graph([
        {"src": e.src, "dst": e.dst, "type": e.ltype.name, "weight": float(getattr(e, "w", 1.0) or 1.0)}
        for e in eng.g.edges
    ])

    tick_records: list[TickRecord] = []
    pack_meta = pack_fingerprint(pack_paths or [])
    plies = 0
    sf_engine = None
    if stockfish_path:
        try:
            sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception:
            sf_engine = None

    while not board.is_game_over() and plies < max_plies:
        env = {"board": board}
        chosen = None
        move_ticks = 0
        while move_ticks < max_ticks_per_move and chosen is None:
            move_ticks += 1
            now_req = eng.step(env)
            chosen = env.get("kpk", {}).get("policy", {}).get("suggested_move") if isinstance(env.get("kpk"), dict) else None
            viz_logger.snapshot(
                engine=eng,
                note=f"tick {move_ticks}",
                env={"fen": board.fen(), "ply": plies + 1},
                new_requests=list(now_req.keys()),
            )
        if not chosen:
            break
        try:
            eval_before = None
            eval_after = None
            if sf_engine is not None:
                try:
                    info_before = sf_engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))
                    score_before = info_before.get("score") if info_before else None
                    eval_before = float(score_before.white().score(mate_score=10000) or 0.0) if score_before else None
                except Exception:
                    eval_before = None
            board.push_uci(chosen)
            if sf_engine is not None:
                try:
                    info_after = sf_engine.analyse(board, limit=chess.engine.Limit(depth=stockfish_depth))
                    score_after = info_after.get("score") if info_after else None
                    eval_after = float(score_after.white().score(mate_score=10000) or 0.0) if score_after else None
                except Exception:
                    eval_after = None
        except Exception:
            break
        plies += 1
        tick_records.append(
            TickRecord(
                tick_id=len(tick_records) + 1,
                board_fen=board.fen(),
                action=chosen,
                active_nodes=[nid for nid, node in eng.g.nodes.items() if node.state != NodeState.INACTIVE],
                eval_before=eval_before,
                eval_after=eval_after,
                reward_tick=(round(eval_after - eval_before, 3) if eval_after is not None and eval_before is not None else None),
                meta={"ply": plies},
            )
        )
        viz_logger.snapshot(
            engine=eng,
            note="applied_move",
            env={"fen": board.fen(), "ply": plies, "move": chosen},
        )
        if board.is_game_over():
            break
        opp_moves = list(board.legal_moves)
        if not opp_moves:
            break
        opp = random.choice(opp_moves)
        board.push(opp)
        plies += 1
        viz_logger.snapshot(
            engine=eng,
            note="opponent_move",
            env={"fen": board.fen(), "ply": plies, "move": opp.uci()},
        )

    if trace_db is not None:
        ep = EpisodeRecord(
            episode_id=trace_episode_id or "kpk-persistent",
            result=board.result() if board.is_game_over() else None,
            ticks=tick_records,
            pack_meta=pack_meta,
            notes={"plies": plies},
        )
        trace_db.add_episode(ep)
    if sf_engine is not None:
        try:
            sf_engine.quit()
        except Exception:
            pass

    out_dir = Path("demos/outputs/persistent")
    out_dir.mkdir(parents=True, exist_ok=True)
    if split_logs and debug_logger is not viz_logger:
        viz_path = out_dir / f"{output_basename}_viz.json"
        debug_path = out_dir / f"{output_basename}_debug.json"
        viz_logger.to_json(str(viz_path))
        debug_logger.to_json(str(debug_path))
    else:
        combined_path = out_dir / f"{output_basename}_visualization.json"
        viz_logger.to_json(str(combined_path))

    return {"plies": plies, "game_over": board.is_game_over(), "result": board.result(), "final_fen": board.fen()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent KPK demo with optional trace output")
    parser.add_argument("--fen", type=str, default=None)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=200)
    parser.add_argument("--engine", type=str, default=None, help="Path to Stockfish (optional)")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth when scoring moves")
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--pack", action="append", type=Path, default=[])
    args = parser.parse_args()

    trace_db = TraceDB(args.trace_out) if args.trace_out else None
    res = play_persistent_game(
        initial_fen=args.fen,
        max_plies=args.max_plies,
        max_ticks_per_move=args.max_ticks,
        trace_db=trace_db,
        pack_paths=args.pack,
        trace_episode_id="kpk-cli",
        stockfish_path=args.engine,
        stockfish_depth=args.depth,
    )
    if trace_db:
        trace_db.flush()
    print(res)


if __name__ == "__main__":
    main()
