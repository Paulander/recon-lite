#!/usr/bin/env python3
"""
Block runner that batches evaluation, checkpoints packs, and emits per-block viz logs.

Workflow per block:
 1) Run batch_eval for N games (KRK or KPK) with optional Stockfish labeling.
 2) Copy current weight packs into a checkpoint directory.
 3) Play ONE illustrative game with the checkpointed pack and save the viz log.

This gives you block-by-block metrics and a visualization sample without manual steps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import chess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for target in (PROJECT_ROOT / "src", PROJECT_ROOT):
    target_str = str(target)
    if target_str not in sys.path:
        sys.path.append(target_str)

from demos.experiments.batch_eval import evaluate_batch, _load_fens, _random_opponent_move  # type: ignore  # pylint: disable=wrong-import-position
from demos.shared.krk_network import build_krk_network  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite import ReConEngine  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.graph import NodeState  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.logger import RunLogger  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite.trace_db import TraceDB, pack_fingerprint  # type: ignore  # pylint: disable=wrong-import-position
from recon_lite_chess.scripts.kpk import build_kpk_network  # type: ignore  # pylint: disable=wrong-import-position
from demos.persistent.krk_persistent_demo import play_persistent_game as play_krk  # type: ignore  # pylint: disable=wrong-import-position


def _copy_packs(paths: Iterable[Path], destination: Path, block_idx: int) -> List[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for p in paths:
        if not p.exists():
            continue
        target = destination / f"{p.stem}_block{block_idx}{p.suffix}"
        shutil.copy2(p, target)
        copied.append(target)
    return copied


def _play_single_game(
    mode: str,
    fen: str,
    pack_paths: Iterable[Path],
    out_json: Path,
    *,
    max_plies: int,
    max_ticks_per_move: int,
) -> None:
    if mode == "krk":
        g = build_krk_network()
        root_id = "krk_root"
        krk_mode = True
    else:
        g = build_kpk_network()
        root_id = "kpk_root"
        krk_mode = False

    eng = ReConEngine(g)
    g.nodes[root_id].state = NodeState.REQUESTED
    board = chess.Board(fen)
    logger = RunLogger()
    logger.attach_graph([
        {"src": e.src, "dst": e.dst, "type": e.ltype.name, "weight": float(getattr(e, "w", 1.0) or 1.0)}
        for e in eng.g.edges
    ])

    env = {"board": board}
    if krk_mode:
        env["chosen_move"] = None  # KRK actuators will populate this

    plies = 0
    while not board.is_game_over() and plies < max_plies:
        move_ticks = 0
        chosen: Optional[str] = None
        while move_ticks < max_ticks_per_move and chosen is None and not board.is_game_over():
            move_ticks += 1
            now_req = eng.step(env)
            chosen = env.get("chosen_move") if krk_mode else env.get("kpk", {}).get("policy", {}).get("suggested_move") if isinstance(env.get("kpk"), dict) else None
            logger.snapshot(
                engine=eng,
                note=f"tick {move_ticks}",
                env={"fen": board.fen(), "ply": plies + 1},
                new_requests=list(now_req.keys()),
                latents=env.get("phase_latents"),
            )
        if chosen is None:
            break
        try:
            board.push_uci(chosen)
        except Exception:
            break
        plies += 1
        logger.snapshot(
            engine=eng,
            note="applied_move",
            env={"fen": board.fen(), "ply": plies, "move": chosen},
            latents=env.get("phase_latents"),
        )
        if board.is_game_over():
            break
        opp = _random_opponent_move(board)
        if opp is None:
            break
        board.push(opp)
        plies += 1
        logger.snapshot(
            engine=eng,
            note="opponent_move",
            env={"fen": board.fen(), "ply": plies, "move": opp.uci()},
            latents=env.get("phase_latents"),
        )
        if krk_mode:
            env["chosen_move"] = None

    out_json.parent.mkdir(parents=True, exist_ok=True)
    logger.to_json(str(out_json))


def main(args: Optional[argparse.Namespace] = None) -> None:
    parser = argparse.ArgumentParser(description="Block runner: batch eval + checkpoints + viz sample.")
    parser.add_argument("--mode", choices=["krk", "kpk"], default="krk", help="Subgraph to run.")
    parser.add_argument("--fen-file", type=Path, required=True, help="Path to FEN file (one per line).")
    parser.add_argument("--runs-per-block", type=int, default=50, help="Games per block.")
    parser.add_argument("--blocks", type=int, default=3, help="Number of blocks.")
    parser.add_argument("--max-plies", type=int, default=100, help="Max plies per game.")
    parser.add_argument("--max-ticks", type=int, default=200, help="Max ticks per move.")
    parser.add_argument("--pack", action="append", type=Path, default=[], help="Weight pack path(s) to hash/copy.")
    parser.add_argument("--engine", type=Path, default=None, help="Optional Stockfish path.")
    parser.add_argument("--depth", type=int, default=2, help="Stockfish depth.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/blocks"), help="Output directory for stats/checkpoints/viz.")
    if args is None:
        args = parser.parse_args()

    fens = _load_fens(args.fen_file)
    if not fens:
        raise SystemExit(f"No FENs in {args.fen_file}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"mode": args.mode, "blocks": []}

    for block_idx in range(1, args.blocks + 1):
        trace_path = out_dir / f"{args.mode}_block{block_idx}_trace.jsonl"
        stats = evaluate_batch(
            args.mode,
            fens,
            args.runs_per_block,
            max_plies=args.max_plies,
            max_ticks_per_move=args.max_ticks,
            pack_paths=args.pack,
            trace_path=trace_path,
            engine_path=args.engine,
            depth=args.depth,
            block_size=args.runs_per_block,  # one summary per block
            checkpoint_dir=None,
        )
        block_dir = out_dir / f"block_{block_idx}"
        copied = _copy_packs(args.pack, block_dir, block_idx)
        # Viz sample from first FEN in this block
        viz_out = block_dir / f"{args.mode}_viz_block{block_idx}.json"
        if args.mode == "krk":
            trace_db = TraceDB(viz_out)  # reuse JSON writer to log viz events
            play_krk(
                initial_fen=fens[(block_idx - 1) % len(fens)],
                max_plies=min(40, args.max_plies),
                tick_watchdog=args.max_ticks,
                split_logs=True,
                output_basename=f"{args.mode}_block{block_idx}_viz",
                use_blended_actuator=True,
                trace_db=trace_db,
                trace_episode_id=f"{args.mode}-block{block_idx}",
                pack_paths=copied or args.pack,
            )
            trace_db.flush()
            # Move viz log into block_dir
            src_viz = Path("demos/outputs/persistent") / f"{args.mode}_block{block_idx}_viz_viz.json"
            if src_viz.exists():
                src_viz.replace(viz_out)
        else:
            _play_single_game(
                args.mode,
                fens[(block_idx - 1) % len(fens)],
                copied or args.pack,
                viz_out,
                max_plies=args.max_plies,
                max_ticks_per_move=args.max_ticks,
            )

        summary["blocks"].append(
            {
                "block": block_idx,
                "stats": stats,
                "packs": pack_fingerprint(copied or args.pack),
                "trace": str(trace_path),
                "viz": str(viz_out),
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
