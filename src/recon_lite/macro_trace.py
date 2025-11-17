"""
Helpers to run the macro engine with trace logging.

This is intentionally light-weight: a single convenience to capture Tick/Episode
records with pack fingerprints for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import chess

from recon_lite.macro_engine import MacroEngine
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint
from recon_lite.graph import NodeState


def run_macro_with_trace(
    board: chess.Board,
    *,
    max_ticks: int = 200,
    pack_paths: Optional[Iterable[Path]] = None,
    trace_db: Optional[TraceDB] = None,
    episode_id: str = "macro-episode",
) -> EpisodeRecord:
    eng = MacroEngine("specs/macrograph_v0.json")
    env: Dict[str, object] = {"board": board.copy()}
    ticks: list[TickRecord] = []
    pack_meta = pack_fingerprint(pack_paths or [])

    for _ in range(max_ticks):
        now_req = eng.step(env)
        ticks.append(
            TickRecord(
                tick_id=len(ticks) + 1,
                phase_estimate=None,
                goal_vector=env.get("macro_frame", {}).get("goal_vector") if isinstance(env.get("macro_frame", {}), dict) else None,
                board_fen=env.get("board").fen() if isinstance(env.get("board"), chess.Board) else None,
                active_nodes=[nid for nid, node in eng.g.nodes.items() if node.state != NodeState.INACTIVE],
                fired_edges=[],
                action=env.get("chosen_move"),
                meta={
                    "macro_frame": env.get("macro_frame"),
                    "new_requests": list(now_req.keys()),
                },
            )
        )
        if env.get("chosen_move"):
            break

    ep = EpisodeRecord(
        episode_id=episode_id,
        result=None,
        ticks=ticks,
        pack_meta=pack_meta,
        notes={"ticks": len(ticks)},
    )
    if trace_db:
        trace_db.add_episode(ep)
    return ep

