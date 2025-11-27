"""
Lightweight TraceDB for logging Tick/Episode data with pack metadata.

Intent:
- Provide a common schema for ticks and episodes so training/eval scripts can
  emit consistent JSONL logs.
- Keep the implementation tiny and dependency-free for laptop runs.
- Capture weight pack metadata (path + sha256) alongside each episode for
  provenance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pack_fingerprint(paths: Iterable[Path]) -> List[Dict[str, str]]:
    """Return [{"path": str, "sha256": str}, ...] for existing paths."""
    out = []
    for p in paths:
        try:
            real = p if p.is_absolute() else p.resolve()
            out.append({"path": str(real), "sha256": _sha256(real)})
        except FileNotFoundError:
            continue
    return out


@dataclass
class TickRecord:
    tick_id: int
    phase_estimate: Optional[str] = None
    goal_vector: Optional[Dict[str, float]] = None
    board_fen: Optional[str] = None
    active_nodes: List[str] = field(default_factory=list)
    fired_edges: List[Dict[str, str]] = field(default_factory=list)
    action: Optional[str] = None  # move UCI or actuator label
    eval_before: Optional[float] = None
    eval_after: Optional[float] = None
    reward_tick: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v not in (None, [], {})}


@dataclass
class EpisodeRecord:
    episode_id: str
    result: Optional[str] = None  # win/draw/loss or custom label
    ticks: List[TickRecord] = field(default_factory=list)
    pack_meta: List[Dict[str, str]] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "result": self.result,
            "ticks": [t.to_dict() for t in self.ticks],
            "pack_meta": list(self.pack_meta),
            "notes": {k: v for k, v in self.notes.items() if v not in (None, [], {})},
        }


class TraceDB:
    """
    Append-only JSONL trace writer. Keeps everything in memory until flush to
    minimize IO during tight loops; call `flush()` periodically or at teardown.
    """

    def __init__(self, path: Path):
        self.path = path
        self.buffer: List[EpisodeRecord] = []

    def add_episode(self, episode: EpisodeRecord) -> None:
        self.buffer.append(episode)

    def flush(self) -> None:
        if not self.buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for ep in self.buffer:
                fh.write(json.dumps(ep.to_dict()) + "\n")
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
