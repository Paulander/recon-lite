"""
Lightweight TraceDB for logging Tick/Episode data with pack metadata.

Intent:
- Provide a common schema for ticks and episodes so training/eval scripts can
  emit consistent JSONL logs.
- Keep the implementation tiny and dependency-free for laptop runs.
- Capture weight pack metadata (path + sha256) alongside each episode for
  provenance.
- M4: Support episode summaries for cross-game consolidation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# M4: Episode Summary for consolidation
# ---------------------------------------------------------------------------


@dataclass
class BanditArmSummary:
    """Summary of a single bandit arm's performance in an episode."""

    pulls: int = 0
    sum_reward: float = 0.0
    mean_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pulls": self.pulls,
            "sum_reward": round(self.sum_reward, 4),
            "mean_reward": round(self.mean_reward, 4),
        }


@dataclass
class AffordanceCrossing:
    """Record of an affordance threshold crossing event."""
    tick: int
    subgraph: str
    direction: str  # "up" or "down"
    prev_value: float
    new_value: float
    move: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "subgraph": self.subgraph,
            "direction": self.direction,
            "prev_value": round(self.prev_value, 4),
            "new_value": round(self.new_value, 4),
            "move": self.move,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffordanceCrossing":
        return cls(
            tick=data.get("tick", 0),
            subgraph=data.get("subgraph", ""),
            direction=data.get("direction", "up"),
            prev_value=data.get("prev_value", 0.0),
            new_value=data.get("new_value", 0.0),
            move=data.get("move"),
        )


@dataclass
class EpisodeSummary:
    """
    Summary of an episode for M4 cross-game consolidation.

    Captures the key signals needed to update persistent weights:
    - Per-edge cumulative deltas from fast plasticity
    - Bandit arm statistics
    - Average reward and outcome
    - Phase usage counts
    - Affordance history and threshold crossings (for bridge discovery)
    """

    edge_delta_sums: Dict[str, float] = field(default_factory=dict)
    bandit_stats: Dict[str, Dict[str, BanditArmSummary]] = field(default_factory=dict)
    avg_reward_tick: float = 0.0
    total_reward_tick: float = 0.0
    reward_tick_count: int = 0
    phase_usage: Dict[str, int] = field(default_factory=dict)
    outcome_score: float = 0.0  # 1.0 win, 0.0 draw, -1.0 loss
    
    # Affordance tracking for implicit lookahead and bridge discovery
    affordance_history: Dict[str, List[float]] = field(default_factory=dict)
    affordance_crossings: List[AffordanceCrossing] = field(default_factory=list)
    final_affordances: Dict[str, float] = field(default_factory=dict)
    max_affordances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        bandit_dict = {}
        for parent_id, arms in self.bandit_stats.items():
            bandit_dict[parent_id] = {
                arm_id: arm.to_dict() for arm_id, arm in arms.items()
            }
        return {
            "edge_delta_sums": {k: round(v, 4) for k, v in self.edge_delta_sums.items()},
            "bandit_stats": bandit_dict,
            "avg_reward_tick": round(self.avg_reward_tick, 4),
            "total_reward_tick": round(self.total_reward_tick, 4),
            "reward_tick_count": self.reward_tick_count,
            "phase_usage": dict(self.phase_usage),
            "outcome_score": self.outcome_score,
            "affordance_history": {
                k: [round(v, 4) for v in vals] for k, vals in self.affordance_history.items()
            },
            "affordance_crossings": [c.to_dict() for c in self.affordance_crossings],
            "final_affordances": {k: round(v, 4) for k, v in self.final_affordances.items()},
            "max_affordances": {k: round(v, 4) for k, v in self.max_affordances.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeSummary":
        """Reconstruct EpisodeSummary from dict (e.g., loaded from JSON)."""
        bandit_stats = {}
        for parent_id, arms in data.get("bandit_stats", {}).items():
            bandit_stats[parent_id] = {}
            for arm_id, arm_data in arms.items():
                bandit_stats[parent_id][arm_id] = BanditArmSummary(
                    pulls=arm_data.get("pulls", 0),
                    sum_reward=arm_data.get("sum_reward", 0.0),
                    mean_reward=arm_data.get("mean_reward", 0.0),
                )
        
        # Parse affordance crossings
        crossings = [
            AffordanceCrossing.from_dict(c)
            for c in data.get("affordance_crossings", [])
        ]
        
        return cls(
            edge_delta_sums=dict(data.get("edge_delta_sums", {})),
            bandit_stats=bandit_stats,
            avg_reward_tick=data.get("avg_reward_tick", 0.0),
            total_reward_tick=data.get("total_reward_tick", 0.0),
            reward_tick_count=data.get("reward_tick_count", 0),
            phase_usage=dict(data.get("phase_usage", {})),
            outcome_score=data.get("outcome_score", 0.0),
            affordance_history=dict(data.get("affordance_history", {})),
            affordance_crossings=crossings,
            final_affordances=dict(data.get("final_affordances", {})),
            max_affordances=dict(data.get("max_affordances", {})),
        )
    
    def record_affordance(
        self,
        affordances: Dict[str, float],
        tick: int = 0,
        move: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Record affordance values for this tick.
        
        Automatically detects threshold crossings.
        
        Args:
            affordances: Current affordance values by subgraph
            tick: Current tick number
            move: Move that was played (if any)
            threshold: Threshold for crossing detection
        """
        for subgraph, value in affordances.items():
            # Initialize history if needed
            if subgraph not in self.affordance_history:
                self.affordance_history[subgraph] = []
            
            history = self.affordance_history[subgraph]
            prev_value = history[-1] if history else 0.0
            
            # Record value
            history.append(value)
            
            # Update max
            if subgraph not in self.max_affordances:
                self.max_affordances[subgraph] = value
            else:
                self.max_affordances[subgraph] = max(self.max_affordances[subgraph], value)
            
            # Update final
            self.final_affordances[subgraph] = value
            
            # Check for crossing
            crossed_up = prev_value < threshold <= value
            crossed_down = prev_value >= threshold > value
            
            if crossed_up or crossed_down:
                self.affordance_crossings.append(AffordanceCrossing(
                    tick=tick,
                    subgraph=subgraph,
                    direction="up" if crossed_up else "down",
                    prev_value=prev_value,
                    new_value=value,
                    move=move,
                ))
    
    def get_affordance_summary(self) -> Dict[str, Any]:
        """Get a summary of affordance activity in this episode."""
        return {
            "total_crossings": len(self.affordance_crossings),
            "crossings_by_subgraph": self._count_crossings_by_subgraph(),
            "final_affordances": dict(self.final_affordances),
            "max_affordances": dict(self.max_affordances),
            "reached_endgame": any(v >= 1.0 for v in self.final_affordances.values()),
        }
    
    def _count_crossings_by_subgraph(self) -> Dict[str, int]:
        """Count crossings per subgraph."""
        counts: Dict[str, int] = {}
        for crossing in self.affordance_crossings:
            counts[crossing.subgraph] = counts.get(crossing.subgraph, 0) + 1
        return counts


def outcome_to_score(result: Optional[str]) -> float:
    """
    Convert game result string to numeric outcome score.

    Args:
        result: Chess result string like "1-0", "0-1", "1/2-1/2", or None

    Returns:
        1.0 for white win, -1.0 for black win, 0.0 for draw/unknown
    """
    if result is None:
        return 0.0
    result = result.strip()
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    elif result in ("1/2-1/2", "draw"):
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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
    summary: Optional[EpisodeSummary] = None  # M4: consolidation summary

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "episode_id": self.episode_id,
            "result": self.result,
            "ticks": [t.to_dict() for t in self.ticks],
            "pack_meta": list(self.pack_meta),
            "notes": {k: v for k, v in self.notes.items() if v not in (None, [], {})},
        }
        if self.summary is not None:
            out["summary"] = self.summary.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeRecord":
        """Reconstruct EpisodeRecord from dict (e.g., loaded from JSONL)."""
        ticks = [
            TickRecord(**{k: v for k, v in t.items()})
            for t in data.get("ticks", [])
        ]
        summary = None
        if "summary" in data and data["summary"]:
            summary = EpisodeSummary.from_dict(data["summary"])
        return cls(
            episode_id=data.get("episode_id", ""),
            result=data.get("result"),
            ticks=ticks,
            pack_meta=data.get("pack_meta", []),
            notes=data.get("notes", {}),
            summary=summary,
        )


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

    @staticmethod
    def load_episodes(path: Path) -> List[EpisodeRecord]:
        """
        Load all episodes from a JSONL trace file.

        Args:
            path: Path to the JSONL file

        Returns:
            List of EpisodeRecord objects
        """
        episodes = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                episodes.append(EpisodeRecord.from_dict(data))
        return episodes
