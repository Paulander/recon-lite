"""
Slow Consolidation Engine (M4)

Cross-game weight consolidation that updates persistent baseline weights (w_base)
from accumulated episode summaries. This module provides both online (per-episode)
and offline (batch) consolidation modes.

Key concepts:
- w_base: Persistent baseline weight that survives across games
- Δw_fast: Per-episode fast plasticity delta (from M3), reset each game
- Consolidation: Updating w_base based on weighted average of episode deltas

The consolidation formula:
    delta_episode = outcome_score * outcome_weight + avg_reward_tick * (1 - outcome_weight)
    w_base_new = w_base + eta_consolidate * mean(edge_delta_sum * delta_episode) over episodes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..graph import Graph, LinkType
from ..trace_db import EpisodeSummary


@dataclass
class ConsolidationConfig:
    """
    Configuration for slow weight consolidation.

    Attributes:
        eta_consolidate: Cross-game learning rate (default 0.01)
        min_episodes: Minimum episodes before applying consolidation (default 10)
        outcome_weight: Weight for outcome vs tick rewards in delta_episode (default 0.5)
        max_base_delta: Maximum allowed change to w_base per consolidation (default 0.5)
        w_min: Minimum allowed w_base value (default 0.1)
        w_max: Maximum allowed w_base value (default 3.0)
        enabled: Global toggle for consolidation
    """

    eta_consolidate: float = 0.01
    min_episodes: int = 10
    outcome_weight: float = 0.5
    max_base_delta: float = 0.5
    w_min: float = 0.1
    w_max: float = 3.0
    enabled: bool = True


@dataclass
class EdgeConsolidationState:
    """
    Per-edge state for slow consolidation.

    Attributes:
        edge_key: Canonical edge identifier (e.g., "src->dst:POR")
        w_base: Persistent baseline weight
        w_init: Original weight from graph (for audit/rollback)
        accumulated_weighted_delta: Sum of (edge_delta_sum * delta_episode) across episodes
        episode_count: Number of episodes that touched this edge
    """

    edge_key: str
    w_base: float = 1.0
    w_init: float = 1.0
    accumulated_weighted_delta: float = 0.0
    episode_count: int = 0

    def mean_weighted_delta(self) -> float:
        """Compute mean weighted delta across accumulated episodes."""
        if self.episode_count == 0:
            return 0.0
        return self.accumulated_weighted_delta / self.episode_count


class ConsolidationEngine:
    """
    Engine for cross-game weight consolidation.

    Accumulates episode summaries and periodically updates w_base values
    for tracked edges.
    """

    def __init__(self, config: Optional[ConsolidationConfig] = None):
        self.config = config or ConsolidationConfig()
        self.edge_states: Dict[str, EdgeConsolidationState] = {}
        self.total_episodes: int = 0
        self.episodes_since_apply: int = 0
        self.last_apply_time: Optional[str] = None
        self.version: str = "0.1"

    def init_from_graph(
        self,
        graph: Graph,
        edge_whitelist: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize edge states from a graph.

        Args:
            graph: The ReCoN graph to extract edges from
            edge_whitelist: Optional list of edge keys to track (tracks all POR/SUB if None)
        """
        whitelist_set = set(edge_whitelist) if edge_whitelist else None

        for e in graph.edges:
            # Only track POR and SUB edges by default
            if e.ltype not in (LinkType.POR, LinkType.SUB):
                continue

            edge_key = f"{e.src}->{e.dst}:{e.ltype.name}"

            if whitelist_set is not None and edge_key not in whitelist_set:
                continue

            if edge_key in self.edge_states:
                continue  # Already initialized

            # Extract current weight
            try:
                w_val = float(e.w[0]) if hasattr(e.w, "__len__") else float(e.w)
            except Exception:
                w_val = 1.0

            self.edge_states[edge_key] = EdgeConsolidationState(
                edge_key=edge_key,
                w_base=w_val,
                w_init=w_val,
            )

    def accumulate_episode(self, summary: EpisodeSummary) -> None:
        """
        Accumulate an episode's data for future consolidation.

        Args:
            summary: Episode summary containing edge deltas, rewards, and outcome
        """
        if not self.config.enabled:
            return

        self.total_episodes += 1
        self.episodes_since_apply += 1

        # Compute delta_episode: weighted combination of outcome and avg reward
        delta_episode = (
            summary.outcome_score * self.config.outcome_weight
            + summary.avg_reward_tick * (1.0 - self.config.outcome_weight)
        )

        # Accumulate weighted deltas for each edge
        for edge_key, edge_delta_sum in summary.edge_delta_sums.items():
            if edge_key not in self.edge_states:
                # Create new state for previously unseen edge
                self.edge_states[edge_key] = EdgeConsolidationState(
                    edge_key=edge_key,
                    w_base=1.0,
                    w_init=1.0,
                )

            state = self.edge_states[edge_key]
            state.accumulated_weighted_delta += edge_delta_sum * delta_episode
            state.episode_count += 1

    def should_apply(self) -> bool:
        """Check if consolidation should be applied based on episode count."""
        return (
            self.config.enabled
            and self.episodes_since_apply >= self.config.min_episodes
        )

    def apply_to_graph(self, graph: Graph) -> Dict[str, float]:
        """
        Apply accumulated consolidation to graph weights.

        Updates w_base values and applies them to the graph. Resets
        accumulation state for the next consolidation cycle.

        Args:
            graph: The ReCoN graph to update

        Returns:
            Dict of edge_key -> delta_w_base applied
        """
        if not self.config.enabled:
            return {}

        applied_deltas: Dict[str, float] = {}

        for edge_key, state in self.edge_states.items():
            if state.episode_count == 0:
                continue

            # Compute mean weighted delta
            mean_delta = state.mean_weighted_delta()

            # Compute new w_base
            delta_w_base = self.config.eta_consolidate * mean_delta

            # Clip to max_base_delta
            delta_w_base = max(
                -self.config.max_base_delta,
                min(self.config.max_base_delta, delta_w_base),
            )

            new_w_base = state.w_base + delta_w_base

            # Clamp to bounds
            new_w_base = max(self.config.w_min, min(self.config.w_max, new_w_base))

            actual_delta = new_w_base - state.w_base

            if abs(actual_delta) > 1e-6:
                # Update state
                state.w_base = new_w_base
                applied_deltas[edge_key] = actual_delta

                # Apply to graph
                self._apply_weight_to_graph(graph, edge_key, new_w_base)

            # Reset accumulation for next cycle
            state.accumulated_weighted_delta = 0.0
            state.episode_count = 0

        self.episodes_since_apply = 0
        self.last_apply_time = datetime.now().isoformat()

        return applied_deltas

    def _apply_weight_to_graph(self, graph: Graph, edge_key: str, weight: float) -> bool:
        """Apply a weight to a specific edge in the graph."""
        # Parse edge key
        parts = edge_key.split("->")
        if len(parts) != 2:
            return False

        src = parts[0]
        dst_ltype = parts[1].split(":")
        if len(dst_ltype) != 2:
            return False

        dst = dst_ltype[0]
        ltype_name = dst_ltype[1]

        # Find and update edge
        for e in graph.edges:
            if e.src == src and e.dst == dst and e.ltype.name == ltype_name:
                e.w = weight
                return True

        return False

    def get_w_base(self, edge_key: str) -> Optional[float]:
        """Get the current w_base for an edge."""
        state = self.edge_states.get(edge_key)
        return state.w_base if state else None

    def get_all_w_base(self) -> Dict[str, float]:
        """Get all w_base values as a dict."""
        return {key: state.w_base for key, state in self.edge_states.items()}

    def save_state(self, path: Path) -> None:
        """
        Save consolidation state to a JSON file.

        Args:
            path: Output file path
        """
        data = {
            "version": self.version,
            "config": {
                "eta_consolidate": self.config.eta_consolidate,
                "min_episodes": self.config.min_episodes,
                "outcome_weight": self.config.outcome_weight,
                "max_base_delta": self.config.max_base_delta,
                "w_min": self.config.w_min,
                "w_max": self.config.w_max,
            },
            "w_base": {
                key: round(state.w_base, 6) for key, state in self.edge_states.items()
            },
            "w_init": {
                key: round(state.w_init, 6) for key, state in self.edge_states.items()
            },
            "consolidation_meta": {
                "total_episodes": self.total_episodes,
                "episodes_since_apply": self.episodes_since_apply,
                "last_apply_time": self.last_apply_time,
                "edges_tracked": len(self.edge_states),
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load_state(self, path: Path) -> None:
        """
        Load consolidation state from a JSON file.

        Args:
            path: Input file path
        """
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.version = data.get("version", "0.1")

        # Load config
        cfg = data.get("config", {})
        self.config.eta_consolidate = cfg.get("eta_consolidate", self.config.eta_consolidate)
        self.config.min_episodes = cfg.get("min_episodes", self.config.min_episodes)
        self.config.outcome_weight = cfg.get("outcome_weight", self.config.outcome_weight)
        self.config.max_base_delta = cfg.get("max_base_delta", self.config.max_base_delta)
        self.config.w_min = cfg.get("w_min", self.config.w_min)
        self.config.w_max = cfg.get("w_max", self.config.w_max)

        # Load edge states
        w_base = data.get("w_base", {})
        w_init = data.get("w_init", {})

        self.edge_states = {}
        for edge_key, base_val in w_base.items():
            self.edge_states[edge_key] = EdgeConsolidationState(
                edge_key=edge_key,
                w_base=base_val,
                w_init=w_init.get(edge_key, base_val),
            )

        # Load meta
        meta = data.get("consolidation_meta", {})
        self.total_episodes = meta.get("total_episodes", 0)
        self.episodes_since_apply = meta.get("episodes_since_apply", 0)
        self.last_apply_time = meta.get("last_apply_time")

    def export_w_base_pack(self, path: Path) -> None:
        """
        Export w_base values as a Subgraph Weight Pack (SWP).

        This format is compatible with the existing weight pack loading system.

        Args:
            path: Output file path
        """
        # Convert edge keys to POR edge format expected by weight packs
        por_edges = {}
        for edge_key, state in self.edge_states.items():
            # Parse edge key: "src->dst:LTYPE"
            parts = edge_key.split("->")
            if len(parts) != 2:
                continue
            src = parts[0]
            dst_ltype = parts[1].split(":")
            if len(dst_ltype) != 2:
                continue
            dst = dst_ltype[0]
            ltype = dst_ltype[1]

            # Only export POR edges (as expected by macro weight pack format)
            if ltype == "POR":
                por_key = f"{src}->{dst}"
                por_edges[por_key] = round(state.w_base, 4)

        data = {
            "version": self.version,
            "por_edges": por_edges,
            "consolidation_meta": {
                "total_episodes": self.total_episodes,
                "last_updated": datetime.now().isoformat(),
                "source": "consolidation_engine",
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def apply_w_base_to_graph(self, graph: Graph) -> int:
        """
        Apply all w_base values to a graph (e.g., at game start).

        This sets graph edge weights to w_base values, which then become
        the starting point for fast plasticity Δw_fast adjustments.

        Args:
            graph: The graph to update

        Returns:
            Number of edges updated
        """
        updated = 0
        for edge_key, state in self.edge_states.items():
            if self._apply_weight_to_graph(graph, edge_key, state.w_base):
                updated += 1
        return updated

    def snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot for logging."""
        return {
            "total_episodes": self.total_episodes,
            "episodes_since_apply": self.episodes_since_apply,
            "edges_tracked": len(self.edge_states),
            "w_base_sample": {
                key: round(state.w_base, 4)
                for key, state in list(self.edge_states.items())[:5]
            },
        }

