"""
Fast plasticity for within-game edge weight adaptation.

This module implements per-tick edge weight updates using eligibility traces and
reward signals. All changes are ephemeral (reset at episode boundaries) and
bounded to prevent runaway drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..graph import Graph, Edge, LinkType


def _edge_key(e: Edge) -> str:
    """Canonical string key for an edge."""
    return f"{e.src}->{e.dst}:{e.ltype.name}"


def _edge_key_from_parts(src: str, dst: str, ltype: LinkType) -> str:
    return f"{src}->{dst}:{ltype.name}"


@dataclass
class EdgePlasticityState:
    """
    Per-edge state for fast plasticity tracking.

    Attributes:
        src: Source node id
        dst: Destination node id
        ltype: Link type
        w_init: Initial weight at episode start (for reset)
        eligibility: Current eligibility trace value
        delta_sum: Cumulative weight change this episode
    """

    src: str
    dst: str
    ltype: LinkType
    w_init: float = 1.0
    eligibility: float = 0.0
    delta_sum: float = 0.0

    def key(self) -> str:
        return _edge_key_from_parts(self.src, self.dst, self.ltype)


@dataclass
class PlasticityConfig:
    """
    Configuration for fast plasticity updates.

    Attributes:
        eta_tick: Base per-tick learning rate
        r_max: Reward clipping bound (absolute value)
        w_min: Minimum allowed edge weight
        w_max: Maximum allowed edge weight
        lambda_decay: Eligibility trace decay factor per tick
        max_delta_episode: Maximum allowed |w - w_init| per episode (optional)
        enabled: Global toggle for plasticity
    """

    eta_tick: float = 0.05
    r_max: float = 2.0
    w_min: float = 0.1
    w_max: float = 3.0
    lambda_decay: float = 0.8
    max_delta_episode: Optional[float] = 1.0
    enabled: bool = True


def init_plasticity_state(
    graph: Graph,
    edge_whitelist: Optional[Iterable[Tuple[str, str, LinkType]]] = None,
) -> Dict[str, EdgePlasticityState]:
    """
    Initialize plasticity state for a subset of edges.

    Args:
        graph: The ReCoN graph
        edge_whitelist: If provided, only track these edges (src, dst, ltype).
                        If None, track all POR and SUB edges.

    Returns:
        Mapping from edge key to EdgePlasticityState
    """
    state: Dict[str, EdgePlasticityState] = {}

    if edge_whitelist is not None:
        whitelist_set: Set[Tuple[str, str, str]] = {
            (src, dst, ltype.name if isinstance(ltype, LinkType) else str(ltype))
            for src, dst, ltype in edge_whitelist
        }
    else:
        whitelist_set = None

    for e in graph.edges:
        # Only track POR and SUB edges by default
        if e.ltype not in (LinkType.POR, LinkType.SUB):
            continue

        if whitelist_set is not None:
            if (e.src, e.dst, e.ltype.name) not in whitelist_set:
                continue

        # Extract current weight
        try:
            w_val = float(e.w[0]) if hasattr(e.w, "__len__") else float(e.w)
        except Exception:
            w_val = 1.0

        key = _edge_key(e)
        state[key] = EdgePlasticityState(
            src=e.src,
            dst=e.dst,
            ltype=e.ltype,
            w_init=w_val,
            eligibility=0.0,
            delta_sum=0.0,
        )

    return state


def update_eligibility(
    state: Dict[str, EdgePlasticityState],
    fired_edges: Iterable[Dict[str, str]],
    lambda_decay: float,
) -> None:
    """
    Update eligibility traces: decay all, then increment for fired edges.

    Args:
        state: Plasticity state dict
        fired_edges: List of {src, dst, ltype} dicts for edges that fired this tick
        lambda_decay: Decay factor (0 = no memory, 1 = full memory)
    """
    # Decay all eligibilities
    for es in state.values():
        es.eligibility *= lambda_decay

    # Increment eligibility for fired edges
    fired_keys: Set[str] = set()
    for fe in fired_edges:
        src = fe.get("src", "")
        dst = fe.get("dst", "")
        ltype_str = fe.get("ltype", fe.get("type", ""))
        # Try to match ltype
        for lt in (LinkType.POR, LinkType.SUB, LinkType.SUR, LinkType.RET):
            if lt.name.lower() == ltype_str.lower():
                fired_keys.add(_edge_key_from_parts(src, dst, lt))
                break

    for key in fired_keys:
        if key in state:
            state[key].eligibility += 1.0


def apply_fast_update(
    state: Dict[str, EdgePlasticityState],
    graph: Graph,
    reward_tick: float,
    eta_eff: float,
    config: PlasticityConfig,
) -> Dict[str, float]:
    """
    Apply fast weight updates to eligible edges.

    Args:
        state: Plasticity state dict
        graph: The ReCoN graph (weights will be mutated)
        reward_tick: The reward signal for this tick
        eta_eff: Effective learning rate (may be modulated by goal)
        config: Plasticity configuration

    Returns:
        Dict of edge_key -> delta_w applied this tick
    """
    if not config.enabled:
        return {}

    # Clip reward
    r_clipped = max(-config.r_max, min(config.r_max, reward_tick))

    deltas: Dict[str, float] = {}

    for key, es in state.items():
        if es.eligibility == 0.0:
            continue

        # Compute delta
        delta_w = eta_eff * r_clipped * es.eligibility

        # Find the edge in the graph and update
        for e in graph.edges:
            if e.src == es.src and e.dst == es.dst and e.ltype == es.ltype:
                # Get current weight
                try:
                    w_current = float(e.w[0]) if hasattr(e.w, "__len__") else float(e.w)
                except Exception:
                    w_current = 1.0

                # Compute new weight
                w_new = w_current + delta_w

                # Clamp to bounds
                w_new = max(config.w_min, min(config.w_max, w_new))

                # Enforce max delta from init if configured
                if config.max_delta_episode is not None:
                    w_new = max(
                        es.w_init - config.max_delta_episode,
                        min(es.w_init + config.max_delta_episode, w_new),
                    )

                # Apply
                actual_delta = w_new - w_current
                e.w = w_new
                es.delta_sum += actual_delta
                deltas[key] = actual_delta
                break

    return deltas


def reset_episode(
    state: Dict[str, EdgePlasticityState],
    graph: Graph,
) -> None:
    """
    Reset all fast plasticity state and restore initial weights.

    Args:
        state: Plasticity state dict
        graph: The ReCoN graph (weights will be restored)
    """
    for key, es in state.items():
        # Restore initial weight
        for e in graph.edges:
            if e.src == es.src and e.dst == es.dst and e.ltype == es.ltype:
                e.w = es.w_init
                break

        # Reset state
        es.eligibility = 0.0
        es.delta_sum = 0.0


def snapshot_plasticity(state: Dict[str, EdgePlasticityState]) -> Dict[str, Any]:
    """
    Create a serializable snapshot of plasticity state for logging.

    Returns:
        Dict with per-edge eligibility and delta_sum
    """
    return {
        key: {
            "eligibility": round(es.eligibility, 4),
            "delta_sum": round(es.delta_sum, 4),
            "w_init": round(es.w_init, 4),
        }
        for key, es in state.items()
        if es.eligibility != 0.0 or es.delta_sum != 0.0
    }


# ---------------------------------------------------------------------------
# M4: Episode summary extraction for consolidation
# ---------------------------------------------------------------------------


def extract_episode_summary(
    plasticity_state: Optional[Dict[str, EdgePlasticityState]],
    bandit_state: Optional[Dict[str, Dict[str, Any]]],
    tick_records: Optional[List[Any]],
    result: Optional[str],
) -> "EpisodeSummary":
    """
    Extract an EpisodeSummary from episode data for M4 consolidation.

    Args:
        plasticity_state: Fast plasticity state dict (edge_key -> EdgePlasticityState)
        bandit_state: Bandit state dict (parent_id -> child_id -> BanditArmState)
        tick_records: List of TickRecord objects from the episode
        result: Game result string (e.g., "1-0", "0-1", "1/2-1/2")

    Returns:
        EpisodeSummary with aggregated episode data
    """
    # Import here to avoid circular dependency
    from ..trace_db import EpisodeSummary, BanditArmSummary, outcome_to_score

    # Extract edge delta sums from plasticity state
    edge_delta_sums: Dict[str, float] = {}
    if plasticity_state:
        for key, es in plasticity_state.items():
            if es.delta_sum != 0.0:
                edge_delta_sums[key] = es.delta_sum

    # Extract bandit statistics
    bandit_stats: Dict[str, Dict[str, BanditArmSummary]] = {}
    if bandit_state:
        for parent_id, arms in bandit_state.items():
            bandit_stats[parent_id] = {}
            for child_id, arm in arms.items():
                # Handle both BanditArmState objects and dicts
                if hasattr(arm, "pulls"):
                    pulls = arm.pulls
                    sum_reward = arm.sum_reward
                    mean_reward = arm.mean_reward() if callable(getattr(arm, "mean_reward", None)) else 0.0
                else:
                    pulls = arm.get("pulls", 0)
                    sum_reward = arm.get("sum_reward", 0.0)
                    mean_reward = sum_reward / pulls if pulls > 0 else 0.0

                if pulls > 0:
                    bandit_stats[parent_id][child_id] = BanditArmSummary(
                        pulls=pulls,
                        sum_reward=sum_reward,
                        mean_reward=mean_reward,
                    )

    # Compute reward statistics from tick records
    total_reward = 0.0
    reward_count = 0
    phase_usage: Dict[str, int] = {}

    if tick_records:
        for tick in tick_records:
            # Handle both TickRecord objects and dicts
            if hasattr(tick, "reward_tick"):
                reward_tick = tick.reward_tick
                phase = tick.phase_estimate
            else:
                reward_tick = tick.get("reward_tick")
                phase = tick.get("phase_estimate")

            if reward_tick is not None:
                total_reward += reward_tick
                reward_count += 1

            if phase:
                phase_usage[phase] = phase_usage.get(phase, 0) + 1

    avg_reward = total_reward / reward_count if reward_count > 0 else 0.0

    # Compute outcome score
    outcome_score = outcome_to_score(result)

    return EpisodeSummary(
        edge_delta_sums=edge_delta_sums,
        bandit_stats=bandit_stats,
        avg_reward_tick=avg_reward,
        total_reward_tick=total_reward,
        reward_tick_count=reward_count,
        phase_usage=phase_usage,
        outcome_score=outcome_score,
    )

