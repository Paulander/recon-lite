"""
Bandit-style selection among sibling scripts.

This module implements UCB (Upper Confidence Bound) selection for choosing
among alternative child nodes under a parent. Statistics are tracked per
episode and reset at episode boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BanditArmState:
    """
    Per-arm (child) state for bandit selection.

    Attributes:
        child_id: Node id of this child
        pulls: Number of times this arm was selected
        sum_reward: Cumulative reward when this arm was active
        sum_sq_reward: Sum of squared rewards (for variance estimation)
        last_reward: Most recent reward assigned
    """

    child_id: str
    pulls: int = 0
    sum_reward: float = 0.0
    sum_sq_reward: float = 0.0
    last_reward: float = 0.0

    def mean_reward(self) -> float:
        """Compute mean reward for this arm."""
        if self.pulls == 0:
            return 0.0
        return self.sum_reward / self.pulls

    def variance(self) -> float:
        """Compute variance of rewards for this arm."""
        if self.pulls < 2:
            return 0.0
        mean = self.mean_reward()
        return (self.sum_sq_reward / self.pulls) - (mean * mean)


@dataclass
class BanditConfig:
    """
    Configuration for bandit selection.

    Attributes:
        c_explore: Exploration coefficient for UCB
        enabled: Global toggle for bandit selection
        min_pulls_before_ucb: Minimum pulls per arm before using UCB (ensures exploration)
    """

    c_explore: float = 1.0
    enabled: bool = True
    min_pulls_before_ucb: int = 1


# Type alias for bandit state structure
BanditStateDict = Dict[str, Dict[str, BanditArmState]]


def init_bandit_state(
    parent_children: Dict[str, List[str]],
) -> BanditStateDict:
    """
    Initialize bandit state for parent-child relationships.

    Args:
        parent_children: Mapping from parent node id to list of child node ids

    Returns:
        Nested dict: parent_id -> child_id -> BanditArmState
    """
    state: BanditStateDict = {}

    for parent_id, children in parent_children.items():
        state[parent_id] = {}
        for child_id in children:
            state[parent_id][child_id] = BanditArmState(child_id=child_id)

    return state


def ucb_score(
    arm: BanditArmState,
    total_pulls: int,
    c_explore: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute UCB score for an arm.

    UCB = mean_reward + c_explore * sqrt(2 * ln(N + 1) / (n + epsilon))

    Args:
        arm: The arm state
        total_pulls: Total pulls across all arms for this parent
        c_explore: Exploration coefficient
        epsilon: Small constant to avoid division by zero

    Returns:
        UCB score (higher = more attractive)
    """
    mean = arm.mean_reward()

    # Exploration bonus
    if arm.pulls == 0:
        # Infinite exploration bonus for untried arms
        return float("inf")

    exploration = c_explore * math.sqrt(
        2.0 * math.log(total_pulls + 1) / (arm.pulls + epsilon)
    )

    return mean + exploration


def choose_child(
    parent_id: str,
    state: BanditStateDict,
    c_explore_eff: float,
    config: BanditConfig,
) -> Optional[str]:
    """
    Choose a child using UCB selection.

    Args:
        parent_id: The parent node id
        state: Bandit state dict
        c_explore_eff: Effective exploration coefficient (may be modulated)
        config: Bandit configuration

    Returns:
        Child id with highest UCB score, or None if parent not in state
    """
    if not config.enabled:
        return None

    if parent_id not in state:
        return None

    arms = state[parent_id]
    if not arms:
        return None

    # Compute total pulls
    total_pulls = sum(arm.pulls for arm in arms.values())

    # Check if we need to force exploration (min pulls not met)
    for child_id, arm in arms.items():
        if arm.pulls < config.min_pulls_before_ucb:
            return child_id

    # Compute UCB scores
    scores: List[Tuple[str, float]] = []
    for child_id, arm in arms.items():
        score = ucb_score(arm, total_pulls, c_explore_eff)
        scores.append((child_id, score))

    # Return child with highest score
    if not scores:
        return None

    best_child, _ = max(scores, key=lambda x: x[1])
    return best_child


def assign_reward(
    parent_id: str,
    child_id: str,
    reward: float,
    state: BanditStateDict,
) -> bool:
    """
    Assign a reward to a specific arm.

    Args:
        parent_id: The parent node id
        child_id: The child node id that was active
        reward: The reward to assign
        state: Bandit state dict

    Returns:
        True if reward was assigned, False if parent/child not found
    """
    if parent_id not in state:
        return False

    if child_id not in state[parent_id]:
        return False

    arm = state[parent_id][child_id]
    arm.pulls += 1
    arm.sum_reward += reward
    arm.sum_sq_reward += reward * reward
    arm.last_reward = reward

    return True


def reset_bandit_episode(state: BanditStateDict) -> None:
    """
    Reset all bandit state for a new episode.

    Args:
        state: Bandit state dict to reset
    """
    for parent_id in state:
        for child_id in state[parent_id]:
            arm = state[parent_id][child_id]
            arm.pulls = 0
            arm.sum_reward = 0.0
            arm.sum_sq_reward = 0.0
            arm.last_reward = 0.0


def snapshot_bandit(state: BanditStateDict) -> Dict[str, Any]:
    """
    Create a serializable snapshot of bandit state for logging.

    Returns:
        Dict with per-parent, per-child statistics
    """
    result: Dict[str, Any] = {}

    for parent_id, arms in state.items():
        result[parent_id] = {}
        for child_id, arm in arms.items():
            if arm.pulls > 0:
                result[parent_id][child_id] = {
                    "pulls": arm.pulls,
                    "mean_reward": round(arm.mean_reward(), 4),
                    "last_reward": round(arm.last_reward, 4),
                }

    # Filter out empty parents
    return {k: v for k, v in result.items() if v}


# ---------------------------------------------------------------------------
# M4: Bandit Priors for cross-game statistics
# ---------------------------------------------------------------------------


@dataclass
class BanditPriors:
    """
    Cross-game bandit priors for initializing arm statistics.

    Stores aggregated statistics across multiple episodes that can be used
    to initialize bandit state at the start of a new game.

    Attributes:
        arm_stats: parent_id -> child_id -> {pulls, sum_reward, mean_reward}
        total_episodes: Number of episodes these priors were computed from
        decay_factor: Factor applied when merging new data (0.9 = 10% decay)
    """

    arm_stats: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    total_episodes: int = 0
    decay_factor: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_stats": self.arm_stats,
            "total_episodes": self.total_episodes,
            "decay_factor": self.decay_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BanditPriors":
        return cls(
            arm_stats=data.get("arm_stats", {}),
            total_episodes=data.get("total_episodes", 0),
            decay_factor=data.get("decay_factor", 0.9),
        )


def init_bandit_state_with_priors(
    parent_children: Dict[str, List[str]],
    priors: Optional[BanditPriors] = None,
    prior_weight: float = 0.5,
) -> BanditStateDict:
    """
    Initialize bandit state with optional priors from previous games.

    Args:
        parent_children: Mapping from parent node id to list of child node ids
        priors: Optional BanditPriors from previous games
        prior_weight: How much to weight priors (0 = ignore, 1 = full prior)

    Returns:
        Nested dict: parent_id -> child_id -> BanditArmState
    """
    state: BanditStateDict = {}

    for parent_id, children in parent_children.items():
        state[parent_id] = {}
        for child_id in children:
            arm = BanditArmState(child_id=child_id)

            # Apply priors if available
            if priors and prior_weight > 0:
                parent_priors = priors.arm_stats.get(parent_id, {})
                child_prior = parent_priors.get(child_id, {})

                if child_prior:
                    # Initialize with weighted prior statistics
                    # Use "virtual" pulls to bias initial estimates
                    prior_pulls = child_prior.get("pulls", 0)
                    prior_mean = child_prior.get("mean_reward", 0.0)

                    # Scale prior influence by weight
                    effective_pulls = int(prior_pulls * prior_weight)
                    if effective_pulls > 0:
                        arm.pulls = effective_pulls
                        arm.sum_reward = prior_mean * effective_pulls
                        arm.sum_sq_reward = (prior_mean ** 2) * effective_pulls

            state[parent_id][child_id] = arm

    return state


def export_priors(state: BanditStateDict) -> BanditPriors:
    """
    Export current bandit state as priors for future games.

    Args:
        state: Current bandit state dict

    Returns:
        BanditPriors with aggregated statistics
    """
    priors = BanditPriors()

    for parent_id, arms in state.items():
        priors.arm_stats[parent_id] = {}
        for child_id, arm in arms.items():
            if arm.pulls > 0:
                priors.arm_stats[parent_id][child_id] = {
                    "pulls": arm.pulls,
                    "sum_reward": round(arm.sum_reward, 4),
                    "mean_reward": round(arm.mean_reward(), 4),
                }

    priors.total_episodes = 1
    return priors


def merge_priors(
    old_priors: BanditPriors,
    new_priors: BanditPriors,
    decay: float = 0.9,
) -> BanditPriors:
    """
    Merge new priors into existing priors with decay.

    This allows gradual updating of cross-game statistics while
    preventing old data from dominating forever.

    Args:
        old_priors: Existing priors to update
        new_priors: New priors to merge in
        decay: Factor to apply to old statistics (0.9 = 10% decay)

    Returns:
        Merged BanditPriors
    """
    merged = BanditPriors(decay_factor=decay)

    # Get all parent/child combinations
    all_parents = set(old_priors.arm_stats.keys()) | set(new_priors.arm_stats.keys())

    for parent_id in all_parents:
        merged.arm_stats[parent_id] = {}

        old_arms = old_priors.arm_stats.get(parent_id, {})
        new_arms = new_priors.arm_stats.get(parent_id, {})
        all_children = set(old_arms.keys()) | set(new_arms.keys())

        for child_id in all_children:
            old_stats = old_arms.get(child_id, {})
            new_stats = new_arms.get(child_id, {})

            # Decay old stats and add new
            old_pulls = old_stats.get("pulls", 0) * decay
            old_sum = old_stats.get("sum_reward", 0.0) * decay

            new_pulls = new_stats.get("pulls", 0)
            new_sum = new_stats.get("sum_reward", 0.0)

            total_pulls = old_pulls + new_pulls
            total_sum = old_sum + new_sum

            if total_pulls > 0:
                merged.arm_stats[parent_id][child_id] = {
                    "pulls": round(total_pulls, 2),
                    "sum_reward": round(total_sum, 4),
                    "mean_reward": round(total_sum / total_pulls, 4),
                }

    merged.total_episodes = old_priors.total_episodes + new_priors.total_episodes
    return merged


def save_priors(priors: BanditPriors, path: "Path") -> None:
    """
    Save bandit priors to a JSON file.

    Args:
        priors: Priors to save
        path: Output file path
    """
    import json
    from pathlib import Path as PathType

    if not isinstance(path, PathType):
        path = PathType(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(priors.to_dict(), fh, indent=2)


def load_priors(path: "Path") -> BanditPriors:
    """
    Load bandit priors from a JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded BanditPriors
    """
    import json
    from pathlib import Path as PathType

    if not isinstance(path, PathType):
        path = PathType(path)

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    return BanditPriors.from_dict(data)


# ---------------------------------------------------------------------------
# Affordance-Enhanced Reward Computation
# ---------------------------------------------------------------------------


@dataclass
class AffordanceRewardConfig:
    """
    Configuration for affordance-based reward enhancement.
    
    Attributes:
        affordance_weight: Weight for affordance delta in total reward (0.0-1.0)
        base_reward_weight: Weight for base reward (typically 1.0 - affordance_weight)
        affordance_threshold: Minimum affordance delta to consider (noise filter)
        clip_range: Max magnitude for affordance contribution
    """
    affordance_weight: float = 0.3
    base_reward_weight: float = 0.7
    affordance_threshold: float = 0.05
    clip_range: float = 1.0


def compute_affordance_delta(
    affordance_before: Dict[str, float],
    affordance_after: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total and per-subgraph affordance delta.
    
    A positive delta means the action moved us closer to activating
    an endgame strategy (e.g., liquidating toward KRK).
    
    Args:
        affordance_before: Affordance values before the action
        affordance_after: Affordance values after the action
        
    Returns:
        Tuple of (total_delta, per_subgraph_deltas)
    """
    deltas: Dict[str, float] = {}
    total_delta = 0.0
    
    # Get all subgraphs mentioned in either dict
    all_subgraphs = set(affordance_before.keys()) | set(affordance_after.keys())
    
    for subgraph in all_subgraphs:
        before = affordance_before.get(subgraph, 0.0)
        after = affordance_after.get(subgraph, 0.0)
        delta = after - before
        deltas[subgraph] = delta
        total_delta += delta
    
    return total_delta, deltas


def compute_reward_with_affordance(
    base_reward: float,
    affordance_before: Dict[str, float],
    affordance_after: Dict[str, float],
    config: Optional[AffordanceRewardConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute enhanced reward that includes affordance gradient.
    
    This enables the Bandit to learn strategies that "climb the hill"
    toward endgame positions, even when material evaluation hasn't changed.
    
    Example: A move that increases KRK affordance from 0.1 to 0.4 is rewarded
    even if material balance is unchanged, because it moves toward a winning
    endgame conversion.
    
    Args:
        base_reward: The base reward from evaluation (e.g., eval_after - eval_before)
        affordance_before: Affordance signals before the action
        affordance_after: Affordance signals after the action
        config: Optional configuration
        
    Returns:
        Tuple of (enhanced_reward, breakdown_dict)
    """
    config = config or AffordanceRewardConfig()
    
    # Compute affordance delta
    total_delta, deltas = compute_affordance_delta(affordance_before, affordance_after)
    
    # Filter out noise
    if abs(total_delta) < config.affordance_threshold:
        total_delta = 0.0
    
    # Clip affordance contribution
    affordance_contrib = max(-config.clip_range, min(config.clip_range, total_delta))
    
    # Compute weighted combination
    enhanced_reward = (
        config.base_reward_weight * base_reward +
        config.affordance_weight * affordance_contrib
    )
    
    # Build breakdown for logging/debugging
    breakdown = {
        "base_reward": round(base_reward, 4),
        "affordance_delta": round(total_delta, 4),
        "affordance_contrib": round(affordance_contrib, 4),
        "enhanced_reward": round(enhanced_reward, 4),
        "per_subgraph_deltas": {k: round(v, 4) for k, v in deltas.items()},
    }
    
    return enhanced_reward, breakdown


def assign_reward_with_affordance(
    parent_id: str,
    child_id: str,
    base_reward: float,
    affordance_before: Dict[str, float],
    affordance_after: Dict[str, float],
    state: BanditStateDict,
    config: Optional[AffordanceRewardConfig] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Assign affordance-enhanced reward to a bandit arm.
    
    Convenience function that combines compute_reward_with_affordance
    and assign_reward.
    
    Args:
        parent_id: The parent node id
        child_id: The child node id that was active
        base_reward: Base reward from evaluation
        affordance_before: Affordance signals before action
        affordance_after: Affordance signals after action
        state: Bandit state dict
        config: Optional reward configuration
        
    Returns:
        Tuple of (success, breakdown_dict)
    """
    enhanced_reward, breakdown = compute_reward_with_affordance(
        base_reward, affordance_before, affordance_after, config
    )
    
    success = assign_reward(parent_id, child_id, enhanced_reward, state)
    
    return success, breakdown


def get_best_affordance_improvement(
    affordance_before: Dict[str, float],
    affordance_after: Dict[str, float],
) -> Optional[str]:
    """
    Get the subgraph with the largest affordance improvement.
    
    Useful for logging and understanding which endgame strategy
    was moved toward.
    
    Args:
        affordance_before: Affordance values before action
        affordance_after: Affordance values after action
        
    Returns:
        Name of subgraph with largest positive delta, or None if all negative
    """
    _, deltas = compute_affordance_delta(affordance_before, affordance_after)
    
    if not deltas:
        return None
    
    best_subgraph = max(deltas.keys(), key=lambda k: deltas[k])
    
    if deltas[best_subgraph] > 0:
        return best_subgraph
    
    return None
