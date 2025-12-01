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

