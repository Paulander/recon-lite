"""M6 Goal Hierarchy modules.

This package implements the goal hierarchy by time-scale:
- Ultimate goals: Win | Draw | Survive (game outcome level)
- Strategic plans: 5-15 move plans that work toward ultimate goals
"""

from .ultimate import (
    UltimateGoal,
    assess_ultimate_goal,
    ultimate_goal_predicate,
    create_ultimate_goal_node,
    build_ultimate_goal_hierarchy,
)
from .strategic import (
    StrategicPlan,
    STRATEGIC_PLANS,
    create_strategic_plan_node,
    build_strategic_layer,
)

__all__ = [
    # Ultimate goals
    "UltimateGoal",
    "assess_ultimate_goal",
    "ultimate_goal_predicate",
    "create_ultimate_goal_node",
    "build_ultimate_goal_hierarchy",
    # Strategic plans
    "StrategicPlan",
    "STRATEGIC_PLANS",
    "create_strategic_plan_node",
    "build_strategic_layer",
]

