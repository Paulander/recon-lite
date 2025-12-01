"""
Plasticity modules for within-game adaptation in ReCoN-lite.

This package provides:
- fast: Edge weight updates based on reward signals (eligibility traces, clipping)
- bandit: UCB-style selection among sibling scripts
- modulation: Goal-aware scaling of learning rate and exploration
"""

from .fast import (
    EdgePlasticityState,
    PlasticityConfig,
    init_plasticity_state,
    update_eligibility,
    apply_fast_update,
    reset_episode,
)
from .bandit import (
    BanditArmState,
    BanditConfig,
    init_bandit_state,
    ucb_score,
    choose_child,
    assign_reward,
    reset_bandit_episode,
)
from .modulation import (
    ModulationConfig,
    compute_modulators,
)

__all__ = [
    # fast
    "EdgePlasticityState",
    "PlasticityConfig",
    "init_plasticity_state",
    "update_eligibility",
    "apply_fast_update",
    "reset_episode",
    # bandit
    "BanditArmState",
    "BanditConfig",
    "init_bandit_state",
    "ucb_score",
    "choose_child",
    "assign_reward",
    "reset_bandit_episode",
    # modulation
    "ModulationConfig",
    "compute_modulators",
]

