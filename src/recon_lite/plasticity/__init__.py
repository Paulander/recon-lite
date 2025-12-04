"""
Plasticity modules for within-game adaptation in ReCoN-lite.

This package provides:
- fast: Edge weight updates based on reward signals (eligibility traces, clipping)
- bandit: UCB-style selection among sibling scripts
- modulation: Goal-aware scaling of learning rate and exploration
- consolidate (M4): Cross-game weight consolidation
"""

from .fast import (
    EdgePlasticityState,
    PlasticityConfig,
    init_plasticity_state,
    update_eligibility,
    apply_fast_update,
    reset_episode,
    snapshot_plasticity,
    extract_episode_summary,
)
from .bandit import (
    BanditArmState,
    BanditConfig,
    BanditPriors,
    init_bandit_state,
    init_bandit_state_with_priors,
    ucb_score,
    choose_child,
    assign_reward,
    reset_bandit_episode,
    snapshot_bandit,
    export_priors,
    merge_priors,
    save_priors,
    load_priors,
)
from .modulation import (
    ModulationConfig,
    compute_modulators,
)
from .consolidate import (
    ConsolidationConfig,
    EdgeConsolidationState,
    ConsolidationEngine,
)

__all__ = [
    # fast
    "EdgePlasticityState",
    "PlasticityConfig",
    "init_plasticity_state",
    "update_eligibility",
    "apply_fast_update",
    "reset_episode",
    "snapshot_plasticity",
    "extract_episode_summary",
    # bandit
    "BanditArmState",
    "BanditConfig",
    "BanditPriors",
    "init_bandit_state",
    "init_bandit_state_with_priors",
    "ucb_score",
    "choose_child",
    "assign_reward",
    "reset_bandit_episode",
    "snapshot_bandit",
    "export_priors",
    "merge_priors",
    "save_priors",
    "load_priors",
    # modulation
    "ModulationConfig",
    "compute_modulators",
    # consolidate (M4)
    "ConsolidationConfig",
    "EdgeConsolidationState",
    "ConsolidationEngine",
]

