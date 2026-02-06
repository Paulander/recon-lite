"""
Training Module for ReCoN Chess.

Implements the Reverse Curriculum training strategy that trains backwards
from fixed "Anchors" (perfect endgames) to discover "Bridge" strategies
(transitions).

Phases:
1. Anchor: Perfect endgame conversion (KRK, KPK, KQK)
2. Bridge: Learn transitions from simplified middlegame
3. Wilderness: Tactical survival in complex positions
4. Integration: Full game from opening

Example:
    from recon_lite_chess.training import CurriculumManager, create_default_curriculum
    
    manager = create_default_curriculum()
    while not manager.is_complete():
        position = manager.get_training_position()
        # ... run training episode ...
        manager.record_episode_result(result)
"""

from .curriculum import (
    CurriculumPhase,
    CurriculumManager,
    PhaseStats,
    create_default_curriculum,
    create_anchor_only_curriculum,
    PHASE_ANCHOR,
    PHASE_BRIDGE,
    PHASE_WILDERNESS,
    PHASE_INTEGRATION,
)
from .generators import (
    generate_anchor_position,
    generate_krk_position,
    generate_kpk_position,
    generate_kqk_position,
    generate_bridge_position,
    generate_wilderness_position,
    generate_wilderness_from_opening,
    generate_integration_position,
    estimate_theoretical_moves,
)
from .krk_curriculum import (
    KRKStage,
    KRKStagePosition,
    KRKCurriculumManager,
    KRKStageStats,
    KRK_STAGES,
    krk_reward,
    box_min_side,
    did_box_grow,
    generate_krk_curriculum_position,
)

__all__ = [
    "CurriculumPhase",
    "CurriculumManager",
    "PhaseStats",
    "create_default_curriculum",
    "create_anchor_only_curriculum",
    "PHASE_ANCHOR",
    "PHASE_BRIDGE",
    "PHASE_WILDERNESS",
    "PHASE_INTEGRATION",
    "generate_anchor_position",
    "generate_krk_position",
    "generate_kpk_position",
    "generate_kqk_position",
    "generate_bridge_position",
    "generate_wilderness_position",
    "generate_wilderness_from_opening",
    "generate_integration_position",
    "estimate_theoretical_moves",
    # KRK Curriculum
    "KRKStage",
    "KRKStagePosition",
    "KRKCurriculumManager",
    "KRKStageStats",
    "KRK_STAGES",
    "krk_reward",
    "box_min_side",
    "did_box_grow",
    "generate_krk_curriculum_position",
]

