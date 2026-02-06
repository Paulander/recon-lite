"""M9: Pattern Recognition Layer.

This module provides position embeddings and pattern memory for 
"I've seen this before" reasoning.
"""

from .embeddings import (
    PositionEmbedding,
    PositionEncoder,
    encode_position,
    encode_positions_batch,
)
from .memory import (
    PatternMemory,
    PatternMatch,
    MemoryConfig,
)
from .boosting import (
    PatternBooster,
    PlanBoost,
    EpisodicMemory,
    Episode,
    EpisodeMatch,
)

__all__ = [
    # Embeddings
    "PositionEmbedding",
    "PositionEncoder", 
    "encode_position",
    "encode_positions_batch",
    # Memory
    "PatternMemory",
    "PatternMatch",
    "MemoryConfig",
    # Boosting
    "PatternBooster",
    "PlanBoost",
    "EpisodicMemory",
    "Episode",
    "EpisodeMatch",
]

