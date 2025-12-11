"""
Unified graph building and subgraph integration for ReCoN chess.
"""

from .unified_builder import (
    build_unified_graph,
    load_all_weights,
    get_subgraph_summary,
    get_edges_for_consolidation,
    get_active_edge_traces,
    reset_edge_traces,
    TACTIC_TYPES,
)
from .subgraph_gates import (
    compute_subgraph_gates,
    compute_subgraph_affordances,
    compute_tactics_context_weights,
    MaterialInfo,
)

__all__ = [
    "build_unified_graph",
    "load_all_weights",
    "get_subgraph_summary",
    "get_edges_for_consolidation",
    "get_active_edge_traces",
    "reset_edge_traces",
    "compute_subgraph_gates",
    "compute_subgraph_affordances",
    "compute_tactics_context_weights",
    "MaterialInfo",
    "TACTIC_TYPES",
]

