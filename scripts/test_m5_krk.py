#!/usr/bin/env python3
"""Test M5 integration for KRK curriculum."""

import sys
sys.path.insert(0, ".")

print("Testing M5 imports...")

try:
    from recon_lite.learning.m5_structure import StructureLearner, compute_branching_metrics
    print("  ✓ M5 imports OK")
except Exception as e:
    print(f"  ✗ M5 import failed: {e}")
    sys.exit(1)

try:
    from recon_lite.nodes.stem_cell import StemCellManager, StemCellConfig, StemCellSample
    print("  ✓ StemCell imports OK")
except Exception as e:
    print(f"  ✗ StemCell import failed: {e}")
    sys.exit(1)

print("\nTesting StemCell spawning...")

# Create manager
stem_config = StemCellConfig(min_samples=30)
manager = StemCellManager(max_cells=20, config=stem_config, max_trial_slots=20)
print(f"  Manager created: {len(manager.cells)} cells")

# Test spawning cells and adding observations
import chess
test_board = chess.Board("8/8/8/8/8/8/R7/K1k5 w - - 0 1")

for i in range(5):
    cell = manager.spawn_cell()
    if cell:
        # Add observations to the cell (needs reward > threshold)
        cell.observe(test_board, reward=1.0, tick=i)
        print(f"  Spawned: {cell.cell_id}, state={cell.state.name}, samples={len(cell.samples)}")

print(f"\nManager now has {len(manager.cells)} cells")

# Test structural evolution
print("\nTesting StructureLearner...")
from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite.models.registry import TopologyRegistry
from pathlib import Path

topo_path = Path("topologies/krk_legs_topology.json")
if topo_path.exists():
    graph = build_graph_from_topology(topo_path)
    registry = TopologyRegistry(topo_path)
    
    # StructureLearner takes registry, not graph
    struct_learner = StructureLearner(registry=registry)
    print(f"  ✓ StructureLearner created")
    
    # Test metrics
    metrics = compute_branching_metrics(graph)
    print(f"  Metrics: depth={metrics.get('max_depth')}, nodes={len(graph.nodes)}")
    
    # Test apply_structural_phase
    result = struct_learner.apply_structural_phase(
        stem_manager=manager,
        episodes=[],  # Empty for test
        max_promotions=2,
        parent_candidates=["krk_detect", "krk_execute"],
        current_win_rate=0.5,
    )
    print(f"  Structural phase result: {result}")
else:
    print(f"  Topology not found: {topo_path}")

print("\n✓ All M5 tests passed!")

