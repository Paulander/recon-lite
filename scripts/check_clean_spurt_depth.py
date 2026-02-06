#!/usr/bin/env python3
"""Check max_depth for the clean structural spurt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.models.registry import TopologyRegistry
from recon_lite.learning.m5_structure import compute_branching_metrics
from recon_lite_chess.graph.builder import build_graph_from_topology

stages = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5"]
base_dir = Path("snapshots/evolution/clean_structural_spurt")

print("="*60)
print("CLEAN STRUCTURAL SPURT - DEPTH ANALYSIS")
print("="*60)

for stage in stages:
    topo_path = base_dir / stage / "snapshots" / "cycle_0015.json"
    if topo_path.exists():
        reg = TopologyRegistry(topo_path)
        graph = build_graph_from_topology(topo_path, reg)
        metrics = compute_branching_metrics(graph)
        
        print(f"\n{stage}:")
        print(f"  Max Depth: {metrics.get('max_depth', 1)}")
        print(f"  Branching Factor: {metrics.get('branching_factor', 1.0):.2f}")
        print(f"  Non-backbone SCRIPTS: {metrics.get('non_backbone_scripts', 0)}")
        print(f"  Speculative ANDs: {metrics.get('speculative_ands', 0)}")
    else:
        print(f"\n{stage}: No final topology found")

# Check stage5 specifically for forced pruning trigger
print("\n" + "="*60)
print("FORCED PRUNING CHECK (Stage 5)")
print("="*60)

stage5_topo = base_dir / "stage5" / "snapshots" / "cycle_0015.json"
if stage5_topo.exists():
    reg = TopologyRegistry(stage5_topo)
    graph = build_graph_from_topology(stage5_topo, reg)
    metrics = compute_branching_metrics(graph)
    max_depth = metrics.get("max_depth", 1)
    
    print(f"Max Depth: {max_depth}")
    
    if max_depth <= 1:
        print("\n⚠️ FORCED PRUNING WOULD TRIGGER!")
        print("   Max depth is 1, hierarchy growth needed")
    else:
        print("\n✅ No forced pruning needed")
        print(f"   Hierarchy achieved: depth = {max_depth}")

