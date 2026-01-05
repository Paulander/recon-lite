#!/usr/bin/env python3
"""Quick test of KRK evolution with transfer tracking."""

import sys
from pathlib import Path
sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from krk_evolution_driver import KRKEvolutionConfig, run_krk_evolution

# Quick test config
config = KRKEvolutionConfig(
    topology_path=Path("topologies/krk_legs_topology.json"),
    output_dir=Path("snapshots/krk_tracking_test"),
    max_cycles=2,
    games_per_cycle=10,  # Small for quick test
    transfer_from=Path("snapshots/krk_bridge_experiment/20260105_163449/experienced_hector/snapshots/stem_cells.json"),
    transfer_top_n=10,  # Transfer top 10 cells
)

print("=" * 60)
print("KRK TRANSFER TRACKING TEST")
print("=" * 60)
print(f"Transfer from: {config.transfer_from}")
print(f"Transfer top N: {config.transfer_top_n}")
print()

# Run a quick test
try:
    results = run_krk_evolution(config)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"Cycle {r.cycle}: Win={r.win_rate:.1%}, Transfer={r.sensor_reuse_ratio:.1%}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

