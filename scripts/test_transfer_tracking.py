#!/usr/bin/env python3
"""Test the new transfer tracking methods."""

from pathlib import Path
import sys
sys.path.insert(0, "src")

from recon_lite.nodes.stem_cell import StemCellManager

# Load the experienced hector stem cells
stem_path = Path("snapshots/krk_bridge_experiment/20260105_163449/experienced_hector/snapshots/stem_cells.json")
if stem_path.exists():
    manager = StemCellManager.load(stem_path)
    
    print("=" * 60)
    print("TRANSFER TRACKING TEST")
    print("=" * 60)
    
    print(f"\nLoaded stem cells: {len(manager.cells)}")
    
    # Get transferred cells
    transferred = manager.get_transferred_cells()
    print(f"Transferred cells: {len(transferred)}")
    
    # Test new method
    active = manager.get_active_transferred_cells()
    print(f"Active transferred cells: {len(active)}")
    print(f"  Cell IDs: {active[:5]}...")
    
    # Test transfer contribution
    contrib = manager.compute_transfer_contribution(game_won=True)
    print(f"\nTransfer contribution: {contrib:.1%}")
    
    # Get reuse stats
    stats = manager.get_reuse_stats()
    print(f"\nReuse stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
else:
    print("Stem cells file not found")

