#!/usr/bin/env python3
"""Quick test of pack template functionality."""

print("Testing pack template import...")
from recon_lite.nodes.pack_template import spawn_goal_delegation_pack
print("  ✓ pack_template import OK")

from recon_lite.nodes.pack_template import get_actuator_weight, prune_pack
print("  ✓ helper functions OK")

print("\nTesting lottery spawning import...")
from recon_lite.nodes.stem_cell import StemCellTerminal, StemCellConfig
print("  ✓ stem_cell import OK")

# Quick instantiation test
cell = StemCellTerminal("test_cell")
print(f"  ✓ StemCellTerminal created: {cell.cell_id}")

# Check spawn_with_lottery exists
assert hasattr(cell, "spawn_with_lottery"), "Missing spawn_with_lottery method!"
print("  ✓ spawn_with_lottery method exists")

print("\n✅ All pack template tests passed!")
