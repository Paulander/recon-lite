#!/usr/bin/env python3
"""Test script to verify KRK curriculum stages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from recon_lite_chess.training.krk_curriculum import KRK_STAGES

print("KRK Curriculum Stages:")
print("=" * 60)
for i, stage in enumerate(KRK_STAGES):
    print(f"Stage {i}: {stage.name}")
    print(f"   - Key Lesson: {stage.key_lesson}")
    print(f"   - Target Win Rate: {stage.target_win_rate:.0%}")
    print(f"   - Positions: {len(stage.positions)}")
    print()

print(f"\nTotal Stages: {len(KRK_STAGES)}")
print("Bridge stage (Anchored_Cut) successfully added!" if any("Anchored" in s.name for s in KRK_STAGES) else "Bridge stage NOT found!")

