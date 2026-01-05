#!/usr/bin/env python3
"""Run Stage 8 Gauntlet Sweep - The Failure Frontier.

This is where flat networks DIE. The enemy King is active,
and only hierarchical scripts can win consistently.
"""

from pathlib import Path
from recon_lite.learning.sweep_engine import HyperSweepEngine, create_gauntlet_sweep

def main():
    print("=" * 60)
    print("STAGE 8 GAUNTLET SWEEP - The Failure Frontier")
    print("=" * 60)
    print()
    print("This is where flat networks DIE.")
    print("Only hierarchical scripts can win consistently.")
    print()

    engine = HyperSweepEngine(base_output_dir=Path("snapshots/sweeps/gauntlet"))
    configs = create_gauntlet_sweep()

    print("Gauntlet Sweep Configs:")
    for c in configs:
        print(f"  - {c.trial_name}:")
        print(f"      stage={c.stage_id}, consistency={c.consistency_threshold}")
        print(f"      forced_hoisting={c.enable_forced_hoisting}, xp_mult={c.leg_link_xp_multiplier}")
    print()

    print("Starting sweep... (3 trials x 30 cycles x 50 games = 4,500 games)")
    print()
    
    results = engine.run_sweep(configs)
    
    print()
    report = engine.generate_report(results)
    print(report)
    
    # Save report
    report_path = Path("snapshots/sweeps/gauntlet/gauntlet_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

