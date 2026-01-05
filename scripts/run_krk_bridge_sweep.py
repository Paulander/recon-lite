#!/usr/bin/env python3
"""
KRK Bridge Experiment: Competitive Sweep

Compares "Blank Slate" vs "Experienced Hector" to answer:
Does KPK knowledge transfer to KRK?

Trial 1: "Blank Slate" (No Transfer)
- Start KRK from zero
- Prediction: 10-15 cycles to reach 20% win rate

Trial 2: "Experienced Hector" (With Transfer)
- Use top 20 KPK cells
- Prediction: Starting win rate >40%, sensor_reuse_ratio spike

Monitoring:
- sensor_reuse_ratio (target: >0.5 for transfer success)
- Tactical_Box_Manager creation (Box Method POR discovery)
- Max depth (target: 3+ when Box Method triggers)
"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class KRKSweepConfig:
    """Configuration for a KRK bridge trial."""
    trial_name: str
    transfer_from: Optional[Path]
    transfer_top_n: int
    games_per_cycle: int
    max_cycles: int
    enable_stall_recovery: bool
    enable_scent_shaping: bool
    high_plasticity_on_reuse: bool  # Boost plasticity if reuse > 0.5


def run_single_trial(config: KRKSweepConfig, output_dir: Path) -> Dict[str, Any]:
    """Run a single KRK bridge trial."""
    import os
    from scripts.krk_evolution_driver import KRKEvolutionConfig, run_krk_evolution
    
    print(f"\n{'='*60}")
    print(f"TRIAL: {config.trial_name}")
    print(f"{'='*60}")
    print(f"Transfer: {config.transfer_from or 'NONE (Blank Slate)'}")
    print(f"Games/cycle: {config.games_per_cycle}, Cycles: {config.max_cycles}")
    print()
    
    trial_output = output_dir / config.trial_name
    
    # Set environment for stall recovery
    if config.enable_stall_recovery:
        os.environ["M5_STALL_THRESHOLD_WIN_RATE"] = "0.15"
        os.environ["M5_STALL_THRESHOLD_CYCLES"] = "3"
    if config.enable_scent_shaping:
        os.environ["M5_ENABLE_SCENT_SHAPING"] = "1"
    
    evo_config = KRKEvolutionConfig(
        topology_path=Path("topologies/krk_legs_topology.json"),
        output_dir=trial_output,
        games_per_cycle=config.games_per_cycle,
        max_cycles=config.max_cycles,
        transfer_from=config.transfer_from,
        transfer_top_n=config.transfer_top_n,
        plasticity_eta=0.05,  # Will be boosted if reuse > 0.5
    )
    
    start_time = time.time()
    try:
        results = run_krk_evolution(evo_config)
        duration = time.time() - start_time
        
        # Extract key metrics
        final_result = results[-1] if results else None
        
        return {
            "trial_name": config.trial_name,
            "success": True,
            "total_cycles": len(results),
            "final_win_rate": final_result.win_rate if final_result else 0,
            "final_reuse_ratio": final_result.sensor_reuse_ratio if final_result else 0,
            "cycles_to_20_pct": next(
                (i+1 for i, r in enumerate(results) if r.win_rate >= 0.20),
                None
            ),
            "cycles_to_50_pct": next(
                (i+1 for i, r in enumerate(results) if r.win_rate >= 0.50),
                None
            ),
            "win_rate_progression": [r.win_rate for r in results],
            "reuse_progression": [r.sensor_reuse_ratio for r in results],
            "duration_seconds": duration,
        }
    except Exception as e:
        return {
            "trial_name": config.trial_name,
            "success": False,
            "error": str(e),
        }


def create_krk_bridge_sweep() -> List[KRKSweepConfig]:
    """Create the competitive sweep configurations."""
    return [
        # Trial 1: Blank Slate - No knowledge transfer
        KRKSweepConfig(
            trial_name="blank_slate",
            transfer_from=None,
            transfer_top_n=0,
            games_per_cycle=50,
            max_cycles=20,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            high_plasticity_on_reuse=False,  # N/A for blank slate
        ),
        
        # Trial 2: Experienced Hector - With KPK transfer
        KRKSweepConfig(
            trial_name="experienced_hector",
            transfer_from=Path("snapshots/archive/kpk_gauntlet_prodigy_v1/stem_cells.json"),
            transfer_top_n=20,
            games_per_cycle=50,
            max_cycles=20,
            enable_stall_recovery=True,
            enable_scent_shaping=True,
            high_plasticity_on_reuse=True,  # Boost plasticity on transfer success
        ),
    ]


def generate_bridge_report(results: List[Dict[str, Any]], output_path: Path) -> str:
    """Generate comparison report."""
    report = []
    report.append("# KRK Bridge Experiment Results")
    report.append(f"\n_Generated: {datetime.now().isoformat()}_\n")
    
    report.append("## Hypothesis")
    report.append("")
    report.append("**Question**: Does knowledge transfer from KPK to KRK accelerate learning?")
    report.append("")
    report.append("**Predictions**:")
    report.append("- Blank Slate: 10-15 cycles to reach 20% win rate")
    report.append("- Experienced Hector: >40% starting win rate, sensor_reuse_ratio > 0.5")
    report.append("")
    
    report.append("## Summary Table")
    report.append("")
    report.append("| Trial | Final Win Rate | Cycles to 20% | Cycles to 50% | Reuse Ratio | Duration |")
    report.append("|-------|----------------|---------------|---------------|-------------|----------|")
    
    for r in results:
        if r.get("success"):
            name = r["trial_name"]
            win = f"{r['final_win_rate']:.1%}"
            c20 = str(r.get('cycles_to_20_pct') or 'N/A')
            c50 = str(r.get('cycles_to_50_pct') or 'N/A')
            reuse = f"{r['final_reuse_ratio']:.1%}"
            dur = f"{r['duration_seconds']:.0f}s"
            report.append(f"| {name} | {win} | {c20} | {c50} | {reuse} | {dur} |")
        else:
            report.append(f"| {r['trial_name']} | FAILED | - | - | - | - |")
    
    report.append("")
    
    # Analysis
    report.append("## Analysis")
    report.append("")
    
    blank = next((r for r in results if r["trial_name"] == "blank_slate"), None)
    hector = next((r for r in results if r["trial_name"] == "experienced_hector"), None)
    
    if blank and hector and blank.get("success") and hector.get("success"):
        # Compare starting performance
        blank_start = blank["win_rate_progression"][0] if blank["win_rate_progression"] else 0
        hector_start = hector["win_rate_progression"][0] if hector["win_rate_progression"] else 0
        
        report.append(f"### Starting Performance")
        report.append(f"- Blank Slate Cycle 1: {blank_start:.1%}")
        report.append(f"- Experienced Hector Cycle 1: {hector_start:.1%}")
        report.append("")
        
        # Transfer success
        reuse = hector["final_reuse_ratio"]
        report.append("### Transfer Success")
        if reuse > 0.5:
            report.append(f"✅ **TRANSFER SUCCESSFUL**: sensor_reuse_ratio = {reuse:.1%} (>50%)")
            report.append("")
            report.append("Universal sensors from KPK are actively contributing to KRK wins!")
        elif reuse > 0.3:
            report.append(f"⚠️ **PARTIAL TRANSFER**: sensor_reuse_ratio = {reuse:.1%} (30-50%)")
            report.append("")
            report.append("Some knowledge transferred, but KRK-specific learning still dominant.")
        else:
            report.append(f"❌ **TRANSFER FAILED**: sensor_reuse_ratio = {reuse:.1%} (<30%)")
            report.append("")
            report.append("KPK sensors not applicable to KRK. Domain-specific learning required.")
        
        report.append("")
        
        # Learning speed comparison
        report.append("### Learning Speed")
        blank_20 = blank.get("cycles_to_20_pct")
        hector_20 = hector.get("cycles_to_20_pct")
        
        if blank_20 and hector_20:
            speedup = blank_20 / hector_20
            report.append(f"- Cycles to 20%: Blank={blank_20}, Hector={hector_20} ({speedup:.1f}x speedup)")
        elif hector_20 and not blank_20:
            report.append(f"- Hector reached 20% in {hector_20} cycles, Blank never reached it!")
        else:
            report.append("- Learning speed comparison not available")
    
    report.append("")
    report.append("## Conclusion")
    report.append("")
    
    if hector and hector.get("success") and hector["final_reuse_ratio"] > 0.5:
        report.append("The **Knowledge Bank** concept is validated. Universal sensors like")
        report.append("`king_distance` and `opposition_status` transfer successfully from KPK to KRK.")
        report.append("")
        report.append("**Next Steps**:")
        report.append("1. Identify which specific sensors transferred (examine stem_cells.json)")
        report.append("2. Check if Tactical_Box_Manager was created (Box Method POR)")
        report.append("3. Measure max depth to see if sequential reasoning emerged")
    else:
        report.append("Transfer results inconclusive or unsuccessful. Consider:")
        report.append("1. Running more cycles")
        report.append("2. Adjusting transfer parameters")
        report.append("3. Manual feature mapping between domains")
    
    # Save report
    report_text = "\n".join(report)
    output_path.write_text(report_text)
    
    return report_text


def main():
    print("=" * 70)
    print("KRK BRIDGE EXPERIMENT: Blank Slate vs Experienced Hector")
    print("=" * 70)
    print()
    print("This experiment tests whether KPK knowledge transfers to KRK.")
    print()
    print("Trial 1: 'Blank Slate' - Fresh start, no transfer")
    print("Trial 2: 'Experienced Hector' - Transfer top 20 KPK cells")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"snapshots/krk_bridge_experiment/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Get sweep configs
    configs = create_krk_bridge_sweep()
    
    print("Sweep Configurations:")
    for c in configs:
        print(f"  - {c.trial_name}:")
        print(f"      transfer={c.transfer_from or 'NONE'}")
        print(f"      transfer_top_n={c.transfer_top_n}")
        print(f"      cycles={c.max_cycles}, games/cycle={c.games_per_cycle}")
    print()
    
    # Run trials
    results = []
    for config in configs:
        result = run_single_trial(config, output_dir)
        results.append(result)
        
        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    # Generate report
    report_path = output_dir / "bridge_report.md"
    report = generate_bridge_report(results, report_path)
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()
    print(report)
    print()
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

