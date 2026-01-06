#!/usr/bin/env python3
"""
KRK Bach-Integrated Training Orchestration

This script runs KRK training with full Bach alignment:
- Progressive gating (30% -> 100% over 100 games)
- Think Harder escalation (no random fallback)
- NodeSpace binding architecture
- Universal King Logic cluster from KPK transfer

Part of the "KPK to KRK Knowledge Transfer" plan.

Usage:
    python scripts/run_krk_bach_training.py

Configuration via environment variables or command-line args.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_gated_krk_topology(
    source_path: Path,
    output_path: Path,
    legacy_stem_cells_path: Optional[Path] = None,
) -> dict:
    """
    Create KRK topology with gating and legacy sensor injection.
    
    Args:
        source_path: Path to base krk_legs_topology.json
        output_path: Where to save the gated topology
        legacy_stem_cells_path: Optional path to legacy stem_cells.json
        
    Returns:
        Summary dict of modifications
    """
    from typing import Optional
    
    # Load base topology
    with open(source_path) as f:
        topo = json.load(f)
    
    summary = {
        "gated_legs": [],
        "legacy_sensors_injected": 0,
        "edges_added": 0,
    }
    
    # Gating should already be enabled in the topology (from earlier modifications)
    # But let's verify/ensure it
    nodes = topo.get("nodes", [])
    gated_legs = ["krk_rook_leg", "krk_king_leg"]
    
    for node in nodes:
        if node.get("id") in gated_legs:
            if "meta" not in node:
                node["meta"] = {}
            if not node["meta"].get("require_child_confirm"):
                node["meta"]["require_child_confirm"] = True
                summary["gated_legs"].append(node.get("id"))
    
    # Inject legacy sensors from KPK if provided
    if legacy_stem_cells_path and legacy_stem_cells_path.exists():
        with open(legacy_stem_cells_path) as f:
            stem_data = json.load(f)
        
        cells = stem_data.get("cells", {})
        edges = topo.get("edges", [])
        
        for cell_id, cell_data in cells.items():
            # Only inject MATURE cells (legacy sensors)
            if cell_data.get("state") != "MATURE":
                continue
            if not cell_data.get("metadata", {}).get("legacy") == "kpk_universal":
                continue
            
            # Create node for this legacy sensor
            legacy_node = {
                "id": f"legacy_{cell_id}",
                "type": "TERMINAL",
                "group": "sensor",
                "factory": None,  # Pattern-based, not factory-based
                "meta": {
                    "legacy": True,
                    "origin": "kpk_transfer",
                    "pattern_signature": cell_data.get("pattern_signature"),
                    "binding_space": "krk_king_binding",
                }
            }
            nodes.append(legacy_node)
            summary["legacy_sensors_injected"] += 1
            
            # Wire to universal_king_logic
            edges.append({
                "src": "universal_king_logic",
                "dst": f"legacy_{cell_id}",
                "type": "SUB",
                "weight": 0.8,
                "consolidate": True,
                "confirmation_count": 0,
            })
            summary["edges_added"] += 1
    
    # Update topology
    topo["nodes"] = nodes
    topo["last_modified"] = datetime.now().isoformat()
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(topo, f, indent=2)
    
    return summary


def main():
    from typing import Optional
    
    parser = argparse.ArgumentParser(description="KRK Bach-Integrated Training")
    
    # Paths
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("topologies/krk_legs_topology.json"),
        help="Base KRK topology"
    )
    parser.add_argument(
        "--legacy-sensors",
        type=Path,
        default=Path("backups/kpk_legacy_20260106/stem_cells_legacy.json"),
        help="Legacy KPK sensors for transfer"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("snapshots/evolution/krk_bach_training"),
        help="Output directory for results"
    )
    
    # Training config
    parser.add_argument("--games-per-cycle", type=int, default=50)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--min-tick-depth", type=int, default=5, 
                        help="Mandatory propagation depth")
    
    # Progressive gating
    parser.add_argument("--gating-initial", type=float, default=0.30,
                        help="Initial gating strictness (training wheels)")
    parser.add_argument("--gating-final", type=float, default=1.0,
                        help="Final gating strictness")
    parser.add_argument("--gating-ramp-games", type=int, default=100,
                        help="Games to reach full strictness")
    parser.add_argument("--gating-win-based", action="store_true",
                        help="Increase strictness only on wins")
    
    # Think Harder
    parser.add_argument("--pure-cognitive", action="store_true",
                        help="No random fallback (pure cognitive stall)")
    parser.add_argument("--enable-curiosity", action="store_true",
                        help="Spawn sensors on stall")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("KRK BACH-INTEGRATED TRAINING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    print("Configuration:")
    print(f"  Topology: {args.topology}")
    print(f"  Legacy Sensors: {args.legacy_sensors}")
    print(f"  Output: {args.output_dir}")
    print()
    print("Progressive Gating:")
    print(f"  Initial Strictness: {args.gating_initial * 100:.0f}%")
    print(f"  Final Strictness: {args.gating_final * 100:.0f}%")
    print(f"  Ramp Games: {args.gating_ramp_games}")
    print(f"  Win-Based: {args.gating_win_based}")
    print()
    print("Think Harder:")
    print(f"  Pure Cognitive Mode: {args.pure_cognitive}")
    print(f"  Curiosity Spawning: {args.enable_curiosity}")
    print(f"  Min Tick Depth: {args.min_tick_depth}")
    print("=" * 70)
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create gated topology with legacy sensors
    print("1. Preparing gated topology with legacy sensors...")
    gated_topo_path = args.output_dir / "topology_gated.json"
    
    summary = create_gated_krk_topology(
        source_path=args.topology,
        output_path=gated_topo_path,
        legacy_stem_cells_path=args.legacy_sensors if args.legacy_sensors.exists() else None,
    )
    
    print(f"   Gated legs: {summary['gated_legs']}")
    print(f"   Legacy sensors injected: {summary['legacy_sensors_injected']}")
    print(f"   Edges added: {summary['edges_added']}")
    print()
    
    # Step 2: Run evolution training
    print("2. Starting KRK evolution training...")
    
    try:
        from scripts.evolution_driver import EvolutionConfig, run_evolution_training
        
        config = EvolutionConfig(
            topology_path=gated_topo_path,
            games_per_cycle=args.games_per_cycle,
            max_cycles=args.max_cycles,
            snapshot_dir=args.output_dir / "snapshots",
            trace_dir=args.output_dir / "traces",
            signature_dir=args.output_dir / "signatures",
            output_dir=args.output_dir / "reports",
            
            # Engine settings
            min_internal_ticks=args.min_tick_depth,
            
            # Progressive gating (Bach-Integrated)
            enable_progressive_gating=True,
            gating_initial_strictness=args.gating_initial,
            gating_final_strictness=args.gating_final,
            gating_ramp_games=args.gating_ramp_games,
            gating_win_based=args.gating_win_based,
            
            # Think Harder (Bach-Integrated)
            enable_think_harder=True,
            pure_cognitive_mode=args.pure_cognitive,
            enable_curiosity_spawning=args.enable_curiosity,
            curiosity_spawn_count=3,
            
            # Sparsity
            max_trial_slots=15,
        )
        
        results = run_evolution_training(config)
        
        # Step 3: Analyze results
        print()
        print("=" * 70)
        print("TRAINING COMPLETE - RESULTS")
        print("=" * 70)
        
        if results:
            final = results[-1]
            print(f"Final Cycle: {final.cycle}")
            print(f"Win Rate: {final.win_rate * 100:.1f}%")
            print(f"Optimal Rate: {final.optimal_rate * 100:.1f}%")
            print(f"Promotions: {len(final.promotions)}")
            
            # Calculate metrics
            total_wins = sum(1 for r in results if r.win_rate > 0.5)
            avg_win_rate = sum(r.win_rate for r in results) / len(results)
            
            print()
            print("Summary:")
            print(f"  Cycles with >50% win rate: {total_wins}/{len(results)}")
            print(f"  Average win rate: {avg_win_rate * 100:.1f}%")
        
        # Save summary
        summary_path = args.output_dir / "bach_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "gating_initial": args.gating_initial,
                    "gating_final": args.gating_final,
                    "pure_cognitive": args.pure_cognitive,
                    "min_tick_depth": args.min_tick_depth,
                },
                "results": [
                    {
                        "cycle": r.cycle,
                        "win_rate": r.win_rate,
                        "optimal_rate": r.optimal_rate,
                        "promotions": len(r.promotions),
                    }
                    for r in results
                ] if results else [],
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

