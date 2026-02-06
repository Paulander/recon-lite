#!/usr/bin/env python3
"""Analyze trace files to find TRIAL node activations."""
import json
from pathlib import Path
import sys

def analyze_traces(trace_dir: Path):
    """Analyze trace files for TRIAL node activations."""
    trace_files = list(trace_dir.glob("*.jsonl"))
    print(f"Analyzing {len(trace_files)} trace files...")
    print()
    
    total_ticks = 0
    ticks_with_trial = 0
    trial_node_counts = {}
    coactivations = []  # LEG + TRIAL in same tick
    
    for trace_file in trace_files:
        with open(trace_file) as f:
            for line in f:
                try:
                    episode = json.loads(line)
                    for tick in episode.get("ticks", []):
                        total_ticks += 1
                        active = tick.get("active_nodes", [])
                        
                        # Count TRIAL nodes
                        trial_nodes = [n for n in active if "TRIAL" in n]
                        if trial_nodes:
                            ticks_with_trial += 1
                            for t in trial_nodes:
                                trial_node_counts[t] = trial_node_counts.get(t, 0) + 1
                        
                        # Check for LEG + TRIAL coactivation
                        leg_nodes = [n for n in active if "_leg" in n.lower()]
                        if leg_nodes and trial_nodes:
                            coactivations.append({
                                "file": trace_file.name,
                                "legs": leg_nodes,
                                "trials": trial_nodes,
                            })
                except json.JSONDecodeError:
                    continue
    
    print(f"Total ticks: {total_ticks}")
    print(f"Ticks with TRIAL nodes: {ticks_with_trial} ({100*ticks_with_trial/total_ticks:.1f}%)")
    print()
    
    if trial_node_counts:
        print("Top 10 TRIAL nodes by activation count:")
        sorted_trials = sorted(trial_node_counts.items(), key=lambda x: -x[1])[:10]
        for name, count in sorted_trials:
            print(f"  {name}: {count}")
    else:
        print("⚠️ No TRIAL nodes found in active_nodes!")
        print("   Check if trace recording is capturing TRUE/CONFIRMED states.")
    
    print()
    if coactivations:
        print(f"🎯 LEG + TRIAL coactivations: {len(coactivations)}")
        for co in coactivations[:5]:
            print(f"  {co['file']}: LEGs={co['legs']}, TRIALs={co['trials'][:3]}...")
    else:
        print("❌ No LEG + TRIAL coactivations found.")


if __name__ == "__main__":
    trace_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("snapshots/evolution/stage11_logic_gauntlet/traces")
    if trace_dir.exists():
        analyze_traces(trace_dir)
    else:
        print(f"Trace directory not found: {trace_dir}")

