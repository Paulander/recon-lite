#!/usr/bin/env python3
"""
Benchmark XP-Weighted Selection vs Legacy.
Runs a set of games in both modes and compares win rates.
"""

import subprocess
import os
import json
from pathlib import Path

def run_test(name, use_maturity):
    print(f"🚀 Running Benchmark: {name} (RECON_USE_MATURITY_WEIGHTING={use_maturity})")
    env = os.environ.copy()
    env["RECON_USE_MATURITY_WEIGHTING"] = use_maturity
    env["M5_HEURISTIC_PROB"] = "0.0" # Pure ReCoN mode
    
    # Run a quick KRK curriculum stage 2 test (Mate in 2)
    # We use Stage 2 because it's non-trivial but fast to converge
    cmd = [
        "/home/paulander/.local/bin/uv", "run", "python3", "scripts/run_krk_curriculum.py",
        "--games-per-cycle", "50",
        "--output-dir", f"snapshots/benchmark/{name}",
        "--mode", "recon",
        "--quick" # Run a shorter version
    ]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(f"✅ {name} completed.")
        # Try to find win rate in output
        for line in result.stdout.split('\n'):
            if "Cycle Win Rate" in line:
                print(f"  📊 {line.strip()}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed with error.")
        print(e.stderr)
        return None

if __name__ == "__main__":
    # Create benchmark dir
    Path("snapshots/benchmark").mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("M5 SELECTION BENCHMARK")
    print("="*60)
    
    # 1. Run Legacy (Maturity Weighting OFF)
    legacy_out = run_test("legacy", "0")
    
    # 2. Run XP-Weighted (Maturity Weighting ON)
    xp_out = run_test("xp_weighted", "1")
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    def get_last_winrate(output):
        if not output: return "N/A"
        rates = []
        for line in output.split('\n'):
            if "Cycle Win Rate" in line:
                try:
                    rates.append(float(line.split(":")[-1].strip().replace("%", "")) / 100.0)
                except: pass
        return rates[-1] if rates else "N/A"

    legacy_rate = get_last_winrate(legacy_out)
    xp_rate = get_last_winrate(xp_out)
    
    print(f"Legacy Win Rate:      {legacy_rate:.2%}" if isinstance(legacy_rate, float) else f"Legacy: {legacy_rate}")
    print(f"XP-Weighted Win Rate: {xp_rate:.2%}" if isinstance(xp_rate, float) else f"XP-Weighted: {xp_rate}")
    
    if isinstance(legacy_rate, float) and isinstance(xp_rate, float):
        diff = xp_rate - legacy_rate
        print(f"Performance Delta:    {diff:+.2%}")
        if diff > 0:
            print("🚀 Result: XP-Weighting IMPROVES performance by focusing on mature patterns.")
        else:
            print("📉 Result: XP-Weighting does not show immediate gains for this short run.")
