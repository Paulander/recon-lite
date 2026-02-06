#!/usr/bin/env python3
"""
Convergence Speed Benchmark.
Measures how many games it takes to reach a target win rate (90%) 
on the Mate_In_2 stage under Legacy vs XP-Weighted modes.
"""

import subprocess
import os
import sys
from pathlib import Path
import time

TARGET_WIN_RATE = 90.0  # 90%
MAX_CYCLES = 20
GAMES_PER_CYCLE = 50
UV_PATH = "/home/paulander/.local/bin/uv"

def run_convergence_test(name, use_maturity):
    print(f"\n🏎️  Starting Convergence Test: {name} (Maturity Weighting: {'ON' if use_maturity == '1' else 'OFF'})", flush=True)
    env = os.environ.copy()
    env["RECON_USE_MATURITY_WEIGHTING"] = use_maturity
    env["M5_HEURISTIC_PROB"] = "0.0" # Pure ReCoN mode
    env["PYTHONUNBUFFERED"] = "1"
    
    # We'll run the curriculum, but we'll parse the output to see when it hits the threshold
    # Note: We don't use --quick here because we want to see actual convergence over time
    cmd = [
        UV_PATH, "run", "python3", "-u", "scripts/run_krk_curriculum.py",
        "--games-per-cycle", str(GAMES_PER_CYCLE),
        "--max-cycles-per-stage", str(MAX_CYCLES),
        "--output-dir", f"snapshots/benchmark/convergence_{name}",
        "--mode", "recon",
        "--win-rate-threshold", str(TARGET_WIN_RATE / 100.0) # Set advancement threshold to our target
    ]
    
    total_games = 0
    start_time = time.time()
    
    try:
        # We'll run it and stream the output to detect convergence
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        reached_stage_0 = False
        reached_stage_1 = False
        mastered_stage_1 = False
        
        for line in process.stdout:
            sys.stdout.write(f"RAW: {line}")
            sys.stdout.flush()
            
            if "STAGE 0:" in line:
                reached_stage_0 = True
                print("  📍 Reached Stage 0 (Mate_In_1)", flush=True)

            if "STAGE 1:" in line:
                reached_stage_1 = True
                print("  📍 Reached Stage 1 (Mate_In_2)", flush=True)
            
            if "Win Rate:" in line and "%" in line:
                try:
                    # Extract rate: "Win Rate: 40.0% (20W/30L...)"
                    rate_str = line.split("Win Rate:")[1].split("%")[0].strip()
                    rate = float(rate_str)
                    total_games += GAMES_PER_CYCLE
                    
                    if reached_stage_1:
                        print(f"    📊 Stage 1 | Games: {total_games:4} | Win Rate: {rate:5.1f}%", flush=True)
                        if rate >= TARGET_WIN_RATE:
                            print(f"    ⭐ Target {TARGET_WIN_RATE}% Reached!", flush=True)
                            mastered_stage_1 = True
                    elif reached_stage_0:
                        print(f"    📊 Stage 0 | Games: {total_games:4} | Win Rate: {rate:5.1f}%", flush=True)
                except ValueError:
                    pass
            
            if "STAGE ADVANCED to" in line and reached_stage_0 and "Mate_In_2" in line:
                 print("  ✅ Stage 0 Complete (Advancing)", flush=True)
                    
            if ("Stage 1 Complete" in line or "STAGE ADVANCED to" in line) and reached_stage_1:
                # We need to see if we reached the target win rate
                if mastered_stage_1:
                    print("  ✅ Stage 1 Complete (Advancing)", flush=True)
                    process.terminate()
                    break

        process.wait()
        duration = time.time() - start_time
        
        if mastered_stage_1:
            return total_games, duration
        else:
            return None, duration
            
    except Exception as e:
        print(f"❌ Error during {name}: {e}")
        return None, 0

if __name__ == "__main__":
    print("="*60)
    print("M5 CONVERGENCE SPEED BENCHMARK")
    print(f"Target: {TARGET_WIN_RATE}% Win Rate on Mate_In_2")
    print("="*60)
    
    # 1. Run Legacy
    # legacy_games, legacy_time = run_convergence_test("legacy", "0")
    legacy_games, legacy_time = None, None
    
    # 2. Run XP-Weighted
    xp_games, xp_time = run_convergence_test("xp_weighted", "1")
    
    print("\n" + "="*60)
    print("CONVERGENCE SPEED RESULTS")
    print("="*60)
    
    if legacy_games:
        print(f"Legacy Mode:      {legacy_games:4} games ({legacy_time:5.1f}s)")
    else:
        print(f"Legacy Mode:      FAILED to reach {TARGET_WIN_RATE}% within {MAX_CYCLES * GAMES_PER_CYCLE} games")
        
    if xp_games:
        print(f"XP-Weighted Mode: {xp_games:4} games ({xp_time:5.1f}s)")
    else:
        print(f"XP-Weighted Mode: FAILED to reach {TARGET_WIN_RATE}% within {MAX_CYCLES * GAMES_PER_CYCLE} games")
        
    if legacy_games and xp_games:
        speedup = legacy_games / xp_games
        reduction = (1 - (xp_games / legacy_games)) * 100
        print(f"\n🏆 Speedup: {speedup:.2f}x faster")
        print(f"📉 Game Reduction: {reduction:.1f}% fewer games needed")
