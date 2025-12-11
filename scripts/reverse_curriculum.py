#!/usr/bin/env python3
"""
Reverse Curriculum Training Loop (CPU Sequential).

Implements the Sensor Flooding + Reverse Curriculum strategy:
1. Teacher Phase: Play games with plasticity, collect traces
2. Dreamer Phase: Consolidate, extract motifs, evaluate stem cells
3. Repeat with updated weights

Phase structure:
- Anchor: KRK/KPK/KQK endgames to 95%+ win rate
- Bridge: Simplified middlegame → discover liquidation strategies
- Wilderness: Full material tactical positions
- Integration: Full games from starting position

Usage:
    uv run python scripts/reverse_curriculum.py --phase anchor --epochs 5
    uv run python scripts/reverse_curriculum.py --phase bridge --epochs 3 --enable-stem-cells
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.trace_db import TraceDB


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""
    name: str
    endgames: List[str]  # List of endgame types to train
    episodes_per_endgame: int
    win_rate_threshold: float
    max_plies: int
    enable_plasticity: bool = True
    enable_consolidation: bool = True
    enable_stem_cells: bool = False
    stockfish_depth: int = 2


@dataclass
class EpochStats:
    """Statistics from a single epoch."""
    phase: str
    epoch: int
    timestamp: str
    endgame_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stem_cell_stats: Dict[str, Any] = field(default_factory=dict)
    total_episodes: int = 0
    total_wins: int = 0
    avg_win_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "endgame_stats": self.endgame_stats,
            "stem_cell_stats": self.stem_cell_stats,
            "total_episodes": self.total_episodes,
            "total_wins": self.total_wins,
            "avg_win_rate": self.avg_win_rate,
        }


# Default phase configurations
PHASE_CONFIGS = {
    "anchor": PhaseConfig(
        name="anchor",
        endgames=["krk", "kpk", "kqk"],
        episodes_per_endgame=100,
        win_rate_threshold=0.95,
        max_plies=200,
        enable_plasticity=True,
        enable_consolidation=True,
    ),
    "bridge": PhaseConfig(
        name="bridge",
        endgames=["bridge"],  # Simplified middlegame positions
        episodes_per_endgame=50,
        win_rate_threshold=0.7,
        max_plies=150,
        enable_plasticity=True,
        enable_consolidation=True,
        enable_stem_cells=True,
    ),
    "wilderness": PhaseConfig(
        name="wilderness",
        endgames=["wilderness"],  # Full material tactical positions
        episodes_per_endgame=50,
        win_rate_threshold=0.5,
        max_plies=200,
        enable_plasticity=True,
        enable_consolidation=True,
        enable_stem_cells=True,
    ),
    "integration": PhaseConfig(
        name="integration",
        endgames=["full_game"],
        episodes_per_endgame=30,
        win_rate_threshold=0.3,
        max_plies=300,
        enable_plasticity=True,
        enable_consolidation=True,
        enable_stem_cells=True,
    ),
}


def get_demo_script(endgame: str) -> str:
    """Get the demo script path for an endgame type."""
    script_map = {
        "krk": "demos/persistent/krk_persistent_demo.py",
        "kpk": "demos/persistent/kpk_persistent_demo.py",
        "kqk": "demos/persistent/kqk_persistent_demo.py",
        "bridge": "demos/persistent/full_game_demo.py",
        "wilderness": "demos/persistent/full_game_demo.py",
        "full_game": "demos/persistent/full_game_demo.py",
    }
    return script_map.get(endgame, "demos/persistent/full_game_demo.py")


def run_teacher_epoch(
    config: PhaseConfig,
    epoch: int,
    output_dir: Path,
    stockfish_path: Optional[str] = None,
    enable_stem_cells: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run Teacher phase: Play games with plasticity, collect traces.
    
    Returns:
        Dict mapping endgame type to stats
    """
    results = {}
    
    for endgame in config.endgames:
        print(f"\n[Teacher] Training {endgame.upper()} ({config.episodes_per_endgame} episodes)...")
        
        trace_file = output_dir / f"{config.name}_{endgame}_epoch{epoch}.jsonl"
        consol_pack = output_dir / f"{endgame}_consol.json"
        
        # Build command
        script = get_demo_script(endgame)
        cmd = [
            "uv", "run", "python", script,
            "--batch", str(config.episodes_per_endgame),
            "--max-plies", str(config.max_plies),
            "--trace-out", str(trace_file),
        ]
        
        if config.enable_plasticity:
            cmd.append("--plasticity")
        
        if config.enable_consolidation:
            cmd.extend(["--consolidate", "--consolidate-pack", str(consol_pack)])
        
        if enable_stem_cells or config.enable_stem_cells:
            cmd.append("--stem-cells")
        
        if stockfish_path:
            cmd.extend(["--engine", stockfish_path, "--depth", str(config.stockfish_depth)])
        
        # Run training
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            # Parse output for stats
            output = result.stdout
            stats = parse_batch_output(output)
            stats["trace_file"] = str(trace_file)
            results[endgame] = stats
            
            print(f"  {endgame}: {stats.get('wins', 0)}W/{stats.get('draws', 0)}D/{stats.get('losses', 0)}L")
            print(f"  Win rate: {stats.get('win_rate', 0):.1%}")
            
            if stats.get("stem_cells"):
                print(f"  Stem cells: {stats['stem_cells']}")
                
        except subprocess.TimeoutExpired:
            print(f"  {endgame}: TIMEOUT")
            results[endgame] = {"error": "timeout"}
        except Exception as e:
            print(f"  {endgame}: ERROR - {e}")
            results[endgame] = {"error": str(e)}
        
        # Cleanup
        gc.collect()
    
    return results


def parse_batch_output(output: str) -> Dict[str, Any]:
    """Parse the JSON output from batch demo runs."""
    stats = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "mates": 0,
        "promotions": 0,
        "win_rate": 0.0,
    }
    
    # Look for JSON output in the last few lines
    lines = output.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line.replace("'", '"'))
                stats["wins"] = data.get("wins", data.get("mates", 0))
                stats["losses"] = data.get("losses", 0)
                stats["draws"] = data.get("draws", data.get("stalls", 0))
                stats["mates"] = data.get("mates", 0)
                stats["promotions"] = data.get("promotions", 0)
                stats["win_rate"] = data.get("win_rate", 0.0)
                stats["stem_cells"] = data.get("stem_cells")
                break
            except json.JSONDecodeError:
                continue
    
    return stats


def run_dreamer_epoch(
    config: PhaseConfig,
    epoch: int,
    output_dir: Path,
    teacher_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run Dreamer phase: Consolidate, extract motifs, evaluate stem cells.
    
    Returns:
        Dreamer phase statistics
    """
    print(f"\n[Dreamer] Processing epoch {epoch} traces...")
    
    dreamer_stats = {
        "motifs_extracted": 0,
        "stem_cell_candidates": 0,
        "consolidation_applied": False,
    }
    
    # Collect all trace files
    trace_files = []
    for endgame, stats in teacher_results.items():
        if "trace_file" in stats:
            trace_file = Path(stats["trace_file"])
            if trace_file.exists():
                trace_files.append(str(trace_file))
    
    if not trace_files:
        print("  No trace files to process")
        return dreamer_stats
    
    # Run motif extraction
    try:
        motif_output = output_dir / f"motifs_epoch{epoch}.jsonl"
        motif_cmd = [
            "uv", "run", "python", "demos/experiments/extract_motifs.py",
            "--traces", *trace_files,
            "--out", str(motif_output),
            "--bridge-mode",  # Enable affordance crossing detection
        ]
        
        result = subprocess.run(motif_cmd, capture_output=True, text=True, timeout=600)
        
        # Count extracted motifs
        if motif_output.exists():
            with open(motif_output) as f:
                dreamer_stats["motifs_extracted"] = sum(1 for _ in f)
            print(f"  Extracted {dreamer_stats['motifs_extracted']} motifs")
    except Exception as e:
        print(f"  Motif extraction error: {e}")
    
    # Generate training report
    try:
        report_cmd = [
            "uv", "run", "python", "scripts/analyze_training.py",
            "--report-dir", str(output_dir),
            "--markdown",
        ]
        subprocess.run(report_cmd, capture_output=True, text=True, timeout=120)
        print(f"  Report generated: {output_dir}/training_report.md")
    except Exception as e:
        print(f"  Report generation error: {e}")
    
    return dreamer_stats


def check_phase_advancement(
    config: PhaseConfig,
    epoch_stats: EpochStats,
) -> bool:
    """Check if we should advance to the next phase."""
    if epoch_stats.avg_win_rate >= config.win_rate_threshold:
        return True
    
    # Also check individual endgame performance
    all_above_threshold = True
    for endgame, stats in epoch_stats.endgame_stats.items():
        win_rate = stats.get("win_rate", 0.0)
        if win_rate < config.win_rate_threshold:
            all_above_threshold = False
            break
    
    return all_above_threshold


def main():
    parser = argparse.ArgumentParser(description="Reverse Curriculum Training Loop")
    parser.add_argument("--phase", choices=list(PHASE_CONFIGS.keys()), default="anchor",
                       help="Training phase to run")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--episodes-per-epoch", type=int, default=None,
                       help="Override episodes per endgame")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory for traces and reports")
    parser.add_argument("--engine", type=str, default="/usr/games/stockfish",
                       help="Path to Stockfish")
    parser.add_argument("--enable-stem-cells", action="store_true",
                       help="Enable stem cell pattern discovery")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous checkpoint")
    
    args = parser.parse_args()
    
    # Get phase config
    config = PHASE_CONFIGS[args.phase]
    
    if args.episodes_per_epoch:
        config.episodes_per_endgame = args.episodes_per_epoch
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(f"reports/curriculum/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Reverse Curriculum Training: {config.name.upper()} Phase")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Episodes per endgame: {config.episodes_per_endgame}")
    print(f"Endgames: {', '.join(config.endgames)}")
    print(f"Win rate threshold: {config.win_rate_threshold:.0%}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    all_epoch_stats = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Teacher phase
        teacher_results = run_teacher_epoch(
            config=config,
            epoch=epoch,
            output_dir=output_dir,
            stockfish_path=args.engine,
            enable_stem_cells=args.enable_stem_cells,
        )
        
        # Dreamer phase
        dreamer_stats = run_dreamer_epoch(
            config=config,
            epoch=epoch,
            output_dir=output_dir,
            teacher_results=teacher_results,
        )
        
        # Compute epoch stats
        total_wins = sum(s.get("wins", 0) for s in teacher_results.values())
        total_losses = sum(s.get("losses", 0) for s in teacher_results.values())
        total_draws = sum(s.get("draws", 0) for s in teacher_results.values())
        total_episodes = total_wins + total_losses + total_draws
        
        epoch_stats = EpochStats(
            phase=config.name,
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            endgame_stats=teacher_results,
            stem_cell_stats=dreamer_stats,
            total_episodes=total_episodes,
            total_wins=total_wins,
            avg_win_rate=total_wins / total_episodes if total_episodes > 0 else 0.0,
        )
        all_epoch_stats.append(epoch_stats)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Total: {total_wins}W/{total_draws}D/{total_losses}L ({total_episodes} episodes)")
        print(f"  Win rate: {epoch_stats.avg_win_rate:.1%}")
        
        # Check for phase advancement
        if check_phase_advancement(config, epoch_stats):
            print(f"\n*** Phase {config.name} COMPLETE - threshold reached! ***")
            break
    
    # Save final summary
    summary_path = output_dir / "epoch_summary.json"
    with open(summary_path, "w") as f:
        json.dump([s.to_dict() for s in all_epoch_stats], f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")
    
    # Final stats
    final_stats = all_epoch_stats[-1] if all_epoch_stats else None
    if final_stats:
        print(f"\nFinal win rate: {final_stats.avg_win_rate:.1%}")
        if final_stats.avg_win_rate >= config.win_rate_threshold:
            print(f"Phase {config.name} PASSED (>= {config.win_rate_threshold:.0%})")
        else:
            print(f"Phase {config.name} needs more training (< {config.win_rate_threshold:.0%})")


if __name__ == "__main__":
    main()

