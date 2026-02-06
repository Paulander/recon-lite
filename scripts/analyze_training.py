#!/usr/bin/env python3
"""
Analyze training trace files and generate statistics.

Usage:
    python scripts/analyze_training.py reports/curriculum/*/anchor_*.jsonl
    python scripts/analyze_training.py --report-dir reports/curriculum/20251211_092937/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingStats:
    """Statistics from training traces."""
    total_episodes: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    incomplete: int = 0
    total_plies: int = 0
    checkmates: int = 0
    promotions: int = 0  # KPK success condition
    avg_plies: float = 0.0
    win_rate: float = 0.0
    
    def compute_derived(self):
        completed = self.wins + self.losses + self.draws
        self.win_rate = self.wins / completed if completed > 0 else 0.0
        self.avg_plies = self.total_plies / self.total_episodes if self.total_episodes > 0 else 0.0


def analyze_jsonl(path: Path) -> TrainingStats:
    """Analyze a single JSONL trace file."""
    stats = TrainingStats()
    
    # Determine phase type from filename (e.g., "anchor_kpk" -> "kpk")
    phase_type = path.stem.lower()
    is_kpk = "kpk" in phase_type and "kqk" not in phase_type
    
    with open(path) as f:
        for line in f:
            try:
                ep = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            stats.total_episodes += 1
            result = ep.get("result")
            notes = ep.get("notes", {})
            plies = notes.get("plies", 0)
            stats.total_plies += plies
            
            if result == "1-0":
                stats.wins += 1
            elif result == "0-1":
                stats.losses += 1
            elif result == "1/2-1/2":
                stats.draws += 1
            else:
                stats.incomplete += 1
            
            # Check for promotion (KPK success condition only)
            # Only count promotions for KPK games, not KQK (which always has a Queen)
            if is_kpk:
                ticks = ep.get("ticks", [])
                if ticks and len(ticks) > 1:
                    # Get initial FEN (first tick)
                    first_tick = ticks[0]
                    if isinstance(first_tick, dict):
                        initial_fen = first_tick.get("board_fen", "")
                        initial_board = initial_fen.split()[0] if initial_fen else ""
                        
                        # Get final FEN (last tick)
                        last_tick = ticks[-1]
                        if isinstance(last_tick, dict):
                            final_fen = last_tick.get("board_fen", "")
                            final_board = final_fen.split()[0] if final_fen else ""
                            
                            # Count White Queens in initial vs final position
                            initial_queens = initial_board.count("Q")
                            final_queens = final_board.count("Q")
                            
                            # Promotion happened if final has more Queens than initial
                            if final_queens > initial_queens:
                                stats.promotions += 1
    
    stats.compute_derived()
    return stats


def analyze_directory(report_dir: Path) -> dict[str, TrainingStats]:
    """Analyze all JSONL files in a directory."""
    results = {}
    
    for jsonl_file in report_dir.glob("*.jsonl"):
        name = jsonl_file.stem
        results[name] = analyze_jsonl(jsonl_file)
    
    return results


def print_stats(name: str, stats: TrainingStats):
    """Print statistics in a readable format."""
    completed = stats.wins + stats.losses + stats.draws
    print(f"\n{name}:")
    print(f"  Total episodes: {stats.total_episodes}")
    print(f"  Completed: {completed} ({stats.incomplete} incomplete)")
    print(f"  Results: {stats.wins}W / {stats.draws}D / {stats.losses}L")
    print(f"  Win rate: {stats.win_rate:.1%}")
    print(f"  Promotions: {stats.promotions}")
    print(f"  Avg plies: {stats.avg_plies:.1f}")


def generate_markdown_report(
    report_dir: Path,
    all_stats: dict[str, TrainingStats],
    output_path: Optional[Path] = None,
) -> str:
    """Generate a markdown summary report."""
    from datetime import datetime
    
    timestamp = report_dir.name
    lines = [
        f"# Training Report: {timestamp}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Phase | Episodes | W/D/L | Win Rate | Promos | Avg Plies |",
        "|-------|----------|-------|----------|--------|-----------|",
    ]
    
    total_episodes = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_promos = 0
    
    for name, stats in sorted(all_stats.items()):
        completed = stats.wins + stats.losses + stats.draws
        wdl = f"{stats.wins}/{stats.draws}/{stats.losses}"
        lines.append(f"| {name} | {stats.total_episodes} | {wdl} | {stats.win_rate:.1%} | {stats.promotions} | {stats.avg_plies:.1f} |")
        total_episodes += stats.total_episodes
        total_wins += stats.wins
        total_losses += stats.losses
        total_draws += stats.draws
        total_promos += stats.promotions
    
    # Totals
    total_completed = total_wins + total_losses + total_draws
    total_win_rate = total_wins / total_completed if total_completed > 0 else 0
    lines.append(f"| **TOTAL** | {total_episodes} | {total_wins}/{total_draws}/{total_losses} | {total_win_rate:.1%} | {total_promos} | - |")
    
    lines.extend([
        "",
        "## Notes",
        "",
        "- Win rate counts only completed games (excludes max_plies timeouts)",
        "- **Promos** = pawn promotions (KPK success condition only)",
        "- For KPK: promotion counts as win (result='1-0')",
        "- KQK and KRK do not have promotions (KQK starts with Queen, KRK has no pawns)",
        "- KPK endgames are harder to win than KRK due to drawing tendencies",
        "",
    ])
    
    content = "\n".join(lines)
    
    if output_path:
        output_path.write_text(content)
        print(f"Report written to {output_path}")
    
    return content


def main():
    parser = argparse.ArgumentParser(description="Analyze training traces")
    parser.add_argument("files", nargs="*", help="JSONL files to analyze")
    parser.add_argument("--report-dir", "-d", type=Path, help="Analyze all files in directory")
    parser.add_argument("--markdown", "-m", action="store_true", help="Generate markdown report")
    parser.add_argument("--output", "-o", type=Path, help="Output file for markdown report")
    
    args = parser.parse_args()
    
    if args.report_dir:
        all_stats = analyze_directory(args.report_dir)
        for name, stats in sorted(all_stats.items()):
            print_stats(name, stats)
        
        if args.markdown:
            output = args.output or args.report_dir / "training_report.md"
            generate_markdown_report(args.report_dir, all_stats, output)
    
    elif args.files:
        all_stats = {}
        for file_path in args.files:
            path = Path(file_path)
            if path.exists():
                stats = analyze_jsonl(path)
                all_stats[path.stem] = stats
                print_stats(path.stem, stats)
        
        if args.markdown and all_stats:
            output = args.output or Path("training_report.md")
            generate_markdown_report(Path.cwd(), all_stats, output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

