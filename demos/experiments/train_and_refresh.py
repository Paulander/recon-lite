#!/usr/bin/env python3
"""Tight loop for labeling datasets, refreshing macro weights, and validating the engine."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demos.experiments.teacher_stockfish import apply_weight_update, label_fens, run_phase_teacher
from recon_lite.macro_engine import MacroEngine

DATASET_ROOT = Path("data/endgames")
DEFAULT_BASE_WEIGHTS = Path("weights/macro_weight_pack.swp")
DEFAULT_VERSION_DIR = Path("weights/versions")
DEFAULT_PHASE_OUTPUT = Path("weights/krk_phase_weight_pack.swp")

VALIDATION_POSITIONS: List[Tuple[str, str]] = [
    ("krk", "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"),
    ("kpk", "8/8/2k5/8/8/6K1/4P3/8 w - - 0 1"),
    ("rook", "4k3/6K1/8/8/3R4/8/8/8 w - - 0 1"),
]


def _load_fens(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def discover_datasets(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.fen") if p.is_file())


def aggregate_labels(files: Iterable[Path], engine_path: Optional[str], depth: int) -> Tuple[Counter, float]:
    total_counts: Counter = Counter()
    weighted_score = 0.0
    total_positions = 0
    for fen_file in files:
        fens = _load_fens(fen_file)
        if not fens:
            continue
        counts, avg_score = label_fens(fens, engine_path, depth)
        dataset_positions = sum(counts.values())
        total_counts.update(counts)
        weighted_score += avg_score * dataset_positions
        total_positions += dataset_positions
    if total_positions == 0:
        raise RuntimeError("No valid FEN positions were discovered across datasets.")
    avg_score = weighted_score / total_positions
    return total_counts, avg_score


def save_versioned_weights(payload: Dict[str, object], base_path: Path, versions_dir: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versions_dir.mkdir(parents=True, exist_ok=True)
    version_path = versions_dir / f"macro_weight_pack_{timestamp}.swp"
    version_path.write_text(json.dumps(payload, indent=2) + "\n")
    if base_path.exists():
        backup = base_path.with_name(base_path.stem + f"_backup_{timestamp}" + base_path.suffix)
        shutil.copy2(base_path, backup)
    shutil.copy2(version_path, base_path)
    return version_path


def run_validation() -> Tuple[bool, str]:
    engine = MacroEngine()
    for label, fen in VALIDATION_POSITIONS:
        board = chess.Board(fen)
        if label == "krk" and not engine._is_krk_board(board):
            return False, "KRK detector failed on validation FEN"
        if label == "kpk" and not engine._is_kpk_board(board):
            return False, "KPK detector failed on validation FEN"
        env = {"board": board}
        engine.step(env)
        macro = env.get("macro_frame")
        if not isinstance(macro, dict):
            return False, f"Macro frame missing for {label} scenario"
    return True, "Validation suite passed"


def summarize_changes(before: Dict[str, object], after: Dict[str, object]) -> List[str]:
    diffs: List[str] = []
    for section in ["por_edges", "por_policies"]:
        before_section = before.get(section, {}) if isinstance(before, dict) else {}
        after_section = after.get(section, {}) if isinstance(after, dict) else {}
        keys = set(before_section) | set(after_section)
        for key in sorted(keys):
            if before_section.get(key) != after_section.get(key):
                diffs.append(f"{section}.{key}: {before_section.get(key)} -> {after_section.get(key)}")
    return diffs or ["No numerical changes recorded (check teacher metadata)."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end loop for labeling datasets and refreshing macro weights.")
    parser.add_argument("--datasets", type=Path, default=DATASET_ROOT, help="Root folder containing *.fen datasets (default: data/endgames)")
    parser.add_argument("--engine", type=str, default=None, help="Optional path to Stockfish binary")
    parser.add_argument("--depth", type=int, default=4, help="Stockfish depth for labeling")
    parser.add_argument("--base-weights", type=Path, default=DEFAULT_BASE_WEIGHTS, help="Live macro weight pack path")
    parser.add_argument("--versions-dir", type=Path, default=DEFAULT_VERSION_DIR, help="Directory for timestamped weight snapshots")
    parser.add_argument("--phase-output", type=Path, default=DEFAULT_PHASE_OUTPUT, help="Path for blended phase weight pack (default: weights/krk_phase_weight_pack.swp)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip MacroEngine validation step")
    parser.add_argument("--skip-phase-weights", action="store_true", help="Skip blended phase weight refresh")
    parser.add_argument("--dry-run", action="store_true", help="Compute candidates but do not write any files")
    parser.add_argument("--with-subgraphs", action="store_true", help="Also train subgraph weights (e.g., KPK) before macro merge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Optional subgraph trainers
    if args.with_subgraphs:
        try:
            from demos.experiments.kpk_train import train_kpk
            kpk_out = Path('weights/subgraphs/kpk_weight_pack.swp')
            train_kpk(Path('data/endgames/kpk'), kpk_out)
            print('Trained KPK subgraph weights ->', kpk_out)
        except Exception as e:
            print('Warning: KPK trainer failed:', e)

    dataset_files = discover_datasets(args.datasets)
    if not dataset_files:
        raise SystemExit(f"No *.fen files found under {args.datasets}.")

    phase_fens: List[str] = []
    for fen_file in dataset_files:
        phase_fens.extend(_load_fens(fen_file))

    counts, avg_score = aggregate_labels(dataset_files, args.engine, args.depth)

    base_payload = json.loads(args.base_weights.read_text()) if args.base_weights.exists() else {"version": "0.1"}
    updated_payload = json.loads(json.dumps(base_payload))  # deep copy via json roundtrip
    apply_weight_update(updated_payload, dict(counts), avg_score)

    print("Labelled datasets:")
    for path in dataset_files:
        print(f" • {path}")
    print("Counts:", dict(counts))
    print("Average score:", round(avg_score, 2))

    if args.dry_run:
        print("Dry run requested; not writing weight files.")
        return

    version_path = save_versioned_weights(updated_payload, args.base_weights, args.versions_dir)
    print("Saved versioned weights to", version_path)

    if not args.skip_validation:
        ok, message = run_validation()
        if not ok:
            raise SystemExit(f"Validation failed: {message}")
        print(message)
    else:
        print("Validation skipped per flag.")

    for line in summarize_changes(base_payload, updated_payload):
        print(line)

    if not args.skip_phase_weights and phase_fens:
        try:
            run_phase_teacher(phase_fens, args.phase_output, args.engine, args.depth)
            print("Updated phase weights ->", args.phase_output)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print("Warning: phase weight refresh failed:", exc)


if __name__ == "__main__":
    main()
