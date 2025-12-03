#!/usr/bin/env python3
"""
M5.5: Extended nightly runner with motif extraction and trust scoring.

Example config (JSON):
{
  "mode": "krk",
  "fen_file": "data/endgames/krk/sample.fen",
  "runs_per_block": 200,
  "blocks": 5,
  "packs": ["weights/subgraphs/kpk_weight_pack.swp"],
  "engine": "/path/to/stockfish",
  "depth": 2,
  "out_dir": "reports/nightly_krk",
  
  "motif_extraction": {
    "enabled": true,
    "reward_threshold": 0.3,
    "activity_threshold": 3
  },
  "trust_scoring": {
    "enabled": true,
    "freeze_threshold": 0.3,
    "promote_threshold": 0.8
  },
  "consolidation": {
    "enabled": true,
    "state_path": "weights/nightly/consol_state.json"
  }
}

Pipeline:
1. Run batch traces (block_runner)
2. Extract motifs from traces
3. Consolidate weights (if enabled)
4. Compute trust scores
5. Generate combined report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_block_evaluation(cfg: Dict[str, Any]) -> Optional[Path]:
    """Run block evaluation and return trace directory."""
    try:
        from demos.experiments import block_runner
    except ImportError:
        print("Warning: block_runner not available")
        return None
    
    args = argparse.Namespace(
        mode=cfg.get("mode", "krk"),
        fen_file=Path(cfg["fen_file"]),
        runs_per_block=int(cfg.get("runs_per_block", 50)),
        blocks=int(cfg.get("blocks", 3)),
        max_plies=int(cfg.get("max_plies", 100)),
        max_ticks=int(cfg.get("max_ticks", 200)),
        pack=[Path(p) for p in cfg.get("packs", [])],
        engine=Path(cfg["engine"]) if cfg.get("engine") else None,
        depth=int(cfg.get("depth", 2)),
        out_dir=Path(cfg.get("out_dir", "reports/blocks")),
    )
    block_runner.main(args=args)
    return args.out_dir


def run_motif_extraction(
    trace_paths: List[Path],
    output_path: Path,
    reward_threshold: float = 0.3,
    activity_threshold: int = 3,
) -> Optional[Path]:
    """Run motif extraction on traces."""
    if not trace_paths:
        return None
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "demos" / "experiments" / "extract_motifs.py"),
        "--traces",
    ]
    cmd.extend([str(p) for p in trace_paths])
    cmd.extend([
        "--out", str(output_path),
        "--reward-threshold", str(reward_threshold),
        "--activity-threshold", str(activity_threshold),
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Motif extraction failed: {result.stderr}")
        return None
    
    print(result.stdout)
    return output_path if output_path.exists() else None


def run_trust_scoring(
    trace_paths: List[Path],
    output_path: Path,
    previous_report: Optional[Path] = None,
    consolidation_state: Optional[Path] = None,
    freeze_threshold: float = 0.3,
    promote_threshold: float = 0.8,
) -> Optional[Path]:
    """Run trust scoring on traces."""
    if not trace_paths:
        return None
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "trust_report.py"),
        "--traces",
    ]
    cmd.extend([str(p) for p in trace_paths])
    cmd.extend([
        "--out", str(output_path),
        "--freeze-threshold", str(freeze_threshold),
        "--promote-threshold", str(promote_threshold),
    ])
    
    if previous_report and previous_report.exists():
        cmd.extend(["--previous", str(previous_report)])
    if consolidation_state and consolidation_state.exists():
        cmd.extend(["--consolidation", str(consolidation_state)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Trust scoring failed: {result.stderr}")
        return None
    
    print(result.stdout)
    return output_path if output_path.exists() else None


def run_consolidation(
    trace_paths: List[Path],
    state_path: Path,
    output_pack: Optional[Path] = None,
) -> Optional[Path]:
    """Run batch consolidation on traces."""
    if not trace_paths:
        return None
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "consolidate_batch.py"),
        "--traces",
    ]
    cmd.extend([str(p) for p in trace_paths])
    cmd.extend(["--state", str(state_path)])
    
    if output_pack:
        cmd.extend(["--out-pack", str(output_pack)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Consolidation failed: {result.stderr}")
        return None
    
    print(result.stdout)
    return state_path if state_path.exists() else None


def generate_combined_report(
    out_dir: Path,
    motifs_path: Optional[Path],
    trust_path: Optional[Path],
    consol_state: Optional[Path],
) -> Path:
    """Generate a combined nightly report."""
    report_path = out_dir / f"nightly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    lines = [
        "# Nightly Pipeline Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]
    
    # Motif summary
    lines.append("## Motif Extraction")
    if motifs_path and motifs_path.exists():
        from recon_lite.motifs.descriptors import MotifDataset
        dataset = MotifDataset.load(motifs_path)
        stats = dataset.statistics()
        lines.append(f"- Total motifs: {stats.get('count', 0)}")
        lines.append(f"- Avg outcome: {stats.get('avg_outcome', 0):.4f}")
        lines.append(f"- Positive ratio: {stats.get('positive_ratio', 0):.1%}")
        if stats.get('type_counts'):
            lines.append("- By type:")
            for t, c in stats['type_counts'].items():
                lines.append(f"  - {t}: {c}")
    else:
        lines.append("- Not run or no motifs extracted")
    lines.append("")
    
    # Trust summary
    lines.append("## Trust Scores")
    if trust_path and trust_path.exists():
        with open(trust_path) as f:
            trust_data = json.load(f)
        summary = trust_data.get("summary", {})
        lines.append(f"- Generation: {trust_data.get('generation', 0)}")
        lines.append(f"- Nodes tracked: {summary.get('total_nodes', 0)}")
        lines.append(f"- Edges tracked: {summary.get('total_edges', 0)}")
        lines.append(f"- Freeze candidates: {summary.get('freeze_candidates', 0)}")
        lines.append(f"- Promote candidates: {summary.get('promote_candidates', 0)}")
    else:
        lines.append("- Not run")
    lines.append("")
    
    # Consolidation summary
    lines.append("## Consolidation")
    if consol_state and consol_state.exists():
        with open(consol_state) as f:
            consol_data = json.load(f)
        lines.append(f"- Episodes processed: {consol_data.get('episode_count', 0)}")
        lines.append(f"- Edges tracked: {len(consol_data.get('edges', {}))}")
    else:
        lines.append("- Not run")
    lines.append("")
    
    # Write report
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    
    return report_path


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full nightly pipeline from config."""
    out_dir = Path(cfg.get("out_dir", "reports/nightly"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "started_at": datetime.now().isoformat(),
        "config": cfg,
    }
    
    # Step 1: Run block evaluation
    print("\n=== Step 1: Block Evaluation ===")
    trace_dir = run_block_evaluation(cfg)
    
    # Find trace files
    trace_paths: List[Path] = []
    if trace_dir and trace_dir.exists():
        trace_paths = list(trace_dir.glob("*.json")) + list(trace_dir.glob("*.jsonl"))
    results["trace_count"] = len(trace_paths)
    
    # Step 2: Motif extraction (optional)
    motif_cfg = cfg.get("motif_extraction", {})
    motifs_path = None
    if motif_cfg.get("enabled", False) and trace_paths:
        print("\n=== Step 2: Motif Extraction ===")
        motifs_path = run_motif_extraction(
            trace_paths,
            out_dir / "motifs.jsonl",
            reward_threshold=motif_cfg.get("reward_threshold", 0.3),
            activity_threshold=motif_cfg.get("activity_threshold", 3),
        )
    results["motifs_path"] = str(motifs_path) if motifs_path else None
    
    # Step 3: Consolidation (optional)
    consol_cfg = cfg.get("consolidation", {})
    consol_state = None
    if consol_cfg.get("enabled", False) and trace_paths:
        print("\n=== Step 3: Consolidation ===")
        consol_state = run_consolidation(
            trace_paths,
            Path(consol_cfg.get("state_path", "weights/nightly/consol_state.json")),
        )
    results["consolidation_state"] = str(consol_state) if consol_state else None
    
    # Step 4: Trust scoring (optional)
    trust_cfg = cfg.get("trust_scoring", {})
    trust_path = None
    if trust_cfg.get("enabled", False) and trace_paths:
        print("\n=== Step 4: Trust Scoring ===")
        trust_path = run_trust_scoring(
            trace_paths,
            out_dir / "trust.json",
            previous_report=Path(trust_cfg.get("previous_report")) if trust_cfg.get("previous_report") else None,
            consolidation_state=consol_state,
            freeze_threshold=trust_cfg.get("freeze_threshold", 0.3),
            promote_threshold=trust_cfg.get("promote_threshold", 0.8),
        )
    results["trust_path"] = str(trust_path) if trust_path else None
    
    # Step 5: Generate combined report
    print("\n=== Step 5: Generate Report ===")
    report_path = generate_combined_report(out_dir, motifs_path, trust_path, consol_state)
    results["report_path"] = str(report_path)
    results["completed_at"] = datetime.now().isoformat()
    
    # Save results manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Complete ===")
    print(f"Report: {report_path}")
    print(f"Manifest: {manifest_path}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended nightly runner with motif extraction and trust scoring."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run_from_config(cfg)


if __name__ == "__main__":
    main()
