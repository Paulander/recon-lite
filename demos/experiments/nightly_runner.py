#!/usr/bin/env python3
"""
Nightly runner stub: executes block_runner with a provided config.

Example config (JSON):
{
  "mode": "kpk",
  "fen_file": "data/endgames/kpk/sample.fen",
  "runs_per_block": 200,
  "blocks": 5,
  "packs": ["weights/subgraphs/kpk_weight_pack.swp"],
  "engine": "/path/to/stockfish",
  "depth": 2,
  "out_dir": "reports/nightly_kpk"
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from demos.experiments import block_runner


def run_from_config(cfg: Dict[str, Any]) -> None:
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
    # Delegate to block_runner.main style function
    block_runner.main(args=args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly runner for block evaluations.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    run_from_config(cfg)


if __name__ == "__main__":
    main()
