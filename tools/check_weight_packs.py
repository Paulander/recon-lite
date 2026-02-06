#!/usr/bin/env python3
"""
Quick sanity check for consolidation packs in weights/{latest,nightly}.

Prints:
- number of tracked edges
- how many weights differ from 1.0
- whether keys look like the expected subgraph (kpk_/kqk_/krk_)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path, *, expect_prefix: str | None = None) -> str:
    if not path.exists():
        return f"{path}: MISSING"

    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return f"{path}: ERROR {exc}"

    w_base = data.get("w_base", {}) or {}
    vals = list(w_base.values())
    if not vals:
        return f"{path}: entries=0"

    non1 = sum(1 for v in vals if abs(float(v) - 1.0) > 1e-9)
    keys = list(w_base.keys())

    prefix_ok = "n/a"
    if expect_prefix is not None:
        good = sum(1 for k in keys if str(k).startswith(expect_prefix))
        prefix_ok = f"{good}/{len(keys)} startwith({expect_prefix})"

    return (
        f"{path}: entries={len(vals)} non1={non1} "
        f"min={min(vals):.4f} max={max(vals):.4f} prefix={prefix_ok}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights/latest"),
        help="Directory containing *_consol.json files (default: weights/latest)",
    )
    args = parser.parse_args()

    wdir = args.weights_dir
    # Note: some packs may store unprefixed edge keys and rely on the unified
    # loader to apply a prefix at load time. This checker mainly catches
    # "polluted packs" (e.g., full-game edges inside a KQK pack).
    packs = [
        ("fullgame_consol.json", None),
        ("krk_consol.json", None),
        ("kpk_consol.json", "kpk_"),
        ("kqk_consol.json", "kqk_"),
    ]

    for fname, prefix in packs:
        print(summarize(wdir / fname, expect_prefix=prefix))


if __name__ == "__main__":
    main()
