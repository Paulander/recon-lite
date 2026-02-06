#!/usr/bin/env python3
"""
Compress visualization logs by removing idle frames.

Keeps frames that:
  - change node states vs previous frame
  - contain new requests
  - contain move/decision markers in env/note
Optionally keep every Nth frame as a heartbeat.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


IMPORTANT_NOTE_TOKENS = (
    "ReCoN ply",
    "Opponent ply",
    "Decision",
    "Diagnostics",
)


def _nodes_signature(nodes: Dict[str, Any]) -> int:
    if not nodes:
        return 0
    # Use a stable hash of (id,state) pairs.
    items: Iterable[Tuple[str, Any]] = nodes.items()
    return hash(tuple(sorted(items)))


def _is_important(frame: Dict[str, Any]) -> bool:
    note = frame.get("note", "") or ""
    if any(token in note for token in IMPORTANT_NOTE_TOKENS):
        return True
    env = frame.get("env", {}) or {}
    keys = ("chosen_move", "recons_move", "opponent_move", "opponents_move")
    return any(env.get(k) for k in keys)


def compress_frames(frames: list[dict], *, keep_every: int = 0) -> list[dict]:
    kept: list[dict] = []
    prev_sig: int | None = None

    for idx, frame in enumerate(frames):
        nodes = frame.get("nodes", {}) or {}
        sig = _nodes_signature(nodes)
        changed = prev_sig is None or sig != prev_sig
        prev_sig = sig

        new_req = frame.get("new_requests") or []
        keep = False

        if changed:
            keep = True
        elif new_req:
            keep = True
        elif _is_important(frame):
            keep = True
        elif keep_every and idx % keep_every == 0:
            keep = True

        if keep:
            kept.append(frame)

    if frames and (kept[-1] is not frames[-1]):
        kept.append(frames[-1])

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress visualization JSON logs.")
    parser.add_argument("--input", required=True, help="Input visualization JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--keep-every", type=int, default=0,
                        help="Keep every Nth frame as heartbeat (0 disables)")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    frames = json.loads(inp.read_text())
    if not isinstance(frames, list):
        raise SystemExit("Expected a list of frames.")

    kept = compress_frames(frames, keep_every=args.keep_every)
    out.write_text(json.dumps(kept, indent=2))
    print(f"Compressed {len(frames)} -> {len(kept)} frames: {out}")


if __name__ == "__main__":
    main()
