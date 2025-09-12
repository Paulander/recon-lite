# Implementation Plan (ReCoN-lite → KRK Demo)

## Phase 0 — Repo + Run
- Install: `uv venv .venv && source .venv/bin/activate && uv pip install -e .`
- Run demo: `uv run python -m demos.sequence_demo`

## Phase 1 — Engine + Logging (done)
- `Graph`: nodes (SCRIPT/TERMINAL), links (SUB/POR).
- `Engine.step(env)`: terminals advance; scripts request children when POR-predecessors TRUE; scripts confirm when last nodes of POR-chains confirm.
- `RunLogger.snapshot(...)`: per-tick frame with `tick, nodes, new_requests, env, thoughts, latents` → JSON.

## Phase 2 — Visualization JSON schema
- Each frame example:
{
  "type": "snapshot",
  "tick": 7,
  "note": "tick 7",
  "nodes": { "ROOT": "WAITING", "PHASE1": "CONFIRMED" },
  "new_requests": ["PHASE2"],
  "env": { "fen": "8/8/8/8/8/8/8/8 w - - 0 1" },
  "thoughts": "Driving BK north; chosen move: Rh8+; box=4x3; opposition=false",
  "latents": { "PHASE2": [0.12, -0.7, 0.1] }
}

## Phase 3 — Plugins API
- Add `src/recon_lite/plugins.py`:
  - `TerminalPlugin` Protocol with `.reset()` and `.step(node, env) -> (done, success)`.
  - Examples: `MatePlugin`, `OnEdgePlugin`, `ChooseMovePhase1`.
- Terminals accept `predicate=lambda n, env: plugin.step(n, env)`.

## Phase 4 — Chess substrate
- Add dependency: `uv add python-chess` (updates `pyproject.toml` and lockfile).
- `env` holds a `chess.Board`, plus helper values (`chosen_move`, features, etc.).
- After each `engine.step`, if `env["chosen_move"]` exists, push it and log `fen`.

## Phase 5 — KRK ReCoN graph
- Scripts: `PHASE1 → PHASE2 → PHASE3 → PHASE4` via POR.
- Terminals per phase:
  - P1: `on_edge(BK)`, `rook_safe`, `choose_move_p1`
  - P2: `box_size <= target`, `rook_safe`, `choose_move_p2`
  - P3: `has_opposition`, `rook_safe`, `choose_move_p3`
  - P4: `is_mate` (or `choose_mate`)
- Rule-based plugins first; ML later if time permits.

## Phase 6 — Visualization (two-pane + board)
- Use the JSON to build a UI:
  - **Board**: render FEN per tick.
  - **Selfie + thoughts**: static image + `thoughts` string.
  - **Node map**: color by state.
  - **Schematic phases**: layered boxes with their terminals, colored by state.

## Phase 7 — Extras (optional)
- `RET` links; explicit alternative-groups (OR).
- Latent view: record small feature vectors per node and embed later.
- Export `run.json` for the viz.
