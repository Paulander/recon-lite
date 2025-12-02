# Visualization Spec (ReCoN-lite Replay)

## JSON frame schema
Fields:
- `type`: "snapshot"
- `tick`: integer, monotonically increasing
- `note`: optional
- `nodes`: map `node_id -> state_name` where state ∈ {INACTIVE, REQUESTED, WAITING, TRUE, CONFIRMED, FAILED}
- `new_requests`: optional list of node_ids requested at this tick
- `env`: optional dict (e.g., chess: `{ "fen": "<FEN string>" }`)
- `thoughts`: optional string (overlay commentary)
- `latents`: optional map `node_id -> [float, ...]`
- `macro_frame`: optional dict describing top-level macrograph state:
  - `version`: schema version (string)
  - `goal_vector`: map goal id → float `[0, 1]`
  - `phase_mix`: map phase id → float (sums ~1)
  - `plan_groups`: list of `{ "id", "activation", "plans": [...], "details": [{"name","highlight"}] }`
  - `feature_groups`: list of `{ "id", "confidence", "features": [...], "details": [{"name","highlight"}] }`
  - `bindings`: map namespace → list of `{ "feature", "items": [...] }`
  - `move_synth`: `{ "weights": {...}, "proposals": [{ "uci", "score", "components": {...} }...], "chosen": "uci" }`

### M3/M4 Plasticity & Consolidation Extensions

Additional optional fields in the JSON frame schema:

- `plasticity`: optional dict describing fast plasticity state:
  - `edge_key -> { "eligibility": float, "delta_sum": float, "w_init": float }`
  - Only includes edges with non-zero eligibility or delta

- `bandit`: optional dict describing bandit arm state:
  - `parent_id -> { child_id -> { "pulls": int, "mean_reward": float } }`

- `consolidation`: optional dict describing slow consolidation state:
  - `w_base`: map `edge_key -> float` (persistent baseline weights)
  - `total_episodes`: int
  - `edges_tracked`: int

- `m3_reward_tick`: optional float, the reward signal for this tick
- `m3_modulators`: optional dict with `{ "eta_tick_eff": float, "c_explore_eff": float }`
- `m3_weight_deltas`: optional map `edge_key -> float` (weight changes this tick)

## Player layout
- **Left (main)**: chess board from `env.fen` (or placeholder).
- **Bottom-left**: "selfie" image + `thoughts` text.
- **Right-top**: node map (graph) colored by state.
- **Right-bottom**: phase schematic (layers), also colored.

## Colors
- INACTIVE gray, REQUESTED blue, WAITING orange, TRUE lime, CONFIRMED green, FAILED red.

### Edge weight visualization (M3/M4)

Edge thickness and color can reflect weight state:

- **Thickness**: proportional to current weight (thicker = higher weight)
- **Color modes** (toggle via `setWBaseMode(mode)`):
  - `"baseline"`: show `w_base` only (consolidation baseline)
  - `"delta"`: show `Δw_fast` (within-episode changes)
  - `"combined"`: show `w_base + Δw_fast` (effective weight)
- **Drift coloring**: edges can be colored by drift from initial:
  - Increased weight: blue gradient
  - Decreased weight: red gradient
  - No change: default gray

## Controls
- Play/pause/step.
- Scrub by tick.
- Hover node → show state, optional latent vector.
- Toggle layouts (force-directed vs layered).
- **M4**: Weight display mode toggle (baseline/delta/combined).

## Data size tips
- Keep `env` compact (FEN only).
- Limit latent dims (2–8) if used.
- Plasticity/consolidation data should only include edges with changes (sparse).

## M4 CLI Tools

For offline analysis and reporting:

- `tools/trace_summarize.py`: aggregate episode metrics from JSONL traces
- `tools/bandit_refresh.py`: update bandit priors from traces
- `tools/consolidate_batch.py`: offline batch consolidation
- `tools/report_consolidation.py`: generate markdown reports with histograms
- `tools/pack_diff.py`: compare two weight packs and show differences
