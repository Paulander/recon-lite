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

## Player layout
- **Left (main)**: chess board from `env.fen` (or placeholder).
- **Bottom-left**: “selfie” image + `thoughts` text.
- **Right-top**: node map (graph) colored by state.
- **Right-bottom**: phase schematic (layers), also colored.

## Colors
- INACTIVE gray, REQUESTED blue, WAITING orange, TRUE lime, CONFIRMED green, FAILED red.

## Controls
- Play/pause/step.
- Scrub by tick.
- Hover node → show state, optional latent vector.
- Toggle layouts (force-directed vs layered).

## Data size tips
- Keep `env` compact (FEN only).
- Limit latent dims (2–8) if used.
