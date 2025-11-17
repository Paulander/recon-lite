# HOWTO: Run, Train, and Evaluate ReCoN-lite (KRK/KPK)

This is the quick-start for running the chess subgraphs with weight packs (SWPs), logging traces, and evaluating progress.

## Weight Packs (SWPs)
- SWPs live under `weights/`: `macro_weight_pack.swp`, `macro_threshold_pack.swp`, `phase_layer_pack.swp`, `krk_phase_weight_pack.swp`, `subgraphs/kpk_weight_pack.swp`, etc.
- Set `RECON_PHASE_WEIGHT_FILE` (for KRK) or pass `--pack` to eval scripts to pin a specific pack. Packs are hashed in logs for provenance.

## Running demos
- Persistent KRK demo (with edge-thickness viz from weights):
  ```bash
  uv run python demos/persistent/krk_persistent_demo.py \
    --use-blended-actuator \
    --phase-microticks 5 --phase-eta 0.3
  ```
  Loads the current `krk_phase_weight_pack.swp` by default; attach `RECON_PHASE_WEIGHT_FILE` to override.

## Batch evaluation
- Single block, KRK/KPK, optional Stockfish labeling:
  ```bash
  uv run python demos/experiments/batch_eval.py \
    --mode krk --fen-file data/endgames/krk/sample.fen \
    --runs 100 --max-plies 100 --max-ticks 200 \
    --pack weights/krk_phase_weight_pack.swp \
    --engine /path/to/stockfish --depth 2 \
    --trace-out reports/krk_trace.jsonl
  ```
- Blocked evaluation with checkpoints + viz samples:
  ```bash
  uv run python demos/experiments/block_runner.py \
    --mode kpk --fen-file data/endgames/kpk/sample.fen \
    --runs-per-block 200 --blocks 5 \
    --pack weights/subgraphs/kpk_weight_pack.swp \
    --engine /path/to/stockfish --depth 2 \
    --out-dir reports/blocks
  ```
  Produces `summary.json`, per-block traces, pack copies, and one viz log per block.

## Tracing
- Trace schema lives in `src/recon_lite/trace_db.py`:
  - `TickRecord`: tick_id, phase_estimate, goal_vector, board_fen, active_nodes, fired_edges, action, eval_before/after, reward_tick, meta.
  - `EpisodeRecord`: episode_id, result, ticks, pack_meta (path + sha256), notes.
- Batch eval and block runner accept `--trace-out` to emit JSONL traces; packs are fingerprinted automatically.

## Training / Pack refresh
- Weight packs are refreshed by the teachers (e.g., `demos/experiments/teacher_stockfish.py`); run with your FEN sets and Stockfish to update SWPs.
- For KPK/rook subgraphs, keep separate SWPs so components can evolve independently.

## Visualization
- Macrograph HTML shows KRK/KPK subgraphs; edge thickness scales with edge weights from the spec/SWPs.
- KRK/KPK per-tick viz: load the viz JSON produced by the persistent demo or block runner sample; weights are embedded in the attached graph.
