# Tracing, Weight Packs, and Evaluation Runs

This short guide summarizes how to log runs, capture weight-pack provenance, and evaluate KRK/KPK variants while we stay CPU-friendly (shallow Stockfish).

## Trace schema
- `TickRecord`: `tick_id`, `phase_estimate`, `goal_vector`, `board_fen`, `active_nodes`, `fired_edges`, `action`, `eval_before`, `eval_after`, `reward_tick`, `meta`.
- `EpisodeRecord`: `episode_id`, `result`, `ticks: [TickRecord...]`, `pack_meta` (list of `{path, sha256}` for weight packs), `notes` (free-form, e.g., engine depth, SWP names).
- JSONL writer lives in `src/recon_lite/trace_db.py` with helpers to hash packs (`pack_fingerprint`).

## Weight packs (SWPs)
- SWPs replace the old “sidecars”: `weights/macro_weight_pack.swp`, `weights/macro_threshold_pack.swp`, `weights/phase_layer_pack.swp`, `weights/krk_phase_weight_pack.swp`, `weights/subgraphs/kpk_weight_pack.swp`, etc.
- Trainers (e.g., `demos/experiments/teacher_stockfish.py`) read/write SWPs; runtime loaders pull them automatically. Include SWP paths + hashes in episode notes for reproducibility.

## Evaluation harness goals
- Iterate over SWPs, run batches of FENs with shallow Stockfish labels, and log win/stall/draw + tick/ply counts (`demos/experiments/batch_eval.py`).
- Stop criteria for KPK (example): >95% win rate on Stockfish-labeled wins across the last block (block metrics emitted when `--block-size` is set).
- Save checkpoints every block (e.g., 100 games): copy SWPs + a sample viz log so we can replay progress (`--checkpoint-dir`).
- Modes: `--mode krk|kpk`, optional `--engine /path/to/stockfish --depth 2`, `--pack` paths recorded with hashes, `--trace-out` for JSONL ticks/episodes.

## Visualization
- Macro viz shows `KRKSubgraph`/`KPKSubgraph` nodes with edge thickness scaled by weights.
- KRK per-tick viz will render any attached graph; attach KPK to see it similarly when you run a KPK-only demo.

## Next steps (implementation checklist)
- [ ] Batch evaluation script: accept `--packs`, `--fen-file`, `--engine`, `--depth`, `--runs`, emit metrics JSON/CSV.
- [ ] Integrate TraceDB into KRK/KPK loops so every run captures episode/tick details plus SWP fingerprints.
- [ ] Nightly wrapper to iterate packs and datasets; write results to `reports/` with timestamps.
- [ ] Optional: snapshot viz logs per block for side-by-side comparison.
