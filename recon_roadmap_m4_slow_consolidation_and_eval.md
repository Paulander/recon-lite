Here’s a fully fleshed-out roadmap for **M4: Slow Consolidation & Eval Upgrade**, mirroring the depth/format of `recon_roadmap_m3_fast_plasticity.md`. You can drop this into a new file such as `recon_roadmap_m4_slow_consolidation.md`.

---

# M4: Slow Consolidation & Eval Upgrade

Goal: carry the fast, within-game adaptations from M3 into a persistent learning layer across games—stabilizing KRK/KPK performance, improving evaluation signals, and producing artifact-ready logs/visuals.

## Phase 0: Preconditions & Scope

- **Inputs**: Rich Tick/Episode traces with `reward_tick`, `goal_vector`, per-edge deltas, `bandit_stats`.
- **Outputs**: Updated `w_base` (persistent weights/packs), improved heuristic evaluation, dashboards capturing per-game learning.
- **Guardrails**:
  - Slow learning never corrupts known-good packs; rely on staging area & rollbacks.
  - Runtime path stays performant on CPU (ThinkPad target).

## Phase 1: Episode Summaries & Trace Enrichment (M4.1)

1. Extend `TraceDB`:
   - Episode-level aggregates (`avg_reward_tick`, `phase_usage`, `edge_delta_sums`, `bandit_means`).
   - Indexing by outcome, opening phase, opponent strength.
2. Summaries per episode:
   - Per-edge cumulative Δw_fast (for reset & consolidation).
   - Bandit pull counts + reward means.
   - Phase win/loss breakdown.
3. CLI tooling:
   - `trace_summarize.py`: export CSV/JSON of aggregated metrics.
   - Visual sanity checks (histograms for rewards, edge drift).

## Phase 2: Slow Weight Consolidation (M4.2)

1. **State split**:
   - Maintain `w_base` (persistent) + `Δw_fast` (per-episode).
   - At game end, compute candidate delta for `w_base`.
2. **Consolidation rules**:
   - Reward-weighted averaging, e.g., `w_base := w_base + η_consolidate * mean(Δw_fast * reward_episode)`.
   - Confidence gating: require N episodes before applying changes.
   - Clip vs baseline to avoid runaway.
3. **Implementation**:
   - `plasticity/consolidate.py` module with:
     - `ConsolidationState`, `ConsolidationConfig`.
     - `accumulate_episode(summary)`.
     - `apply_to_graph(graph, state)`.
   - Hooks in demos to call consolidation after episodes or batches.
4. **Storage**:
   - Persist `w_base` to new SWP variant (e.g., `weights/krk_fast_base.swp`).
   - Keep diffs for audit.

## Phase 3: Cross-Game Bandit Refresh (M4.3)

1. Batch stats:
   - Aggregate rewards per arm across episodes.
   - Compute confidence intervals.
2. Refresh algorithm:
   - Replace bandit priors (`pulls`, `sum_reward`) with smoothed cross-game stats.
   - Optional decay for aging data.
3. CLI:
   - `bandit_refresh.py --trace reports/krk_trace.jsonl --out weights/bandit_priors.json`.
4. Visualization:
   - Table of arm means, counts; highlight shifts vs previous pack.

## Phase 4: Evaluation Upgrade & Hybrid Signals (M4.4)

1. Expand heuristic eval (`eval/light.py`):
   - Material, king safety, pawn structure, piece activity, mobility weighting.
   - Add tactical heuristics hooks (forks/pins detectors from existing sensors).
2. Hybrid pipeline:
   - `EvalManager` chooses between Stockfish, improved heuristic, or cached evals.
   - Optionally sample Stockfish every N plies and train a lightweight regressor to mimic it (distillation).
3. Logging:
   - Store both heuristic and Stockfish evals when available.
   - Track eval error when teacher present to gauge heuristic accuracy.

## Phase 5: Dashboards & Artifact Hooks (M4.5)

1. **Consolidation dashboard**:
   - Plot `w_base` trajectory for top edges.
   - Show phase win rates over time.
   - Display bandit arm performance.
2. **Replays**:
   - Integrate consolidated weights into viz (edge thickness baseline vs fast delta).
   - Provide toggles: “baseline only”, “fast delta”, “combined”.
3. **Report scripts**:
   - `report_consolidation.py` generates markdown/HTML summary per training block.
   - Include before/after pack checksums.

## Phase 6: Tests & Benchmarks (M4.6)

1. Unit tests:
   - Consolidation math (expected updates given sample summaries).
   - Bandit prior refresh (ensures smoothing, gating).
   - Eval upgrade (regression tests on curated positions).
2. Integration tests:
   - KRK persistent run: ensure `w_base` changes only via consolidation, `Δw_fast` resets.
   - Batch evaluation comparing baseline vs consolidated packs.
3. Performance checks:
   - Measure CPU impact of new eval and consolidation steps.

## Phase 7: Rollout Plan & Safety (M4.7)

1. Staging workflow:
   - Run consolidation on trace subset → produce candidate packs.
   - Validate via `pack_tournament` vs baseline.
   - Promote only if metrics improve.
2. Rollback strategy:
   - Keep versioned packs (`weights/m4_baseline_v1.swp`, etc.).
   - CLI `pack_diff.py` to inspect differences (top edges changed).
3. Documentation:
   - Update `updates_continuous.md` (M4 section) once implemented.
   - Document CLI usage and expected outputs.

## File Summary

**New modules**
- `src/recon_lite/plasticity/consolidate.py`
- `src/recon_lite_chess/eval/light.py` (expanded heuristic)
- `tools/trace_summarize.py`
- `tools/bandit_refresh.py`
- `tools/report_consolidation.py`

**Modified**
- `demos/persistent/krk_persistent_demo.py` and `kpk_persistent_demo.py` (hook consolidation).
- `src/recon_lite/plasticity/fast.py` (integration points for `w_base`).
- `demos/visualization/network-visualization.js` (baseline vs delta view).
- `README.md`, `updates_continuous.md`, `VIS_SPEC.md` (to reflect M4 flow).

**Tests**
- `tests/test_consolidation.py`
- `tests/test_eval_light.py`
- Updates to `tests/test_plasticity_integration.py` to cover slow/fast interplay.

## Deliverables Checklist

- [ ] Episode summary enrichment landed (`TraceDB` + CLI exports).
- [ ] Consolidation engine adjusts `w_base` safely, with audit logs.
- [ ] Bandit priors refreshed from cross-game stats.
- [ ] Upgraded heuristic eval + optional distillation path.
- [ ] Dashboards/reporting for weight drift, bandit performance.
- [ ] Regression/batch tests showing no KRK regressions, measurable improvements.
- [ ] Documentation & visualization updates completed.

---

Let me know if you’d like this inserted into a file (switch to agent mode) or adjusted for specific milestones.