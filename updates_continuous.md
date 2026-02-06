# ReCoN-lite Chess: Continuous Activations, Binding, and Strategic Layer

We want the KRK demo to feel like a living ReCoN brain: continuous activations that settle over micro-ticks, bindings that keep track of which squares anchor each hypothesis, and a strategic layer that explains why a move is happening. This document keeps that intent front-and-center so we can add the harder engineering pieces without losing the intuition that the system is thinking through a plan.

## Scope and Goals
- Add continuous activations, binding, and micro-ticks without regressing existing KRK demos or scripts.
- Start with KRK (M1) so we can phase in latents, binding tables, and visualization upgrades in a controlled setting.
- Keep `python-chess` and reuse current KRK scripts/sensors; only incremental re-organization for now.
- Treat the KRK loop as the place where we practice running soft activations before we scale to KPK and the rest of the full game stack.
- For the detailed **M3 fast-plasticity & bandit-control plan** that builds on top of this continuous layer, see `recon_roadmap_m3_fast_plasticity.md`.

## Key Integrations (Use Existing APIs)
- Engine already exposes a `step` hook; we extend it with an optional micro-tick pre-phase when `env["microticks"] > 0` (default 0, so the legacy flow stays intact).
- Logger can already handle `latents`; we just need to populate per-phase activations into that payload.
- `krk_nodes` already names the phase nodes we care about; we use those labels to map logits/latents to the scripts that exist today.

## M1 (2 Weeks): KRK + Continuous Activations + Binding + Micro-ticks + Viz

**Intent:** give KRK a soft strategic layer and visual feedback so we can watch reasoning settle before a move fires. Every sub-piece is incremental and guarded by flags so ongoing demos keep working.

### 1. Core Continuous Activations (non-breaking)
- New `src/recon_lite/core/activations.py` for vector helpers (`sigmoid`, `softmax`) and an `ActivationState` per node that tracks current value, target, and settling rate.
- New `src/recon_lite/time/microtick.py` with `settle(y, target, eta, k)` plus `run_microticks(callback_compute_targets, steps, eta)` so we can iterate the activations a few times before committing the tick.
- Update `ReConEngine.step(env)` to optionally run micro-ticks (e.g., `env.get("microticks", 0)`); default path is zero micro-ticks so nothing breaks for current demos.

### 2. Binding Manager (instances + namespaces)
- New `src/recon_lite/binding/manager.py` with:
  - `BindingInstance(feature_id: str, items: set[str])` to describe “rook on a7” style bindings.
  - `BindingTable.begin_tick(namespace: str)`, `reserve(instance)`, `conflicts(instance)`, and `invalidate_on_board_change(board)`.
- Namespaces will follow the KRK structure (`krk/p1/drive`, `krk/p2/shrink`, etc.) so bindings stay local to each hypothesis.
- First wave of binding instances: rook square, our king square, enemy king square, target fence line squares, and box corners so we can visualize anchoring in the viz.

### 3. Strategic Layer for KRK
- New `src/recon_lite_chess/strategy.py` with `compute_phase_logits(board)` built on existing detectors (`has_stable_cut`, `enemy_at_edge`, `box_min_side`, `has_opposition_after`, `can_deliver_mate`).
- Apply a softmax with a mild temperature to produce `phase_latents`; surface them via `env["phase_latents"]` and log as `latents` keyed to script IDs (`phase0_establish_cut`, …, `phase4_deliver_mate`).
- Add placeholder scaffolding for `OutcomeMode` and `StyleBias`, returning neutral vectors for now so we can slot them in later without rethreading.

### 4. Persistent KRK Loop Integration
- In `demos/persistent/krk_persistent_demo.py`, run micro-ticks for a few steps (e.g., 5 with `eta=0.3`) before calling `engine.step`.
- Maintain a `BindingTable` across plies; call `invalidate_on_board_change` after either side moves; place bindings in `env["binding"]`.
- Log activations and bindings every N ticks, keeping the existing phase scripts and actuators so behavior stays recognizable.

### 5. Move Selection: Soft Blending (optional)
- New `src/recon_lite_chess/actuators_blend.py` as an optional move chooser that blends candidate moves from phases P1/P2/P3.
- Score candidates via `phase_weight * phase_score + cheap_eval`, falling back to existing phase-specific choosers.
- Make integration flag-controlled so the default KRK loop continues to use the current discrete chooser unless explicitly enabled.

### 6. Visualization Upgrades
- Update `demos/visualization/network-visualization.js` to consume `frame.latents` and animate node fill alpha + a subtle pulse when activation increases.
- Update `demos/visualization/chess-board.js` to overlay bindings (rook, kings, target fence) with colors keyed by namespace.
- Use the existing wrapper collapse toggle to add a rationale ribbon driven by `env["last_reason"]`.

### 7. Tests and Benchmarks
- Unit tests for binding conflicts, micro-tick convergence, and monotonicity of phase logits with respect to the existing detectors.
- Regression test: random KRK positions should mate vs random defense within 50 plies; explicitly assert no stalemate regressions.

## Control Loop (M1 Persistent)
1. Sense: update board state and invalidate bindings if the board changed.
2. Micro-ticks: compute `phase_latents` via `strategy.compute_phase_logits` and settle them with `microtick.run_microticks`.
3. Requests: keep the existing POR/SUB request flow so scripts still wake up in the same order.
4. Actions: push candidate moves through existing actuators; optionally weight them with the blended chooser.
5. Score: reuse current actuator scoring with a cheap eval term in the blender to keep move choice stable.
6. Act: execute the chosen move and log a trace with activations and bindings for visualization.

## M2 (4–6 Weeks): KPK + Rook Techniques + Script-weight Learning
- Extend sensors with KPK cues and rook technique detectors under `src/recon_lite_chess/sensors/{structure.py,tactics.py}`.
- Add scripts in `src/recon_lite_chess/scripts/{kpk.py,rook_endings.py}` that reuse the new binding namespaces and activations.
- Build a lightweight teacher (`demos/experiments/teacher_stockfish.py`) running Stockfish depth 4 to label script choices and learn child weights `a_i`; emit them to `weights/krk_phase_weight_pack.swp` (Subgraph Weight Pack, formerly “sidecar”) and have `actuators_blend` consume the pack (with overrides for experiments) so blended scoring reflects the learned preferences.

## M3 (8–10 Weeks): Minimal Opening/Middlegame Layer + Tactical Detectors
- Flesh out the strategic layer with `GamePhase`, `OutcomeMode`, and `StyleBias` continuous nodes, routed into plan-selection weights.
- Expand sensors with tactical detectors (forks, pins, hanging pieces) and structural cues (open files, outposts) returning confidence `p`.
- Author 3–5 opening and 3–5 middlegame scripts that hook into the binding system; run 100 rapid games vs Stockfish depth 1/2 aiming for Elo 1200–1400.

## M4 (12–14 Weeks): Self-play Fine-tuning + Improved Eval + Artifact
- Launch self-play to fine-tune child weights and upgrade the cheap eval (`material + king safety + mobility`) in `src/recon_lite_chess/eval/light.py`.
- Ship an interactive viz and a short write-up that walks through the architecture and showcases activation/binding timelines.

## Deliverables Snapshot
- KRK M1: 100% mate from random KRK vs random defense; pulsing viz; binding overlays; reproducible demo script with flags for micro-ticks and blended actuators.
- Documentation: refresh `ARCHITECTURE.md` and `VIS_SPEC.md` to reflect continuous activations, bindings, and visualization changes.
- Publication track: artifact-ready repo, interactive demo, and a short paper outline ("Hierarchical Hypothesis-Driven Control with ReCoNs: A Case Study in Chess").

## M2 kickoff: macrograph Subgraph Weight Packs + endgame mounts (2025-11-05)

- Macrograph SWPs wired: `weights/macro_weight_pack.swp`, `weights/macro_threshold_pack.swp`; loader applies POR weights and node policies at instantiation (name change from “sidecar” documented here for legacy context).
- Macrograph spec extended with endgame subgraphs: `KPKSubgraph`, `RookTechniquesSubgraph`, mounted via `mount_builders`.
- Skeleton KPK and rook technique subgraphs added (script chains with placeholder terminals).
- Tests added for SWP application and subgraph mounting (`tests/test_macro_weight_pack.py`, `tests/test_endgame_components.py`).
- `MacroEngine` accepts multiple builders to mount all endgame subgraphs cleanly. Visualization unchanged (macrograph-static auto-places new nodes).
- TraceDB + metrics: KRK persistent emits Tick/Episode traces with pack fingerprints; batch/block runners write traces, checkpoints, per-block viz logs (KRK playback); nightly runner stub added for automation.

---

## Assistant handoff (2025‑12‑01)

This section is a quick orientation for a future AI instance that needs to talk the user through the system (e.g. over voice) without lots of back‑and‑forth.

### 1. Where we are on the roadmap

- M1 (continuous activations, bindings, micro‑ticks) is implemented and tested (`src/recon_lite/core/activations.py`, `src/recon_lite/binding/manager.py`, `src/recon_lite/time/microtick.py`, `tests/test_continuous.py`).
- M2 / M2.5 (macrograph, SWPs, Stockfish teacher, TraceDB, KRK/KPK tooling) are implemented:
  - Macrograph: `specs/macrograph_v0.json`, `src/recon_lite/macro_engine.py`, `tests/test_macrograph.py`, `tests/test_macro_weight_pack.py`.
  - TraceDB schema + helpers: `src/recon_lite/trace_db.py`, `src/recon_lite/macro_trace.py`.
  - Teachers / eval: `demos/experiments/teacher_stockfish.py`, `demos/experiments/kpk_train.py`.
  - Evaluation tools: `demos/experiments/batch_eval.py`, `demos/experiments/block_runner.py`, `demos/experiments/pack_tournament.py`.
  - Persistent KRK/KPK demos with traces: `demos/persistent/krk_persistent_demo.py`, `demos/persistent/kpk_persistent_demo.py`.
- M3 fast‑plasticity & bandit control is **partially implemented**:
  - Core modules: `src/recon_lite/plasticity/fast.py`, `bandit.py`, `modulation.py`.
  - Chess eval used for reward: `src/recon_lite_chess/eval/heuristic.py`.
  - KRK wiring: `demos/persistent/krk_persistent_demo.py` supports optional plasticity/bandit flags and eval modes.
  - Tests: `tests/test_plasticity.py`, `tests/test_plasticity_integration.py`.
- Full M3 consolidation and polishing (e.g. applying plasticity beyond KRK, tuning bandits, tying into full‑game macro) is still ongoing; treat `recon_roadmap_m3_fast_plasticity.md` as the authoritative design.

### 2. Files to skim first (for a new assistant)

When you start a fresh session, read or be ready to reference:

- `recon_roadmap_m2.md` – overall project roadmap from M2.5 upward (big picture).
- `recon_roadmap_m3_fast_plasticity.md` – detailed M3 spec and status summary (what fast plasticity/bandits should do).
- `updates_continuous.md` (this file) – history and intent for continuous activations + how KRK is wired.
- `docs/HOWTO_RUN_TRAIN_EVAL.md` – exact commands the user runs for demos, teachers, and evaluations.
- `ARCHITECTURE.md` – conceptual ReCoN architecture and terminology.
- `VIS_SPEC.md` – how the visualization expects bindings/latents/weights.

### 3. Typical commands the user runs

These are the main entry points to confirm the system is healthy and to generate data:

- KRK persistent demo with eval and trace:
  - `uv run python demos/persistent/krk_persistent_demo.py --engine /usr/games/stockfish --depth 2 --trace-out reports/krk_trace.jsonl`
- KPK persistent demo:
  - `uv run python demos/persistent/kpk_persistent_demo.py --engine /usr/games/stockfish --depth 2 --trace-out reports/kpk_trace.jsonl`
- KRK teacher (macro + phase packs):
  - `uv run python demos/experiments/teacher_stockfish.py data/endgames/krk/random.fen --engine /usr/games/stockfish --depth 4 --output weights/macro_weight_pack.swp --phase-output weights/krk_phase_weight_pack.swp`
- Batch evaluation:
  - `uv run python demos/experiments/batch_eval.py --mode krk --fen-file data/endgames/krk/random.fen --runs 200 --pack weights/krk_phase_weight_pack.swp --engine /usr/games/stockfish --depth 2 --trace-out reports/krk_batch_trace.jsonl`
- Pack tournament:
  - `uv run python demos/experiments/pack_tournament.py --mode krk --fen-file data/endgames/krk/random.fen --pack weights/krk_phase_weight_pack.swp --engine /usr/games/stockfish --depth 2 --runs 200 --output reports/krk_pack_tournament.json`
- Full‑game macro driver:
  - `uv run python demos/gameplay/full_game_macro.py --engine /usr/games/stockfish --depth 2 --trace-out reports/fullgame_trace.jsonl`

The user cares a lot about **metrics, traces, and viz**. When you suggest steps, try to preserve trace/logging options and explain how to interpret them (e.g. expected win rate, `reward_tick` trends, phase usage).

### 4. Reward and learning signals (for M3/M4)

- Tick‑level reward:
  - `reward_tick = eval_after − eval_before` in centipawns (clipped).
  - Eval source:
    - Stockfish when `--engine` is provided (preferred).
    - Heuristic eval (`recon_lite_chess.eval.heuristic.eval_position`) when no engine is available.
  - Only meaningful on ticks where a move is chosen; other ticks may have `reward_tick = null`.
- Episode‑level reward:
  - `result` in `EpisodeRecord` (win/draw/loss) and summary stats in `notes`.
- Fast plasticity:
  - Implemented as within‑episode updates to edge weights for a whitelisted set of KRK edges; state resets between games.
- Bandit control:
  - Implemented (in code) but still needs tuning; used to choose between sibling scripts based on accumulated reward.

When guiding the user, keep this mental model: **M2.5 gives you rich traces; M3 uses those traces online inside KRK; M4 will use them offline to adjust baseline weights across games.**

### 5. What the user is aiming for next

From recent notes and roadmaps, likely next steps the user will ask about:

- Better understanding and tuning of M3 behavior:
  - Visualizing how weights change during a KRK run (edge thickness/colors).
  - Comparing “plasticity off” vs “plasticity on” using batch_eval / pack_tournament.
- Extending plasticity/bandits beyond KRK:
  - KPK endgames.
  - Eventually macrograph decisions (phase/plan selection) once KRK/KPK look stable.
- Moving toward full‑game play:
  - More endgame subgraphs and heuristics.
  - Shallow Stockfish teacher plus ReCoN control for opening/middlegame.
- Laying groundwork for M4:
  - Summaries per episode (per‑edge Δw_fast, bandit stats) that can be aggregated across traces.

When in doubt, ask the user which **milestone** they want to push (M2.5 polish, M3 tuning, or early M4), and anchor your suggestions to the relevant roadmap file.

---

## M4: Slow Consolidation & Eval Upgrade (2025-12)

M4 carries the fast, within-game adaptations from M3 into a persistent learning layer across games—stabilizing KRK/KPK performance, improving evaluation signals, and producing artifact-ready logs/visuals.

### What's Implemented

#### 1. Episode Summaries & Trace Enrichment (M4.1)

- **EpisodeSummary** in `src/recon_lite/trace_db.py`:
  - `edge_delta_sums`: cumulative Δw_fast per edge
  - `bandit_stats`: per-arm pull counts and reward means
  - `avg_reward_tick`, `total_reward_tick`, `reward_tick_count`
  - `phase_usage`: count of ticks per phase
  - `outcome_score`: +1 win, -1 loss, 0 draw

- **extract_episode_summary()** in `src/recon_lite/plasticity/fast.py` aggregates episode data at game end.

- **CLI tools**:
  - `tools/trace_summarize.py`: aggregate metrics from JSONL traces to JSON/CSV
  - Usage: `uv run python tools/trace_summarize.py reports/krk_trace.jsonl -o reports/summary.json`

#### 2. Slow Weight Consolidation (M4.2)

- **ConsolidationEngine** in `src/recon_lite/plasticity/consolidate.py`:
  - `ConsolidationConfig`: `eta_consolidate`, `min_episodes`, `max_base_delta`, `w_min`, `w_max`
  - `EdgeConsolidationState`: tracks `w_base`, `w_init`, accumulated weighted deltas
  - `accumulate_episode(summary)`: adds episode data to running statistics
  - `should_apply()`: checks if enough episodes accumulated
  - `apply_to_graph(graph)`: updates graph edge weights with consolidated w_base
  - `save_state()` / `load_state()`: persist to JSON
  - `export_w_base_pack()`: export as SWP-compatible format

- **Demo integration** in `demos/persistent/krk_persistent_demo.py`:
  - `--consolidate`: enable slow consolidation
  - `--consolidate-pack PATH`: load/save consolidation state
  - `--consolidate-eta FLOAT`: learning rate (default 0.01)
  - `--consolidate-min-episodes INT`: threshold before applying (default 10)

#### 3. Cross-Game Bandit Refresh (M4.3)

- **BanditPriors** in `src/recon_lite/plasticity/bandit.py`:
  - Stores aggregated arm statistics across episodes
  - `init_bandit_state_with_priors()`: warm-start new games with prior knowledge
  - `export_priors()`, `merge_priors()`, `save_priors()`, `load_priors()`

- **CLI**: `tools/bandit_refresh.py`
  - Aggregates bandit stats from traces with optional decay
  - Usage: `uv run python tools/bandit_refresh.py reports/*.jsonl --existing weights/priors.json --decay 0.9 -o weights/priors.json`

- **Demo integration**: `--bandit-priors PATH` in krk_persistent_demo.py

#### 4. Evaluation Upgrade & Hybrid Signals (M4.4)

- **Expanded heuristic** in `src/recon_lite_chess/eval/heuristic.py`:
  - `_material_score()`: piece values in pawn units
  - `_king_safety_score()`: castling bonus, exposure penalty, attacker count
  - `_mobility_score()`: legal move count differential
  - `_pawn_structure_score()`: doubled, isolated, passed pawn detection
  - `_piece_activity_score()`: centralization, outposts, bishop pair, rook files
  - `_tactical_tension_score()`: hanging/under-defended pieces
  - `_krk_specific_score()`: enemy king distance from center (endgame)
  - `_endgame_factor()`: material-based phase interpolation
  - Variants: `eval_position()`, `eval_position_fast()`, `eval_position_full()`

- **EvalManager** in `src/recon_lite_chess/eval/manager.py`:
  - Unified interface with modes: `HEURISTIC_FAST`, `HEURISTIC`, `HEURISTIC_FULL`, `STOCKFISH`, `HYBRID`, `DISTILLED`
  - Caching with configurable max size
  - Statistics tracking (`get_stats()`)

- **Distillation stub** in `src/recon_lite_chess/eval/distill.py`:
  - Interface defined for future ML-based evaluation
  - `DistillationConfig`, `DistillationSample`, `DistillationDataset`
  - `DistilledEvaluator`, `train_distilled_eval()`, `collect_distillation_data()`
  - Currently raises `NotImplementedError` – placeholder for M5

#### 5. Dashboards & Artifact Hooks (M4.5)

- **CLI tools**:
  - `tools/consolidate_batch.py`: offline batch consolidation from traces
  - `tools/report_consolidation.py`: generate markdown reports with checksums and comparisons
  - `tools/pack_diff.py`: compare two weight packs and show top changes

- **Visualization** in `demos/visualization/network-visualization.js`:
  - `edgeWBase`: tracks baseline weights from consolidation
  - `setWBaseMode(mode)`: toggle between "baseline", "delta", "combined" display
  - `getEdgeWBaseDriftColor()`: color edges by drift from initial weights
  - Edge thickness/color reflects consolidation state

#### 6. Tests & Benchmarks (M4.6)

- `tests/test_consolidation.py`: 25 tests covering config, state, engine, save/load, bounds
- `tests/test_eval_light.py`: 32 tests covering all heuristic components and EvalManager

#### 7. Rollout Plan & Safety (M4.7)

- **Versioning**: `save_state()` includes version field; packs have checksums
- **Comparison**: `tools/pack_diff.py` and `report_consolidation.py --compare` show differences
- **Validation workflow**:
  1. Run consolidation on trace subset → produce candidate pack
  2. Validate via `pack_tournament` vs baseline
  3. Promote only if metrics improve

### Typical M4 Commands

```bash
# Run KRK with consolidation enabled (batch of 20 games)
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 20 --consolidate --consolidate-pack weights/krk_consol.json \
  --consolidate-min-episodes 10 --plasticity --bandit \
  --bandit-priors weights/bandit_priors.json \
  --trace-out reports/krk_trace.jsonl

# Summarize traces
uv run python tools/trace_summarize.py reports/krk_trace.jsonl -o reports/summary.json

# Refresh bandit priors with decay
uv run python tools/bandit_refresh.py reports/*.jsonl \
  --existing weights/bandit_priors.json --decay 0.9 -o weights/bandit_priors.json

# Batch consolidation from multiple trace files
uv run python tools/consolidate_batch.py reports/*.jsonl \
  --existing weights/krk_consol.json --min-episodes 20 -o weights/krk_consol_new.json

# Generate consolidation report
uv run python tools/report_consolidation.py weights/krk_consol.json -o reports/consol_report.md

# Compare two consolidation packs
uv run python tools/pack_diff.py weights/krk_consol_old.json weights/krk_consol_new.json

# Validate consolidated pack vs baseline
uv run python demos/experiments/pack_tournament.py --mode krk \
  --fen-file data/endgames/krk/random.fen \
  --pack weights/krk_consol_new.json --runs 100
```

---

## M5 (Implemented 2024-12): Structure Discovery & Script Induction

Goal: Move from "tune existing scripts" to "propose and vet new ones" with trust-based pruning/promotion, while keeping explainability and safety.

### What's New in M5

#### 1. Motif Extraction (`src/recon_lite/motifs/`)

- **BindingDescriptor**: Compact schema for "interesting patches" extracted from traces
- **Extractors**: Functions to analyze board positions for tactical/structural patterns
- **CLI**: `demos/experiments/extract_motifs.py` to extract motifs from trace files

```bash
uv run python demos/experiments/extract_motifs.py \
  --traces reports/*.json --out reports/motifs/extracted.jsonl --stats
```

#### 2. Clustering & Script Proposals

- **Motif Clustering**: `demos/experiments/cluster_motifs.py` groups patterns by type and context
- **Script Proposals**: `demos/experiments/propose_scripts.py` generates candidate subgraph diffs
- **Human Review**: `tools/review_proposal.py` for accept/reject workflow

```bash
# Cluster extracted motifs
uv run python demos/experiments/cluster_motifs.py \
  --motifs reports/motifs/extracted.jsonl --out reports/motifs/clusters.json

# Generate script proposals
uv run python demos/experiments/propose_scripts.py \
  --clusters reports/motifs/clusters.json --out proposals/

# Review proposals
uv run python tools/review_proposal.py --list proposals/
uv run python tools/review_proposal.py --proposal proposals/fork_v1.yaml --action accept
```

#### 3. Trust Scoring & Pruning (`src/recon_lite/trust/`)

- **NodeTrustScore** and **EdgeTrustScore**: Track activation counts, rewards, and variance
- **Trust formula**: `trust = α * norm(activations) + β * norm(reward) - γ * norm(variance)`
- **Actions**: `freeze` (disable plasticity), `remove` (candidate for deletion), `promote` (boost weight)

```bash
uv run python tools/trust_report.py \
  --traces demos/outputs/persistent/*.json \
  --out reports/trust.json
```

#### 4. Tactical Subgraph (`src/recon_lite_chess/scripts/tactics.py`)

New nodes for tactical pattern detection:
- **Sensors**: `detect_fork`, `detect_pin`, `detect_hanging`
- **Actuators**: `exploit_fork`, `capture_hanging`, `protect_hanging`
- **Weight pack**: `weights/subgraphs/tactics_weight_pack.swp`

#### 5. Rook Endgame Subgraph (`src/recon_lite_chess/scripts/rook_endgame.py`)

Implements classic rook endgame techniques:
- Lucena position detection and bridge-building
- Philidor defense
- King cutoff moves
- **Weight pack**: `weights/subgraphs/rook_weight_pack.swp`

#### 6. Extended Nightly Pipeline

`demos/experiments/nightly_runner.py` now supports:
- Motif extraction after trace generation
- Trust scoring with incremental generation tracking
- Combined report generation

Config example (`configs/nightly/m5_full.json`):
```json
{
  "mode": "krk",
  "motif_extraction": {"enabled": true, "reward_threshold": 0.3},
  "trust_scoring": {"enabled": true, "promote_threshold": 0.8},
  "consolidation": {"enabled": true}
}
```

#### 7. Benchmark Suites

- `data/benchmarks/tactics_suite.fen`: Fork, pin, hanging piece positions
- `data/benchmarks/rook_endgame_suite.fen`: Lucena, Philidor, cutoff positions
- `demos/experiments/benchmark_eval.py`: Evaluate success rate on finding best moves

```bash
uv run python demos/experiments/benchmark_eval.py \
  --suite data/benchmarks/tactics_suite.fen --out reports/benchmarks/tactics.json -v
```

#### 8. Versioning

Weight pack manifest in `weights/manifest.json`:
```json
{
  "tactics": {"current": "weights/subgraphs/tactics_weight_pack.swp", "generation": 1},
  "rook_endgame": {"current": "weights/subgraphs/rook_weight_pack.swp", "generation": 1}
}
```

### M5 File Summary

**New modules**:
- `src/recon_lite/motifs/` - Descriptors, extractors
- `src/recon_lite/trust/` - Trust scoring
- `src/recon_lite_chess/scripts/tactics.py`
- `src/recon_lite_chess/scripts/rook_endgame.py`

**New CLIs**:
- `demos/experiments/extract_motifs.py`
- `demos/experiments/cluster_motifs.py`
- `demos/experiments/propose_scripts.py`
- `demos/experiments/benchmark_eval.py`
- `tools/trust_report.py`
- `tools/review_proposal.py`

**New data**:
- `data/benchmarks/tactics_suite.fen`
- `data/benchmarks/rook_endgame_suite.fen`
- `weights/subgraphs/tactics_weight_pack.swp`
- `weights/subgraphs/rook_weight_pack.swp`
- `weights/manifest.json`

**Tests**:
- `tests/test_motif_extraction.py` (20 tests)
- `tests/test_trust_scoring.py` (25 tests)
- `tests/test_tactics_subgraph.py` (14 tests)

---

## M6 (Implemented 2025-12): Full-Game Architecture & Multi-Scale Dynamics

**Goal**: Restructure ReCoN around a time-scale goal hierarchy with fan-in sensor terminals, plan persistence, and full-game capability from opening to checkmate.

### Key Features

#### 1. Fan-In Terminals (ReCoN Extension)
Sensor terminals can have multiple parent scripts querying them:

```python
# Multiple plans query the same sensor
g.add_edge("DevelopMinorPieces", "CenterControlSensor", LinkType.SUB)
g.add_edge("ControlCenter", "CenterControlSensor", LinkType.SUB)  # Fan-in

# Graph tracks all parents
parents = g.all_parents("CenterControlSensor")  # ["DevelopMinorPieces", "ControlCenter"]
g.is_fanin_terminal("CenterControlSensor")  # True
```

#### 2. Goal Hierarchy
```
ULTIMATE (WIN/DRAW/SURVIVE)
    ↓
STRATEGIC (AttackKing, Simplify, Develop...)
    ↓  
TACTICAL (Forks, Pins, Hanging pieces)
    ↓
SENSORS (Material, Phase, KingSafety...)
```

#### 3. Plan Persistence via Activation
Plans maintain activation over time (inertia and decay):

```python
from recon_lite.dynamics.persistence import apply_persistence_to_node

# Evidence accumulates with inertia, decays over time
apply_persistence_to_node(plan_node, evidence=0.7)

# Check if plan is active
from recon_lite.dynamics.persistence import is_plan_active
is_plan_active(plan_node)  # True if accumulated >= threshold
```

#### 4. Soft Phase Gating
Phase is continuous weights, not hard gates:

```python
from recon_lite_chess.sensors.phase import estimate_phase
phase = estimate_phase(board)  # PhaseWeights(opening=0.2, middlegame=0.6, endgame=0.2)
```

#### 5. Full Game Demo
Play complete games from starting position:

```bash
uv run python demos/persistent/full_game_demo.py --max-moves 200
uv run python demos/persistent/full_game_demo.py --vs-random --output game.json
```

### M6 File Summary

**New/Modified modules**:
- `src/recon_lite/graph.py` - Fan-in support for terminals
- `src/recon_lite/dynamics/persistence.py` - Plan persistence logic
- `src/recon_lite_chess/goals/ultimate.py` - WIN/DRAW/SURVIVE assessment
- `src/recon_lite_chess/goals/strategic.py` - Strategic plan definitions
- `src/recon_lite_chess/sensors/material.py` - Material category sensor
- `src/recon_lite_chess/sensors/phase.py` - Soft phase sensor
- `src/recon_lite_chess/scripts/opening.py` - Opening plans and sensors
- `src/recon_lite_chess/scripts/middlegame.py` - Middlegame plans and sensors

**New demos**:
- `demos/persistent/full_game_demo.py` - Full game from start to finish

**Updated specs**:
- `specs/macrograph_v1.json` - M6 goal hierarchy with fan-in

**Tests** (47 passing):
- `tests/test_fanin_terminals.py`
- `tests/test_goal_hierarchy.py`
- `tests/test_persistence.py`
- `tests/test_opening_middlegame.py`

---

## M7 (Implemented 2025-12): Distillation & Evaluation Upgrade

**Goal**: Train a lightweight neural network to mimic Stockfish for fast, accurate evaluation without runtime engine dependency.

### Key Features

#### 1. Feature Extraction
Extract ~77 features from chess positions for ML training:

```python
from recon_lite_chess.eval.features import extract_features

board = chess.Board()
fv = extract_features(board)
print(f"Features: {len(fv)}")  # 77 features
print(fv.feature_names)  # ["mat_w_P", "mat_w_N", ...]
```

Features include:
- Material counts (12)
- Material balance (7)
- Piece positions (16)
- King positions (8)
- Pawn structure (12)
- King safety (8)
- Mobility (4)
- Phase indicators (4)
- Tactical features (6)

#### 2. Stockfish Data Collection
Collect training data by annotating positions with Stockfish:

```bash
# From random positions
uv run python tools/collect_stockfish_evals.py \
  --random 5000 --depth 15 --out data/distillation/evals.jsonl

# From game traces
uv run python tools/collect_stockfish_evals.py \
  --traces reports/*.jsonl --out data/distillation/evals.jsonl

# From PGN games
uv run python tools/collect_stockfish_evals.py \
  --pgn games.pgn --out data/distillation/evals.jsonl
```

#### 3. Model Training
Train distilled model (supports PyTorch or sklearn backend):

```bash
uv run python tools/train_distilled_eval.py \
  --data data/distillation/evals.jsonl \
  --out weights/distilled_eval.pt \
  --epochs 100 --lr 0.001 --hidden 256,128
```

#### 4. Integrated Evaluation
Use distilled model in EvalManager:

```python
from recon_lite_chess.eval import EvalMode, EvalConfig, EvalManager

# Pure distilled mode
config = EvalConfig(
    mode=EvalMode.DISTILLED,
    distilled_model_path="weights/distilled_eval.pt"
)
manager = EvalManager(config)
result = manager.evaluate(board)

# Distilled + tactical bonuses
config = EvalConfig(mode=EvalMode.DISTILLED_HYBRID, ...)
```

### M7 File Summary

**New modules**:
- `src/recon_lite_chess/eval/features.py` - Feature extraction for ML
- `tools/train_distilled_eval.py` - Model training script

**Modified**:
- `src/recon_lite_chess/eval/distill.py` - Now fully implemented
- `src/recon_lite_chess/eval/manager.py` - DISTILLED and DISTILLED_HYBRID modes

**Tests** (17 passing):
- `tests/test_distillation.py`

### M7 Acceptance Criteria

- [x] Feature extraction produces consistent 77-feature vectors
- [x] Data collection tool supports traces, PGN, FENs, and random positions
- [x] Training script supports both PyTorch and sklearn backends
- [x] EvalManager supports DISTILLED and DISTILLED_HYBRID modes
- [ ] 10,000+ positions collected (user task)
- [ ] Model achieves >0.85 correlation with Stockfish (depends on training data)

---

## M8 (Implemented 2025-12): Reverse Curriculum & FeatureHub

**Goal**: Enable "Implicit Lookahead" and "Structural Discovery" by implementing continuous affordance signals, global feature hoisting, and a reverse curriculum training strategy that trains backwards from perfect endgames to discover bridge strategies.

### Key Features

#### 1. Continuous Affordance Signals (`src/recon_lite_chess/affordance/`)

Move from binary gates to continuous [0.0, 1.0] "scent" signals that measure distance to applicability:

```python
from recon_lite_chess.affordance import compute_all_affordances

board = chess.Board("8/8/4k3/8/8/4K3/4R3/8 w - - 0 1")
affs = compute_all_affordances(board)
print(affs["krk"].value)  # 0.95 - very close to pure KRK
print(affs["kpk"].value)  # 0.0 - no pawns present
```

- **AffordanceSignal**: Contains `subgraph`, `value`, `components`, `is_exact_match`
- **Sensors**: `compute_krk_affordance()`, `compute_kpk_affordance()`, `compute_kqk_affordance()`
- **Purpose**: Creates gradient for M3 Bandit to "climb the hill" of a strategy

#### 2. FeatureHub - Global Feature Registry (`src/recon_lite_chess/features/`)

Hoists tactical and geometric sensors from local subgraphs to a global registry:

```python
from recon_lite_chess.features import create_default_hub

hub = create_default_hub()
features = hub.compute_all(board)

# 18+ features including:
# - Tactical: detect_fork, detect_pin, detect_hanging, detect_back_rank
# - Material: material_balance, material_advantage, piece_count
# - Positional: king_safety, center_control, color_complex_weakness
# - Phase: phase_opening, phase_middlegame, phase_endgame
# - Geometric: opposition_status, affordance_krk, affordance_kpk, affordance_kqk
```

- **FeatureDefinition**: Declarative feature specs with categories and dependencies
- **FeatureCategory**: TACTICAL, GEOMETRIC, MATERIAL, POSITIONAL, DYNAMIC, PHASE
- **Integration**: `wire_feature_sensors_to_graph()` creates terminal nodes from hub features

#### 3. Reverse Curriculum Training (`src/recon_lite_chess/training/`)

Trains backwards from "Anchors" (perfect endgames) to discover "Bridge" strategies:

```python
from recon_lite_chess.training import CurriculumManager, create_default_curriculum

curriculum = create_default_curriculum()
# Phase 1: Anchor - KRK/KPK/KQK to 99% win rate
# Phase 2: Bridge - Simplified middlegame → discover liquidation
# Phase 3: Wilderness - Full material tactical positions
# Phase 4: Integration - Full games from starting position
```

**Position Generators** (`training/generators.py`):
- `generate_krk_position()`, `generate_kpk_position()`, `generate_kqk_position()`
- `generate_bridge_position()`, `generate_wilderness_position()`

#### 4. Curriculum Training Script (`scripts/curriculum_training.sh`)

End-to-end training with evaluation and markdown reports:

```bash
# Quick test (50 games per endgame)
./scripts/curriculum_training.sh --quick --phase anchor

# Full training (500 games, all phases)
./scripts/curriculum_training.sh

# With Stockfish evaluation
./scripts/curriculum_training.sh --engine /usr/games/stockfish --depth 4
```

**Output**:
- JSONL traces: `reports/curriculum/{timestamp}/anchor_krk.jsonl`, etc.
- Markdown report: `reports/curriculum/{timestamp}/training_report.md`
- Weight checkpoints: `weights/nightly/{endgame}_consol.json`

#### 5. Training Analysis (`scripts/analyze_training.py`)

Generate statistics from JSONL traces:

```bash
uv run python scripts/analyze_training.py --report-dir reports/curriculum/20251211_130006/ --markdown
```

**Output format**:
| Phase | Episodes | W/D/L | Win Rate | Promos | Avg Plies |
|-------|----------|-------|----------|--------|-----------|
| anchor_krk | 50 | 45/5/0 | 90.0% | 0 | 13.5 |
| anchor_kpk | 50 | 40/10/0 | 80.0% | 40 | 7.4 |
| anchor_kqk | 50 | 35/15/0 | 70.0% | 0 | 18.2 |

#### 6. KQK Endgame Network (`src/recon_lite_chess/scripts/kqk.py`)

New endgame network for King+Queen vs King with stalemate protection:

- **Position detection**: `is_kqk_position()`, `create_random_kqk_board()`
- **Move strategies**: `can_deliver_queen_mate()`, `get_restriction_moves()`, `can_approach_for_mate()`, `get_waiting_queen_moves()`
- **Stalemate protection**: All queen moves checked for `is_stalemate()` before selection
- **Demo**: `demos/persistent/kqk_persistent_demo.py`

#### 7. KPK Promotion Detection

KPK success condition is pawn promotion (not checkmate):

```python
# In kpk_persistent_demo.py
def _check_pawn_promoted(board: chess.Board) -> bool:
    """Check if attacking side promoted a pawn to Queen."""
    # ...

# Game ends with "1-0" when pawn promotes
if _check_pawn_promoted(board):
    game_result = "1-0"
    result["promoted"] = True
```

### M8 File Summary

**New modules**:
- `src/recon_lite_chess/affordance/` - Continuous affordance signals
- `src/recon_lite_chess/features/` - Global FeatureHub
- `src/recon_lite_chess/training/` - CurriculumManager and generators
- `src/recon_lite_chess/scripts/kqk.py` - KQK endgame network

**New scripts**:
- `scripts/curriculum_training.sh` - End-to-end curriculum training
- `scripts/analyze_training.py` - Training statistics and reports
- `scripts/generate_endgame_fens.py` - Position generation

**New demos**:
- `demos/persistent/kqk_persistent_demo.py` - KQK training demo

**Modified**:
- `src/recon_lite_chess/graph/subgraph_gates.py` - Uses affordance signals
- `src/recon_lite/plasticity/bandit.py` - Affordance delta in reward
- `demos/experiments/extract_motifs.py` - Bridge motif extraction

### M8 Training Results (Quick Mode)

| Endgame | Win Rate | Notes |
|---------|----------|-------|
| KRK | 90% | 45 checkmates, 5 stalemates (edge positions) |
| KPK | 80% | 40 promotions, 10 theoretical draws |
| KQK | 70%+ | With stalemate protection fixes |

### M8 Acceptance Criteria

- [x] Affordance sensors produce continuous [0,1] signals for KRK/KPK/KQK
- [x] FeatureHub registers 18+ features across 6 categories
- [x] CurriculumManager orchestrates 4-phase training
- [x] Position generators produce valid endgame positions
- [x] KQK network achieves 70%+ win rate with stalemate protection
- [x] KPK network detects promotion as success condition
- [x] Training script produces JSONL traces and markdown reports
- [x] Analysis script generates statistics from traces
