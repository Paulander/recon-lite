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
