# ReCoN-lite Chess: Continuous Activations, Binding, and Strategic Layer

We want the KRK demo to feel like a living ReCoN brain: continuous activations that settle over micro-ticks, bindings that keep track of which squares anchor each hypothesis, and a strategic layer that explains why a move is happening. This document keeps that intent front-and-center so we can add the harder engineering pieces without losing the intuition that the system is thinking through a plan.

## Scope and Goals
- Add continuous activations, binding, and micro-ticks without regressing existing KRK demos or scripts.
- Start with KRK (M1) so we can phase in latents, binding tables, and visualization upgrades in a controlled setting.
- Keep `python-chess` and reuse current KRK scripts/sensors; only incremental re-organization for now.
- Treat the KRK loop as the place where we practice running soft activations before we scale to KPK and the rest of the full game stack.

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
- Build a lightweight teacher (`demos/experiments/teacher_stockfish.py`) running Stockfish depth 4 to label script choices and learn child weights `a_i`; emit them to `weights/phase_child_weights.json` and have `actuators_blend` consume the sidecar (with overrides for experiments) so blended scoring reflects the learned preferences.

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

## M2 kickoff: macrograph sidecars + endgame mounts (2025-11-05)

- Weights sidecars wired: `weights/macro_weights.json`, `weights/macro_thresholds.json`; loader applies POR weights and node policies at instantiation.
- Macrograph spec extended with endgame subgraphs: `KPKSubgraph`, `RookTechniquesSubgraph`, mounted via `mount_builders`.
- Skeleton KPK and rook technique subgraphs added (script chains with placeholder terminals).
- Tests added for sidecar application and subgraph mounting (`tests/test_macro_sidecar.py`, `tests/test_endgame_components.py`).
- `MacroEngine` accepts multiple builders to mount all endgame subgraphs cleanly. Visualization unchanged (macrograph-static auto-places new nodes).
