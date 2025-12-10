# ReCoN-lite Chess — AI Orientation Brief

## Purpose and Philosophy
- Implements Joscha Bach / Priska Herger’s Request–Confirmation Network (ReCoN) ideas for chess: top-down requests via `sub/por`, bottom-up confirmations via `sur/ret`, continuous activations, bindings, and explicit traceability.
- Goal: an explainable ReCoN-based chess agent (~1900 Elo target) that orchestrates hand-written heuristics, search engines, and learned components rather than a single end-to-end model.
- Paper reference: “Request confirmation networks for neuro-symbolic script execution” (Bach & Herger). Use this as the canonical conceptual anchor.

## History & Milestones (M1→M6)
- M1 (KRK, continuous layer): micro-ticks and soft activations (`core/activations.py`, `time/microtick.py`), binding tables (`binding/manager.py`), KRK strategic layer + viz (`recon_lite_chess/strategy.py`, `demos/persistent/krk_persistent_demo.py`). Source doc: `updates_continuous.md`.
- M2 / M2.5 (macrograph + instrumentation): macrograph spec and Subgraph Weight Packs (SWPs) for KRK/KPK/rook techniques; TraceDB (`trace_db.py`, `macro_trace.py`); Stockfish teacher and batch/block eval pipelines. Source doc: `recon_roadmap_m2.md`, `updates_continuous.md`.
- M3 (fast plasticity + bandits, partial at first, now integrated): within-game weight updates on selected POR/SUB edges, bandit gating for sibling scripts, goal-aware modulation, KRK wiring + tests (`plasticity/fast.py`, `bandit.py`, `modulation.py`). Source doc: `recon_roadmap_m3_fast_plasticity.md`.
- M4 (slow consolidation + eval upgrade): cross-game consolidation of `w_base` from episode summaries (`plasticity/consolidate.py`), bandit priors refresh, richer heuristic/Stockfish/hybrid eval manager, consolidation dashboards and reports (`tools/consolidate_batch.py`, `tools/report_consolidation.py`). Source: `updates_continuous.md`.
- M5 (structure learning): motif extraction, clustering, script proposals, trust scoring, tactical + rook subgraphs, nightly runner wiring. Source: `recon_roadmap_m5_structure_learning.md`, `updates_continuous.md`.
- M6 (full-game, multi-scale dynamics): fan-in terminals, goal hierarchy (Ultimate→Strategic→Tactical→Sensor), plan persistence, opening/middlegame scripts, full-game demo. Source: `recon_roadmap_m6_full_game_architecture.md`.
- Later tracks: M7 (distillation) and beyond are stubbed (see `recon_roadmap_m7_distillation.md`); distillation interfaces exist but are not implemented.

## Current Architecture Snapshot
- Core ReCoN engine (`src/recon_lite`): graph/state machine with `sub/sur/por/ret`, continuous activations, micro-ticks, bindings, TraceDB logging, fan-in terminals.
- Chess layer (`src/recon_lite_chess`): KRK/KPK/rook tactics endgame scripts, opening/middlegame plans, strategy/phase logits, evaluation manager (heuristic, Stockfish, hybrid), actuator blending.
- Macrograph and subgraphs: macro spec mounts KRK/KPK/rook/tactics subgraphs; SWPs (`weights/*.swp`, `weights/subgraphs/*`) supply baseline weights/thresholds per subgraph.
- Persistence & dynamics: plan persistence (`dynamics/persistence.py`), activation settling, binding namespaces carried across plies, goal hierarchy sensors and strategic biasing.
- Tooling: demos for persistent play and evaluation (`demos/persistent/*.py`, `demos/experiments/*.py`), viz for activations/bindings/weights (`demos/visualization/*`), reporting/pack diff tools.

## Learning & Training Mechanics (implemented today)
- **Subgraph Weight Packs (SWPs):** Serialized weight/threshold packs for macro and subgraphs; loaded at build time; used as baselines for fast/slow learning.
- **Fast plasticity (M3):** Per-episode Δw on whitelisted POR/SUB edges, driven by clipped `reward_tick = eval_after − eval_before`; supports eligibility traces and goal-aware scaling; reset each episode.
- **Bandit control (M3):** UCB/softmax gating among sibling scripts; per-episode stats; optional priors for cross-episode warm starts.
- **Slow consolidation (M4):** Aggregates episode summaries (Δw_fast, bandit stats, phase usage) into updated `w_base` with bounds and versioning; exportable as packs and visualized as edge drift.
- **Eval signals (M4):** Hierarchical eval manager with heuristic, Stockfish, hybrid, and future distilled modes; tick reward from eval deltas, episode reward from game result.
- **Structure learning (M5):** Motif extraction → clustering → scripted proposals; human review (`tools/review_proposal.py`); trust scoring to freeze/promote/prune edges; tactical and rook subgraphs generated from this loop.
- **Plan persistence & horizon mixing (M6):** Activation-based inertia for plans, interrupts for urgent tactics, strategic goal modulation—gives multiple horizons (ply-level tactics, phase plans, whole-game goals) a shared substrate.

## What You Can Train/Run Now
- KRK/KPK/rook/tactics episodes with trace logging and optional plasticity/bandits: `demos/persistent/krk_persistent_demo.py`, `kpk_persistent_demo.py`, `full_game_demo.py`.
- Stockfish-labeled teacher for KRK (macro + phase packs): `demos/experiments/teacher_stockfish.py`.
- Batch/block eval and tournaments: `batch_eval.py`, `block_runner.py`, `pack_tournament.py`.
- Consolidation and bandit refresh offline: `tools/consolidate_batch.py`, `tools/report_consolidation.py`, `tools/bandit_refresh.py`.
- Structure loop: extract motifs → cluster → propose scripts → review (`demos/experiments/extract_motifs.py`, `cluster_motifs.py`, `propose_scripts.py`, `tools/review_proposal.py`).

## Outlook and Design Strategy (Joscha’s lens)
- Maintain pure ReCoN semantics (only `sub/sur/por/ret`, request→confirm flow) while layering continuous activations and plasticity for smooth bottom-up confirmations and top-down intent.
- Unify horizons: fast tactical scripts (1 ply), phase plans (3–5 ply), and strategic/goal nodes (whole game) via shared activation/persistence and bandit gating; fan-in sensors let multiple plans share evidence.
- Future push: distill Stockfish into lightweight eval, extend plasticity/bandits to macro decisions, automate proposal acceptance with trust scores, and improve viz for within-game weight drift and plan persistence.
- Safety/explainability remain constraints: hard weight bounds, pack versioning/checksums, human-in-the-loop proposal review, trace-driven dashboards.

## Mental Model for the AI Assistant
- Treat ReCoN as an **orchestrator** over heterogeneous modules. Training chiefly adjusts edge preferences (fast/slow plasticity) and scripts via SWPs; topology changes go through the proposal/trust pipeline.
- When suggesting changes, align with milestone scope: M3 = within-game adaptation, M4 = cross-game consolidation, M5 = structure learning, M6 = full-game multi-scale control.
- Encourage experiments that compare modes (plasticity on/off, different packs, heuristic vs Stockfish eval) and preserve trace outputs so learning tools can consume them.
