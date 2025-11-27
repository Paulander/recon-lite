# ReCoN‑lite (Request–Confirmation Network) — Chess Sandbox

ReCoN‑lite is a small, dependency‑light Python implementation of a **Request–Confirmation Network (ReCoN)** plus a set of chess‑focused demos (KRK/KPK endgames, macrograph, visualization, and training/eval tooling). The goal is to explore ReCoNs as an **orchestrator** for scripts, heuristics, and learned components — not as a monolithic learner — while keeping the internal graph state explainable and easy to visualize.

Current medium‑term target: a ReCoN‑driven chess player that can play full games in the ~1900 Elo range, with a clear story for *why* it plays the moves it plays.

---

## 1. What’s in this repo

- **Core ReCoN engine** (`src/recon_lite`)
  - Graph, node and edge types, request→confirm executor, POR/RET sequencing.
  - Continuous activations + micro‑ticks, binding manager, and a lightweight `TraceDB`.
- **Chess integration** (`src/recon_lite_chess`)
  - KRK (King + Rook vs King) and KPK (King + Pawn vs King) endgame scripts.
  - Helpers for evaluation, feature extraction, and phase/strategy logic.
- **Demos & experiments** (`demos/`)
  - One‑shot KRK demos, persistent KRK loop, macrograph chess episodes.
  - Batch/block evaluation, trace generation, and visualization data.
- **Docs** (`docs/`)
  - How‑to run/train/evaluate, trace schema, and visualization/spec notes.
- **Roadmaps & architecture**
  - `ARCHITECTURE.md` — overall ReCoN‑lite design and decisions.
  - `updates_continuous.md` — continuous activations, micro‑ticks, binding, and the KRK persistent demo.
  - `recon_roadmap_m2.md` — M2.x plan: instrumentation, macrograph, script‑weight learning.
  - `recon_roadmap_m3_fast_plasticity.md` — M3 plan: fast plasticity & bandit control (within‑game).

---

## 2. Project status (high‑level)

- **M1: Continuous activations & binding (KRK)**
  - Each node has a continuous activation state; micro‑ticks settle activations before each discrete tick.
  - A `BindingTable` tracks which pieces/squares are “claimed” by which features, per namespace.
  - Visualization overlays show activations and bindings during KRK games.

- **M2.x: Macrograph, KPK/KRK subgraphs, TraceDB**
  - Macrograph with endgame subgraphs (KRK, KPK, rook techniques), configured via Subgraph Weight Packs (SWPs) in `weights/`.
  - `TraceDB` captures `TickRecord`/`EpisodeRecord` with evals, rewards, and fired edges for analysis and training.
  - Batch/block runners generate traces and visualizations for offline inspection.

- **M3 (planned, design complete)**
  - Fast, within‑game plasticity on a subset of edge weights using `reward_tick`.
  - Bandit‑style control for choosing among alternative scripts.
  - Goal‑aware modulation of learning rate and exploration.
  - See `recon_roadmap_m3_fast_plasticity.md` for the detailed plan.

---

## 3. Install & setup (uv)

Use `uv` for environment and dependencies:

```bash
uv venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -e .
```

You can sanity‑check the install by running a tiny demo:

```bash
uv run python -m demos.evaluation.sequence_demo
```

---

## 4. Running the chess demos

### 4.1 Basic KRK checkmate demo

A simple KRK ReCoN graph that chooses a move, logs, and exits:

```bash
uv run python demos/gameplay/krk_play_demo.py
```

This demonstrates a hand‑authored hierarchical KRK strategy:

- **Phase 1**: Drive enemy king to the edge.
- **Phase 2**: Shrink the “safe box”.
- **Phase 3**: Take opposition.
- **Phase 4**: Deliver checkmate.

To visualize a run, open one of the HTML viewers:

- `demos/visualization/chessboard_view.html`
- `demos/visualization/onepage_view.html`

…then use “Load JSON” to select a `_viz.json` produced by the demo.

### 4.2 Persistent KRK (continuous activations, binding, micro‑ticks)

The persistent KRK loop retains state between plies, uses continuous activations, and logs more data:

```bash
uv run python demos/persistent/krk_persistent_demo.py \
  --max-plies 40 \
  --seed 0 \
  --output-basename krk_persistent_review
```

This produces:

- Visualization logs under `demos/outputs/persistent/` (e.g. `*_viz.json` and `*_debug.json`).
- Binding overlays and phase latents that can be inspected in the HTML viewers.

### 4.3 Macrograph episodes and evaluation

Once you have weight packs and macros wired (see `recon_roadmap_m2.md` and `docs/HOWTO_RUN_TRAIN_EVAL.md`):

- Use batch/block runners in `demos/experiments/` to:
  - Generate KRK/KPK positions.
  - Run episodes with a configured macrograph.
  - Emit traces for analysis and training.

See:

- `docs/HOWTO_RUN_TRAIN_EVAL.md`
- `docs/TRACE_AND_EVAL.md`

for up‑to‑date CLI examples.

---

## 5. Traces, training, and evaluation

ReCoN‑lite uses a minimal `TraceDB` (`src/recon_lite/trace_db.py`) to log:

- **Per‑tick**:
  - `tick_id`, `phase_estimate`, `goal_vector`
  - `board_fen`, `active_nodes`, `fired_edges`
  - `eval_before`, `eval_after`, `reward_tick`
- **Per‑episode**:
  - `episode_id`, `result`, list of ticks
  - `pack_meta` (weight pack fingerprints) and notes

You can:

- Run experiments in `demos/experiments/` to generate JSONL traces.
- Use these traces to:
  - Analyze which edges/scripts are helpful.
  - Train simpler models (e.g. eval approximators, phase predictors).
  - Drive M2/M3 learning loops described in `recon_roadmap_m2.md` and `recon_roadmap_m3_fast_plasticity.md`.

---

## 6. Architecture & design docs

If you want to understand or extend the system:

- `ARCHITECTURE.md` — high‑level design of the ReCoN‑lite engine and chess integration.
- `updates_continuous.md` — continuous activations, micro‑ticks, binding, KRK persistent loop, and viz upgrades.
- `recon_roadmap_m2.md` — roadmap from static scripts to a self‑organizing, explainable graph.
- `recon_roadmap_m3_fast_plasticity.md` — detailed plan for fast plasticity & bandit control within games.
- `VIS_SPEC.md` — visualization structure and expectations.

---

## 7. Outlook and future work

ReCoN here is used as an **executive/orchestrator**, not a single end‑to‑end learner:

- Nodes can be:
  - Hand‑written heuristics.
  - Chess engines (e.g. Stockfish).
  - Neural models (CNNs/MLPs for vision, small policies, etc.).
- ReCoN provides:
  - Hierarchical sequencing (SUB/SUR).
  - Temporal ordering (POR/RET).
  - Request/confirm semantics with clear traces and logs.

Planned / aspirational directions:

- **Sensors & vision terminals**
  - Terminals that:
    - Parse chessboards from APIs/web UIs.
    - Recognize boards from images (PNG screenshots) or camera frames.
    - Eventually support human‑vs‑ReCoN via a physical board or robot.
- **Engine integration**
  - Nodes that wrap Stockfish or learned eval/policy networks as terminals.
  - Reuse the existing graph to provide context, control, and explanation for engine‑backed moves.
- **Self‑organization**
  - M3 fast plasticity + bandit control (within‑game, per‑episode).
  - M4 slow consolidation across games, feature/script induction, and structural evolution.

---

## 8. Acknowledgements

This implementation is inspired by:

> **Request confirmation networks for neuro-symbolic script execution**  
> Joscha Bach and Priska Herger

and uses `python-chess` for chess logic. The project aims to be a practical, inspectable playground for ReCoN ideas in a concrete domain (chess), not a polished chess engine.
