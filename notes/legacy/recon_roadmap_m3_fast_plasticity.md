## ReCoN Roadmap – M3: Fast Plasticity & Bandit Control (Within‑Game)

This document details the **M3 fast‑plasticity and bandit‑control layer** that sits on top of the M2.5 instrumentation and continuous activations. It assumes:

- You already have:
  - Stable KRK/KPK graphs and macrograph.
  - Continuous node activations + micro‑ticks.
  - `TraceDB` with `TickRecord` / `EpisodeRecord` (including `reward_tick`).
  - Binding tables and visualization for activations/bindings.
- You want:
  - **Within‑game adaptation only** (per‑episode; no topology changes).
  - Bandit‑style selection among alternative scripts.
  - A clean bridge to M4’s slow consolidation (`w_base + Δw_fast`) without rewiring.

---

## 1. Goals and Non‑Goals

### 1.1 Goals (M3 Scope)

- **Fast, per‑episode plasticity**:
  - Adjust a subset of edge weights during a game based on `reward_tick`.
  - Reset fast changes at the start of each new episode.
- **Bandit control over alternatives**:
  - For selected parent nodes, treat child scripts as bandit arms.
  - Use tick‑level rewards to adapt which children get attention.
- **Goal‑aware learning**:
  - Modulate learning rate and exploration using the `goal_vector`.
- **Safety & explainability**:
  - Hard bounds on weights and simple, inspectable update rules.
  - Visualization that exposes which edges/scripts adapted and why.

### 1.2 Non‑Goals (Deferred to M4+)

- No structural graph edits (no new nodes, no pruning/merging).
- No persistent weight updates to disk (`*.swp` remain frozen).
- No learned eval network yet; still rely on Stockfish/heuristic eval for rewards.
- No across‑game consolidation of `Δw_fast` back into `w_base` (that is explicitly M4).

---

## 2. Fast Plasticity Targets (What Can Change)

### 2.1 Primary Targets – KRK/KPK Script Edges

- **POR / SUB edges inside KRK and KPK endgame scripts**:
  - Edges that:
    - Gate sub‑steps within a phase (e.g., which heuristic to try first).
    - Connect scripts to actuators (proposal and scoring).
  - In code: `Graph.edges` where `e.ltype in {POR, SUB}` and `e.src` / `e.dst` are KRK/KPK script nodes.
- These edges’ weights (`e.w`) become **fast‑plastic parameters**:
  - Interpreted as within‑game preferences.
  - Reset to initial pack values at episode boundaries.

### 2.2 Secondary Targets – Macro Gating Edges (Optional Later in M3)

- SUB edges from macro nodes into endgame subgraphs (e.g. phase selectors).
- Same update rule as internal KRK/KPK edges, but guarded by:
  - A global config flag: `macro_plasticity_enabled`.
  - Per‑node flag: `node.meta["plasticity_enabled"] = False` to freeze some macros.

### 2.3 Soft Targets – Exploration & Policy Parameters

- Per‑parent **exploration coefficients** for bandits (e.g. `c_explore` in UCB).
- Optional per‑parent **softmax temperature** for sampling among children.
- Stored in `node.meta` and updated per game, but not persisted.

---

## 3. Reward Signals and Credit Assignment

### 3.1 Reward Definitions

- **Tick‑level reward** (already present in M2.5):
  - `reward_tick = eval_after − eval_before`, clipped to `[-r_max, +r_max]`.
  - Recommended `r_max` in `[1.0, 3.0]` pawns.
- **Episode‑level reward**:
  - `reward_episode` from game result (win = +1, draw = 0, loss = −1).
  - In M3, used mostly for logging and sanity checks, not for updates.

### 3.2 Edge‑Level Credit Assignment

- For each tick `t` we have:
  - `E_t`: set of edges that contributed this tick (from `TickRecord.fired_edges`).
  - `r_t`: the (clipped) `reward_tick`.
- Each edge `e ∈ E_t` receives credit proportional to `r_t`, optionally smoothed by **eligibility traces**:
  - Maintain per‑edge `z_e` (eligibility) within the episode:
    - `z_e ← λ * z_e + I[e ∈ E_t]`, with `λ ∈ [0, 1]` (e.g. `0.8`).
  - This allows earlier decisions to share some credit/blame for rewards that materialize a few ticks later.

### 3.3 Bandit‑Level Credit Assignment

- For a parent `P` with bandit‑controlled children `{c_i}`:
  - Track which child `c_i` is “active” for a given behavioral phase or ply.
  - Aggregate rewards over that phase:
    - `R_i = Σ_t r_t` for ticks where `c_i` was the chosen child.
  - At the end of that phase / ply:
    - `pulls[P][c_i] += 1`
    - `sum_reward[P][c_i] += R_i`
- These stats live only **within the episode** in M3; they are reset between games.

---

## 4. Fast Edge Update Rules

### 4.1 Base Update Formula

For each tick `t` and eligible edge `e ∈ E_t`:

- Let:
  - `r_t_clipped = clip(reward_tick, -r_max, +r_max)`.
  - `z_e` = current eligibility for `e`.
  - `η_tick` = base per‑tick learning rate.
  - `[w_min, w_max]` = hard bounds for edge weights.
- Compute:
  - `Δw_e = η_tick * r_t_clipped * z_e`
  - `w_e = clip(w_e + Δw_e, w_min, w_max)`

Recommended defaults (for KRK experiments):

- `η_tick ≈ 0.05–0.1`
- `r_max ≈ 1.0–2.0` (pawns)
- `w_min = 0.1`, `w_max = 3.0` (avoid sign flips in v1)
- `λ ≈ 0.8` for eligibility traces

### 4.2 Simpler Sign‑Based Variant (First Implementation)

If you want to start even simpler:

- Define a monotone scaling `f(|r_t|) = min(|r_t|, r_max) / r_max`.
- Update rule:
  - `w_e = clip(w_e + η_tick * sign(r_t) * f(|r_t|), w_min, w_max)`
- This matches the informal rule in `recon_roadmap_m2.md` and can later be upgraded to full eligibility‑based updates.

### 4.3 Frequency of Updates (Tick vs Ply)

- **Initial approach**:
  - Update **per tick** but only when:
    - `reward_tick` is defined and `|reward_tick| > ε` (e.g. `ε = 0.01`), or
    - The tick is part of a move decision (e.g. around `env["chosen_move"]`).
- **Later refinement**:
  - Aggregate tick rewards into a **per‑ply reward**:
    - `R_ply = Σ_{t in ply} r_t`
  - Apply a single weight update at the move boundary using `R_ply`.

### 4.4 Lifetime and Reset

- At **episode start**:
  - Reset all `z_e ← 0` for tracked edges.
  - Reset `w_e` to initial values from the current weight pack or graph fixture.
- Optional **within‑episode decay** (for very long games):
  - Periodically:
    - `w_e ← w_init_e + γ * (w_e − w_init_e)` with `γ ∈ (0, 1)` (e.g. `0.9`).
  - This gently pulls weights back toward baseline over time.

---

## 5. Bandit Gating for Sibling Selection

### 5.1 State Per Parent–Child Pair

For a bandit parent `P` with children `{c_i}`:

- Maintain in `P.meta["bandit"]`:
  - `pulls[c_i]`: number of times child `c_i` was selected.
  - `sum_reward[c_i]`: sum of rewards assigned to `c_i`.
  - Optionally `sum_sq_reward[c_i]` for variance estimates.

All of this lives in memory per episode and is discarded at game end.

### 5.2 UCB‑Style Scoring

On a decision where `P` chooses among `{c_i}`:

- Let:
  - `n_i = pulls[c_i]`
  - `N = Σ_j pulls[c_j]`
  - `μ_i = 0` if `n_i == 0`, else `sum_reward[c_i] / n_i`
  - `c_explore` = exploration coefficient (tunable, e.g. `0.7–1.5`)
  - `ε` = small constant to avoid divide‑by‑zero (e.g. `1e-6`)
- Compute:
  - `explore_i = c_explore * sqrt(2 * ln(N + 1) / (n_i + ε))`
  - `score_i = μ_i + explore_i`
- Choice rules:
  - **Deterministic**: pick `argmax_i score_i`.
  - **Stochastic**: softmax over `score_i` with a temperature controlled by `goal_vector`.

### 5.3 Reward Assignment to Bandit Arms

- While a particular child `c_i` is the active plan:
  - Accumulate a running `R_i` from successive `reward_tick` values.
- When the phase/ply ends or a move is committed:
  - `pulls[c_i] += 1`
  - `sum_reward[c_i] += R_i`
- Optionally, you can also track:
  - `last_reward[c_i]` for diagnostics.
  - `phase_context[c_i]` (phase estimate, material pattern) for analysis scripts.

---

## 6. GoalVector Modulation

### 6.1 Derived Scalars: Risk and Urgency

From the `goal_vector` (already present in M2.5), define:

- `risk ∈ [0, 1]` – how dangerous the current situation is.
- `urgency ∈ [0, 1]` – how quickly we need to force progress.

Implementation:

- Start with a simple hand‑crafted mapping based on:
  - Material imbalance.
  - King safety.
  - Distance to checkmate / stalemate patterns.
  - Phase estimate (endgame vs middlegame).
- Log `risk` and `urgency` into `TickRecord.meta` for tuning.

### 6.2 Modulating Learning Rate

- Effective tick learning rate:
  - `η_tick_eff = η_tick_base * (1 + α_risk * risk)`
  - With `α_risk ≥ 0` (e.g. `0.5`).
- Intuition:
  - In high‑risk situations, adapt faster.
  - In very safe or won positions, keep behavior conservative.

### 6.3 Modulating Exploration

- Effective exploration coefficient:
  - `c_explore_eff = c_explore_base * (1 + α_urgency * urgency)`
- Intuition:
  - When urgency is high, try more alternatives (higher exploration).
  - When we just need to “convert” a win, reduce exploration.

### 6.4 Implementation Hooks

- Implement a small helper:
  - `compute_m3_modulators(goal_vector) -> dict(risk, urgency, eta_tick_eff, c_explore_eff)`
- Call this helper:
  - Inside the KRK persistent loop before applying plasticity/bandit updates.
  - Log outputs for later analysis.

---

## 7. TraceDB Integration and Data Flow

### 7.1 Online Usage (Within Game)

- During the KRK persistent decision cycle:
  - Construct a `TickRecord` per tick with:
    - `tick_id`, `phase_estimate`, `goal_vector`, `board_fen`.
    - `active_nodes` (optional for M3 logic but useful for analysis).
    - `fired_edges` for credit assignment.
    - `eval_before`, `eval_after`, `reward_tick`.
  - Immediately after logging (or building) the `TickRecord`, call:
    - `apply_fast_plasticity(tick_record, engine_state, modulators)`
    - `update_bandits(tick_record, engine_state, modulators)`
- These helpers:
  - Do **not** need to know about `TraceDB`; they operate on the same data structure used for logging.

### 7.2 Offline Usage (Prep for M4)

- TraceDB continues to store:
  - `TickRecord` and `EpisodeRecord` with the new `meta` fields.
- M3 adds the **expectation** that offline analysis scripts will:
  - Read traces.
  - Produce diagnostics (e.g., per‑edge reward histograms, bandit arm performance).
  - Feed into M4’s slow consolidation design.

---

## 8. Safety, Constraints, and Visualization

### 8.1 Hard Safety Constraints

- **Weight bounds**:
  - Every update enforces `w_min ≤ w_e ≤ w_max`.
- **Max drift per episode**:
  - Optional check: `|w_e − w_init_e| ≤ max_delta_w` (e.g. `1.0`).
  - Violations are logged and clamped.
- **Toggles**:
  - Global: `PLASTICITY_ENABLED`, `BANDIT_ENABLED` flags in config/CLI.
  - Per‑subgraph: `node.meta["plasticity_enabled"] = False` and/or `node.meta["bandit_enabled"] = False`.

### 8.2 Debug and Shadow Modes

- **Shadow mode**:
  - Compute hypothetical updates without mutating live `Edge.w`.
  - Log `(w_e_current, w_e_hypothetical)` into `TickRecord.meta` for a small test batch.
- **Diff reporting**:
  - At episode end:
    - For each tracked edge, compute `Δw_e = w_e − w_init_e`.
    - Log top‑K edges by `|Δw_e|` along with their aggregated reward statistics.

### 8.3 Visualization Hooks

- Extend viz payload to include:
  - Current edge weights for tracked edges.
  - Recent updates (`Δw_e` signs) over a short time window.
- Suggested visual encodings:
  - Line thickness ∝ `w_e` magnitude.
  - Green/red tint for recent strengthening/weakening.
  - Simple tooltip showing:
    - `w_e`, `Δw_e` over episode.
    - For bandit parents: `pulls` and `mean_reward` per child.

---

## 9. Implementation Order (M3 Phases)

### 9.1 M3.1 – Minimal KRK‑Only Fast Plasticity

1. **Plasticity core module**:
   - Implement `plasticity.fast_update_edges(tick_record, graph, state, modulators)` as a pure function.
   - Restrict it to a **whitelist** of KRK edges initially (e.g. configured by node/edge IDs).
2. **Wire into KRK persistent loop**:
   - After each tick with defined `reward_tick`, call the plasticity helper.
3. **Sanity tests**:
   - Unit‑test the helper on a tiny synthetic graph with known reward sequences and expected `w_e` changes.

### 9.2 M3.2 – Single Bandit Parent (KRK)

1. Choose one high‑leverage parent (e.g. a phase script that chooses between alternative shrink strategies).
2. Implement:
   - `bandit.update_state(parent_node, chosen_child, reward_segment, modulators)`.
   - `bandit.choose_child(parent_node, modulators)` using UCB.
3. Wire this into the KRK decision flow:
   - Use bandit choice to bias which child scripts get requested or prioritized.
4. Log bandit decisions in `TickRecord.meta` and verify behavior with small experiments.

### 9.3 M3.3 – GoalVector Modulation

1. Implement `compute_m3_modulators(goal_vector)` returning `risk`, `urgency`, `η_tick_eff`, `c_explore_eff`.
2. Thread this into both plasticity and bandit helpers.
3. Tune qualitatively:
   - Inspect traces and viz to see how modulators behave in sharp vs quiet positions.

### 9.4 M3.4 – Generalization and Tuning

1. Extend fast plasticity to:
   - Additional KRK edges.
   - KPK endgame scripts.
2. Optionally enable macro edge plasticity under a feature flag.
3. Run batches of games with/without plasticity:
   - Track mate speed, stalemate frequency, and stability.
4. Calibrate hyper‑parameters (`η_tick`, `r_max`, `w_min/max`, `c_explore`) based on observed stability and performance.

---

## 10. Testing and Evaluation

### 10.1 Unit Tests

- **Plasticity**:
  - Verify that a deterministic sequence of `reward_tick` values produces the expected `w_e` trajectories and respects clipping.
- **Bandit**:
  - Toy environment where one child has higher mean reward; ensure selection frequency shifts toward that child over time.
- **Goal modulation**:
  - Given fixed `goal_vector`s, test that `η_tick_eff` and `c_explore_eff` scale monotonically with `risk` and `urgency`.

### 10.2 Integration and Regression Tests

- **KRK performance**:
  - With plasticity off vs on:
    - Assert no regression in “mate within N plies” success rate.
    - Track average plies to mate and stalemate frequency.
- **Stability**:
  - Long‑run KRK persistent games with plasticity on:
    - Assert no edge ever violates `w_min/max` or `max_delta_w`.
    - Assert the engine loop terminates decisions within expected tick bounds.
- **Logging**:
  - Ensure `TickRecord` and `EpisodeRecord` remain valid and JSON‑serializable with added fields.

---

## 11. Hooks for M4 – Slow Consolidation

Although M4 is out of scope for this document, M3 should lay clean hooks:

- Treat current `Edge.w` as `w_base + Δw_fast` conceptually, even if only `w` is stored.
- At episode end:
  - Summarize per‑edge and per‑bandit‑arm statistics into a compact structure that M4 can consume:
    - Mean `Δw_fast` over the episode.
    - Cumulative reward where the edge/arm was active.
- M4 will:
  - Aggregate these episode‑level summaries over many games.
  - Fold a fraction into persistent `w_base` and reset `Δw_fast` before new games.


# Summary after implementation 1 Dec 2025: 

## M3 Fast Plasticity & Bandit Control Implementation Complete
I've implemented the full M3 plan as specified. Here's a summary of what was created:
### New Files Created
  Core Plasticity Module (src/recon_lite/plasticity/):
  __init__.py - Package exports
    fast.py - Edge weight adaptation with eligibility traces
    EdgePlasticityState dataclass
    PlasticityConfig with tunable parameters
    init_plasticity_state(), update_eligibility(), apply_fast_update(), reset_episode()
  bandit.py - UCB-style selection among sibling scripts
    BanditArmState dataclass
    BanditConfig with exploration coefficient
    init_bandit_state(), ucb_score(), choose_child(), assign_reward()
  modulation.py - Goal-aware parameter scaling
    ModulationConfig with alpha coefficients
    Modulators dataclass with risk, urgency, eta_tick_eff, c_explore_eff
    compute_modulators(), compute_modulators_from_board()

### Heuristic Evaluation (src/recon_lite_chess/eval/):
  __init__.py - Package exports
  heuristic.py - Position evaluation for reward computation
  eval_position() - material + king safety + mobility + KRK bonus
  compute_reward_tick() - clipped delta from eval changes
  eval_position_stockfish() - optional Stockfish evaluation
### Tests:
tests/test_plasticity.py - 23 unit tests for plasticity, bandit, and modulation
tests/test_plasticity_integration.py - 12 integration tests with KRK graph


### Modified Files
- KRK Persistent Demo (demos/persistent/krk_persistent_demo.py):
- Added imports for plasticity, bandit, modulation, and eval modules
- Added constants for KRK plasticity edges and bandit parents
### Extended play_persistent_game() with new parameters:
  --plasticity - Enable fast plasticity
  --bandit - Enable bandit control
  --eval-mode {heuristic,stockfish} - Evaluation mode
  --plasticity-eta, --plasticity-r-max, --plasticity-lambda
  --bandit-c-explore
### Modified _decision_cycle() to:
  Compute eval before/after each tick
  Update eligibility traces for fired edges
  Apply fast weight updates with goal modulation
  Track M3 data in env for logging
### Network Visualization (demos/visualization/network-visualization.js):
  Added tracking for edge weights, deltas, and bandit stats
  Added updatePlasticityData() method
  Added getEdgePlasticityColor() for green/red tinting based on weight changes
  Added getEdgeWeightMultiplier() for thickness based on weight
  Modified edge drawing to use plasticity visualization

## Usage
Run KRK with plasticity enabled:
uv run python demos/persistent/krk_persistent_demo.py \  --plasticity \  --bandit \  --eval-mode heuristic
Or with Stockfish evaluation:
uv run python demos/persistent/krk_persistent_demo.py \  --plasticity \  --bandit \  --eval-mode stockfish \  --engine /usr/games/stockfish
All 35 new tests pass, and the implementation is backward-compatible (all new behavior is opt-in via CLI flags).