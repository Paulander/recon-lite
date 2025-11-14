# ReCoN Roadmap (M2.5–M8): From Static Scripts to Self‑Organizing Graph

This document captures the "best" consolidated plan for evolving the current ReCoN chess system from M2.x into a genuinely self‑organizing architecture while remaining explainable and CPU‑friendly.

---

## 0. Assumptions and High‑Level Goals

- You already have:
  - Macrograph with phases (opening / middlegame / endgame).
  - Local scripts for KPK / rook endgames.
  - Continuous activations, micro‑ticks, and request→confirm dynamics.
  - A visualization layer for nodes, edges, and activations.

- You want:
  - Online adaptation within a game ("fast" plasticity).
  - Cross‑game consolidation and learning ("slow" plasticity).
  - Automatic discovery of mid‑level features / motifs.
  - Script induction and structural editing in line with ReCoN philosophy.
  - Strong explainability and visualization of how the graph evolves.

---

## 1. M2.5 – Semantics and Instrumentation

### 1.1 Define Evaluation / Reward

You need a clear objective signal for learning and structural edits. For chess:

- Use an evaluation function `eval(board)`:
  - Initially this can be:
    - A simple handcrafted eval (material + piece‑square tables + king safety), or
    - A call to an engine (e.g. Stockfish) with shallow depth for select positions.
- Derive per‑tick and per‑episode rewards:
  - `reward_tick = eval_after − eval_before` (clipped).
  - `reward_episode = game_result` (win = +1, draw = 0, loss = −1).

Engine‑based eval can be used more sparsely (e.g. once per move, or on sampled ticks) if cost is an issue.

### 1.2 Tick and Episode Schema

Define a standard structure for logging:

- **Tick**
  - `tick_id`
  - `phase_estimate`
  - `goal_vector` (risk/initiative/phase progress, etc.)
  - `board_encoding` (compact internal representation)
  - `active_nodes` (IDs)
  - `fired_edges` (list of request→confirm edges)
  - `action` (if a move was chosen at this tick)
  - `eval_before`, `eval_after`
  - `reward_tick`

- **Episode (Game)**
  - `ticks: [Tick…]`
  - `result` (win/draw/loss, score)

### 1.3 TraceDB

Implement an append‑only, queryable storage for episodes:

- Circular buffer (e.g. up to N games) to bound memory.
- Index by:
  - Outcome (win/draw/loss).
  - Phase / material pattern.
  - Large `|reward_tick|` (big swings).
  - Node/edge participation.

### 1.4 Node and Edge Metadata

Extend node/edge data structures with runtime metadata:

- Node:
  - `id`
  - `type: {feature, script, policy, eval, sensor}`
  - `lifecycle: {TRAINING, LATENT, BINDING, INITIALIZED}`
  - `usage: {activations, successes, failures, last_used}`

- Edge:
  - `id`
  - `src`, `dst`
  - `role: {request, confirm, modulate}`
  - `weight`
  - `plasticity_params` (learning rates, bounds)
  - `usage: {successes, failures, last_used}`

This instrumentation is mandatory before any genuine self‑organization.

---

## 2. M3 – Fast Plasticity & Bandit Control (Within‑Game)

Goal: make the ReCoN **adapt during a single game** without changing topology.

### 2.1 Fast Edge Updates (Hebbian / Delta‑Rule)

For each edge used in a tick:

- If `reward_tick > 0`, strengthen the edge.
- If `reward_tick < 0`, weaken it.
- Clamp weights to `[w_min, w_max]`.

Conceptually:

- `w ← clamp(w + sign(reward_tick) * η * f(|reward_tick|), w_min, w_max)`

These are **fast**, potentially reset or decayed at the start of each new game.

### 2.2 Bandit Gating for Sibling Selection

For parent node choosing between child subgraphs `{cᵢ}`:

- Maintain per‑child statistics: mean reward when chosen, count, variance.
- Compute a bandit score (e.g. UCB):

  - `score[cᵢ] = mean_reward[cᵢ] + exploration_term[cᵢ]`

- Use `score` to bias which child gets attention / time / priority.

This gives context‑sensitive selection of strategies without overcomplicating the policy.

### 2.3 GoalVector Modulation

Use `goal_vector` to modulate:

- Exploration rate (e.g. in desperate positions or sharp tactics).
- Risk appetite (e.g. pushing for win vs securing draw).

This keeps behavior coherent with high‑level intent.

---

## 3. M4 – Slow Consolidation Across Games

Goal: separate **fast** within‑game plasticity from **slow** across‑game learning.

### 3.1 Baseline vs Fast Offsets

Maintain for each edge:

- `w_base` – slow, persistent weight.
- `Δw_fast` – fast, per‑game adjustment.

During a game:

- Use `w = w_base + Δw_fast`.
- Update `Δw_fast` via fast rules.

Between games:

- Aggregate `Δw_fast` from good episodes and fold a fraction into `w_base`:

  - `w_base ← w_base + α * mean(Δw_fast_over_good_games)`

- Reset `Δw_fast`.

### 3.2 Node / Edge Trust Scores

From TraceDB, compute per‑node/edge:

- Average reward contribution when active.
- Context distribution (phase, material, king safety).
- Variance and reliability of effect.

These scores later guide feature/script promotion and pruning.

---

## 4. M5 – BindingDescriptors and Representation Design

Before discovery, define precisely what you will cluster.

### 4.1 Local BindingDescriptor

For each relevant focus (e.g. around last move, king, critical squares), create a descriptor:

- `patch_3x3_piece_channels` (one‑hot piece type + color per square).
- `patch_3x3_attack_mask` (attacked/defended indicators).
- `king_distances` (distance to both kings).
- `phase_estimate` (opening/middlegame/endgame).
- `last_move_vector` (from→to, normalized).
- `local_material_signals` (material in/near patch).

### 4.2 Sampling Strategy

Sample descriptors from ticks where:

- `|reward_tick|` is high (big eval swings).
- Games flip from winning→losing or vice versa.
- Stable conversion of advantage occurs.

Tag each descriptor with:

- Episode outcome.
- Context labels (phase, open/closed, material type).
- Local reward stats.

### 4.3 Dimensionality Reduction

Reduce descriptors to a compact vector (8–16 dims), e.g. via PCA.

This embedding is what you feed into k‑means or similar clustering.

---

## 5. M6 – Feature Discovery ("Dreaming")

Now you can implement the Gemini‑style "dreaming" in a grounded way.

### 5.1 Clustering

Offline (or during idle time):

- Run k‑means (or GMM) on the reduced descriptor embeddings.
- For each cluster, compute:
  - centroid,
  - member count,
  - average reward_tick and outcome,
  - context distribution.

### 5.2 Propose Feature Nodes

For each promising cluster:

- Require minimum support (enough samples).
- Require meaningful reward correlation (e.g. often appears before good outcome, or reliably signals danger).

Create a `FeatureNode` with:

- A pattern‑matching function approximating the cluster centroid.
- Initial low‑impact connections to relevant parents.
- Lifecycle state = `LATENT`.

### 5.3 Canary Evaluation

Enable a handful of new latent features as observers:

- Let them fire and log their activations.
- Measure:
  - Predictive power (mutual information with reward_tick).
  - Redundancy vs existing features.

Promote or discard:

- Promote to `INITIALIZED` if useful.
- Discard if redundant or noise.

---

## 6. M7 – Script Induction from Traces

Goal: build new mid‑level scripts (subgraphs) from repeated successful behavior.

### 6.1 Mining Subpaths

From winning (or high‑score) episodes in TraceDB:

- Extract windows of ticks with positive cumulative reward.
- Consider sequences of:
  - context summaries (phase, material, king safety),
  - active feature nodes,
  - actions (candidate moves).

Identify frequent subpaths that:

- Occur in similar contexts.
- Have consistently positive cumulative reward.
- Involve overlapping sets of features.

### 6.2 Construct Script Nodes

For each candidate subpath:

- Build a `ScriptNode` encapsulating:
  - a sequence of subgoals (feature activations or positional patterns),
  - typical action proposals,
  - conditions to terminate or bail out.

Connect script nodes:

- As children of higher‑level phase/strategy nodes.
- With initial conservative weights.

### 6.3 Transactional Graph Updates

Treat topology edits as transactions:

- Fork a `GraphCandidate` from current graph.
- Insert new script nodes and required edges.
- Run:
  - self‑play mini‑matches,
  - curated test positions.

Compare metrics against baseline:

- Win rate / eval stability / computation cost.

Only if non‑regressive:

- Promote `GraphCandidate` to the new main graph.
- Keep a versioned snapshot of the old graph for rollback.

---

## 7. M8 – Structural Evolution, Pruning, and Meta‑Stability

Now you manage long‑term complexity.

### 7.1 Pruning

Periodically prune:

- Nodes with extremely low usage and poor reward contribution.
- Edges that are consistently harmful.

Use hysteresis to avoid constant add/remove thrashing.

### 7.2 Merging and Compression

Detect:

- Feature nodes with highly similar activation profiles / descriptors.
- Scripts that differ only by minor details.

Merge into more general nodes and adjust associations.

### 7.3 Versioned Snapshots

Maintain explicit versions:

- `Graph_vN`, `Graph_vN+1`, etc.
- Store key metrics and example games per version.

This is both a safety net and a research artifact: you can **show how the ReCoN evolved**.

### 7.4 Meta‑Policies for Structural Change

Over longer timescales, adjust:

- How aggressively you allow new nodes.
- How conservative pruning thresholds are.
- How often you run dreaming / induction.

So the system can have a "growth phase" and later a more stable, exploitative phase.

---

## 8. Recommended Coding Order (Practical Implementation Sequence)

1. Implement eval → reward pipeline (possibly via Stockfish or simple heuristic eval).
2. Define Tick/Episode schemas and implement TraceDB.
3. Extend node/edge structs with metadata (type, lifecycle, usage, trust).
4. Implement fast within‑game plasticity rules (edge deltas) and bandit gating.
5. Implement slow cross‑game consolidation for edge baselines.
6. Define BindingDescriptor and descriptor sampling strategy.
7. Add dimensionality reduction + offline clustering for feature discovery.
8. Implement latent feature nodes + canary evaluations and promotion.
9. Implement subpath mining and script induction from successful traces.
10. Add transactional graph updates (candidate graphs, test harness, regression checks).
11. Add pruning, merging, and versioned snapshots.

Following this roadmap will move your current M2.x ReCoN from a static, scripted system into a self‑organizing, explainable graph that discovers its own useful features and scripts over time, while remaining debuggable and CPU‑feasible.



---

## 9. GPU Rig Considerations (2× GTX 5070, 12 GB)

Once a dual‑GPU setup is available, several stages of this roadmap can be accelerated dramatically:

- **Eval Network Training**
  - Use Stockfish (or another engine) initially as an oracle to label board positions from TraceDB.
  - Train a compact neural eval model on GPUs to approximate engine scores.
  - Replace most engine calls with this learned eval, keeping occasional engine calls for calibration.

- **Representation Learning for BindingDescriptors**
  - Replace purely hand‑crafted descriptors + PCA with a learned embedding model (e.g. small CNN or transformer over board patches).
  - Train this embedding on GPUs using contrastive or reconstruction objectives.
  - Use the learned embeddings as inputs to clustering / feature discovery.

- **Faster Feature Discovery and Script Induction**
  - Run clustering and other unsupervised steps on large batches using GPU kernels.
  - Use the learned embeddings to mine higher‑level, more general motifs and scripts.

Important: the **early phases (M2.5–M4)**—instrumentation, plasticity logic, bandit gating—remain **CPU‑centric**, as they are mostly control‑flow and graph logic. The GPUs become crucial once you introduce:

- Learned eval models.
- Learned feature/patch representations.
- Larger‑scale self‑play and offline training.

---

### 9.1 Multi‑GPU Strategy for ReCoN

Given a dual‑GPU setup (e.g. 2× 12 GB cards), a pragmatic division of labor is:

- **GPU 0 – Online / Inference‑Oriented Tasks**
  - Batched inference for node‑local MLPs (feature detectors, small script policies).
  - Inference for the global eval model (approximate Stockfish).
  - Optionally, lightweight policy/value heads used during actual ReCoN play.

- **GPU 1 – Offline / Training‑Oriented Tasks**
  - Background training of node MLPs from TraceDB (supervised or RL‑style updates).
  - Representation learning for BindingDescriptors (autoencoders / contrastive models).
  - Clustering and other heavy discovery steps for M6/M7.
  - Self‑play simulation batches if/when they become GPU‑bound.

Key principles:

- Keep the **ReCoN graph logic and request→confirm orchestration on CPU**; send batched queries (many node evaluations at once) to GPU(s) for efficient MLP inference.
- Avoid splitting a *single* logical inference step across GPUs (inter‑GPU communication overhead can dominate for small models).
- Exploit **task parallelism** rather than model parallelism:
  - While GPU 0 runs inference for live games, GPU 1 trains new or updated nodes/scripts using stored traces.

Visualization should typically stay on the display/host GPU and is not expected to be a primary load compared to ML training/inference.

---

## 10. Recommended Coding Order (Practical Implementation Sequence)

1. Implement eval → reward pipeline (possibly via Stockfish or simple heuristic eval).
2. Define Tick/Episode schemas and implement TraceDB.
3. Extend node/edge structs with metadata (type, lifecycle, usage, trust).
4. Implement fast within‑game plasticity rules (edge deltas) and bandit gating.
5. Implement slow cross‑game consolidation for edge baselines.
6. Define BindingDescriptor and descriptor sampling strategy.
7. Add dimensionality reduction + offline clustering for feature discovery.
8. Implement latent feature nodes + canary evaluations and promotion.
9. Implement subpath mining and script induction from successful traces.
10. Add transactional graph updates (candidate graphs, test harness, regression checks).
11. Add pruning, merging, and versioned snapshots.

Following this roadmap will move your current M2.x ReCoN from a static, scripted system into a self‑organizing, explainable graph that discovers its own useful features and scripts over time, while remaining debuggable and CPU‑feasible, and ready to exploit a dual‑GPU rig when available.

