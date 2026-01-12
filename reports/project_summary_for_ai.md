# ReCoN-lite Chess — AI Orientation Brief

## Purpose and Philosophy
- Implements Joscha Bach / Priska Herger's Request–Confirmation Network (ReCoN) ideas for chess: top-down requests via `sub/por`, bottom-up confirmations via `sur/ret`, continuous activations, bindings, and explicit traceability.
- Goal: an explainable ReCoN-based chess agent (~1900 Elo target) that orchestrates hand-written heuristics, search engines, and learned components rather than a single end-to-end model.
- Paper reference: "Request confirmation networks for neuro-symbolic script execution" (Bach & Herger). Use this as the canonical conceptual anchor.

## History & Milestones (M1→M8)
- **M1** (KRK, continuous layer): micro-ticks and soft activations (`core/activations.py`, `time/microtick.py`), binding tables (`binding/manager.py`), KRK strategic layer + viz.
- **M2/M2.5** (macrograph + instrumentation): macrograph spec and Subgraph Weight Packs (SWPs) for KRK/KPK/rook techniques; TraceDB (`trace_db.py`, `macro_trace.py`); Stockfish teacher and batch/block eval pipelines.
- **M3** (fast plasticity + bandits): within-game weight updates on selected POR/SUB edges, bandit gating for sibling scripts, goal-aware modulation (`plasticity/fast.py`, `bandit.py`, `modulation.py`).
- **M4** (slow consolidation + eval upgrade): cross-game consolidation of `w_base` from episode summaries (`plasticity/consolidate.py`), bandit priors refresh, richer heuristic/Stockfish/hybrid eval manager.
- **M5** (structure learning + EVOLUTION): **Current focus.** Motif extraction, clustering, script proposals, trust scoring, stem cell lifecycle with XP system, recursive branching, vertical topology growth.
- **M6** (full-game, multi-scale dynamics): fan-in terminals, goal hierarchy (Ultimate→Strategic→Tactical→Sensor), plan persistence, opening/middlegame scripts.
- **M7** (distillation): Feature extraction for ML, Stockfish data collection, distilled eval model training (stub implementation).
- **M8** (reverse curriculum + FeatureHub): Continuous affordance signals, global feature hoisting, CurriculumManager for 4-phase reverse training.

Source docs: `updates_continuous.md`, `recon_roadmap_m3_fast_plasticity.md`, `recon_roadmap_m5_structure_learning.md`, `recon_roadmap_m6_full_game_architecture.md`.

---

## Current Architecture Snapshot

### Core ReCoN Engine (`src/recon_lite/`)

#### Graph Structure (`graph.py`)
- **Node Types**: `SCRIPT` (intermediate goals) and `TERMINAL` (sensors/actuators)
- **Node States**: INACTIVE, REQUESTED, ACTIVE, SUPPRESSED, WAITING, TRUE, CONFIRMED, FAILED
- **Link Types**:
  - `SUB` (subgraph): Parent → Child (top-down request)
  - `SUR` (super): Child → Parent (bottom-up confirmation)  
  - `POR` (predecessor): Previous → Next in sequence (temporal ordering)
  - `RET` (return): Reverse temporal confirmation
- **Key Constraint**: TERMINAL nodes can only receive SUB and send SUR. POR/RET require SCRIPT nodes.

#### Engine (`engine.py`)
- Discrete-time state machine with request/confirmation cycles
- Subgraph locking: `lock_subgraph(root_id, sentinel_fn)` keeps specific network active
- Microtick propagation for continuous activations

#### Continuous Activations
- Each node has `ActivationState` with value in [0.0, 1.0]
- Propagation: `z_i = Σ(w_ij * a_j)` smoothed via exponential moving average
- Supports aggregation modes: "avg" (OR-like), "and" (min, TRUE AND gate)

### Plasticity System

| Layer | File | Scope | Description |
|-------|------|-------|-------------|
| M3 Fast | `plasticity/fast.py` | Per-episode | Δw on whitelisted edges, eligibility traces |
| M3 Bandit | `plasticity/bandit.py` | Per-episode | UCB/softmax gating among siblings |
| M4 Slow | `plasticity/consolidate.py` | Cross-game | Aggregates summaries to update `w_base` |

---

## M5 Evolution System ⭐ (Latest Focus)

The M5 system enables the ReCoN graph to **grow its own topology** through a biological-inspired lifecycle.

### Three-Tier Stem Cell Lifecycle

```
DORMANT → EXPLORING → CANDIDATE → TRIAL → MATURE
                                    ↓
                                DEMOTED (XP ≤ 0)
```

| Tier | State | Description |
|------|-------|-------------|
| 1 | EXPLORING | Collects samples during high-reward moments |
| 1 | CANDIDATE | Has enough samples (≥50), awaits trial |
| 2 | TRIAL | Transient vertex in graph, earns XP to prove utility |
| 3 | MATURE | Permanent node in topology.json, fully trusted |

### XP System (TRIAL tier)
- **Initial XP**: 50 (on promotion to TRIAL)
- **Success** (positive affordance delta): +10 XP
- **Failure** (negative affordance delta): -10 XP
- **Decay**: -1 XP per cycle (cost of living)
- **Solidify threshold**: XP ≥ 100 → MATURE
- **Demotion threshold**: XP ≤ 0 → back to EXPLORING

### Key M5 Concepts

#### Recursive Branching
When a stem cell reaches MATURE (100 XP), it can **spawn children**:
- Children inherit parent's `pattern_signature` as starting context
- Children link to parent as their `local_root_id` (vertical parenting)
- Creates hierarchical tactical reasoning trees instead of flat topology

#### Vertical Parenting vs Backbone
- **Backbone nodes**: kpk_root, kpk_detect, kpk_execute, kpk_finish, kpk_wait
- **Old behavior**: All new sensors wire to backbone (flat)
- **M5 behavior**: Children wire to their MATURE parent (hierarchical)
- **Depth limit**: MAX_BRANCH_DEPTH = 5 (prevents O(n²) propagation)

#### Sparsity Constraint
A new node must be >10% better (z_sur) than its parent to survive. Forces "elegant" solutions over redundant clusters.

#### Survival Bond
If a parent is pruned, its children experience 2x XP decay (accelerated death of orphaned branches).

#### AND-Gate Hoisting
When TRIAL cells correlate ≥85% in win-coactivations, they're hoisted into an intermediate SCRIPT node using `aggregation="and"` (min function). This creates TRUE logical AND gates.

#### POR Chain Discovery
Analyzes temporal patterns (A fires before B in wins) to create POR links between SCRIPT-wrapped sensors. Enables sequential reasoning: Opposition → Protect → Promote.

### Evolution Data Flow

```
evolution_driver.py
    ├── Online Phase (play games)
    │   ├── ReConEngine runs microticks
    │   ├── StemCellManager collects samples
    │   └── TraceDB logs episodes
    │
    └── Structural Phase (analyze & grow)
        ├── StructureLearner.scan_for_affordance_spikes()
        ├── StructureLearner.find_high_impact_stem_cells()
        ├── Promote CANDIDATE → TRIAL (if consistency ≥ 0.40)
        ├── Decay XP for all TRIAL cells
        ├── Solidify TRIAL → MATURE (if XP ≥ 100)
        ├── Demote TRIAL → EXPLORING (if XP ≤ 0)
        ├── Spawn children from MATURE cells
        ├── Hoist correlated clusters into AND gates
        ├── Discover POR chains from temporal patterns
        └── Save snapshot to snapshots/evolution/
```

### snapshots/evolution/ Directory Structure

```
snapshots/evolution/
├── {run_name}/
│   ├── stage0/
│   │   ├── cycle_0001.json  # Topology snapshot
│   │   ├── cycle_0002.json
│   │   ├── ...
│   │   └── stem_cells.json  # StemCellManager state
│   ├── stage1/
│   │   ├── cycle_0001.json
│   │   └── stem_cells.json  # Inherited from stage0
│   └── ...
```

Each `cycle_XXXX.json` contains:
- `nodes`: Dict of node specs (including transient TRIAL cells)
- `edges`: Dict of edge specs with weights
- `timestamp`, `cycle` metadata

Each `stem_cells.json` contains:
- All `StemCellTerminal` instances with samples, XP, state
- Win-coactivation tracking data for AND-gate discovery
- Can be loaded for next stage (weight inheritance)

---

## Chess Layer (`src/recon_lite_chess/`)

### Endgame Networks (`scripts/`)
| Network | File | Goal | Win Rate |
|---------|------|------|----------|
| KRK | `krk_nodes.py`, `krk_strategy.py` | Checkmate | 90% |
| KPK | `scripts/kpk.py` | Pawn promotion | 80% |
| KQK | `scripts/kqk.py` | Checkmate | 70%+ |
| Rook Endings | `scripts/rook_endgame.py` | Lucena/Philidor | Partial |

### KPK Backbone Structure
```
kpk_root
├── kpk_detect
│   ├── kpk_material_check (TERMINAL/sensor)
│   └── kpk_push_window (TERMINAL/sensor)
├── kpk_execute
│   ├── kpk_move_selector (TERMINAL/actuator)
│   └── kpk_opposition_probe (TERMINAL/sensor)
├── kpk_finish
│   └── kpk_promotion_probe (TERMINAL/sensor)
└── kpk_wait
    └── kpk_wait_for_change (TERMINAL/sensor)

POR sequence: detect → execute → finish → wait
```

### Curriculum System (13 Stages)

Training uses reverse curriculum: start from easy endgames, work backward.

| Idx | Stage | Description |
|-----|-------|-------------|
| 0 | SPRINTER | Pawn on 7th, king far. Just push! |
| 1-5 | Discovery Bridge | Baby steps: guardian, step-aside, shouldering |
| 6 | ESCORT | King support (original Stage 1) |
| 7 | SQUARE_RULE | Racing calculation |
| 8 | FRONTAL_BLOCKADE | Shouldering |
| 9 | KEY_SQUARES | Direct opposition |
| 10 | PIVOT | Distant opposition |
| 11 | CORNER_TRAP | Rook pawn draws |
| 12 | ZUGZWANG | Triangulation |

### FeatureHub (`features/hub.py`)
Global registry of 30+ hoisted features across 6 categories:
- **TACTICAL**: fork_available, pin_present, hanging_piece, skewer, back_rank_vulnerable
- **GEOMETRIC**: opposition_status
- **MATERIAL**: material_advantage
- **POSITIONAL**: king_safety, center_control, pawn_structure
- **DYNAMIC**: mobility
- **PHASE**: phase_opening, phase_endgame, affordance_krk, affordance_kpk, affordance_kqk

---

## Key File Reference

### Core ReCoN
| File | Purpose |
|------|---------|
| `src/recon_lite/graph.py` | Node, Edge, Graph, LinkType definitions |
| `src/recon_lite/engine.py` | ReConEngine with subgraph locking |
| `src/recon_lite/nodes/stem_cell.py` | StemCellTerminal, StemCellManager, XP system |
| `src/recon_lite/learning/m5_structure.py` | StructureLearner, POR discovery, AND-gate hoisting |
| `src/recon_lite/models/registry.py` | TopologyRegistry for dynamic graph loading |
| `src/recon_lite/viz/evolution_viz.py` | Graph visualization, diff rendering |

### Chess Domain
| File | Purpose |
|------|---------|
| `src/recon_lite_chess/scripts/kpk.py` | KPK network with promotion detection |
| `src/recon_lite_chess/training/generators.py` | Position generators, KPK_STAGES curriculum |
| `src/recon_lite_chess/features/hub.py` | Global FeatureHub |
| `src/recon_lite_chess/affordance/sensors.py` | Continuous affordance signals |

### Training & Scripts
| File | Purpose |
|------|---------|
| `scripts/evolution_driver.py` | Main evolution training loop |
| `scripts/curriculum_training.sh` | End-to-end training script |
| `scripts/analyze_training.py` | Statistics and markdown reports |
| `topologies/kpk_topology.json` | Base KPK network definition |

---

## ⭐ HOW WE GOT HERE: The Project Journey

### The Core Challenge
We wanted to prove that a ReCoN network can **autonomously discover tactical chess patterns** (opposition, key squares, zugzwang) without being explicitly programmed with those concepts - using only game rules and reward signals.

### Key Design Decisions (Why We Made Them)

| Decision | Rationale | Alternative Rejected |
|----------|-----------|---------------------|
| **Pure strategies** (no hardcoded heuristics) | Fair comparison - network learns WHAT moves are good | Could have pre-weighted promotions, king centralization |
| **Stem cell lifecycle** | Evolutionary pressure selects useful patterns | Could have used static hand-picked sensors |
| **XP system** | Darwinian selection mechanism - good sensors live | Could have used simple threshold pruning |
| **Curriculum learning** | Start easy (pawn on 7th), build complexity | Could have trained on random positions |
| **Pack templates** (AND/OR gates) | Composable primitives for complex patterns | Could have used monolithic sensor nodes |
| **Vertical parenting** | True hierarchical reasoning (depth 3+) | Could have kept flat topology |
| **Graph.to_snapshot()** | Persist spawned packs across cycles | Was losing packs on each cycle! |

### Evolution of the System (Chronological)

```
PHASE 1: Rigid Decision Graph (M1-M2)
├── Hand-coded KRK/KPK networks with hardcoded strategies
├── Fixed topology - humans designed all nodes and edges
└── Problem: No adaptation, no learning

PHASE 2: Weight Learning (M3-M4)
├── Added plasticity - network learns WHICH existing nodes to trust
├── Bandit gating - learn to select between strategy alternatives
└── Problem: Can't discover NEW patterns, only weight existing ones

PHASE 3: Structure Learning (M5)
├── Stem cells - exploratory sensors that collect samples
├── XP system - survival of the fittest patterns
├── AND-gate hoisting - combine correlated sensors
└── Problem: Flat topology - all sensors at depth 1-2

PHASE 4: Vertical Growth (M5.2) ← Current
├── Packs spawn under existing packs (depth 3+)
├── SURVIVOR BYPASS - auto-promote survivors
├── Dual-depth spawning - L1 + deeper simultaneously
└── Result: Hierarchical pattern detectors emerge
```

### What's Hardcoded vs Learned

This is critical for the article's validity:

| Component | Hardcoded? | Details |
|-----------|------------|---------|
| **Chess rules** | ✅ Yes | Legal moves from python-chess library |
| **Piece types** | ✅ Yes | "These are pawn moves, these are king moves" |
| **Goal** | ✅ Yes | Promote pawn = win (reward signal) |
| **Move weights** | ❌ Learned | All moves start with neutral 0.5 weight |
| **Move selection** | ❌ Learned | Softmax over `learned_weights` from M3/M4 |
| **Pattern discovery** | ❌ Learned | Stem cells discover opposition, timing, etc. |
| **Structure** | ❌ Learned | Pack spawning creates AND/OR hierarchies |

**The defensible claim**: "Given legal moves and binary win/loss rewards, the network autonomously discovers hierarchical tactical patterns corresponding to known chess theory."

### Pure Strategy Implementation

The `strategy_actuator.py` strategies output ALL moves with **neutral weights**:

```python
# push_strategy: outputs ALL pawn moves, learned arbiter selects
for move in board.legal_moves:
    if piece.piece_type == chess.PAWN:
        pawn_moves.append({
            "move": move.uci(),
            "weight": 0.5,  # NEUTRAL - learned via plasticity
        })
```

The arbiter uses `learned_weights` (starts empty, updated by M3/M4):
```python
# Apply LEARNED multipliers - these are what actually select moves
learned_weights = node.meta.get("learned_weights", {})
selected = _softmax_sample(candidate_moves, temperature=0.3)
```

**No heuristics like "prefer promotions" or "stay near pawn" are hardcoded.**

---

## Current Status & Next Steps

### What's Working Well (January 2025)
- **Stem Cell Lifecycle**: Full EXPLORING → TRIAL → MATURE pipeline
- **XP System**: Proper +10/-10/decay mechanics with solidification
- **Evolution Snapshots**: JSON topology + stem_cells.json saved per cycle
- **Vertical Parenting**: Children wire to MATURE parents, not just backbone
- **AND-Gate Hoisting**: Correlated cells merged into min() nodes
- **POR Discovery**: Sequential patterns detected from temporal correlation
- **Curriculum**: 13-stage KPK training with position generators
- **Pack Persistence** ⭐: AND/OR gates and their children persist across cycles
- **Vertical Growth** ⭐: Packs spawn under existing pack gates for depth 3+
- **Graph Reuse** ⭐: Single graph instance preserved across structural phase steps
- **SURVIVOR BYPASS** ⭐ NEW: Auto-promote CANDIDATEs after 2+ cycles
- **Dual-Depth Spawning** ⭐ NEW: Spawn packs at L1 AND deeper simultaneously

### Latest Fixes (January 2025)

#### SURVIVOR BYPASS
**Problem**: Stage transitions killed too many cells. Easy stages (100% win) → aggressive pruning → 0 TRIALs left for hard stages.

**Solution** (in `m5_structure.py`):
```python
# Auto-promote CANDIDATEs that survived 2+ structural phases
survivor_bypass = (
    cell.state == StemCellState.CANDIDATE and
    cycles_survived >= 2 and
    sample_count >= 15
)
```

#### Dual-Depth Pack Spawning
**Problem**: 50% backbone OR 50% deep spawning meant sometimes only L1, sometimes only deep → inconsistent structure.

**Solution** (in `stem_cell.py`):
```python
# 1. ALWAYS spawn one pack at backbone (level 1 sensor)
pack_ids = spawn_and_gate_pack(parent_id=backbone_parent, ...)

# 2. ADDITIONALLY spawn one deeper pack if candidates exist
if trial_parent_candidates:
    deep_pack_ids = spawn_and_gate_pack(parent_id=deep_parent, ...)
```

**Result**: 45+ pack nodes (was 9), healthy TRIAL/CANDIDATE ratios.

### Validated Results

| Run | Stage 7 Win Rate | Total Games | Notes |
|-----|------------------|-------------|-------|
| kpk_hybrid_growth | 93.2% | 6000 | Best result |
| kpk_vertical | 92.7% | 6000 | With vertical growth |
| kpk_expansion2 | 92.0% | 6000 | Baseline |

### Active Development Areas
1. **Depth 3+ Verification**: Convergence studies on harder stages (S5-S7)
2. **Consolidation Phase**: Prune unused nodes while maintaining win rate
3. **Spawn Probability Decay**: 1/n or moving window for spawning
4. **Sparsity Audit**: Ensure children improve >10% over parent

### Known Issues
- Some KPK starting positions cause immediate stalemate (edge cases)
- TRIAL cells sometimes have 0 consistency (feature extraction gaps)
- Bridge phase generators need tuning for realistic positions

---

## ⭐ CRITICAL: Learning Mechanisms Hierarchy

Understanding how these learning mechanisms fit together is essential. Here's the complete picture:

### Conceptual Layers
```
Layer 3: WEIGHTS (learned values)
         ├── Edge weights (0.73, 0.92, etc.)
         ├── Node meta (promotion_bonus, learned_weights)
         └── Trained on specific curriculum data

Layer 2: TOPOLOGY (structure)
         ├── Which nodes exist (kpk_root, TRIAL_stem_*, AND-gates, etc.)
         ├── Which edges connect them (SUB, POR, SUR, RET)
         ├── Node types (SCRIPT, TERMINAL)
         └── Grows during M5 structural learning

Layer 1: REGISTRY (definitions/templates)
         ├── Factory functions for node types
         ├── Base topology templates (kpk_topology.json)
         └── Like "class definitions" you instantiate from
```

### Learning Mechanisms Summary

| Mechanism | File | Scope | What It Learns | Timescale |
|-----------|------|-------|----------------|-----------|
| **M3 Fast Plasticity** | `plasticity/fast.py` | Per-tick | Edge weights (POR/SUB) | Milliseconds |
| **M3 Bandit Gating** | `plasticity/bandit.py` | Per-episode | Sibling selection priors | Per-game |
| **M4 Slow Consolidation** | `plasticity/consolidate.py` | Cross-game | Base weight aggregation | Per-cycle |
| **M5 Structure Learning** | `learning/m5_structure.py` | Cross-cycle | Topology growth | Per-stage |
| **Stem Cell XP** | `nodes/stem_cell.py` | Per-episode | Node survival | Per-cycle |
| **AND-Gate Hoisting** | `m5_structure.py` | Cross-cycle | Composite structures | Per-stage |
| **POR Chain Discovery** | `m5_structure.py` | Cross-cycle | Sequential patterns | Per-stage |
| **Vertical Growth** | `stem_cell.py` | Cross-cycle | Hierarchy depth | Per-stage |

### How They Fit Together

```
┌─────────────────────────────────────────────────────────────────┐
│                     ONLINE PHASE (Per Game)                     │
├─────────────────────────────────────────────────────────────────┤
│  1. ReConEngine activates nodes via microticks                  │
│  2. M3 Fast Plasticity updates edge weights (Δw ~ eligibility)  │
│  3. Stem cells collect samples during high-reward moments       │
│  4. Trace logged to TraceDB (episode record)                    │
│                                                                 │
│  After game:                                                    │
│  5. M3 Bandit updates sibling selection priors                  │
│  6. XP awarded to TRIAL nodes based on outcome (+10/-10)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STRUCTURAL PHASE (Per Cycle)                   │
├─────────────────────────────────────────────────────────────────┤
│  7. M5 scans for affordance spikes (high reward patterns)       │
│  8. High-impact stem cells identified                           │
│  9. CANDIDATE → TRIAL promotion (if consistency ≥ 40%)          │
│ 10. XP decay for all TRIAL cells (-1 per cycle)                 │
│ 11. TRIAL → MATURE solidification (if XP ≥ 100)                 │
│ 12. TRIAL → EXPLORING demotion (if XP ≤ 0)                      │
│                                                                 │
│ GROWTH MODE (if win_rate ≥ 80%):                                │
│ 13. Pack spawning (AND/OR gates) from TRIAL cells               │
│ 14. VERTICAL GROWTH: 50% chance to spawn under existing gates   │
│                                                                 │
│ HOISTING:                                                       │
│ 15. AND-Gate hoisting (if cell pair correlates ≥ 85%)           │
│ 16. POR chain discovery (temporal precedence patterns)          │
│                                                                 │
│ 17. Save snapshot with Graph.to_snapshot()                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CONSOLIDATION (Per Stage)                      │
├─────────────────────────────────────────────────────────────────┤
│ 18. M4 aggregates edge weights from successful games            │
│ 19. Topology inherited to next stage (stem_cells.json)          │
│ 20. Mastered sensors marked for backward chaining               │
└─────────────────────────────────────────────────────────────────┘
```

### Pack Persistence (January 2025 Fix)

**Problem**: AND/OR gate packs were spawned in-memory but NOT persisted to snapshots. Packs vanished at cycle end.

**Root Cause**: Graph was rebuilt 7 times during structural phase, discarding spawned packs each time.

**Solution**:
1. `Graph.to_snapshot()` method added to export full graph state
2. Graph initialized ONCE at start of `apply_structural_phase()`
3. All 7 rebuild locations now check `if graph is None` before rebuilding
4. Structural phase returns graph in stats for snapshot saving

**Key Files Changed**:
- `src/recon_lite/graph.py` - Added `to_snapshot()` method
- `src/recon_lite/learning/m5_structure.py` - Graph reuse throughout phase
- `scripts/evolution_driver.py` - Uses structural phase graph for snapshot

### Vertical Growth (January 2025 Addition)

**Motivation**: All TRIAL nodes were flat (depth 1-2). No true hierarchical reasoning depth.

**Solution**: When spawning packs, 50% chance to use a TRIAL/SOLID/gate node as parent instead of backbone.

**Implementation** (in `stem_cell.py`):
```python
# Find TRIAL/SOLID/gate nodes as potential parents
trial_parent_candidates = []
for nid, node in graph.nodes.items():
    if nid.startswith("TRIAL_") or nid.startswith("SOLID_"):
        if node.ntype.name == "SCRIPT":
            trial_parent_candidates.append(nid)
    elif "_gate" in nid and node.ntype.name == "SCRIPT":
        trial_parent_candidates.append(nid)

# 50% chance to use TRIAL parent
if trial_parent_candidates and random.random() < 0.5:
    parent_id = random.choice(trial_parent_candidates)
    print(f"🌲 VERTICAL GROWTH: Spawning under {parent_id}")
else:
    parent_id = "kpk_execute"  # Fallback to backbone
```

**Result**: Pack gates can now appear at depth 3, 4, 5+.

---

## Article Constraints (CRITICAL)

The implementation follows the 2015 ReCoN paper strictly. These constraints MUST be preserved:

| Constraint | Description | Where Enforced |
|------------|-------------|----------------|
| TERMINAL only SUB/SUR | TERMINAL nodes can ONLY receive SUB edges and send SUR edges | `graph.py:add_edge()` |
| POR/RET require SCRIPT | POR and RET edges can only connect SCRIPT nodes | `graph.py:add_edge()` |
| Single parent for SCRIPT | SCRIPT nodes have exactly one SUB parent | `graph.py:add_edge()` |
| Fan-in for TERMINAL | TERMINAL nodes can have multiple SUB parents | `graph.py:add_edge()` |
| Confirm flows up | Confirmation propagates from TERMINAL through SUR to parent | `engine.py:step()` |
| Request flows down | Requests propagate from parent through SUB to children | `engine.py:step()` |

### Pack Template Design

**Why packs have SCRIPT wrappers**: TERMINAL nodes can't have POR edges. Each pack creates:
- Gate (SCRIPT) - Aggregates conditions
- Condition nodes (TERMINAL) - Sense environment
- Action node (TERMINAL) - Execute behavior

**Edge structure**:
```
parent (backbone SCRIPT)
  └─ SUB → gate (SCRIPT, aggregation="and")
             └─ SUB → cond_0 (TERMINAL)
             └─ SUB → cond_1 (TERMINAL)
             └─ SUB → action (TERMINAL)
```

---

## Mental Model for AI Assistant

1. **ReCoN = Request/Confirm Network**: Top-down requests (SUB/POR), bottom-up confirmations (SUR/RET)
2. **Stem Cells = Exploratory Sensors**: Collect samples, earn XP, mature into permanent nodes
3. **XP = Darwinian Selection**: Good sensors live, bad ones die
4. **Vertical Growth**: MATURE nodes spawn children, creating hierarchies
5. **Pack Persistence**: Packs are saved via Graph.to_snapshot(), not registry.get_snapshot()
6. **Graph Reuse**: Graph is initialized ONCE per structural phase to preserve spawned packs
7. **Snapshots = Evolution History**: Every cycle saves topology for replay/analysis
8. **Backbone = Stable Trunk**: kpk_root/detect/execute/finish/wait are permanent
9. **TRIAL = Probation**: Nodes exist in graph but aren't permanent until 100 XP

When suggesting changes:
- **DO NOT** rebuild graph multiple times in structural phase (breaks pack persistence)
- **DO NOT** remove `Graph.to_snapshot()` method (essential for pack saving)
- **DO NOT** change TERMINAL nodes to receive POR edges (violates article)
- Preserve trace outputs and snapshot compatibility
- Test with `--quick` flag first
- Check `snapshots/evolution/` for latest topology state
- Align with XP thresholds (50 initial, 100 solidify, 0 demote)

---

## Additional Notes

For detailed training logs and experimental results, see:
- **KRK Curriculum Training**: `notes/krk_curriculum_training.md` - Staged curriculum results (Stage 0-10)
- **KRK Draw Fixes**: `notes/logbook.md` - Draw scent sampling, M5 hybrid growth fixes
- **Goal Delegation Packs**: `nodes/pack_template.py` - POR-based hierarchical structures
- **Pack Template Design**: `notes/pack_template_design.md` - Design rationale and article constraints

Latest significant run: `snapshots/evolution/krk_curriculum/curriculum_summary.json`


