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

## Typical Commands

```bash
# Quick M5 evolution test (10 games, 2 cycles)
uv run python scripts/evolution_driver.py --quick

# Full evolution run (100 games/cycle, 10 cycles)
uv run python scripts/evolution_driver.py \
  --topology topologies/kpk_topology.json \
  --games-per-cycle 100 \
  --cycles 10 \
  --output-dir reports/evolution/

# Multi-stage evolution (runs all 13 curriculum stages)
uv run python scripts/evolution_driver.py --all-stages \
  --games-per-cycle 50 --win-threshold 0.9

# Single stage evolution
uv run python scripts/evolution_driver.py \
  --stage 5 --cycles 20 --run-name my_experiment
```

---

## Current Status & Next Steps

### What's Working Well (Jan 2025)
- **Stem Cell Lifecycle**: Full EXPLORING → TRIAL → MATURE pipeline
- **XP System**: Proper +10/-10/decay mechanics with solidification
- **Evolution Snapshots**: JSON topology + stem_cells.json saved per cycle
- **Vertical Parenting**: Children wire to MATURE parents, not just backbone
- **AND-Gate Hoisting**: Correlated cells merged into min() nodes
- **POR Discovery**: Sequential patterns detected from temporal correlation
- **Curriculum**: 13-stage KPK training with position generators

### Active Development Areas
1. **Sparsity Audit**: Ensure children improve >10% over parent
2. **Survival Bond**: Accelerated decay for orphaned children
3. **Link-XP Pruning**: Neural Darwinism for weak edges (25-game fast kill)
4. **Pattern Signature Clustering**: Spatial dedup (95% similarity = merge)

### Known Issues
- Some KPK starting positions cause immediate stalemate (edge cases)
- TRIAL cells sometimes have 0 consistency (feature extraction gaps)
- Bridge phase generators need tuning for realistic positions

---

## M5.1.1: Hybrid Growth-During-Failure

Addresses the "no wins → no XP → no growth" deadlock at 0% win rate stages.

### Engagement XP (Gradual Growth)
Nodes accumulate XP from participation, not just wins:
```
+0.5 XP per activation (node participates)
+0.2 XP per consistent pattern match
+0.1 XP per depth level > 2 (hierarchical reasoning)
Cap: +2 XP max per game
```

### Failure-Driven Spawning
When `win_rate < 10%` for 50+ games:
- Spawn from TRIAL nodes (not just MATURE)
- Children start at 30 XP (not 50)
- Children have 1.5x decay rate
- Requires `activation_count >= 30`

### Key Methods
| Method | File | Purpose |
|--------|------|---------|
| `accumulate_engagement_xp()` | `stem_cell.py` | Award XP for participation |
| `spawn_exploration_children()` | `stem_cell.py` | Spawn during failure |
| Step 8b in `apply_structural_phase()` | `m5_structure.py` | Trigger exploration spawning |

---

## M5.1: Structural Hyper-Sweeping

M5.1 extends M5 with systematic hyperparameter exploration and stall recovery mechanisms.

### HyperSweep Engine (`learning/sweep_engine.py`)

Orchestrates multiple isolated training runs to find optimal configurations:

```python
from recon_lite.learning.sweep_engine import HyperSweepEngine, SweepConfig

engine = HyperSweepEngine()
configs = [
    SweepConfig(trial_name="conservative", consistency_threshold=0.50, hoist_threshold=0.90),
    SweepConfig(trial_name="speculative", consistency_threshold=0.30, hoist_threshold=0.75),
    SweepConfig(trial_name="recursive", enable_success_bypass=True, enable_speculative_hoisting=True),
]
results = engine.run_sweep(configs)
```

### M5.1 Unblock Mechanisms

#### 1. Success-Based Promotion ("Bypass")
If `reward_average > 0.90` over 50+ samples, force-promote even if consistency math is undefined (Zero-Variance Trap escape):
```
Cell with avg_reward=0.95, samples=60 → Promoted despite consistency=NaN
```

#### 2. Speculative Hoisting on CANDIDATEs
Don't wait for TRIAL - hoist CANDIDATE cells at 85%+ win-coactivation:
```
CANDIDATE(A) + CANDIDATE(B) co-activate 90% → Hoisted to AND-gate immediately
```

#### 3. Stall Recovery
If `win_rate < 10%` for 3 consecutive cycles:
- Double `spawn_rate` for more exploration
- Enable **Scent-Based Shaping**: +0.1 reward for draws showing King approaching Pawn's promotion path
- Increase `plasticity_eta` by 50%

#### 4. Micro-Temporal POR Discovery
For short games (<10 ticks), analyze tick-by-tick sequences to find strong King → Pawn dependencies:
```
Game 1 (7 ticks): King fires tick 2, Pawn fires tick 5 → King precedes Pawn
Game 2 (6 ticks): King fires tick 1, Pawn fires tick 4 → Consistent!
→ Create POR link: goal_king → goal_pawn
```

### Healthy Growth Signature

Monitor these metrics to distinguish "flat memorization" from "hierarchical reasoning":

| Metric | Flat Behavior | Healthy Target | Meaning |
|--------|---------------|----------------|---------|
| Max Depth | 2 | >= 4 | Sensor → Sub-goal → Leg → Backbone |
| Branching Factor | 1.0 | >= 1.5 | Non-backbone SCRIPTs have multiple children |
| POR Edges | 0 | > 0 | Sequential patterns discovered |
| Edge Types | 99% SUB | 70% SUB, 30% POR | Sequences encoded |
| Vertical Promotions | 0% | >= 50% | New nodes parent to SOLID, not backbone |
| Speculative ANDs | 0 | Growing | Tactical patterns forming before solidification |

### Sweep Analysis

Use `scripts/analyze_sweep.py` to generate comparison reports:
```bash
python scripts/analyze_sweep.py snapshots/sweeps/stage1_validation/
```

Generates markdown with:
- Summary comparison table
- Configuration differences
- Win rate progression
- Learning speed analysis
- Structural maturity assessment
- Recommendations

### Environment Variable Overrides

Sweep configurations can be injected via environment:
```bash
M5_CONSISTENCY_THRESHOLD=0.30 \
M5_HOIST_THRESHOLD=0.75 \
M5_ENABLE_SUCCESS_BYPASS=1 \
M5_ENABLE_SCENT_SHAPING=1 \
python scripts/evolution_driver.py --stage 1
```

---

## M5.2: Knowledge Bank & Cross-Endgame Transfer

The Knowledge Bank enables transfer learning between endgames (e.g., KPK → KRK).

### Universal Sensors

Sensors that work across multiple endgames are identified and preserved:

| Feature | KPK Source | KRK Application | Transfer Score |
|---------|------------|-----------------|----------------|
| `king_distance` | Distance between kings | Same - kings always interact | High |
| `opposition_status` | Direct/distant opposition | Same mechanic applies | High |
| `enemy_edge_distance` | King proximity to board edge | Drive enemy to edge | High |
| `cut_established` | KPK: pawn path clear | KRK: rook fence established | Medium |

### Knowledge Transfer API

```python
from recon_lite.nodes.stem_cell import StemCellManager

# Load KPK stem cells and rename for KRK
stem_manager = StemCellManager.load_with_transfer(
    source_path="snapshots/kpk_run/stem_cells.json",
    prefix_map={
        "kpk_sensor_": "universal_sensor_",
        "stem_": "universal_stem_",
    },
    states_to_transfer=["TRIAL", "MATURE"],
    top_n=20,  # Transfer top 20 by XP
    new_domain="krk",
)

# Track transfer success
reuse_stats = stem_manager.get_reuse_stats()
print(f"Transferred: {reuse_stats['transferred_count']}")
print(f"Avg reuse ratio: {reuse_stats['avg_reuse_ratio']:.1%}")
```

### Bridge Metric: `sensor_reuse_ratio`

Tracks how many "KPK-born" sensors contribute to KRK wins:

| Ratio | Interpretation | Action |
|-------|----------------|--------|
| < 0.3 | Transfer failed | More KRK-specific exploration needed |
| 0.3-0.5 | Partial transfer | Some universal sensors working |
| > 0.5 | **Successful transfer** | Increase plasticity, lock in universals |

When `sensor_reuse_ratio > 0.5`, the system automatically:
- Increases `plasticity_eta` by 20% to reward general strategies
- Flags transferred sensors for potential solidification

### KRK Box Method Discovery

The "Box Method" is a specific checkmating technique in KRK:

```
Rook_Cuts_Rank → King_Approaches → Rook_Shrinks_Box
```

This sequence **cannot be vibed** - doing step 3 before step 2 lets the enemy escape.

When detected in >70% of wins, the system creates a `Tactical_Box_Manager`:
- POR-linked SCRIPT nodes for each step
- Enforces proper sequencing via soft-POR gates

### KRK Evolution Driver

```bash
# Fresh KRK training
python scripts/krk_evolution_driver.py \
    --topology topologies/krk_legs_topology.json \
    --games-per-cycle 100 \
    --cycles 30

# With knowledge transfer from KPK
python scripts/krk_evolution_driver.py \
    --transfer-from snapshots/kpk_run/stem_cells.json \
    --transfer-top-n 20 \
    --topology topologies/krk_legs_topology.json
```

### Expected Transfer Outcomes

| Metric | Cold Start | Successful Transfer |
|--------|------------|---------------------|
| Initial Win Rate | <20% | >30% |
| Sensor Reuse Ratio | 0% | >50% |
| First Hybrid AND-gate | Never | Cycles 3-5 |
| Box Method POR | Never | Cycles 10-15 |

### File Reference for Knowledge Transfer

| File | Purpose |
|------|---------|
| `scripts/export_krk_topology.py` | Export KRK network to JSON |
| `scripts/krk_evolution_driver.py` | KRK training with transfer |
| `src/recon_lite_chess/features/krk_features.py` | KRK feature extraction |
| `src/recon_lite/nodes/stem_cell.py` | `load_with_transfer()` method |
| `src/recon_lite/learning/m5_structure.py` | `discover_krk_box_method_por()` |
| `topologies/krk_legs_topology.json` | KRK network definition |

---

## Mental Model for AI Assistant

1. **ReCoN = Request/Confirm Network**: Top-down requests (SUB/POR), bottom-up confirmations (SUR/RET)
2. **Stem Cells = Exploratory Sensors**: Collect samples, earn XP, mature into permanent nodes
3. **XP = Darwinian Selection**: Good sensors live, bad ones die
4. **Vertical Growth**: MATURE nodes spawn children, creating hierarchies
5. **Snapshots = Evolution History**: Every cycle saves topology for replay/analysis
6. **Backbone = Stable Trunk**: kpk_root/detect/execute/finish/wait are permanent
7. **TRIAL = Probation**: Nodes exist in graph but aren't permanent until 100 XP

When suggesting changes:
- Preserve trace outputs and snapshot compatibility
- Test with `--quick` flag first
- Check `snapshots/evolution/` for latest topology state
- Align with XP thresholds (50 initial, 100 solidify, 0 demote)
