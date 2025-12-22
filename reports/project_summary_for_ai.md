# ReCoN-lite Chess — AI Orientation Brief

## Purpose and Philosophy
- Implements Joscha Bach / Priska Herger's Request–Confirmation Network (ReCoN) ideas for chess: top-down requests via `sub/por`, bottom-up confirmations via `sur/ret`, continuous activations, bindings, and explicit traceability.
- Goal: an explainable ReCoN-based chess agent (~1900 Elo target) that orchestrates hand-written heuristics, search engines, and learned components rather than a single end-to-end model.
- Paper reference: "Request confirmation networks for neuro-symbolic script execution" (Bach & Herger). Use this as the canonical conceptual anchor.

## History & Milestones (M1→M8)
- **M1** (KRK, continuous layer): micro-ticks and soft activations (`core/activations.py`, `time/microtick.py`), binding tables (`binding/manager.py`), KRK strategic layer + viz (`recon_lite_chess/strategy.py`).
- **M2/M2.5** (macrograph + instrumentation): macrograph spec and Subgraph Weight Packs (SWPs) for KRK/KPK/rook techniques; TraceDB (`trace_db.py`, `macro_trace.py`); Stockfish teacher and batch/block eval pipelines.
- **M3** (fast plasticity + bandits): within-game weight updates on selected POR/SUB edges, bandit gating for sibling scripts, goal-aware modulation (`plasticity/fast.py`, `bandit.py`, `modulation.py`).
- **M4** (slow consolidation + eval upgrade): cross-game consolidation of `w_base` from episode summaries (`plasticity/consolidate.py`), bandit priors refresh, richer heuristic/Stockfish/hybrid eval manager.
- **M5** (structure learning): motif extraction, clustering, script proposals, trust scoring, tactical + rook subgraphs (`motifs/`, `trust/`, `scripts/tactics.py`, `scripts/rook_endgame.py`).
- **M6** (full-game, multi-scale dynamics): fan-in terminals, goal hierarchy (Ultimate→Strategic→Tactical→Sensor), plan persistence, opening/middlegame scripts.
- **M7** (distillation): Feature extraction for ML, Stockfish data collection, distilled eval model training (stub implementation).
- **M8** (reverse curriculum + FeatureHub): Continuous affordance signals, global feature hoisting, CurriculumManager for 4-phase reverse training, KQK endgame network. **Current focus**.

Source docs: `updates_continuous.md`, `recon_roadmap_m3_fast_plasticity.md`, `recon_roadmap_m5_structure_learning.md`, `recon_roadmap_m6_full_game_architecture.md`.

## Current Architecture Snapshot

### Core ReCoN Engine (`src/recon_lite/`)
- Graph/state machine with `sub/sur/por/ret` link types
- Continuous activations and micro-ticks
- Bindings and namespaces
- TraceDB logging (JSONL format)
- Fan-in terminals (multiple parents per sensor)
- **Plasticity**: M3 fast (within-game) + M4 slow (cross-game) consolidation
- **Bandits**: UCB/softmax gating among sibling scripts

### Chess Layer (`src/recon_lite_chess/`)

**Endgame Networks** (`scripts/`):
| Network | File | Goal | Current Win Rate |
|---------|------|------|-----------------|
| KRK | `krk_nodes.py`, `krk_strategy.py` | Checkmate | 90% |
| KPK | `scripts/kpk.py` | Pawn promotion | 80% |
| KQK | `scripts/kqk.py` | Checkmate | 70%+ |
| Rook Endings | `scripts/rook_endgame.py` | Lucena/Philidor | Partial |

**Affordance Sensors** (`affordance/sensors.py`):
- Continuous [0.0, 1.0] signals for endgame proximity
- `compute_krk_affordance()`, `compute_kpk_affordance()`, `compute_kqk_affordance()`
- Used by M3 bandit for "scent" gradients toward winning positions

**FeatureHub** (`features/hub.py`):
Global registry of 30+ hoisted features across 6 categories:
- **TACTICAL**: `fork_available`, `pin_present`, `hanging_piece`, `skewer`, `back_rank_vulnerable`, `discovered_attack`, `double_check`
- **GEOMETRIC**: `opposition_status`
- **MATERIAL**: `material_advantage`
- **POSITIONAL**: `king_safety`, `center_control`, `pawn_structure`, `color_complex_weakness`
- **DYNAMIC**: `mobility`
- **PHASE**: `phase_opening`, `phase_endgame`, `affordance_krk`, `affordance_kpk`, `affordance_kqk`
- Plus 20+ additional sensors from `sensors_v2.py` for sensor flooding

**Evaluation** (`eval/`):
- `heuristic.py`: Material, king safety, mobility, pawn structure, piece activity
- `manager.py`: EvalManager with modes (HEURISTIC, STOCKFISH, HYBRID, DISTILLED)
- `features.py`: 77-feature extraction for ML training

**Goals & Strategy** (`goals/`, `sensors/`):
- Ultimate goals: WIN/DRAW/SURVIVE
- Strategic plans: AttackKing, Simplify, Develop
- Phase detection: Soft weights (opening/middlegame/endgame)

### Training Infrastructure (`training/`, `scripts/`)

**CurriculumManager** (`training/curriculum.py`):
4-phase reverse curriculum strategy:
1. **Anchor**: Perfect endgame conversion (KRK, KPK, KQK) → 99% win rate
2. **Bridge**: Simplified middlegame → discover liquidation strategies
3. **Wilderness**: Full material tactical positions
4. **Integration**: Full games from starting position

**Position Generators** (`training/generators.py`):
- `generate_krk_position()`, `generate_kpk_position()`, `generate_kqk_position()`
- `generate_bridge_position()`, `generate_wilderness_position()`

**Training Scripts**:
```bash
# Quick curriculum training (50 games per endgame)
./scripts/curriculum_training.sh --quick --phase anchor

# Full curriculum (500 games, all phases)
./scripts/curriculum_training.sh

# Analysis with markdown report
uv run python scripts/analyze_training.py --report-dir reports/curriculum/latest/ --markdown
```

## Learning & Training Mechanics

### Subgraph Weight Packs (SWPs)
Serialized weight/threshold packs loaded at build time; used as baselines:
- `weights/macro_weight_pack.swp` - Macro-level weights
- `weights/subgraphs/*.swp` - Per-subgraph weights
- `weights/nightly/*_consol.json` - Consolidated weights from training

### Fast Plasticity (M3)
- Per-episode Δw on whitelisted POR/SUB edges
- Reward: `reward_tick = eval_after − eval_before` (clipped centipawns)
- Eligibility traces and goal-aware scaling
- Reset each episode

### Bandit Control (M3)
- UCB/softmax gating among sibling scripts
- **Affordance delta** now incorporated in reward computation
- Per-episode stats with optional priors for warm starts

### Slow Consolidation (M4)
- Aggregates episode summaries into updated `w_base`
- Bounds and versioning for safety
- Exportable as packs

### Structure Learning (M5)
- Motif extraction → clustering → script proposals
- Trust scoring to freeze/promote/prune edges
- Human review via `tools/review_proposal.py`

## Key Files Reference

### Endgame Networks
- `src/recon_lite_chess/scripts/kpk.py` - KPK with promotion detection
- `src/recon_lite_chess/scripts/kqk.py` - KQK with stalemate protection
- `src/recon_lite_chess/krk_nodes.py`, `krk_strategy.py` - KRK network

### Affordance & Features
- `src/recon_lite_chess/affordance/sensors.py` - Continuous affordance signals
- `src/recon_lite_chess/features/hub.py` - Global FeatureHub (18+ features)
- `src/recon_lite_chess/features/integration.py` - Wire features to graph

### Training
- `src/recon_lite_chess/training/curriculum.py` - CurriculumManager
- `src/recon_lite_chess/training/generators.py` - Position generators
- `scripts/curriculum_training.sh` - End-to-end training script
- `scripts/analyze_training.py` - Statistics and markdown reports

### Demos
- `demos/persistent/krk_persistent_demo.py` - KRK with plasticity/bandit
- `demos/persistent/kpk_persistent_demo.py` - KPK with promotion
- `demos/persistent/kqk_persistent_demo.py` - KQK with stalemate protection
- `demos/persistent/full_game_demo.py` - Full game from start
- `demos/persistent/full_game_train.py` - Full game training with plasticity, consolidation, and stem cells

### Plasticity
- `src/recon_lite/plasticity/fast.py` - M3 fast plasticity
- `src/recon_lite/plasticity/bandit.py` - Bandit control + affordance delta
- `src/recon_lite/plasticity/consolidate.py` - M4 slow consolidation

### Tools
- `tools/trace_summarize.py` - Aggregate trace metrics
- `tools/consolidate_batch.py` - Offline batch consolidation
- `tools/bandit_refresh.py` - Update bandit priors
- `demos/experiments/extract_motifs.py` - M5 motif extraction

## Typical Commands

```bash
# KRK training with plasticity and bandit
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 50 --plasticity --bandit --consolidate \
  --trace-out reports/krk_trace.jsonl

# KPK training (promotion as success)
uv run python demos/persistent/kpk_persistent_demo.py \
  --batch 50 --plasticity --consolidate \
  --trace-out reports/kpk_trace.jsonl

# KQK training (checkmate with stalemate protection)
uv run python demos/persistent/kqk_persistent_demo.py \
  --batch 50 --plasticity --consolidate \
  --trace-out reports/kqk_trace.jsonl

# Full curriculum training
./scripts/curriculum_training.sh --quick --phase anchor

# Generate training report
uv run python scripts/analyze_training.py \
  --report-dir reports/curriculum/latest/ --markdown
```

## Current Status & Next Steps

### What's Working Well
- **KRK**: 90% win rate, mature network
- **KPK**: 80% win rate with promotion detection
- **KQK**: 70%+ with stalemate protection (recently fixed)
- **FeatureHub**: 18+ hoisted features for global visibility
- **Affordance signals**: Continuous scent for bridge discovery
- **Training infrastructure**: JSONL traces, markdown reports, curriculum phases

### Ready for Training
The system is ready for "Reverse Curriculum" training:
1. Train Anchor phase (KRK/KPK/KQK) to 95%+ win rates
2. Let Bridge phase discover liquidation strategies via affordance crossings
3. Apply learned patterns to full-game play

### Candidate Next Steps
1. **More endgames**: KBB (two bishops), KBN (bishop+knight), KRR, KRRK
2. **Bridge discovery**: Run M5 motif extraction on affordance crossings
3. **Tactical coverage**: Expand FeatureHub with discovered pins/forks patterns
4. **Stem cells**: Pattern induction pipeline (`src/recon_lite/nodes/stem_cell.py`, `src/recon_lite/motifs/induction.py`) - exists but needs wiring

### Known Issues
- Some KRK starting positions cause immediate stalemate (edge cases)
- KQK win rate can be improved with better king approach logic
- Bridge phase generators need tuning for realistic positions

## Mental Model for AI Assistant

- **ReCoN as orchestrator**: Training adjusts edge preferences (fast/slow plasticity) and scripts via SWPs; topology changes go through proposal/trust pipeline
- **Milestone alignment**: M3=within-game, M4=cross-game, M5=structure learning, M6=full-game, M7=distillation, M8=curriculum
- **Affordance signals**: The key innovation for bridge discovery - continuous scent toward winning positions
- **FeatureHub**: Hoisted features enable M5 to detect patterns in novel contexts
- **Reverse curriculum**: Train backwards from anchors to discover bridges

When suggesting changes, preserve trace outputs and align with milestone scope. Encourage experiments comparing modes (plasticity on/off, different packs, affordance thresholds).
