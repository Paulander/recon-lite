# ReCoN Demos - Learning Progression

This directory contains demos that showcase ReCoN's capabilities, from basic concepts to full game play with learning.

## 📁 Directory Structure

```
demos/
├── evaluation/       # Static position evaluation demos
├── experiments/      # Training and batch evaluation tools
├── gameplay/         # Full game playing demos
├── persistent/       # Stateful demos with learning (M3-M7)
├── shared/           # Shared network builders
├── testing/          # Test utilities
└── visualization/    # HTML/JS visualization suite
```

## 🎯 Demo Types by Milestone

### M1-M2: Core ReCoN & KRK Endgame

| Demo | Purpose |
|------|---------|
| `evaluation/sequence_demo.py` | Basic ReCoN execution validation |
| `evaluation/krk_checkmate_demo.py` | Static KRK position analysis |
| `gameplay/krk_play_demo.py` | Interactive KRK game playing |

### M3-M4: Plasticity & Consolidation

| Demo | Purpose |
|------|---------|
| `persistent/krk_persistent_demo.py` | KRK with fast plasticity & slow consolidation |
| `persistent/kpk_persistent_demo.py` | KPK with batch training support (M8) |
| `experiments/batch_eval.py` | Batch evaluation with timeout handling (M8) |
| `experiments/pack_tournament.py` | Compare weight pack performance |

### M5: Structure Discovery

| Demo | Purpose |
|------|---------|
| `experiments/extract_motifs.py` | Extract patterns from game traces |
| `experiments/cluster_motifs.py` | Cluster similar patterns |
| `experiments/propose_scripts.py` | Generate candidate script proposals |
| `experiments/benchmark_eval.py` | Evaluate against benchmark suites |

### M6-M7: Full Game Architecture

| Demo | Purpose |
|------|---------|
| `persistent/full_game_demo.py` | Complete games with goal hierarchy |
| `gameplay/full_game_macro.py` | Macrograph-driven full game |

## 🚀 Quick Start

### Play a Full Game (M6)
```bash
# Play against random opponent
uv run python demos/persistent/full_game_demo.py --vs-random --max-moves 100

# With trained weights
uv run python demos/persistent/full_game_demo.py \
  --vs-random --weights weights/nightly/krk_consol.json

# Output visualization data
uv run python demos/persistent/full_game_demo.py \
  --vs-random --output game.json --viz
```

### Train KRK with Consolidation (M4)
```bash
# Batch training with consolidation
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 50 --plasticity --consolidate \
  --consolidate-pack weights/nightly/krk_consol.json \
  --engine /usr/games/stockfish --depth 2

# Evaluate results
uv run python demos/experiments/batch_eval.py \
  --mode krk --fen-file data/endgames/krk/random.fen \
  --runs 100 --episode-timeout 60 -v
```

### Run Overnight Training (M8)
```bash
# Quick test mode (~30 min)
./scripts/overnight_training.sh --quick

# Full overnight (~6+ hours)
./scripts/overnight_training.sh
```

## 📊 Visualization

| View | File | Purpose |
|------|------|---------|
| Full Game | `visualization/full_game_view.html` | M6 goal hierarchy visualization |
| Consolidation | `visualization/consolidation_dashboard.html` | Training progress monitoring |
| Network | `visualization/chessboard_view.html` | Detailed network + board view |
| Quick View | `visualization/onepage_view.html` | Simple frame-by-frame viewer |

## 📁 Output Locations

| Type | Location |
|------|----------|
| Visualization JSON | `demos/outputs/persistent/` |
| Trace JSONL | `reports/` |
| Consolidation State | `weights/nightly/` |
| Training Reports | `reports/overnight/` |

## 🔧 Key Arguments (Common Across Demos)

| Argument | Purpose |
|----------|---------|
| `--batch N` | Run N games in batch mode |
| `--plasticity` | Enable fast weight updates |
| `--consolidate` | Enable slow consolidation |
| `--consolidate-pack PATH` | Load/save consolidation state |
| `--engine PATH` | Use Stockfish for evaluation |
| `--depth N` | Stockfish search depth |
| `--trace-out PATH` | Output JSONL trace |
| `--output PATH` | Output visualization JSON |

## 📈 Current Capabilities (After M8)

- ✅ Full game play from opening to checkmate
- ✅ Goal hierarchy (Ultimate → Strategic → Tactical)
- ✅ Fast plasticity (in-game weight updates)
- ✅ Slow consolidation (cross-game learning)
- ✅ Batch training with timeout handling
- ✅ KRK and KPK endgame training
- ✅ Multiple visualization options
- ✅ Automated overnight training
- ✅ Standardized report generation

## 🔜 Coming in M9-M14

- Pattern recognition ("I've seen this before")
- Stem cells (emergent sensor discovery)
- Expert sub-ReCoNs (hierarchical experts)
- Feature binding & instances
- Hypothesis testing & creative exploration
