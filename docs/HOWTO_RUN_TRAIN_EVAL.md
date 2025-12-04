# HOWTO: Run, Train, and Evaluate ReCoN-lite (KRK/KPK)

This is the comprehensive guide for running the chess subgraphs, training with plasticity and consolidation, logging traces, and evaluating progress.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Weight Packs (SWPs)](#weight-packs-swps)
3. [Running Demos](#running-demos)
4. [Training Workflow](#training-workflow)
5. [Slow Consolidation (M4)](#slow-consolidation-m4)
6. [Batch Evaluation](#batch-evaluation)
7. [Comparing Configurations](#comparing-configurations)
8. [Visualization](#visualization)
9. [Nightly Training Automation](#nightly-training-automation)
10. [ThinkPad / Laptop Performance Guide](#thinkpad--laptop-performance-guide)
11. [Tracing](#tracing)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Setup (one-time)
./scripts/setup_nightly.sh

# 2. Run a quick training cycle (no Stockfish, ~2 min)
./scripts/training_cycle.sh --games 10

# 3. View the consolidation dashboard
# Open demos/visualization/consolidation_dashboard.html in browser
# Load: weights/nightly/krk_consol.json
```

---

## Weight Packs (SWPs)

SWPs (Subgraph Weight Packs) store edge weights and metadata:

| File | Purpose |
|------|---------|
| `weights/macro_weight_pack.swp` | Top-level macro controller weights |
| `weights/krk_phase_weight_pack.swp` | KRK-specific phase layer weights |
| `weights/subgraphs/kpk_weight_pack.swp` | KPK pawn endgame weights |
| `weights/nightly/krk_consol.json` | Consolidated weights from training |

**Override with environment:**
```bash
RECON_PHASE_WEIGHT_FILE=weights/my_custom.swp uv run python demos/persistent/krk_persistent_demo.py
```

**Fingerprinting**: All logs include SHA256 hash of the weight pack for reproducibility.

---

## Running Demos

### Single KRK Game (Quick)
```bash
uv run python demos/persistent/krk_persistent_demo.py \
  --use-blended-actuator \
  --output-basename krk_demo
```

### With Plasticity and Visualization
```bash
uv run python demos/persistent/krk_persistent_demo.py \
  --plasticity --bandit \
  --log-full-state \
  --output-basename krk_plasticity_demo
```

### With Stockfish Evaluation
```bash
uv run python demos/persistent/krk_persistent_demo.py \
  --engine /usr/games/stockfish --depth 2 \
  --plasticity --bandit \
  --output-basename krk_stockfish_demo
```

---

## Training Workflow

ReCoN-lite uses a **two-speed learning** system:

### Fast Plasticity (per-tick)
- **What**: Edge weights adjust based on immediate reward signals
- **Controlled by**: `--plasticity` flag
- **Parameters**: `--phase-eta 0.3` (learning rate), `--phase-microticks 5`
- **Scope**: Within a single game

### Slow Consolidation (cross-game)
- **What**: Baseline weights (`w_base`) updated from aggregated episode data
- **Controlled by**: `--consolidate` flag
- **Storage**: JSON file via `--consolidate-pack`
- **Scope**: Across many games

### Recommended Training Command
```bash
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 20 \
  --plasticity \
  --bandit \
  --consolidate \
  --consolidate-pack weights/nightly/krk_consol.json \
  --engine /usr/games/stockfish --depth 2 \
  --output-basename krk_train
```

This runs 20 games with:
- Per-tick reward from Stockfish
- Edge plasticity (eligibility traces)
- Bandit exploration for action selection
- Cross-game consolidation of successful patterns

---

## Slow Consolidation (M4)

### What Gets Consolidated?

After each episode, the consolidation engine tracks:
- **Edge delta sums**: Cumulative weight changes per edge
- **Bandit statistics**: Success rates for each action arm
- **Outcome scores**: Win/draw/loss mapped to numeric rewards

### How Consolidation Works

1. **Accumulation Phase** (per episode):
   - Extract episode summary (edge deltas, bandit stats, outcome)
   - Store in pending buffer

2. **Application Phase** (after `min_episodes`):
   - Compute weighted average of deltas
   - Apply learning rate `eta_consolidate` to baseline weights
   - Clamp within `[w_min, w_max]`

### Configuration

```python
ConsolidationConfig(
    eta_consolidate=0.01,   # Learning rate for w_base updates
    min_episodes=10,        # Episodes before applying updates
    outcome_weight=0.5,     # Weight for win/loss signal vs tick rewards
    max_base_delta=0.5,     # Maximum per-edge change
    w_min=0.1,              # Floor for w_base
    w_max=3.0,              # Ceiling for w_base
)
```

### Viewing Consolidation Progress

**CLI Report:**
```bash
uv run python tools/report_consolidation.py weights/nightly/krk_consol.json -o reports/progress.md
```

**Dashboard (GUI):**
1. Open `demos/visualization/consolidation_dashboard.html` in browser
2. Click "Load State" and select your consolidation JSON
3. View histograms, top changes, and configuration

**Compare Two States:**
```bash
uv run python tools/pack_diff.py weights/baseline.json weights/trained.json
```

### Key Metrics to Watch

| Metric | Good Sign | Concern |
|--------|-----------|---------|
| `total_episodes` | Growing steadily | Flat = training not running |
| `edges_tracked` | Includes key phase edges | Missing important edges |
| Weight drift (Δ) | Small positive shifts on winning patterns | Large swings = unstable |
| Histogram spread | Narrowing over time | Extreme values (near min/max) |

---

## Batch Evaluation

### Single Block
```bash
uv run python demos/experiments/batch_eval.py \
  --mode krk --fen-file data/endgames/krk/sample.fen \
  --runs 100 --max-plies 100 --max-ticks 200 \
  --pack weights/krk_phase_weight_pack.swp \
  --engine /usr/games/stockfish --depth 2 \
  --trace-out reports/krk_trace.jsonl
```

### Blocked Evaluation (with checkpoints)
```bash
uv run python demos/experiments/block_runner.py \
  --mode krk --fen-file data/endgames/krk/sample.fen \
  --runs-per-block 50 --blocks 5 \
  --pack weights/krk_phase_weight_pack.swp \
  --engine /usr/games/stockfish --depth 2 \
  --out-dir reports/blocks
```

Produces: `summary.json`, per-block traces, pack copies, viz samples.

---

## Comparing Configurations

### Pack Tournament
Compare multiple weight configurations on the same positions:

```bash
uv run python demos/experiments/pack_tournament.py \
  --mode krk --fen-file data/endgames/krk/random.fen \
  --pack weights/baseline.swp \
  --pack weights/trained.swp \
  --engine /usr/games/stockfish --depth 2 \
  --runs 200 \
  --output reports/tournament.json
```

### Side-by-Side Visualization
Run the same position with different packs and compare visualizations:

```bash
# Run with pack A
uv run python demos/persistent/krk_persistent_demo.py \
  --fen "4k3/6K1/8/8/8/8/R7/8 w - - 0 1" \
  --consolidate-pack weights/pack_a.json \
  --log-full-state \
  --output-basename krk_pack_a

# Run with pack B
uv run python demos/persistent/krk_persistent_demo.py \
  --fen "4k3/6K1/8/8/8/8/R7/8 w - - 0 1" \
  --consolidate-pack weights/pack_b.json \
  --log-full-state \
  --output-basename krk_pack_b

# Load both viz JSONs in separate browser tabs/windows
# Or use pack_diff for weight comparison:
uv run python tools/pack_diff.py weights/pack_a.json weights/pack_b.json
```

---

## Visualization

### Available Viewers

| File | Best For |
|------|----------|
| `demos/visualization/onepage_view.html` | Full dashboard (chess + network) |
| `demos/visualization/consolidation_dashboard.html` | Training progress over time |
| `demos/visualization/standalone_html_example.html` | Quick 3D network demo |
| `demos/visualization/macrograph_view.html` | Macrograph skeleton |

### Generating Visualization Data

```bash
# Full state logging for rich visualization
uv run python demos/persistent/krk_persistent_demo.py \
  --log-full-state \
  --output-basename my_run

# Output: demos/outputs/persistent/my_run_viz.json
```

### What's Visualized

- **Node states**: Active/inactive/pending by color
- **Edge weights**: Thickness proportional to weight
- **Edge plasticity**: Delta coloring (blue=increased, red=decreased)
- **w_base modes**: Baseline / delta / combined (toggle in JS)

---

## Nightly Training Automation

### One-Time Setup
```bash
./scripts/setup_nightly.sh
```

This creates:
- `configs/nightly/` - Example configurations
- `scripts/run_nightly.sh` - Manual run script
- `scripts/training_cycle.sh` - Full training loop
- Required directories

### Manual Runs
```bash
# Quick validation (~5 min)
./scripts/run_nightly.sh krk_fast

# Full training with Stockfish (~30-60 min)
./scripts/run_nightly.sh krk_stockfish
```

### Training Cycle (Recommended)
```bash
# Run 20 games, consolidate, generate report
./scripts/training_cycle.sh --games 20

# With Stockfish for better reward signal
./scripts/training_cycle.sh --games 20 --engine /usr/games/stockfish
```

### Automated Cron Job
```bash
# Edit crontab
crontab -e

# Add line (runs at 2am daily):
0 2 * * * cd /home/paulander/git/recon-lite && uv run python demos/experiments/nightly_runner.py --config configs/nightly/krk_stockfish.json >> logs/nightly.log 2>&1
```

---

## ThinkPad / Laptop Performance Guide

ReCoN-lite is designed to run on CPU-only machines like a ThinkPad.

### Expected Performance

| Configuration | Time per KRK Game | Notes |
|--------------|-------------------|-------|
| Heuristic eval only | 2-5 seconds | Fastest, good for iteration |
| Stockfish depth 2 | 10-20 seconds | Better reward signal |
| Stockfish depth 4 | 30-60 seconds | High quality, slower |

### Recommended Settings for Laptop

```bash
# Quick iteration (heuristic eval)
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 20 --plasticity --consolidate \
  --consolidate-pack weights/nightly/krk_consol.json

# Overnight training (Stockfish)
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 100 --plasticity --consolidate \
  --consolidate-pack weights/nightly/krk_consol.json \
  --engine /usr/games/stockfish --depth 2
```

### Power Management Tips

1. **Plug in** for long training runs
2. **Lower Stockfish depth** (depth 2 vs 4) for faster iteration
3. **Batch size**: 20-50 games is reasonable for interactive training
4. **Background runs**: Use `nohup` or `screen` for overnight runs

```bash
# Background training with logging
nohup ./scripts/training_cycle.sh --games 100 --engine /usr/games/stockfish > logs/overnight.log 2>&1 &
```

### Resource Usage

- **CPU**: Single-core primarily (Stockfish is the bottleneck)
- **RAM**: <500MB for typical runs
- **Disk**: ~1-10MB per 100 episodes (traces + consolidation state)

---

## Tracing

### Schema

Trace records are defined in `src/recon_lite/trace_db.py`:

- **TickRecord**: tick_id, phase_estimate, goal_vector, board_fen, active_nodes, fired_edges, action, eval_before/after, reward_tick, meta
- **EpisodeRecord**: episode_id, result, ticks, pack_meta (path + sha256), notes
- **EpisodeSummary** (M4): edge_delta_sums, bandit_stats, avg_reward_tick, phase_usage, outcome_score

### Analyzing Traces

```bash
# Aggregate episode summaries
uv run python tools/trace_summarize.py reports/krk_trace.jsonl -o reports/summary.md

# Refresh bandit priors from summaries
uv run python tools/bandit_refresh.py reports/krk_trace.jsonl -o weights/bandit_priors.json
```

---

## Troubleshooting

### "No FEN files found"
```bash
./scripts/setup_nightly.sh  # Creates sample FEN files
```

### "Stockfish not found"
```bash
sudo apt install stockfish  # Ubuntu/Debian
# Or download from https://stockfishchess.org/download/
```

### Training seems stuck
- Check `reports/nightly/` for output files
- View consolidation dashboard for progress
- Run with `--batch 1` to debug single game

### Weights not changing
- Ensure `--consolidate` flag is set
- Check `consolidation_meta.total_episodes` in JSON
- Verify `min_episodes` threshold is met (default: 10)

### Visualization not loading
- Check browser console for errors
- Verify JSON file exists and is valid
- Try a different viz HTML (standalone_html_example.html for testing)
