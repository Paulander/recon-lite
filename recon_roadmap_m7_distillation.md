# M7 Roadmap: Distillation & Evaluation Upgrade

**Status**: ✅ Implemented (December 2025)

## Goal

Improve evaluation quality by training a lightweight neural network to mimic Stockfish, enabling fast and accurate position assessment without runtime engine dependency.

---

## Phase 1: Feature Extraction (M7.1)

### Deliverables
- [x] Feature extraction for ~77 position features
- [x] Material counts and balance
- [x] Piece positions (compressed)
- [x] Pawn structure (isolated, doubled, passed)
- [x] King safety (shield, attackers)
- [x] Mobility features
- [x] Phase indicators
- [x] Tactical features

### Files Created
- `src/recon_lite_chess/eval/features.py`

---

## Phase 2: Data Collection (M7.2)

### Deliverables
- [x] Stockfish annotation pipeline
- [x] Support for traces, PGN, FEN inputs
- [x] Random position generation
- [x] Phase-balanced sampling option

### Files Created
- `tools/collect_stockfish_evals.py`

---

## Phase 3: Model Training (M7.3)

### Deliverables
- [x] PyTorch backend (preferred)
- [x] sklearn fallback (MLPRegressor)
- [x] Early stopping
- [x] Validation metrics (MSE, correlation)

### Files Created
- `tools/train_distilled_eval.py`

---

## Phase 4: Integration (M7.4)

### Deliverables
- [x] `DistilledEvaluator` class with load/save
- [x] `EvalMode.DISTILLED` in EvalManager
- [x] `EvalMode.DISTILLED_HYBRID` (distilled + tactical bonuses)
- [x] Fallback to heuristic if model unavailable

### Files Modified
- `src/recon_lite_chess/eval/distill.py`
- `src/recon_lite_chess/eval/manager.py`
- `src/recon_lite_chess/eval/__init__.py`

---

## Testing

All tests passing:
- `tests/test_distillation.py` (17 tests)

---

## Usage

### Step 1: Collect Training Data

```bash
# From random positions
uv run python tools/collect_stockfish_evals.py \
  --random 10000 --depth 15 --out data/distillation/evals.jsonl

# From game traces
uv run python tools/collect_stockfish_evals.py \
  --traces reports/*.jsonl --out data/distillation/evals.jsonl --max-positions 10000

# Balance across phases
uv run python tools/collect_stockfish_evals.py \
  --random 15000 --balance-phases --out data/distillation/balanced_evals.jsonl
```

### Step 2: Train Model

```bash
# PyTorch (recommended)
uv run python tools/train_distilled_eval.py \
  --data data/distillation/evals.jsonl \
  --out weights/distilled_eval.pt \
  --epochs 100 --lr 0.001 --hidden 256,128

# sklearn fallback
uv run python tools/train_distilled_eval.py \
  --data data/distillation/evals.jsonl \
  --out weights/distilled_eval.joblib \
  --backend sklearn
```

### Step 3: Use in Evaluation

```python
from recon_lite_chess.eval import EvalMode, EvalConfig, EvalManager

# Pure distilled
config = EvalConfig(
    mode=EvalMode.DISTILLED,
    distilled_model_path="weights/distilled_eval.pt"
)
manager = EvalManager(config)
result = manager.evaluate(board)
print(f"Score: {result.score} (source: {result.source.name})")

# Hybrid: distilled + tactical bonuses
config = EvalConfig(
    mode=EvalMode.DISTILLED_HYBRID,
    distilled_model_path="weights/distilled_eval.pt"
)
```

---

## Acceptance Criteria

- [x] Feature extraction produces consistent 77-feature vectors
- [x] Data collection tool supports traces, PGN, FENs, random positions
- [x] Training script supports PyTorch and sklearn backends
- [x] EvalManager supports DISTILLED and DISTILLED_HYBRID modes
- [x] Graceful fallback to heuristic when model unavailable
- [ ] Collect 10,000+ positions (user task, requires Stockfish)
- [ ] Model achieves >0.85 correlation (depends on training data quality)

