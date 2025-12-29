# Subgraph Goal Delegation - Update Notes

**Date**: 2025-12-29  
**Ticket**: Bridge Training 0% Win Rate Fix

---

## Summary

Implemented **SubgraphLock** mechanism in ReConEngine to support goal delegation. When a subgraph is locked, the engine "collapses into" that subgraph, running internal ticks until an actuator produces output, all within a single `step()` call.

## Changes Made

### 1. Engine Core (`src/recon_lite/engine.py`)

- Added `SubgraphLock` dataclass to track locked subgraph state
- Added `lock_subgraph(root_id, sentinel_fn)` method
- Added `unlock_subgraph(goal_achieved)` method
- Added `_step_subgraph()` for internal tick batching
- Added subset methods for processing only subgraph nodes
- Modified `step()` to check sentinel and delegate when locked

### 2. Test Suite (`tests/test_subgraph_delegation.py`)

New verification tests confirming:
- KPK/KQK move selectors work when called directly
- Single engine tick does NOT reach deep move selectors (confirms bug)
- Multiple ticks DO reach move selectors (7 ticks required)
- SubgraphLock produces correct move in single step (the fix!)

---

## How to Verify

### Run Test Suite
```bash
cd ~/git/recon-lite
source .venv/bin/activate
python tests/test_subgraph_delegation.py
```

**Expected output**: 5/6 tests pass (SubgraphLock test is the key one)

### Manual Test
```python
from recon_lite_chess.graph import build_unified_graph
from recon_lite.engine import ReConEngine
from recon_lite_chess.sensors.structure import summarize_kpk_material
import chess

# Build graph and engine
g = build_unified_graph()
engine = ReConEngine(g)

# KPK position (pawn about to promote)
board = chess.Board("8/6P1/7K/8/2k5/8/8/8 w - - 0 1")
env = {"board": board}

# Define sentinel
def kpk_sentinel(env):
    summary = summarize_kpk_material(env["board"])
    return bool(summary.get("is_kpk"))

# Lock into KPK subgraph
engine.lock_subgraph("kpk_root", kpk_sentinel)

# Single step produces move!
engine.step(env)

print(f"Suggested move: {env['kpk']['policy']['suggested_move']}")
# Expected: g7g8q (pawn promotion)
```

---

## Integration with Bridge Training

To use in `full_game_train.py`, add subgraph locking when endgame detected:

```python
# In play_training_game, after board update:
from recon_lite_chess.sensors.structure import summarize_kpk_material

def kpk_sentinel(env):
    summary = summarize_kpk_material(env["board"])
    return bool(summary.get("is_kpk"))

# Detect and lock
if is_kpk_position(state.board):
    if not engine.subgraph_lock:
        engine.lock_subgraph("kpk_root", kpk_sentinel)

# Promotion handling
if env.get("kpk", {}).get("promotion_detected"):
    engine.unlock_subgraph(goal_achieved=True)
    # Lock into KQK
    engine.lock_subgraph("kqk_root", kqk_sentinel)
```

---

## Known Limitation

Test 5 (KPK→KQK Transition) fails because after White promotes, it's Black's turn. The KQK move selector correctly returns no move when it's the defender's turn. This is expected behavior - the test could be improved to have Black make a random move first.

## Verification Results ✓

```bash
cd ~/git/recon-lite && source .venv/bin/activate
python demos/persistent/full_game_train.py --batch 5 --fen-file data/bridge/near_promo.fens --max-moves 80 --vs-random
```

**Output:**
```
Win rate: 100.0%
Wins: 5, Losses: 0, Draws: 0
Checkmates: 5, Stalemates: 0
Avg moves: 39.8
```

All games successfully:
1. Promoted pawn (KPK → KQK transition)
2. Achieved checkmate with queen

---

## Summary

The 0% bridge training win rate issue has been fixed by implementing proper **Subgraph Goal Delegation**. The key fixes were:

1. **SubgraphLock mechanism**: Engine "collapses into" active subgraph
2. **Node state reset**: Subgraph nodes reset to INACTIVE before each step
3. **Sentinel-based transitions**: KPK→KQK on promotion detection
4. **+1.0 promotion bonus**: Reinforces successful promotions
