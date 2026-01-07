# ReCoN-lite Development Logbook

## 2026-01-07: M5 Evolution Improvements

**Extended Run Analysis** (75K games):
- Zero pack spawning despite 30K games at 0% win rate
- XP decay to avg 5.8 (should be 50+)
- Depth stuck at 3

**Fixes Implemented**:
1. **Pack spawning threshold**: 50 → 20 games, 10% → 20% win rate
2. **XP floor**: Min 10 XP to prevent death spiral in `decay_xp()`  
3. **Dynamic lottery**: 60/30/10 in stalls (<10%), 30/50/20 in success (>30%)
4. **Plateau detection**: Abort after 2000 games if <5% delta over 10 cycles

**Verified Working**:
```
[M5] Failure check: win=8.0%, games=50, trials=12
[M5] ⚡ FAILURE MODE TRIGGERED: Spawning packs/singles from 12 TRIALs
```

---

## 2026-01-07: KRK Evolution Fails to Form Chains

**Run:** `krk_hybrid_chain_test` - 30 cycles, 3000 games
**Win Rate:** 1.7% avg
**Result:** Flat topology - 0 TRIAL promotions, 0 spikes, 50 stuck CANDIDATEs

**Analysis:**
The hybrid growth features work fine on KPK (93% win rate), but KRK is far more difficult (only 1.7% wins). 
The structure learner found zero reward-correlated patterns because:
1. `min_spike_reward=0.2` is too high when KRK tick rewards are 0.1-0.3
2. Consistency threshold (0.40) too strict for chaotic 98% draw patterns  
3. Failure-driven spawning requires TRIAL cells, but no cells reached TRIAL tier

**Fixes Applied:**
- Lower `min_spike_reward` from 0.2 → 0.05
- Add extreme_failure_bypass in trial promotion (win_rate < 5%, 1000+ games)
- Add Step 2b: Direct CANDIDATE selection when high_impact is empty (bypasses spike deadlock)
- M5_MIN_SAMPLES env var (default: 15, lowered from 30) for configurable threshold
- **NEW: Draw Scent Sampling** - `get_draw_scent()` in `krk_features.py` assigns micro-rewards:
  - +0.08 for rook cut established
  - +0.06 for enemy king at edge
  - +0.05 for small box area (≤16)
  - +0.04 for opposition
  - +0.02 for rook safe, king coordination
  - Enabled via `M5_ENABLE_DRAW_SAMPLING=1` env var

**Learning:** Simple endgames (KPK) don't force hierarchical reasoning - the flat topology is "good enough." 
Complex endgames (KRK) require explicit Box Method sequences, but current thresholds are too strict to 
discover patterns in the chaos. Need more aggressive exploration in failure states. Also discovered that
the spike detection system was blocking promotion entirely - no spikes means no high_impact cells, which
blocks the entire promotion pipeline. Draw scent sampling addresses the root cause: sample starvation.
By rewarding partial Box Method progress even in draws, cells collect enough samples for consistency
scoring and promotion.

---

## 2026-01-07: KRK Staged Curriculum Validated

**Run:** `krk_simple_full` - 11 stages, simple heuristic mode
**Result:** Curriculum positions are solvable but heuristics fail on mid-stages

| Stage | Name | Win Rate | Notes |
|-------|------|----------|-------|
| 0 | Mate_In_1 | **100%** | ✅ Trivial |
| 1 | Mate_In_2 | 39% | Heuristics struggle |
| 2 | Edge_Trapped_Tempo | **77%** | Almost passed |
| 3 | Anchored_Cut | 22% | King coordination |
| 4-7 | Edge_Cut_Hold... | **0%** | Heuristics fail completely |
| 8 | Box_Small | 31% | Partial recovery |
| 9-10 | Box_Medium/Full | 0% | Too complex |

**Analysis:**
- Stages 0-2 can be solved with simple logic → good for M5 to learn from
- Stages 4-7 require multi-step planning → need ReCoN hierarchies
- Random position runs skip the easy stages → sample starvation

**Recommendation:**
Use staged curriculum with ReCoN mode + M5 enabled. The easy stages (0-2) 
generate wins that trigger TRIAL promotions, enabling hierarchical growth 
for harder stages. Need to fix ReCoN mode topology loading first.

---
