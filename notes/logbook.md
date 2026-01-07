# ReCoN-lite Development Logbook

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

**Learning:** Simple endgames (KPK) don't force hierarchical reasoning - the flat topology is "good enough." 
Complex endgames (KRK) require explicit Box Method sequences, but current thresholds are too strict to 
discover patterns in the chaos. Need more aggressive exploration in failure states. Also discovered that
the spike detection system was blocking promotion entirely - no spikes means no high_impact cells, which
blocks the entire promotion pipeline. Step 2b fixes this by directly selecting CANDIDATE cells when in
extreme failure mode.

---
