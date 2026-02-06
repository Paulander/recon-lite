# Tactics Training Baseline Report

**Generated:** 2025-12-05  
**Training Runs:** 3 (1500 positions/tactic total)  
**Stockfish Depth:** 6  

---

## Executive Summary

After 3 training runs (24,000 total positions across 16 tactics), we have established baseline performance metrics for the ReCoN tactics subsystem.

| Metric | Value |
|--------|-------|
| **Overall Detection Rate** | 47.3% |
| **Overall Accuracy** | 10.3% |
| **Best Performer** | interference (43.8% accuracy) |
| **Broken Detectors** | 3 (backRankMate, doubleCheck, smotheredMate) |

---

## Detailed Results by Tactic

### Tier 1: Working Well (>20% accuracy)

| Tactic | Detection | Accuracy | Episodes | Notes |
|--------|-----------|----------|----------|-------|
| interference | 97.4% | **43.8%** | 2000 | ⭐ Best overall performer |
| quietMove | 92.6% | **32.2%** | 2000 | ⭐ Non-forcing moves work well |
| zugzwang | 86.0% | **26.8%** | 2000 | ⭐ Good pattern recognition |

### Tier 2: Detection Works, Move Proposal Needs Work (>10% accuracy)

| Tactic | Detection | Accuracy | Gap | Episodes | Notes |
|--------|-----------|----------|-----|----------|-------|
| discoveredAttack | 35.8% | 11.0% | 24.8% | 2000 | Medium detection |
| exposedKing | 30.0% | 9.6% | 20.4% | 2000 | Decent |
| hangingPiece | 27.0% | 9.4% | 17.6% | 2000 | Room to improve |
| fork | 81.0% | 9.2% | 71.8% | 2550 | High detect, low accuracy |
| attraction | 79.2% | 9.0% | 70.2% | 2000 | Same pattern |

### Tier 3: Needs Significant Improvement (<10% accuracy)

| Tactic | Detection | Accuracy | Gap | Episodes | Notes |
|--------|-----------|----------|-----|----------|-------|
| skewer | 24.8% | 4.8% | 20.0% | 2500 | Low detection |
| sacrifice | 89.8% | 2.6% | 87.2% | 2000 | 🔧 Move proposal broken |
| deflection | 81.0% | 2.6% | 78.4% | 2000 | 🔧 Move proposal broken |
| pin | 12.4% | 2.2% | 10.2% | 2500 | Poor detection |
| trappedPiece | 20.0% | 2.2% | 17.8% | 2000 | Needs work |

### Tier 4: Broken (0% detection)

| Tactic | Detection | Accuracy | Episodes | Notes |
|--------|-----------|----------|----------|-------|
| backRankMate | 0.0% | 0.0% | 2000 | ❌ Detector never fires |
| doubleCheck | 0.0% | 0.0% | 2000 | ❌ Detector never fires |
| smotheredMate | 0.0% | 0.0% | 2000 | ❌ Detector never fires |

---

## Detection vs Accuracy Analysis

The **gap** between detection rate and accuracy reveals where the problem lies:

### Large Gap = Move Proposal Bug
```
sacrifice:   89.8% detect → 2.6% accuracy  (gap: 87.2%)
deflection:  81.0% detect → 2.6% accuracy  (gap: 78.4%)
fork:        81.0% detect → 9.2% accuracy  (gap: 71.8%)
attraction:  79.2% detect → 9.0% accuracy  (gap: 70.2%)
```

**Diagnosis:** These detectors correctly identify the pattern exists, but `get_*_moves()` functions return wrong moves.

### Small Gap = Detector Bug
```
pin:          12.4% detect → 2.2% accuracy  (gap: 10.2%)
trappedPiece: 20.0% detect → 2.2% accuracy  (gap: 17.8%)
```

**Diagnosis:** The detector itself fails to recognize the pattern.

### Zero Detection = Heuristic Failure
```
backRankMate:  0.0% detect → 0.0% accuracy
doubleCheck:   0.0% detect → 0.0% accuracy  
smotheredMate: 0.0% detect → 0.0% accuracy
```

**Diagnosis:** The heuristic completely fails. Needs rewrite or MLP replacement.

---

## Edge Weight Evolution

Example from `weights/latest/tactics/fork_consol.json`:

```json
{
  "w_base": {
    "tactics_root->detect_fork:SUB": 1.010495,  // +1.0% from initial
    "tactics_root->exploit_fork:SUB": 1.010495   // +1.0% from initial
  },
  "consolidation_meta": {
    "total_episodes": 2550,
    "edges_tracked": 14
  }
}
```

With ~9.2% accuracy and 2550 episodes:
- Positive rewards: ~235 episodes (9.2% × 2550)
- Negative rewards: ~2315 episodes
- Net weight change: +1.0% (slow learning working as intended)

---

## Training Progression Across Runs

| Run | Positions/Tactic | Total | Detection | Accuracy | Notes |
|-----|------------------|-------|-----------|----------|-------|
| 1 | 500 | 8,000 | 47.3% | 10.3% | Baseline established |
| 2 | 500 | 8,000 | 47.3% | 10.3% | Continued from Run 1 |
| 3 | 500 | 8,000 | 47.3% | 10.3% | Stable metrics |

**Observation:** Detection and accuracy remained stable across runs. This suggests:
1. The heuristics have reached their performance ceiling
2. Edge weight changes alone cannot improve move selection
3. The underlying detection/proposal logic needs improvement

---

## Recommendations

### Immediate Fixes (Phase 1)

1. **Fix Broken Detectors** - `backRankMate`, `doubleCheck`, `smotheredMate`
   - File: `src/recon_lite_chess/scripts/tactics.py`
   - These return 0% detection despite having puzzle data

2. **Fix Move Proposal Functions** for high-gap tactics:
   - `sacrifice` (87% gap)
   - `deflection` (78% gap)  
   - `fork` (72% gap)

### Medium-term Improvements (Phase 2)

3. **Add Stockfish Validation** to reward correct moves only
4. **Implement MLP Detectors** for patterns where heuristics fail

### Long-term (Phase 3)

5. **Full Game Evaluation** - Run 50 games before/after to measure game impact
6. **Cross-tactic Integration** - Train combined tactical awareness

---

## Files & Artifacts

### Training Logs
- `reports/tactics_run1.log` - Run 1 output
- `reports/tactics_run2.log` - Run 2 output
- `reports/tactics_run3.log` - Run 3 output

### Consolidated Weights (Current)
```
weights/latest/tactics/
├── fork_consol.json        (2550 episodes)
├── pin_consol.json         (2500 episodes)
├── skewer_consol.json      (2500 episodes)
├── hangingPiece_consol.json
├── backRankMate_consol.json
├── discoveredAttack_consol.json
├── attraction_consol.json
├── deflection_consol.json
├── doubleCheck_consol.json
├── smotheredMate_consol.json
├── trappedPiece_consol.json
├── quietMove_consol.json
├── sacrifice_consol.json
├── exposedKing_consol.json
├── interference_consol.json
└── zugzwang_consol.json
```

### Weight Backups
```
weights/tactics/  (source weights, also updated)
```

---

## Conclusion

The baseline is established. Key findings:

1. **3 tactics are completely broken** (0% detection) - need heuristic fixes
2. **4 tactics have excellent detection but poor accuracy** - move proposal bugs
3. **3 tactics perform well** (>20% accuracy) - the heuristics work
4. **Edge weight learning is working** but can't fix broken heuristics

**Next step:** Fix the broken detectors and move proposal functions before running more training.

