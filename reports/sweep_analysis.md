# M5.1 Hyperparameter Sweep Analysis

**Generated:** 2026-01-05T15:20:12.136715
**Sweep Directory:** snapshots/sweeps
**Trials Analyzed:** 3

---

## Summary Table

| Trial | Win Rate | Cycles to 80% | SOLID | Hoisted | POR | Max Depth | Branching |
|-------|----------|---------------|-------|---------|-----|-----------|-----------|
| conservative | 100.0% | 1 | 0 | 0 | 3 | 1 | 0.00 |
| speculative | 100.0% | 1 | 0 | 0 | 3 | 1 | 0.00 |
| recursive | 100.0% | 1 | 0 | 0 | 3 | 1 | 0.00 |

---

## Configuration Comparison

| Trial | Consistency | Hoist | Success Bypass | Speculative | Stall Recovery | Scent Shaping |
|-------|-------------|-------|----------------|-------------|----------------|---------------|
| conservative | 0.5 | 0.9 | No | No | Yes | Yes |
| speculative | 0.3 | 0.75 | No | No | Yes | Yes |
| recursive | 0.4 | 0.85 | Yes | Yes | Yes | Yes |

---

## Win Rate Progression

**conservative:** 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% ... (20 cycles)

**speculative:** 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% ... (20 cycles)

**recursive:** 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% -> 100% ... (20 cycles)


---

## Learning Speed Analysis

**Fastest to 80%:** conservative (1 cycles)
**Slowest to 80%:** recursive (1 cycles)

### Final Win Rates

🥇 conservative: 100.0%
🥈 speculative: 100.0%
🥉 recursive: 100.0%

---

## Structural Maturity Analysis

**Deepest Hierarchy:** conservative (depth 1)
**Most POR Edges:** conservative (3 edges)
**Highest Branching Factor:** conservative (0.00)

### Healthy Growth Signature Check

| Trial | Depth >= 4 | Branch >= 1.5 | POR > 0 | Status |
|-------|------------|---------------|---------|--------|
| conservative | No | No | Yes | Growing |
| speculative | No | No | Yes | Growing |
| recursive | No | No | Yes | Growing |

---

## Recommendations

**Recommended Configuration:** conservative

```
consistency_threshold: 0.5
hoist_threshold: 0.9
enable_success_bypass: False
enable_speculative_hoisting: False
enable_stall_recovery: True
enable_scent_shaping: True
```