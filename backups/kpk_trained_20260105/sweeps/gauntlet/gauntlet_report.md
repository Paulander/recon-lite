# Hyperparameter Sweep Report

**Generated:** 2026-01-05T15:51:31.528734
**Trials:** 3

## Summary Table

| Trial | Win Rate | Cycles to 80% | SOLID | Hoisted | POR | Max Depth | Branching |
|-------|----------|---------------|-------|---------|-----|-----------|-----------|
| gauntlet_baseline | 96.0% | 1 | 0 | 0 | 3 | 1 | 0.00 |
| gauntlet_aggressive | 84.0% | 1 | 0 | 0 | 3 | 1 | 0.00 |
| forced_hierarchy | 88.0% | 1 | 0 | 0 | 3 | 1 | 0.00 |

## Configuration Comparison

| Trial | Consistency | Hoist | Success Bypass | Speculative | Stall Recovery |
|-------|-------------|-------|----------------|-------------|----------------|
| gauntlet_baseline | 0.4 | 0.85 | No | No | Yes |
| gauntlet_aggressive | 0.2 | 0.6 | Yes | Yes | Yes |
| forced_hierarchy | 0.15 | 0.5 | Yes | Yes | Yes |

## Win Rate Progression

**gauntlet_baseline:** 94% → 88% → 86% → 84% → 90% → 92% → 94% → 90% → 94% → 90% ... (30 cycles)

**gauntlet_aggressive:** 94% → 90% → 90% → 90% → 96% → 94% → 90% → 88% → 100% → 88% ... (30 cycles)

**forced_hierarchy:** 84% → 88% → 90% → 90% → 90% → 94% → 86% → 92% → 90% → 98% ... (30 cycles)
