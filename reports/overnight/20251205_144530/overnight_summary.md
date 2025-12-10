# 🌙 Overnight Training Report

**Generated:** Fri Dec  5 14:47:26 CET 2025
**Started:** 20251205_144530
**Duration:** 1 minutes

## 📊 Win Rate Comparison (vs Random)

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Games** | 10 | 10 | - |
| **Wins** | 0 |  | 0 |
| **Losses** | 0 | 0 | 0 |
| **Draws** | 0 | 0 | 0 |
| **Win Rate** | 0% | 0% | 0% |

## 🎓 Training Summary

### Endgame Training
- **KRK games:** 50
- **KPK games:** 20
- **Engine depth:** 2 (endgames), 2 (full games), 4 (tactics)

### Full Game Training
- **Games trained:** 20
- **Stem cells:** Enabled
- **Patterns promoted:** 0

### Tactical Training
- **Tactic types trained:** 13
- **Total patterns detected:** 268
- **Positions per type:** 50

## 🎯 Tactical Results

- `attraction`: - **50** positions, **8.0%** accuracy
- `backRankMate`: - **50** positions, **0.0%** accuracy
- `deflection`: - **50** positions, **4.0%** accuracy
- `discoveredAttack`: - **50** positions, **12.0%** accuracy
- `doubleCheck`: - **50** positions, **0.0%** accuracy
- `fork`: - **50** positions, **10.0%** accuracy
- `hangingPiece`: - **50** positions, **6.0%** accuracy
- `pin`: - **50** positions, **0.0%** accuracy
- `quietMove`: - **50** positions, **32.0%** accuracy
- `sacrifice`: - **50** positions, **2.0%** accuracy
- `skewer`: - **50** positions, **8.0%** accuracy
- `smotheredMate`: - **50** positions, **0.0%** accuracy
- `trappedPiece`: - **50** positions, **4.0%** accuracy

## 📦 Weight Packs

| Pack | Path |
|------|------|
| KRK (before) | `/home/paulander/git/recon-lite/weights/backups/20251205_144530/krk_consol_FIRST.json` |
| KRK (after) | `/home/paulander/git/recon-lite/weights/backups/20251205_144530/krk_consol_FINAL.json` |
| Full Game | `/home/paulander/git/recon-lite/weights/backups/20251205_144530/fullgame_consol_FINAL.json` |
| KPK | `/home/paulander/git/recon-lite/weights/backups/20251205_144530/kpk_consol_FINAL.json` |

## 📁 Files Generated

- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251205_144530/`
- Backups: `/home/paulander/git/recon-lite/weights/backups/20251205_144530/`
- Logs: `/home/paulander/git/recon-lite/logs/overnight_20251205_144530.log`

## 🔍 Compare Weight Changes

```bash
# KRK changes
uv run python tools/pack_diff.py \
  /home/paulander/git/recon-lite/weights/backups/20251205_144530/krk_consol_FIRST.json \
  /home/paulander/git/recon-lite/weights/backups/20251205_144530/krk_consol_FINAL.json
```

Pack Diff Report
================

Old: /home/paulander/git/recon-lite/weights/backups/20251205_144530/krk_consol_FIRST.json
New: /home/paulander/git/recon-lite/weights/backups/20251205_144530/krk_consol_FINAL.json

Summary:
  Total edges compared: 4
  Modified: 0
  New: 0
  Removed: 0

No differences found.
