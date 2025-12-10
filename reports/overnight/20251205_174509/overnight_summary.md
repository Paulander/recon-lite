# 🌙 Overnight Training Report

**Generated:** Fri Dec  5 17:48:25 CET 2025
**Started:** 20251205_174509
**Duration:** 3 minutes

## 📊 Win Rate Comparison (vs Random)

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Games** | 10 | 10 | - |
| **Wins** | 0 | 4 | 4 |
| **Losses** | 0 | 0 | 0 |
| **Draws** | 10 | 6 | -4 |
| **Win Rate** | 0% | 40.0% | 40.0% |

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
| KRK (before) | `/home/paulander/git/recon-lite/weights/backups/20251205_174509/krk_consol_FIRST.json` |
| KRK (after) | `/home/paulander/git/recon-lite/weights/backups/20251205_174509/krk_consol_FINAL.json` |
| Full Game | `/home/paulander/git/recon-lite/weights/backups/20251205_174509/fullgame_consol_FINAL.json` |
| KPK | `/home/paulander/git/recon-lite/weights/backups/20251205_174509/kpk_consol_FINAL.json` |

## 📁 Files Generated

- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251205_174509/`
- Backups: `/home/paulander/git/recon-lite/weights/backups/20251205_174509/`
- Logs: `/home/paulander/git/recon-lite/logs/overnight_20251205_174509.log`

## 🔍 Compare Weight Changes

```bash
# KRK changes
uv run python tools/pack_diff.py \
  /home/paulander/git/recon-lite/weights/backups/20251205_174509/krk_consol_FIRST.json \
  /home/paulander/git/recon-lite/weights/backups/20251205_174509/krk_consol_FINAL.json
```

Pack Diff Report
================

Old: /home/paulander/git/recon-lite/weights/backups/20251205_174509/krk_consol_FIRST.json
New: /home/paulander/git/recon-lite/weights/backups/20251205_174509/krk_consol_FINAL.json

Summary:
  Total edges compared: 4
  Modified: 3
  New: 0
  Removed: 0

Top 3 Changes (by absolute delta):

Edge                                            Old        New      Delta        %
---------------------------------------- ---------- ---------- ---------- --------
p0_check → p0_move (POR)                     0.1138     0.1000    -0.0138   -12.1%
p1_check → p1_move (POR)                     0.8046     0.8055    +0.0009    +0.1%
phase0_establish_cut → phase1_drive...       0.8046     0.8055    +0.0009    +0.1%
