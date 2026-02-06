# 🌙 Overnight Training Report

**Generated:** Fri Dec  5 00:02:14 CET 2025
**Started:** 20251204_234521
**Duration:** 16 minutes

## 📦 Weight Packs for Comparison

| Pack | Path |
|------|------|
| **FIRST (before training)** | `/home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FIRST.json` |
| **FINAL (after training)** | `/home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FINAL.json` |
| **After KRK** | `/home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_after_KRK.json` |

## 📊 Full-Game Win Rate vs Random

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Wins** | 0 | 0 | 0 |
| **Draws** | 0 | 0 | 0 |
| **Losses** | 0 | 0 | 0 |
| **Win Rate** | 0% | 0% | - |

## 🎓 Training Summary

- **KRK games:** 2000
- **KPK games:** 100
- **Engine depth:** 2
- **Tactics processed:** 0

## 🎯 Tactical Training Results

- `suite`: - **13** positions, **0.0%** accuracy

## 📁 Files Generated

- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251204_234521/`
- Backups: `/home/paulander/git/recon-lite/weights/backups/20251204_234521/`
- Logs: `/home/paulander/git/recon-lite/logs/overnight_20251204_234521.log`

## 🔍 Compare Weight Changes

Run this command to see what changed:
```bash
uv run python tools/pack_diff.py \
  /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FIRST.json \
  /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FINAL.json
```

Pack Diff Report
================

Old: /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FIRST.json
New: /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FINAL.json

Summary:
  Total edges compared: 4
  Modified: 4
  New: 0
  Removed: 0

Top 4 Changes (by absolute delta):

Edge                                            Old        New      Delta        %
---------------------------------------- ---------- ---------- ---------- --------
p0_move → p0_wait (POR)                      0.7872     0.1065    -0.6808   -86.5%
p0_check → p0_move (POR)                     0.6619     0.1108    -0.5511   -83.3%
p1_check → p1_move (POR)                     0.9574     0.8032    -0.1543   -16.1%
phase0_establish_cut → phase1_drive...       0.9574     0.8032    -0.1543   -16.1%
