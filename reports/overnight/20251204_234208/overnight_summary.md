# 🌙 Overnight Training Report

**Generated:** Thu Dec  4 23:44:15 CET 2025
**Started:** 20251204_234208
**Duration:** 2 minutes

## 📦 Weight Packs for Comparison

| Pack | Path |
|------|------|
| **FIRST (before training)** | `/home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_FIRST.json` |
| **FINAL (after training)** | `/home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_FINAL.json` |
| **After KRK** | `/home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_after_KRK.json` |

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

- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251204_234208/`
- Backups: `/home/paulander/git/recon-lite/weights/backups/20251204_234208/`
- Logs: `/home/paulander/git/recon-lite/logs/overnight_20251204_234208.log`

## 🔍 Compare Weight Changes

Run this command to see what changed:
```bash
uv run python tools/pack_diff.py \
  /home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_FIRST.json \
  /home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_FINAL.json
```

Pack Diff Report
================

Old: /home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_FIRST.json
New: /home/paulander/git/recon-lite/weights/backups/20251204_234208/krk_consol_FINAL.json

Summary:
  Total edges compared: 4
  Modified: 2
  New: 0
  Removed: 0

Top 2 Changes (by absolute delta):

Edge                                            Old        New      Delta        %
---------------------------------------- ---------- ---------- ---------- --------
p0_check → p0_move (POR)                     0.6850     0.6619    -0.0231    -3.4%
p0_move → p0_wait (POR)                      0.8012     0.7872    -0.0140    -1.7%
