# 🌙 Overnight Training Report

**Generated:** Fri Dec  5 14:36:51 CET 2025
**Started:** 20251205_143648
**Duration:** 0 minutes

## 📊 Win Rate Comparison (vs Random)

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Games** | 10 | 10 | - |
| **Wins** | 0 | 0 | 0 |
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
- **Tactic types trained:** 0
- **Total patterns detected:** 0
- **Positions per type:** 50

## 🎯 Tactical Results


## 📦 Weight Packs

| Pack | Path |
|------|------|
| KRK (before) | `/home/paulander/git/recon-lite/weights/backups/20251205_143648/krk_consol_FIRST.json` |
| KRK (after) | `/home/paulander/git/recon-lite/weights/backups/20251205_143648/krk_consol_FINAL.json` |
| Full Game | `/home/paulander/git/recon-lite/weights/backups/20251205_143648/fullgame_consol_FINAL.json` |
| KPK | `/home/paulander/git/recon-lite/weights/backups/20251205_143648/kpk_consol_FINAL.json` |

## 📁 Files Generated

- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251205_143648/`
- Backups: `/home/paulander/git/recon-lite/weights/backups/20251205_143648/`
- Logs: `/home/paulander/git/recon-lite/logs/overnight_20251205_143648.log`

## 🔍 Compare Weight Changes

```bash
# KRK changes
uv run python tools/pack_diff.py \
  /home/paulander/git/recon-lite/weights/backups/20251205_143648/krk_consol_FIRST.json \
  /home/paulander/git/recon-lite/weights/backups/20251205_143648/krk_consol_FINAL.json
```

Pack Diff Report
================

Old: /home/paulander/git/recon-lite/weights/backups/20251205_143648/krk_consol_FIRST.json
New: /home/paulander/git/recon-lite/weights/backups/20251205_143648/krk_consol_FINAL.json

Summary:
  Total edges compared: 4
  Modified: 0
  New: 0
  Removed: 0

No differences found.
