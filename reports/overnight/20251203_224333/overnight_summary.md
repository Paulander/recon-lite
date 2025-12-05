# 🌙 Overnight Training Report

**Generated:** Wed Dec  3 22:53:01 CET 2025
**Duration:** Started 20251203_224333

## 📦 Weight Packs for Comparison

| Pack | Path |
|------|------|
| **FIRST (before training)** | `/home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_FIRST.json` |
| **FINAL (after training)** | `/home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_FINAL.json` |
| **After KRK** | `/home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_after_KRK.json` |

## 📊 Full-Game Win Rate vs Random

| Metric | BEFORE | AFTER | Change |
|--------|--------|-------|--------|
| **Wins** | 0 | 0 | 0 |
| **Draws** | 0 | 0 | 0 |
| **Losses** | 0 | 0 | 0 |
| **Win Rate** | 0% | 0% | 0% |

## 🎓 Training Summary

- **KRK games:** 200
- **KPK games:** 100
- **Engine depth:** 2

## 📁 Files Generated

- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251203_224333/`
- Backups: `/home/paulander/git/recon-lite/weights/backups/20251203_224333/`
- Logs: `/home/paulander/git/recon-lite/logs/overnight_20251203_224333.log`

## 🔍 Compare Weight Changes

Run this command to see what changed:
```bash
uv run python tools/pack_diff.py \
  /home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_FIRST.json \
  /home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_FINAL.json
```

## 🎮 Replay Games with Different Packs

To compare gameplay:
```bash
# With FIRST (before training)
uv run python demos/persistent/full_game_demo.py \
  --vs-random --max-moves 200

# Manual comparison - load different consolidation packs
```

Pack Diff Report
================

Old: /home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_FIRST.json
New: /home/paulander/git/recon-lite/weights/backups/20251203_224333/krk_consol_FINAL.json

Summary:
  Total edges compared: 4
  Modified: 4
  New: 0
  Removed: 0

Top 4 Changes (by absolute delta):

Edge                                            Old        New      Delta        %
---------------------------------------- ---------- ---------- ---------- --------
p0_check → p0_move (POR)                     0.9302     0.6850    -0.2452   -26.4%
p0_move → p0_wait (POR)                      0.9652     0.8012    -0.1640   -17.0%
phase0_establish_cut → phase1_drive...       0.9966     0.9574    -0.0392    -3.9%
p1_check → p1_move (POR)                     0.9966     0.9574    -0.0392    -3.9%
