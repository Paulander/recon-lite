# Overnight Training Report - 20251204_234521

**Generated:** 2025-12-05 00:02:14

## Consolidation State

- **Total Episodes Accumulated:** 0
- **Tracked Edges:** 0

## File References

- Consolidation: `/home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FINAL.json`
- Traces: `/home/paulander/git/recon-lite/reports/overnight/20251204_234521/krk_training.jsonl`
- Initial State: `/home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FIRST.json`

## Analysis Commands

```bash
# View weight changes in detail
uv run python tools/pack_diff.py /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FIRST.json /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FINAL.json
uv run python tools/report_consolidation.py /home/paulander/git/recon-lite/weights/backups/20251204_234521/krk_consol_FINAL.json

# Visualize consolidation history
# Open demos/visualization/consolidation_dashboard.html and load the JSON
```
