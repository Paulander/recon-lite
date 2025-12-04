# {{ title }}

**Generated:** {{ timestamp }}
**Run Duration:** {{ duration }}

## Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Win Rate** | {{ before_win_rate }}% | {{ after_win_rate }}% | {{ win_rate_delta }}% |
| **Checkmate Rate** | {{ before_checkmate_rate }}% | {{ after_checkmate_rate }}% | {{ checkmate_delta }}% |
| **Avg Plies to Mate** | {{ before_avg_plies }} | {{ after_avg_plies }} | {{ plies_delta }} |

## Training Details

- **Mode:** {{ mode }}
- **Games Trained:** {{ games_trained }}
- **Engine:** {{ engine }} (depth {{ depth }})
- **Consolidation Applied:** {{ consolidation_count }} times

## Top Weight Changes

| Edge | Initial | Final | Delta | % Change |
|------|---------|-------|-------|----------|
{{ weight_changes }}

## Files Generated

- **Weights (Before):** `{{ weights_before }}`
- **Weights (After):** `{{ weights_after }}`
- **Traces:** `{{ traces_path }}`
- **Log:** `{{ log_path }}`

## Next Steps

1. Compare weights: `uv run python tools/pack_diff.py {{ weights_before }} {{ weights_after }}`
2. View dashboard: Open `demos/visualization/consolidation_dashboard.html`
3. Run validation: `uv run python demos/experiments/pack_tournament.py --pack {{ weights_after }}`

## Issues

{{ issues }}

