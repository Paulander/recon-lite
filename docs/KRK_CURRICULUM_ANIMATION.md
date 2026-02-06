# KRK Curriculum Animation Metadata

The KRK curriculum training saves metadata snapshots for animation, similar to KPK epoch training.

## Snapshot Structure

### Cycle Snapshots
Saved to: `snapshots/evolution/krk_curriculum/stage{N}/cycle_{NNNN}.json`

```json
{
  "stage_id": 0,
  "stage_name": "Mate_In_1",
  "cycle": 1,
  "games": 50,
  "win_rate": 0.85,
  "avg_moves": 2.3,
  "avg_reward": 0.72,
  "escape_rate": 0.05,
  "total_games": 50,
  "timestamp": "2026-01-06T12:34:56",
  "w_base": {
    "krk_root->krk_detect:SUB": 1.2,
    "krk_detect->krk_execute:POR": 0.9,
    ...
  },
  "consolidation_applied": false
}
```

### Final Weights Snapshot
Saved to: `snapshots/evolution/krk_curriculum/final_weights.json`

Contains all `w_base` weights after training completes, for visualization.

### Final Consolidation State
Saved to: `snapshots/evolution/krk_curriculum/final_consolidation.json`

Full consolidation engine state (can be loaded to resume training or transfer to new runs).

## Transfer Learning

Weights learned in Phase 0 transfer to Phase 1, etc.:

1. **Within Run**: Consolidation engine persists across stages
   - Weights learned in Stage 0 (Mate in 1) are used in Stage 1 (Mate in 2)
   - No explicit stage transition needed - consolidation engine is shared

2. **Across Runs**: Load `final_consolidation.json` to resume
   ```python
   consolidation_engine.load_state(Path("snapshots/evolution/krk_curriculum/final_consolidation.json"))
   consolidation_engine.apply_w_base_to_graph(graph)
   ```

## Animation

The cycle snapshots can be used to animate:
- Weight changes over cycles (`w_base` field)
- Win rate progression (`win_rate` field)
- Stage transitions (when `stage_id` changes)

Each snapshot represents one training cycle (typically 50 games), showing how weights evolve as the network learns.

