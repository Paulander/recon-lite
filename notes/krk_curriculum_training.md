# KRK Curriculum Training Notes

## 2026-01-07: Staged KRK Curriculum Completed

### Run Summary
**Path:** `snapshots/evolution/krk_curriculum/curriculum_summary.json`
**Mode:** ReCoN Engine + M5 stem cells enabled
**Total Games:** ~8,030

### Stage Results

| Stage | Name | Win Rate | Games |
|-------|------|----------|-------|
| 0 | Mate_In_1 | **100%** ✅ | 30 |
| 1 | Mate_In_2 | 37.1% | 1000 |
| 2 | Edge_Trapped_Tempo | **86.7%** ✅ | 30 |
| 3 | Anchored_Cut | 19.4% | 1000 |
| 4 | Edge_Cut_Hold | 0% | 1000 |
| 5 | King_Close_1 | 0% | 1000 |
| 6 | King_Close_2 | 0% | 1000 |
| 7 | King_Far_Cut_Held | 0% | 1000 |
| 8 | Box_Small | **34.5%** | 1000 |
| 9 | Box_Medium | 0% | 1000 |
| 10 | Full_KRK | 0% | 1000 |

### Key Observations

1. **Easy stages pass quickly**: Stage 0 (100%) and Stage 2 (86.7%) hit threshold fast
2. **Mid-stages stall**: Stages 4-7 require multi-step planning → 0% win rate
3. **Stage 8 partial recovery**: Box_Small gets 34.5% suggesting simpler box positions are learnable
4. **Gap Analysis**: The curriculum jumps from "easy" (2-move mates) to "complex" (full cut management)

### M5 Stem Cell Activity
- TRIAL promotions observed in Stages 2-3
- Samples collecting (10,000+)
- Stall recovery spawning new cells

### Technical Details
- Uses ReCoN engine with heuristic fallback
- Goal Delegation Pack template implemented (`nodes/pack_template.py`)
- Lottery spawning: 40% pack, 40% single, 20% variant
- Pack spawning enabled via `M5_PACK_PROB=0.40`

### Recommendations

1. **Reorder stages for knowledge transfer:**
   - Current: Linear difficulty increase
   - Proposed: Group by "skill clusters" (cut_skills → king_approach → box_shrink)
   
2. **Add intermediate "bridge" stages:**
   - Between Stage 3 (Anchored_Cut) and Stage 4 (Edge_Cut_Hold)
   - Teach cut MAINTENANCE before cut APPROACH
   
3. **Train failing stages longer:**
   - Stages 4-7 need 2000+ games to trigger failure-driven pack spawning
   - Or increase `M5_PACK_PROB` to 0.6 for more aggressive exploration

### Open Questions
- Are the 15 TRIAL cells learning useful patterns or random noise?
- Does the 86.7% on Stage 2 indicate genuine skill or heuristic fallback?
- Would POR chains emerge with longer training on harder stages?
