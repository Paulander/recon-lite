# Stem Cell Inspector: Visualization & Naming Tool

**Date**: 2026-01-07 

**Context**: Need to understand what stem cells have learned, verify bindings, and potentially auto-name sensors based on their pattern signatures.## ProblemCurrently, stem cells are opaque:- `stem_cells.json` contains raw feature vectors- No easy way to visualize what patterns a cell has learned- No navigation between cells (prev/next)- No connection to actual chess positions- AND-gate clusters and POR chains are hard to trace## Proposed Tool: `scripts/inspect_stem_cells.py`

### Core Features1. **Cell Listing & Navigation**   
$ python scripts/inspect_stem_cells.py --list
Stem Cells (19 TRIAL, 3 CANDIDATE, 0 MATURE):
[0] stem_0000 (TRIAL) XP: 47/100 "SmallBox_CutStable"
[1] stem_0001 (TRIAL) XP: 52/100 "KingsClose_Opposition"
[2] stem_0002 (TRIAL) XP: 38/100 "AtEdge_MateReady"
...
2. **Detailed Cell View**
$ python scripts/inspect_stem_cells.py --cell stem_0015
============================================================
Stem Cell: stem_0015 (TRIAL) | XP: 47/100 → MATURE
============================================================
Pattern Signature (11 features):
king_distance: 3.2 [■■■░░░░░] (0-7)
opposition_status: 0.8 [■■■■■■■░] (0-1)
enemy_king_edge_dist: 2.1 [■■░░░░░░] (0-4)
box_area: 12.5 [■░░░░░░░] (0-64) ← SMALL!
box_min_side: 4.0 [■■■■░░░░] (0-8)
rook_fence_distance: 2.3 [■■░░░░░░] (0-8)
cut_established: 0.9 [■■■■■■■■] STABLE
rook_safe: 1.0 [■■■■■■■■] SAFE
king_rook_distance: 2.1 [■■░░░░░░] (0-8)
can_mate_now: 0.0 [░░░░░░░░] NO
stalemate_danger: 0.0 [░░░░░░░░] NO
Auto-Name: "SmallBox_KingsClose_CutStable"
Connections:
← Parent: krk_detect (SUB, w=0.50)
→ Wired to: krk_rook_leg (SUB, w=0.50)
→ Wired to: krk_king_leg (SUB, w=0.50)
🔗 AND-gate: cluster_0019 (combined with stem_0016)
Sample Positions (5 of 127):
[1] 4k3/R7/K7/8/8/8/8/8 w - - (r=1.0, t=5)
[2] 5k2/1R6/3K4/8/8/8/8/8 w - - (r=0.8, t=12)
[3] 6k1/2R5/4K3/8/8/8/8/8 w - - (r=0.9, t=18)
3. **Chessboard Visualization (Optional)**
$ python scripts/inspect_stem_cells.py --cell stem_0015 --board 0
Position [0]: 4k3/R7/K7/8/8/8/8/8 (reward: 1.0)
a b c d e f g h
8 . . . . ♚ . . . Features:
7 ♖ . . . . . . . King distance: 3
6 ♔ . . . . . . . Cut: STABLE
5 . . . . . . . . Box: 1×5 = 5
4 . . . . . . . . Edge dist: 0
3 . . . . . . . .
2 . . . . . . . .
1 . . . . . . . .
4. **AND-Gate / Cluster View**
$ python scripts/inspect_stem_cells.py --clusters
AND-Gate Clusters (33 total):
[cluster_0019] stem_0000 AND stem_0001
→ Parent: krk_execute
→ Activations: 127, Confirms: 45
[cluster_0023] stem_0000 AND stem_0001 (forced_crisis)
...
5. **Auto-Naming Logic**   def auto_name(sig: List[float]) -> str:       parts = []       names = KRKFeatures.feature_names()              # King distance       if sig[0] < 2: parts.append("KingsAdjacent")       elif sig[0] < 4: parts.append("KingsClose")              # Opposition       if sig[1] > 0.5: parts.append("Opposition")              # Edge       if sig[2] < 1.5: parts.append("AtEdge")              # Box       if sig[3] < 12: parts.append("SmallBox")       elif sig[3] < 24: parts.append("MediumBox")              # Cut       if sig[6] > 0.7: parts.append("CutStable")              # Mate       if sig[9] > 0.5: parts.append("MateReady")              return "_".join(parts) or "Exploring"### Export Options- `--export-csv`: Dump all cells to CSV for spreadsheet analysis- `--export-graphviz`: Generate DOT file showing cell → cluster → leg connections- `--find-similar PATTERN`: Find cells matching a named pattern## Implementation Notes1. Load from `stem_cells.json` or `--path` argument2. Feature names come from `KRKFeatures.feature_names()`3. Cluster info embedded in cell metadata4. Chess board rendering via `python-chess` ASCII or custom## Use Cases- **Sanity Check**: "Is stem_0015 actually detecting what I think?"- **Naming**: Manually rename important sensors for clarity- **Debugging**: "Why did this cluster form? Show me the samples"- **Transfer Analysis**: "Which KPK sensors transferred to KRK?"## Related- `krk_features.py` - Feature definitions- `stem_cell.py` - StemCellTerminal.to_dict() for saved data- `viz/` - Existing topology visualizer could integrate this
Export Options
--export-csv: Dump all cells to CSV for spreadsheet analysis
--export-graphviz: Generate DOT file showing cell → cluster → leg connections
--find-similar PATTERN: Find cells matching a named pattern
Implementation Notes
Load from stem_cells.json or --path argument
Feature names come from KRKFeatures.feature_names()
Cluster info embedded in cell metadata
Chess board rendering via python-chess ASCII or custom
Use Cases
Sanity Check: "Is stem_0015 actually detecting what I think?"
Naming: Manually rename important sensors for clarity
Debugging: "Why did this cluster form? Show me the samples"
Transfer Analysis: "Which KPK sensors transferred to KRK?"
Related
krk_features.py - Feature definitions
stem_cell.py - StemCellTerminal.to_dict() for saved data
