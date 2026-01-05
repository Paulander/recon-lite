# ReCoN Evolution Training System

## Overview

This system trains chess endgame networks using **M5-Evolution**: an alternating 
Online/Structural training paradigm with stem cell-based feature discovery.

## Key Components

### 1. Evolution Driver (`scripts/evolution_driver.py`)
Main training script. Alternates between:
- **Online Phase**: Play games, apply plasticity, collect stem cell samples
- **Structural Phase**: Promote cells to nodes, prune weak edges

```bash
# Full 8-stage curriculum
uv run python scripts/evolution_driver.py \
    --topology topologies/kpk_legs_topology.json \
    --cycles 20 --games-per-cycle 200 \
    --run-name kpk_full --all-stages
```

### 2. Legs Architecture
KPK uses a **legs** architecture instead of monolithic move selector:

```
kpk_execute
├── kpk_pawn_leg   (proposes pawn moves, calculates activation)
├── kpk_king_leg   (proposes king moves, calculates activation)  
└── kpk_arbiter    (picks winner based on activations)
```

**Critical**: Legs are SCRIPT nodes. ReConEngine.step() only calls TERMINAL 
predicates, so evolution_driver.py calls leg predicates explicitly (line ~325).

### 3. Topologies
- `topologies/kpk_topology.json` - Old monolithic (deprecated)
- `topologies/kpk_legs_topology.json` - New legs architecture ✓

### 4. Curriculum Stages (KPK)
| Stage | Name | Description |
|-------|------|-------------|
| 0 | Sprinter | Pawn on 7th, enemy far |
| 1 | Guardian_E | King escorts pawn on 6th |
| 2-5 | ... | Progressively harder |
| 6-7 | Rook pawn | Edge file with draws |

### 5. Stem Cells (`src/recon_lite/nodes/stem_cell.py`)
- **EXPLORING** → **CANDIDATE** → **TRIAL** → **MATURE**
- XP system: +5 success, -3 failure, -1 decay
- 100 XP = solidification + spawn_neighbors()

### 6. M5 Intelligent Growth
- `spawn_neighbors()` - Success-triggered spawning at 100 XP
- `orphan_sweep()` - 2x decay for cells whose parent was pruned
- `identity_audit()` - Prune if not >20% better than simpler branch

## Output Directories
```
snapshots/evolution/{run_name}/stage{N}/  - Topology snapshots
reports/evolution/{run_name}/stage{N}/    - Summary JSON
traces/evolution/{run_name}/stage{N}/     - Episode traces
```

## Visualization
Open `tools/evolution_visualizer.html` and load a snapshot folder.
- Spheres = SCRIPT nodes
- Diamonds = TERMINAL nodes (sensors/actuators)
- "Show all labels" checkbox for persistent labels

## Common Issues

1. **0% wins on Stage 1+**: Legs not being called. Check evolution_driver.py 
   has explicit leg predicate calls.

2. **No stem cell promotions**: Check `--topology` uses legs architecture.

3. **Draws**: Usually means pawn is blocked but king isn't helping. 
   Legs architecture should handle this.
