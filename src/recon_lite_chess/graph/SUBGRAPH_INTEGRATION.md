# Subgraph Integration Guide

## Overview
When integrating a new endgame or tactical subgraph into the unified ReCoN graph, you MUST use **factory functions/classes with predicates**, not placeholder `Node()` calls.

## The Problem
This creates a **dead node** that auto-confirms without behavior:
```python
# ❌ WRONG - Placeholder without predicate
node = Node("kqk_move_selector", NodeType.TERMINAL, meta={...})
```

This creates a **working node** with actual logic:
```python
# ✅ CORRECT - Factory with predicate
from recon_lite_chess.scripts.kqk import create_kqk_move_selector
g.add_node(create_kqk_move_selector("kqk_move_selector"))
```

## Checklist for New Subgraphs

1. **Define factory functions** in `src/recon_lite_chess/scripts/{subgraph}.py`:
   ```python
   def create_{subgraph}_move_selector(nid: str) -> Node:
       def _predicate(node: Node, env: Dict[str, Any]):
           # Actual logic here
           env.setdefault("{subgraph}", {}).setdefault("policy", {})["suggested_move"] = move_uci
           return True, True
       return Node(nid=nid, ntype=NodeType.TERMINAL, predicate=_predicate)
   ```

2. **Import factories** in `unified_builder.py`:
   ```python
   from recon_lite_chess.scripts.{subgraph} import (
       create_{subgraph}_material_detector,
       create_{subgraph}_move_selector,
       # ... all terminal factories
   )
   ```

3. **Use factories** in `_integrate_{subgraph}_subgraph()`:
   ```python
   g.add_node(create_{subgraph}_move_selector(f"{prefix}move_selector"))
   ```

4. **Populate `env["{subgraph}"]["policy"]["suggested_move"]`** in your move selector predicate.

5. **Check `full_game_train.py`** line ~427: ensure your subgraph key is in the lookup:
   ```python
   for key in ("kqk", "krk", "kpk", "{your_subgraph}"):
       pol = env.get(key, {}).get("policy")
       ...
   ```

## Existing Subgraph Factories

| Subgraph | Factory Module | Pattern |
|----------|----------------|---------|
| KQK | `scripts/kqk.py` | `create_kqk_*()` functions |
| KPK | `scripts/kpk.py` | `create_kpk_*()` functions |
| KRK | `krk_nodes.py` | `ClassBasedDetector(nid)` classes |

## Testing
After integration, verify with:
```bash
wsl bash -l -c "uv run python demos/persistent/full_game_train.py --batch 5 --debug-draws"
```
Check that `suggested_move` is populated in the logs.
