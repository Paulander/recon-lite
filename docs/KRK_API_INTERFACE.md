# KRK Subgraph API Interface Template

This document defines the **API Interface Template** that all KRK subgraph implementations must follow, regardless of their internal structure (phase0-4 vs legs).

## Interface Contract

All KRK subgraphs must:

1. **Root Node**: Always named `krk_root` (SCRIPT type)
2. **Move Output**: Write moves to `env["krk_root"]["policy"]["suggested_move"]`
3. **Legacy Support**: Also write to `env["chosen_move"]` for backward compatibility
4. **Subgraph Locking**: Support `engine.lock_subgraph("krk_root", sentinel_fn)`

## Implementation Variants

### Variant 1: Phase0-4 Topology (Original/Heuristic)
- Structure: `krk_root` → `krk_phase0_establish_cut` → ... → `krk_phase4_deliver_mate`
- Move writers: `krk_choose_phase0`, `krk_mate_moves`, etc.
- All use `_set_suggested_move(env, mv)` which writes to both paths

### Variant 2: Legs Topology (M5 Evolution)
- Structure: `krk_root` → `krk_detect` → `krk_execute` → `krk_finish` → `krk_wait`
- Move writer: `krk_arbiter` (uses `create_phase0_choose_moves` factory)
- Must use `_set_suggested_move(env, mv)` to match API

## Required Function: `_set_suggested_move()`

```python
def _set_suggested_move(env: Dict[str, Any], mv: str) -> None:
    """
    Set move in standard ReCoN interface paths.
    
    Engine expects: env["<root>"]["policy"]["suggested_move"]
    This ensures KRK actuators work with the standard game loop.
    """
    env["chosen_move"] = mv  # Legacy path
    # Standard interface (matches KPK/KQK pattern)
    env.setdefault("krk_root", {}).setdefault("policy", {})["suggested_move"] = mv
```

## External Code Compatibility

All external code (full_game_train.py, curriculum scripts, tests) should read from:

```python
# Primary path (standard)
suggested = env.get("krk_root", {}).get("policy", {}).get("suggested_move")

# Fallback (legacy)
if not suggested:
    suggested = env.get("chosen_move")
```

## Interchangeability

Both topologies are **interchangeable** - they can be swapped without changing external code:

- `build_unified_graph()` → uses phase0-4 structure
- `build_graph_from_topology("krk_legs_topology.json")` → uses legs structure
- Both write to same env paths
- Both support same locking mechanism
- Both work with same consolidation/plasticity code

This is like different implementations of the same abstract class - same interface, different internal structure.

