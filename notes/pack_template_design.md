# Pack Template Design Notes

## Article Constraints (Bach 2015, lines 120-123)

From Article.md:

> "Script nodes can be the source and target of links of all types, whereas **terminal nodes can only be targeted by links of type sub, and be the origin of links of type sur**."

### Link Type Matrix

| Node Type | Can ORIGINATE | Can be TARGETED by |
|-----------|--------------|-------------------|
| SCRIPT | SUB, SUR, POR, RET | SUB, SUR, POR, RET |
| TERMINAL | **SUR only** | **SUB only** |

### Implications for Packs

1. **POR edges can only connect SCRIPT nodes**
2. **TERMINAL nodes (sensors/actuators) must have SCRIPT parents**
3. **Temporal sequencing requires SCRIPT wrappers**

---

## Pack Architecture Pattern

### Wrong (POR → TERMINAL)
```
detect (TERMINAL) --POR--> execute (SCRIPT)  <- ILLEGAL!
```

### Correct (SCRIPT wrapper with TERMINAL child)
```
detect_phase (SCRIPT) --POR--> execute_phase (SCRIPT)
     |                              |
     +-SUB-> detect_sensor         +-SUB-> actuator
            (TERMINAL)                    (TERMINAL)
```

---

## Pack Types Implemented

### 1. AND-Gate Pack (spawn_and_gate_pack)
```
parent (SCRIPT)
   |
   +-SUB-> gate (SCRIPT, aggregation="min")
              |
              +-SUB-> cond_0 (TERMINAL)
              +-SUB-> cond_1 (TERMINAL)
              +-SUB-> actuator (TERMINAL)
```
Logic: Gate confirms only when ALL condition children are True (min aggregation).

### 2. OR-Gate Pack (spawn_or_gate_pack)
```
parent (SCRIPT)
   |
   +-SUB-> gate (SCRIPT, aggregation="max")
              |
              +-SUB-> cond_0 (TERMINAL)
              +-SUB-> cond_1 (TERMINAL)
              +-SUB-> actuator (TERMINAL)
```
Logic: Gate confirms when ANY condition child is True (max aggregation).

### 3. Goal Delegation Pack (spawn_goal_delegation_pack)
```
parent (SCRIPT)
   |
   +-SUB-> pack_root (SCRIPT)
              |
              +-SUB-> detect (SCRIPT) --POR--> execute (SCRIPT) --POR--> finish (SCRIPT) --POR--> wait (SCRIPT)
              |          |                       |                         |                         |
              |          +-SUB-> sensor          +-SUB-> actuator          +-SUB-> sentinel          +-SUB-> wait_sensor
              |               (TERMINAL)               (TERMINAL)               (TERMINAL)                  (TERMINAL)
```
Logic: Sequential execution via POR chain of SCRIPT phases, each with TERMINAL children.

---

## Bugs Fixed (2026-01-09)

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| registry.topology_path not found | Wrong attribute name | Use registry.path |
| TRIAL node as parent_id | TRIAL nodes are TERMINAL | Use kpk_execute (SCRIPT) |
| POR to TERMINAL edges | Article constraint violation | Use SCRIPT wrappers |
| Lottery never called | Win rate > 20% | Added GROWTH_MODE (30% on success) |

---

## Growth Mode

When win rate >= 80%, we still want to build structure. The GROWTH_MODE triggers pack spawning with 30% probability per cycle even when "winning" to continue building topology depth.

```python
GROWTH_MODE_CHANCE = 0.30  # Spawn packs even when winning
if current_win_rate >= 0.80 and random() < GROWTH_MODE_CHANCE:
    # Trigger lottery for pack spawning
```

---

## Key Files

- pack_template.py - Pack spawning functions
- stem_cell.py - spawn_with_lottery() 
- m5_structure.py - Growth mode trigger
- graph.py - Link type constraints
