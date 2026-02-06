# Unified Plan: Pre-trained ReCoN + Runtime Stem Cell Growth

## Vision Summary

**Pre-trained foundation with runtime adaptation:**
- **Offline**: Baseline learning discovers sensor/actuator patterns
- **Compile**: Build proper ReCoN graph (SCRIPT hierarchy, Article.md compliant)
- **Runtime**: Fast plasticity (weights) + stem cell exploration (topology)
- **Recursive**: Each leg spawns children independently following ReCoN rules

---

## Architecture

### Pre-trained Graph Structure (CORRECTED)

```
KRK_entry (SCRIPT, root)
├─ SUB → KRK_Hub (SCRIPT, bandit selector)
│   ├─ SUB → Leg_0 (SCRIPT, actuator pattern 0)
│   │   ├─ SUB → Precondition_Gate (SCRIPT)
│   │   │   ├─ SUB → sensor_3 (TERMINAL)
│   │   │   └─ SUB → sensor_15 (TERMINAL)
│   │   ├─ POR → actuator_0 (TERMINAL, move selector)
│   │   └─ POR → Postcondition_Gate (SCRIPT)
│   │       ├─ SUB → sensor_3_post (TERMINAL)
│   │       └─ SUB → sensor_15_post (TERMINAL)
│   ├─ SUB → Leg_1 (SCRIPT, actuator pattern 1)
│   ├─ SUB → Leg_2 (SCRIPT, actuator pattern 2)
│   └─ SUB → Leg_3, Leg_4... (all parallel alternatives)
└─ [stem cell spawn point at root level]
```

**Key properties (CORRECTED):**
- ✅ Legs are **parallel alternatives** under Hub (SUB, not POR)
- ✅ **POR only for sequences** within a Leg (precondition → actuator → postcondition)
- ✅ Actuators are **TERMINAL nodes** (not separate entity type)
- ✅ TERMINAL nodes only receive SUB, send SUR
- ✅ Each Leg is a 3-part micro-script (sensorimotor script pattern)
- ✅ Recursive: Legs can spawn child Legs

### Baseline → Graph Compilation

**Step 1: Learned patterns → Node specs (CORRECTED)**

```python
# From baseline learning:
mature_sensors = [s for s in sensors if s.xp >= 0.7]  # 28 sensors
actuators = [...]  # 5 actuator patterns

# Create TERMINAL specs (sensors + actuators):
for sensor in mature_sensors:
    node_spec = {
        "id": f"sensor_{sensor.id}",  # Stable ID
        "type": "TERMINAL",
        "predicate": create_sensor_predicate(sensor.spec),
        "meta": {
            "readout_type": sensor.spec.readout_type,
            "feature_mask_keys": sensor.spec.feature_keys,  # NOT indices
            "baseline_xp": sensor.xp,
            "origin": "baseline",
            "stage": 0
        }
    }

for actuator in actuators:
    # Actuator is also a TERMINAL
    actuator_spec = {
        "id": f"actuator_{actuator.id}",  # Stable ID
        "type": "TERMINAL",
        "predicate": create_actuator_predicate(actuator.spec),
        "meta": {
            "targets": [f"sensor_{i}" for i in actuator.sensor_indices],  # Stable IDs
            "goal_delta": {  # Keyed by stable IDs
                f"sensor_{i}": float(delta)
                for i, delta in zip(actuator.sensor_indices, actuator.goal_delta)
            },
            "baseline_xp": actuator.xp,
            "origin": "baseline",
            "stage": 0
        }
    }
    
    # Leg is a 3-part micro-script
    leg_spec = {
        "id": f"leg_{actuator.id}",
        "type": "SCRIPT",
        "children": [
            {"id": f"precond_{actuator.id}", "type": "SCRIPT"},
            {"id": f"actuator_{actuator.id}", "type": "TERMINAL"},
            {"id": f"postcond_{actuator.id}", "type": "SCRIPT"}
        ]
    }
```

**Step 2: Wire into hierarchy (CORRECTED)**

```python
graph = Graph()

# Root with blackboard/binding cache
root = create_script_node("krk_entry")
root.meta["blackboard"] = {}  # Cache for features + sensor outputs

# Hub (bandit selector)
hub = create_script_node("krk_hub", bandit_enabled=True)
graph.add_edge(root.id, hub.id, EdgeType.SUB)

# Legs (parallel alternatives, NO POR-chaining)
for actuator in actuators:
    leg = create_3part_leg(actuator, mature_sensors)
    
    # All Legs are parallel alternatives under Hub
    graph.add_edge(hub.id, leg.id, EdgeType.SUB)
    # NO POR between Legs - they compete via bandit

def create_3part_leg(actuator, sensors):
    """Create precondition → actuator → postcondition micro-script"""
    leg = create_script_node(f"leg_{actuator.id}")
    
    # Part 1: Precondition gate
    precond = create_script_node(f"precond_{actuator.id}", aggregation="and")
    for sensor_id in actuator.targets:
        sensor = create_sensor_terminal(sensor_id, sensors)
        graph.add_edge(precond.id, sensor.id, EdgeType.SUB)
    graph.add_edge(leg.id, precond.id, EdgeType.SUB)
    
    # Part 2: Actuator terminal
    actuator_node = create_actuator_terminal(actuator)
    graph.add_edge(leg.id, actuator_node.id, EdgeType.SUB)
    graph.add_edge(precond.id, actuator_node.id, EdgeType.POR)  # Sequence
    
    # Part 3: Postcondition gate (verify Δs happened)
    postcond = create_script_node(f"postcond_{actuator.id}", aggregation="and")
    for sensor_id in actuator.targets:
        sensor_post = create_sensor_terminal(f"{sensor_id}_post", sensors)
        graph.add_edge(postcond.id, sensor_post.id, EdgeType.SUB)
    graph.add_edge(leg.id, postcond.id, EdgeType.SUB)
    graph.add_edge(actuator_node.id, postcond.id, EdgeType.POR)  # Sequence
    
    return leg
```

### Runtime Stem Cell Integration (CORRECTED)

**Spawn points create micro-scripts, not isolated sensors:**

```python
class StemCellSpawnPoint:
    """Attached to SCRIPT nodes for runtime growth"""
    parent_script_id: str
    candidate_sensors: List[SensorSpec]  # From baseline (XP < 0.7)
    candidate_actuators: List[ActuatorSpec]  # Weak patterns
    spawn_budget: int = 5
    
    def spawn_child_micro_script(self, trigger_condition):
        """Spawn complete 3-part micro-script (not isolated sensor)"""
        # Select best candidate pattern
        sensor = max(self.candidate_sensors, key=lambda s: s.xp)
        actuator = self._create_actuator_from_sensor(sensor)
        
        # Create 3-part micro-script
        child_leg = create_script_node(f"{parent_script_id}_child_{n}")
        
        # Part 1: Precondition (new sensor)
        precond = create_script_node(f"{child_leg.id}_precond")
        sensor_node = create_sensor_terminal(f"sensor_{sensor.id}_inst_{n}", sensor.spec)
        graph.add_edge(precond.id, sensor_node.id, EdgeType.SUB)
        graph.add_edge(child_leg.id, precond.id, EdgeType.SUB)
        
        # Part 2: Actuator (new move selector)
        actuator_node = create_actuator_terminal(f"actuator_{n}", actuator.spec)
        graph.add_edge(child_leg.id, actuator_node.id, EdgeType.SUB)
        graph.add_edge(precond.id, actuator_node.id, EdgeType.POR)
        
        # Part 3: Postcondition (verify)
        postcond = create_script_node(f"{child_leg.id}_postcond")
        sensor_post = create_sensor_terminal(f"{sensor_node.id}_post", sensor.spec)
        graph.add_edge(postcond.id, sensor_post.id, EdgeType.SUB)
        graph.add_edge(child_leg.id, postcond.id, EdgeType.SUB)
        graph.add_edge(actuator_node.id, postcond.id, EdgeType.POR)
        
        # Wire to parent
        graph.add_edge(parent_script_id, child_leg.id, EdgeType.SUB)
        
        # Child can spawn its own children (recursive)
        child_leg.meta["stem_cell_spawn_point"] = StemCellSpawnPoint(...)
        
        return child_leg
```

**Spawn triggers:**
- **Affordance spike**: High reward moment during execution
- **Failure recovery**: Leg fails, spawn alternative
- **Exploration budget**: Periodic spawning for diversity

---

## ReCoN Compliance (Article.md)

### Node Types & States

**SCRIPT nodes:**
- States: `inactive, requested, active, suppressed, waiting, true, confirmed, failed`
- Can have: SUB (to children), SUR (from children), POR (to successor), RET (from predecessor)

**TERMINAL nodes:**
- States: `inactive, active, confirmed`
- Can have: SUB (from parent), SUR (to parent)
- **Cannot have POR/RET** (Article constraint)

### Execution Flow

**Request (top-down):**
1. Root requested → sends SUB to Hub
2. Hub requested → bandit selects Leg → sends SUB
3. Leg requested → sends SUB to all children (sensors + actuator)
4. Terminals activate → perform measurements

**Confirmation (bottom-up):**
1. Sensors measure → send SUR to Leg
2. Actuator selects move → sends SUR to Leg
3. Leg aggregates → sends SUR to Hub
4. Hub confirms → sends SUR to Root

**POR sequencing (Hub → Legs):**
- Leg_0 active → sends POR inhibit to Leg_1
- Leg_0 confirms → releases inhibit
- Leg_1 becomes active

---

## Learning Mechanisms

### 1. Pre-training (Offline)

**Baseline learning:**
- Input: KRK mate-in-1 positions
- Output: 40 sensor specs, 5 actuator specs
- Method: Δt computation, XP metric, pattern extraction

**Graph compilation:**
- Convert specs → ReCoN nodes
- Wire hierarchy (Root → Hub → Legs)
- Save as `krk_entry_topology.json`

### 2. Fast Plasticity (Runtime, M3)

**Per-tick weight updates:**
- Eligibility traces on SUB/POR edges
- Δw ∝ reward × trace
- Whitelisted edges only (not all)

### 3. Stem Cell Exploration (Runtime, M5-lite)

**Simplified from current M5:**
- No EXPLORING/CANDIDATE/TRIAL states
- Just: **spawn point** → **spawn** → **evaluate** → **keep or prune**

**Lifecycle:**
```
Spawn Point (attached to SCRIPT)
    ↓ (trigger: affordance spike)
Child Leg spawned
    ↓ (evaluate over N games)
XP > threshold → Keep (solidify)
XP < threshold → Prune (remove)
```

**XP system:**
- +10 on success (mate delivered)
- -10 on failure (no mate)
- -1 decay per cycle
- Threshold: 50 to keep

---

## Implementation Phases

### Phase 1: Graph Compilation (Week 1)

**Files:**
- `src/recon_lite_chess/krk_baseline_graph.py` - Compilation logic
- `topologies/krk_entry_topology.json` - Pre-trained graph

**Tasks:**
- [ ] Convert baseline sensors → TERMINAL node specs
- [ ] Convert baseline actuators → Leg SCRIPT specs
- [ ] Build Root → Hub → Legs hierarchy
- [ ] Add SUB/SUR/POR/RET edges (Article compliant)
- [ ] Save topology JSON

### Phase 2: Runtime Execution (Week 2)

**Files:**
- `src/recon_lite_chess/krk_features.py` - Feature extraction
- `src/recon_lite_chess/krk_actuators.py` - Move selection

**Tasks:**
- [ ] Hook feature extraction into Root predicate
- [ ] Implement sensor predicates (read features → apply readout)
- [ ] Implement actuator predicates (score moves → select best)
- [ ] Test: KRK_entry plays mate-in-1 (>90% win rate)

### Phase 3: Stem Cell Spawn Points (Week 3)

**Files:**
- `src/recon_lite/nodes/spawn_point.py` - Spawn point logic
- `src/recon_lite/learning/runtime_growth.py` - Growth triggers

**Tasks:**
- [ ] Attach spawn points to Leg nodes
- [ ] Implement spawn triggers (affordance spike)
- [ ] Spawn child Legs at runtime
- [ ] XP evaluation and pruning
- [ ] Test: Child Legs spawn and solidify

### Phase 4: Full Integration (Week 4)

**Files:**
- `scripts/krk_entry_training.py` - Training loop
- `demos/visualization/krk_entry_viz.html` - Visualization

**Tasks:**
- [ ] Integrate with full-game affordance selection
- [ ] Bandit selection at Hub
- [ ] Visualization of runtime growth
- [ ] End-to-end testing

---

## Key Design Decisions

### 1. Pre-trained vs Runtime

| Component | Pre-trained | Runtime |
|-----------|-------------|---------|
| Mature sensors (XP >= 0.7) | ✅ Compiled into graph | ❌ |
| Actuator Legs | ✅ Compiled into graph | ❌ |
| Root → Hub → Legs structure | ✅ Fixed hierarchy | ❌ |
| Candidate sensors (XP < 0.7) | ❌ | ✅ Spawn points |
| Child Legs | ❌ | ✅ Spawned on trigger |
| Edge weights | ✅ Initialized from baseline | ✅ Updated via M3 |

### 2. Stem Cell Simplification

**Old M5 (complex):**
- EXPLORING → CANDIDATE → TRIAL → MATURE states
- Win-coactivation tracking
- AND-gate hoisting
- POR chain discovery

**New M5-lite (simple):**
- Just spawn points + spawn + evaluate
- No complex state machine
- No automatic hoisting (manual pack design)
- Focus on recursive branching

### 3. Recursive Branching

**Each Leg can spawn children:**
```
Leg_0 (pre-trained)
├─ Child_0_0 (spawned at runtime)
│   └─ Child_0_0_0 (spawned recursively)
├─ Child_0_1 (spawned at runtime)
└─ ...
```

**Depth limit:** 5 levels (prevent explosion)

---

## Success Criteria

1. **Compilation**: Baseline → ReCoN graph with proper hierarchy
2. **Execution**: KRK_entry achieves >90% win rate on mate-in-1
3. **Article compliance**: All constraints satisfied (TERMINAL SUB/SUR only, etc.)
4. **Runtime growth**: Stem cells spawn child Legs during execution
5. **Recursive**: Child Legs spawn their own children (depth 3+)
6. **Visualization**: Can view pre-trained + runtime-grown graph

---

## Open Questions

1. **Bandit at Hub**: UCB or softmax? Warmup period?
2. **Spawn trigger threshold**: What affordance delta triggers spawn?
3. **XP decay rate**: -1 per cycle or faster?
4. **Pruning strategy**: Immediate or batch pruning?
5. **Feature caching**: Cache at Root or recompute per sensor?
