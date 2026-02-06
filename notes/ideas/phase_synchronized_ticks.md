# Phase-Synchronized Ticks ("Fire Together, Wire Together")

**Date**: 2025-12-29  
**Context**: Discovered during bridge training analysis - deep subgraphs require multiple engine ticks to propagate.

## Core Idea

"Neurons that fire together, wire together" - distant brain regions synchronize via oscillation phase-locking (gamma ~40Hz). Apply this to ReCoN: nodes with matching `tick_phase` evaluate together, enabling deep hierarchies to "resonate" in sync.

## Current Problem

Engine propagates 1 hierarchy level per tick:
- GameRoot → subgraph roots (tick 1)
- roots → internal scripts (tick 2)  
- scripts → terminals (tick 3)
- terminals execute predicates (tick 4+)

Deep paths (KPK: 4 levels, tactics: 3 levels) need N ticks before actuators fire.

## Proposed Mechanism

### Option A: Phase Tags
```python
# Each node has a phase (0 to N-1)
node.meta["tick_phase"] = 2  

# Engine cycles through phases
def step(self, env):
    self.current_phase = (self.current_phase + 1) % self.total_phases
    for nid, node in self.g.nodes.items():
        if node.meta.get("tick_phase", 0) == self.current_phase:
            self._process_node(node, env)
```

- Local/fast paths: phase 0-1 (tactics detection)
- Deep/strategic paths: phase 2-4 (endgame subgraphs)
- One "logical move" = one full phase cycle

### Option B: Per-Subgraph Tick Batching
When subgraph root is activated, run internal ticks until actuator is ready:

```python
def _activate_subgraph(self, root_id: str, env):
    # Collect all nodes in subgraph
    subgraph_nodes = self._get_subgraph_nodes(root_id)
    
    # Run ticks internally until done
    for _ in range(MAX_INTERNAL_TICKS):
        changed = self._step_nodes(subgraph_nodes, env)
        if self._subgraph_actuator_ready(root_id):
            break
```

### Option C: Priority Queues
Nodes have `urgency` score. Engine processes high-urgency first, depth-first within subtree:

```python
def step(self, env):
    queue = PriorityQueue()
    for nid in self._get_requested():
        queue.put((node.meta.get("urgency", 1.0), nid))
    
    while not queue.empty():
        _, nid = queue.get()
        self._process_node(nid, env)
        for child in self.g.children(nid):
            queue.put((child_urgency, child))
```

## Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Phase Tags | Elegant, brain-inspired, configurable | Adds complexity, needs phase assignment algorithm |
| Subgraph Batching | Local, minimal change | Doesn't generalize to non-subgraph cases |
| Priority Queue | Flexible, urgent paths complete first | Breaks temporal guarantees |

## Relation to Microticks

Current microticks (`time/microtick.py`) are for continuous activation settling within a tick - analog smoothing. This proposal is for discrete hierarchical propagation - different purpose.

Could combine: microticks for activation values, phase-ticks for request/confirm propagation.

## Next Steps

1. Implement Option B (subgraph batching) as immediate fix - minimal invasive
2. Prototype Option A on a branch to evaluate complexity
3. Consider formalizing in ReCoN spec if Option A proves valuable

## References

- Gamma oscillations: https://en.wikipedia.org/wiki/Gamma_wave
- Neural synchrony: Engel et al., "Dynamic predictions: oscillations and synchrony in top-down processing"
- Original insight from user comment during bridge training analysis
