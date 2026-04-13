# Architecture

ReCoN Lite keeps the reusable Request-Confirmation Network machinery separate
from domain projects.

## Public Surface

- `Graph`, `Node`, `Edge`, `NodeType`, `NodeState`, and `LinkType` define the
  graph model.
- `ReConEngine` executes pragmatic high-level network ticks for examples and
  applications.
- `FormalReConEngine` executes explicit symbolic SUB/SUR/POR/RET message
  passing.
- `EngineConfig` and `ActivationMode` configure discrete or continuous
  activation behavior.
- `BindingTable` and `BindingInstance` track explicit feature-to-object
  reservations across environment states.
- `RunLogger`, `TraceDB`, and `Graph.to_snapshot()` provide lightweight
  observability hooks without requiring a visualization dependency.
- `recon_lite.examples.gridworld` demonstrates the smallest complete
  sensor-actuator loop and optional trace export.

## Execution Model

`SUB` edges request work down the hierarchy. `SUR` is the corresponding
confirmation direction. `POR` edges encode temporal prerequisites, and `RET`
is the corresponding return direction.

`ReConEngine` is the pragmatic executor. It uses the graph structure for
request propagation and predecessor gating, and it observes child node states to
confirm scripts. This keeps the learning path small and preserves compatibility
with earlier examples.

`FormalReConEngine` is the article-style executor. It expects paired `SUB`/`SUR`
hierarchy links and paired `POR`/`RET` sequence links. Every tick emits explicit
messages from the tick-start node states, groups them by target, and applies all
state updates simultaneously. This covers the Bach/Herger message-passing
definition with symbolic states and messages.

The compact neural/activation equations discussed later in the ReCoN paper are
future work. The current continuous activation microticks are useful for traces,
diagnostics, and visual intuition, but they are not claimed to be a full
implementation of that compact neural formulation.

Terminal predicates are the boundary between the graph and an environment. They
may read sensors, call tools, update node metadata, or mutate a simulated world.
Script nodes remain coordination nodes.

## Subgraph Lock Extension

`ReConEngine.lock_subgraph()` is an implementation extension, not pure ReCoN in
the Bach and Herger 2015 formalism. In a strict ReCoN model, the same effect
should be expressed with normal graph structure and `POR` logic.

The extension exists as a practical way to zoom in on a subgraph's timescale:
the engine temporarily ticks only that subgraph until it finishes, produces
domain-level output, reaches a tick limit, or its sentinel exits. This is useful
when a requested subgraph needs several internal ticks before returning control
to its caller. Projects that need formal purity should avoid this helper or
treat it as an executor optimization layered on top of the ReCoN graph.

## Observability

Interpretable state is part of the core design. Applications should be able to
record:

- graph structure
- node states per tick
- newly requested nodes
- activation values
- binding-table snapshots
- microtick history in continuous mode

The grid-world trace JSON is the reference shape for small examples. Larger
applications may emit richer domain payloads alongside the same core fields.
