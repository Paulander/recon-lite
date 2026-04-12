# Grid-World Example

This example is the smallest ReCoN loop in the standalone package. A toy agent
starts at `(0, 0)` in a 5x5 grid and moves toward a goal at `(4, 4)`.

The path planner is intentionally simple. The useful part is the traceable
network loop around it.

## Network

- `root` is the top-level script node.
- `sense` is a script node that asks the sensor terminal to inspect the world.
- `goal_sensor` is a terminal predicate. It records activation and reserves
  bindings for the current agent and goal cells.
- `act` is a script node that can run after `sense` confirms.
- `move_agent` is a terminal predicate. It mutates the grid by taking one step
  toward the goal.

`SUB` edges form the hierarchy:

```text
root
  sense
    goal_sensor
  act
    move_agent
```

The `sense -> act` `POR` edge encodes order: acting waits until sensing has
confirmed.

## Outer Steps And Engine Ticks

One printed grid step is not the same thing as one ReCoN tick.

Each outer grid step:

1. invalidates stale bindings if the grid-state signature changed
2. resets node states for a fresh request
3. requests `root`
4. advances the engine until `move_agent` confirms
5. prints the visible world state

The engine may need several internal ticks inside one visible step because
requests, predicate execution, script confirmation, and temporal gating happen
as separate state transitions.

## Discrete And Continuous Modes

Discrete mode runs the request/confirmation state machine.

Continuous mode uses the same graph and the same visible agent behavior, but it
also performs activation microticks between discrete engine ticks. The example
records these activation values in the trace so a visualizer can show how the
network settles while the discrete logic runs.

## Bindings

The example reserves two bindings in the `grid/sense` namespace:

- `agent -> cell:x,y`
- `goal -> cell:x,y`

`GridState.signature()` describes the current world state. When the agent moves,
the signature changes and the binding table clears stale reservations before
the next sense step. This is a tiny version of the object-binding problem that
larger applications need to handle explicitly.

## Commands

Compact output:

```bash
uv run python -m recon_lite.examples.gridworld --mode discrete --steps 3
```

Explanatory output:

```bash
uv run python -m recon_lite.examples.gridworld --mode continuous --steps 3 --microticks 2 --explain
```

Trace export:

```bash
uv run python -m recon_lite.examples.gridworld --mode continuous --steps 3 --microticks 2 --trace-json traces/gridworld.json
```

The trace is a single JSON document with:

- `metadata`: mode, seed, requested steps, microticks
- `graph`: nodes and edges
- `steps`: visible grid steps, including world state before and after acting
- `ticks`: per-engine-tick node states, new requests, activations, bindings,
  and optional activation history

This format is deliberately visualizer-neutral. A later UI can render the graph,
timeline, activation curves, and binding table without needing to understand
the grid-world Python code.
