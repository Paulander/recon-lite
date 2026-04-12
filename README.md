# ReCoN Lite

ReCoN Lite is a dependency-light Request-Confirmation Network core for building
interpretable agent control loops. It provides graph primitives, a discrete
executor, optional continuous activation settling, lightweight tracing, and a
small grid-world example that can export visualization-ready JSON.

The package is intentionally domain-free. Projects can add their own sensors,
actuators, world models, and visualizers without bringing those dependencies
into the core library.

## Install

ReCoN Lite assumes `uv` for local development and examples.

<details>
<summary>Beginner note: what is uv?</summary>

`uv` is a fast Python project tool. It installs Python packages, creates local
virtual environments, runs commands inside the right environment, and keeps
dependency locks reproducible. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal and check:

```bash
uv --version
```

</details>

From a checkout:

```bash
uv run --with-editable . python -c "import recon_lite; print(recon_lite.__version__)"
```

From another local project during development:

```bash
uv add --editable /path/to/recon-lite
```

## Quickstart

```python
from recon_lite import Graph, LinkType, Node, NodeState, NodeType, ReConEngine


def terminal(_node, _env):
    return True, True


graph = Graph()
graph.add_node(Node("root", NodeType.SCRIPT))
graph.add_node(Node("sensor", NodeType.TERMINAL, predicate=terminal))
graph.add_edge("root", "sensor", LinkType.SUB)

engine = ReConEngine(graph)
graph.nodes["root"].state = NodeState.REQUESTED

for _ in range(4):
    engine.step({})

assert graph.nodes["sensor"].state == NodeState.CONFIRMED
```

## Core Concepts

- `Graph` stores nodes and typed edges.
- `NodeType.SCRIPT` nodes coordinate requests and confirmations.
- `NodeType.TERMINAL` nodes call predicates that connect the graph to a world,
  tool, model, or sensor.
- `LinkType.SUB` expresses hierarchy: a parent requests a child.
- `LinkType.POR` expresses temporal order: a successor waits for a predecessor  to confirm.
- `LinkType.SUR` child confirms upward to parent.
- `LinkType.RET` RET successor returns/blocks predecessor.
- `ReConEngine.step(env)` advances the network one discrete tick.
- `EngineConfig(activation_mode=ActivationMode.CONTINUOUS)` enables activation
  settling between discrete ticks for diagnostics and visualization.

Important notes/TODO:
SUR implementation: Partial. It is represented and allowed by the graph, but engine confirmation is mostly implicit via child node states, not actual SUR message propagation.
RET	successor returns/blocks predecessor	Current support: Mostly data-model only. It exists in LinkType and validation, but the engine does not really execute RET semantics.

These will be adressed ASAP.

## Grid-World Example

Run the simple agent:

```bash
uv run python -m recon_lite.examples.gridworld --mode discrete --steps 3
```

Ask for the explanatory output:

```bash
uv run python -m recon_lite.examples.gridworld --mode continuous --steps 3 --microticks 2 --explain
```

Export a trace for visualization:

```bash
uv run python -m recon_lite.examples.gridworld --mode continuous --steps 3 --microticks 2 --trace-json traces/gridworld.json
```

The trace JSON contains run metadata, graph structure, visible world steps,
per-tick node states, activation values, binding snapshots, and continuous
activation history when microticks are enabled.

See `src/recon_lite/examples/README.md` for the walkthrough.

See `docs/architecture.md` for the package architecture and observability
surface.

## Development

Run the isolated core tests:

```bash
uv run --with pytest --with-editable . pytest
```

Check the lockfile:

```bash
uv lock --check
```

Build the package:

```bash
uv build
```

The core package has no runtime dependencies. Test tooling is optional and
installed only for development.
