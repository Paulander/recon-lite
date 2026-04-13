# ReCoN Lite

ReCoN Lite is a dependency-light Request-Confirmation Network core for building
interpretable agent control loops. It provides graph primitives, a pragmatic
discrete executor, an explicit formal message-passing executor, optional
continuous activation settling, lightweight tracing, and small examples that
can export visualization-ready JSON.

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
- `LinkType.SUR` is the reverse hierarchy channel: child-to-parent wait,
  confirm, or fail.
- `LinkType.POR` expresses temporal order: a predecessor inhibits a successor
  request until the predecessor is true enough for the successor to start.
- `LinkType.RET` is the reverse sequence channel: a successor inhibits a
  predecessor confirmation so only the final sequence element confirms upward.
- `Graph.add_hierarchy_pair(parent, child)` creates the paired `SUB`/`SUR`
  links expected by the formal executor.
- `Graph.add_sequence_pair(predecessor, successor)` creates the paired
  `POR`/`RET` links expected by the formal executor.
- `EngineConfig(activation_mode=ActivationMode.CONTINUOUS)` enables activation
  settling between discrete ticks for diagnostics and visualization.

## Executor Modes

`ReConEngine` is the pragmatic legacy/high-level executor. It keeps a small API
for examples and existing projects, uses `POR` for predecessor gating, and can
run optional continuous activation microticks. Its confirmation logic is state
based rather than a literal edge-message interpreter.

`FormalReConEngine` is the article-style symbolic executor. It explicitly emits
`request`, `wait`, `confirm`, `fail`, `inhibit_request`, and
`inhibit_confirm` messages over `SUB`, `SUR`, `POR`, and `RET` links. Each tick
uses a two-phase update: emit messages from all node states, group incoming
messages by target, then compute all next states simultaneously.

The current continuous activation/microtick implementation is useful for
diagnostics and visualization, but it is not yet the compact neural
implementation described later in the ReCoN paper.

## Grid-World Example

The grid-world example uses `ReConEngine`, not `FormalReConEngine`, because it
is meant to teach the smallest practical agent loop first.

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

## Formal Trace Example

Generate an article-style symbolic message trace:

```bash
uv run python -m recon_lite.examples.formal_trace --trace-json traces/formal.json
```

Open `examples/formal_trace_viewer.html` in a browser and load the generated
JSON file. The viewer is a static HTML file with no build step or runtime
dependencies. If your browser blocks `file://` pages, especially from WSL, run
it through a tiny local server:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/examples/formal_trace_viewer.html`.

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
