# ReCoN-lite (Request–Confirmation Network) — Minimal Scaffold

A small, dependency-light Python implementation of a ReCoN-style executor with structured logging.

## Features

- ✅ **Complete ReCoN Foundation**: Core graph, engine, and execution logic
- ✅ **Chess Integration**: King+Rook vs King endgame solver
- ✅ **Visualization**: Structured logging for replay and debugging
- ✅ **Extensible**: Plugin system for domain-specific nodes


## State of the project and quick overview
See `ARCHITECTURE.md` for a more complete description of the project. 
On a top level we have the following hierarchy: 

```
├── src                   # The ReCoN implementation
   ├── recon_lite         # Core ReCoN data structures and graph management
   └── recon_lite_chess   # Chess specific ReCoN data structures and helper functions
└──demos                 # Several example implementations of ReCoN networks. 
    └──visualization      # Html visualizations of ReCoN outputs. (Run scripts to produce json)

```

## Install (uv)
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv run python -m demos.sequence_demo
```

## Chess Demo

Try the King+Rook vs King checkmate solver:

```bash
uv run python demos/gameplay/krk_play_demo.py
```

This demonstrates the hierarchical KRK strategy:
- **Phase 1**: Drive enemy king to edge \*
- **Phase 2**: Shrink the safe "box"
- **Phase 3**: Take opposition
- **Phase 4**: Deliver checkmate

Some comments and outlook:
This solver is far from perfect; in fact it's quite mediocre as a chess engine. I think it serves as a good demo of how 
to set up a ReCoN network though. It could easily be expanded/improved in several ways (that is out of scope for now as I am running out of time):
    - Terminal Nodes could added:
        -- a small CNN or MLP to identify chess positions from pngs (or other images)
        -- a more sophisticated image recognition (CNN or other) to identify positions from photos or videos of real chessboards
    - Use a true chess engine:
      -- instead of using the "phases" and heuristics above an API call to e.g. stockfish could be wrapped into one node and connected to the ReCoN. This would report back with suggested move. So the rest of the graph could stay the same, as long as the new node adheres to the interface spec. 

## Other demos and file structure

## Architecture

See `ARCHITECTURE.md` for detailed design decisions and implementation rationale.
