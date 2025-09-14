# ReCoN-lite (Request–Confirmation Network) — Minimal Scaffold

A small, dependency-light Python implementation of a ReCoN-style executor with structured logging.

## Features

- ✅ **Complete ReCoN Foundation**: Core graph, engine, and execution logic
- ✅ **Chess Integration**: King+Rook vs King endgame solver
- ✅ **Visualization**: Structured logging for replay and debugging
- ✅ **Extensible**: Plugin system for domain-specific nodes

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
uv run python demos/krk_checkmate_demo.py
```

This demonstrates the hierarchical KRK strategy:
- **Phase 1**: Drive enemy king to edge
- **Phase 2**: Shrink the safe "box"
- **Phase 3**: Take opposition
- **Phase 4**: Deliver checkmate

## Architecture

See `ARCHITECTURE.md` for detailed design decisions and implementation rationale.
