# Request Confirmation Networks – Implementation & Demos

Quick summary and background for CIMC. This will be moved after initial review (regular README further
down in this document).

## Intro/Background
This repo is my one-week exploration of Request Confirmation Networks (ReCoNs), a cognitive architecture proposed for hierarchical, sequential control and feature detection.

## What I built

Core Engine: a modular ReCoN class with node types, message passing, and logging — able to run arbitrary ReCoN graphs.

Abstract Demo: a “toy” ReCoN animation that shows requests, confirmations, POR/RET gating and the message dynamics.

Chess Endgame Demo: a heuristic solver for King + Rook vs King built as a ReCoN graph, demonstrating sequential scripts and terminal conditions without external chess engines. Obviously this is nowhere near as good as actual chess engines 
at playing chess, but I thought it was a good example to try ReCoN on. Furthermore it would be quite straight forward to "hook up" a SOTA chess engine (like Stockfish) to my ReCoN implementation; the heuristics subtree boils down to recommend a "next move" to its parent node. One could replace this whole subtree with one node (a wrapper around an API call to e.g. Stockfish) and leave the rest of the network untouched. Terminal nodes would still communicate with the game environment and other parts would facilitate updating the board state. Herein lies one of the strengths of ReCoN. 

## What I learned

ReCoN is an orchestrator, not a learner. It’s a representational / control fabric, not a monolithic ML model. Nodes can be heuristics, detectors, or neural nets; ReCoN provides the sequencing and decision layer. I love abstractions and ReCoN's strength in my view is that it *is* very abstract; the nodes can be anything (as long as they adhere to the rules of the interface: states and connections).

### Strengths:

- Compositional:  mix symbolic and learned nodes.

- Transparent:  each node has local state, making the reasoning traceable.

- Neuro-inspired: resembles how brains coordinate distributed modules.

### Scope for growth:

Plug in vision nodes (CNNs, image parsers) for board recognition.

Replace heuristics with a chess engine or learned policies.

Explore dynamic/learned graph structure beyond hand-crafted scripts.

## Outlook

ReCoNs strike me as a step toward more brain-like AI architectures: not end-to-end gradient blobs, but modular systems where perception, action, and reasoning interlock through structured control. Even in this small project, I’ve seen how they can serve as an “executive” layer above raw learning; a promising direction for building AI systems that are both more interpretable and more general.
Specifically for this project I'd want to use a SOTA chess engine (as mentioned above) instead of my heuristic approach. Furthermore I would like to add terminal nodes (sensors) that could encode different representations of chessboards (API to websites, image recognnition of computer 2D-chessboards or photos or even live video of a real chess board; making it possible to play vs human opponents via a robot). This can quite easily be added by adding the terminal nodes (implemented reasonably e.g. via CNNs or similar) and maybe a selector script node as a parent that automatically identifies which sensor is active. 





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

To view it, start a server (e.g)

```bash
uv run python -m http.server 8000
```

And open http://localhost:8000/visualization/chessboard_view.html

## Some comments and outlook:

This solver is far from perfect; in fact it's quite mediocre as a chess engine. I think it serves as a good demo of how 
to set up a ReCoN network though. It could easily be expanded/improved in several ways (that is out of scope for now as I am running out of time):
- Terminal Nodes could added:
    - a small CNN or MLP to identify chess positions from pngs (or other images)
    - a more sophisticated image recognition (CNN or other) to identify positions from photos or videos of real chessboards
- Use a true chess engine:
    - instead of using the "phases" and heuristics above an API call to e.g. stockfish could be wrapped into one node and connected to the ReCoN. This would report back with suggested move. So the rest of the graph could stay the same, as long as the new node adheres to the interface spec. 

## Other demos and file structure

## Architecture

See `ARCHITECTURE.md` for detailed design decisions and implementation rationale.
