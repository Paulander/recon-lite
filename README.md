# Request Confirmation Networks – Implementation & Demos

## Intro/Background
This repo started as my one-week exploration of Request Confirmation Networks (ReCoNs), a cognitive architecture proposed for hierarchical, sequential control and feature detection. I have kept working on it and the current goal is to have a Chess playing demo 
that can play a full game of chess at at least 1900 ELO (which falls short of top chess engines, but is better than general LLMs such as GPT-5 or Grok 4). One core design philosophy is that the internal state of the graph should always be visibile/accesible.

## What I built

Core Engine: a modular ReCoN class with node types, message passing, and logging — able to run arbitrary ReCoN graphs.

Abstract Demo: a “toy” ReCoN animation that shows requests, confirmations, POR/RET gating and the message dynamics.

Chess Endgame Demo: a persistent King + Rook vs King solver built as a ReCoN graph, now handling the full leg‑2 choreography: keep the defender boxed, force zugzwang, and deliver the mate without re-entering shrink phases. Obviously this is nowhere near as strong as modern chess engines, but it demonstrates how ReCoN orchestrates sequential scripts. Plugging in Stockfish would simply mean swapping the heuristic actuator subtree for one node that queries the engine while the rest of the network keeps handling state and terminals—one of ReCoN’s strengths.

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

My original plan to build (which I didn't quite have time to implement - I partly blame a mancold, but the main issue 
was probably me being overly optimistic and wasting time on trying to improve heuristics which was really just a side track; even though it's interesting from a ReCoN sense):
- Implement "empty" ReCoN (the base class) and visualize dummy (done)
- Solve King + Rook vs King (ROOT: Checkmate)
- Solve King + Pawn vs King (ROOT: Promote)
- Identify that King + Pawn -> Promote leaves us at King + Rook so we can expand with a "super ReCoN" that contains the two earlier examples as subtrees. *this showcases the distributed decision-making of ReCoN*; we can reuse the two earlier recons by simply hooking their respective ROOT nodes as subnodes to a higher level network that identifies what kind of position we are in and what strategy to employ. This way of thinking can be generalized to all kinds of problems/simulations. 

- Fancier animations (3D, better schematic groups of subnetworks and different states in nodes/subnetworks and vertices)
- Add image recognition, plug and play chess engines (basically making the ReCoN engine and board "environment" agnostic)


--- 
"Standard" ReadMe below: 



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

To replay the execution, open `demos/visualization/chessboard_view.html` (double-click the file) or the compact `demos/visualization/onepage_view.html`, then use the “Load JSON” button to select any `_viz.json` produced by the demo.

### Persistent deterministic leg 2 demo

```bash
uv run python demos/persistent/krk_persistent_demo.py --max-plies 40 --seed 0 --output-basename krk_persistent_review
```

- Produces `demos/outputs/krk_persistent_review_viz.json` and `_debug.json` (matched pair).
- A pre-generated example (`krk_persistent_phasefix_viz.json`) is already in `demos/outputs` if you just want to load data immediately.
- Visualization workflow is identical: open the HTML viewer of choice and load the `_viz.json` file.

## Some comments and outlook:

This solver is far from perfect; in fact it's quite mediocre as a chess engine. I think it serves as a good demo of how 
to set up a ReCoN network though. It could easily be expanded/improved in several ways (that is out of scope for now as I am running out of time):
- Terminal Nodes could be added:
    - a small CNN or MLP to identify chess positions from pngs (or other images)
    - a more sophisticated image recognition (CNN or other) to identify positions from photos or videos of real chessboards
- Use a true chess engine:
    - instead of using the "phases" and heuristics above an API call to e.g. stockfish could be wrapped into one node and connected to the ReCoN. This would report back with suggested move. So the rest of the graph could stay the same, as long as the new node adheres to the interface spec. 

## Other demos and file structure

## Architecture

See `ARCHITECTURE.md` for detailed design decisions and implementation rationale.

## Docs & Howtos

- `docs/HOWTO_RUN_TRAIN_EVAL.md` — quickstart for running, training, and evaluating with Subgraph Weight Packs (SWPs), batch/block evaluation, tracing, and visualization.
- `docs/TRACE_AND_EVAL.md` — trace schema, weight pack provenance, and evaluation harness details.
- `demos/experiments/krk_generate_fens.py` — utility to generate random legal KRK FEN suites for training/eval.

## Acknowledgements

The ReCoN implementation is based on 
**Request confirmation networks for neuro-symbolic script execution** by Joscha Bach and Priska Herger. 
