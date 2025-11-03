# ReCoN Macrograph Skeleton

The macrograph is a stable, top-down description of the full chess ReCoN.
It defines the high-level control, phase, plan, feature, synthesis, and
learning nodes that orchestrate lower-level sub-networks (such as the KRK
endgame ReCoN). By keeping this skeleton versioned and lightweight we can
expand the system without repeatedly rewriting top-level wiring.

## Spec

- Path: `specs/macrograph_v0.json`
- Versioned (currently `0.2`)
- Nodes include:
  - `GameControl`, `GoalVector`, `OutcomeMode`
  - `PhaseLayer`, `PlanHub`, grouped plan buckets (Opening/Middlegame/Endgame)
  - `FeatureHub` with tactic/structure/endgame bundles
  - `MoveSynth`, `LightEval`, `LearningSupervisor`
  - `KRKSubgraph` placeholder for the existing KRK network
- Edges use semantic kinds (`sub`, `request`, `confirm`, `gate`, `tune`, etc.).

## Loader

`src/recon_lite/macrograph.py` provides:

- `load_macrograph(path)` &rarr; `MacroGraphSpec` (nodes, edges, notes)
- `describe_macrograph(spec)` for quick CLI inspection
- `MacroNode`, `MacroEdge` dataclasses with metadata (goal vectors, phase lists, mounts)
- Support for locating subgraph mounts (`spec.subgraph_mounts()`)

## Usage

```python
from recon_lite.macrograph import load_macrograph, describe_macrograph

spec = load_macrograph("specs/macrograph_v0.json")
print(describe_macrograph(spec))
```

Future stages will:

- Instantiate the macrograph into a running `Graph` (mapping semantic edge kinds to ReCoN link policies).
- Mount the existing KRK sub-network under `PlanEndgame` / `KRKSubgraph`.
- Store learned weights/thresholds in sidecar JSON referenced in the spec.

Because the spec is separate from implementation, we can evolve the internal
nodes (adding new plans, sensors, or learning hooks) without altering the macrograph.
