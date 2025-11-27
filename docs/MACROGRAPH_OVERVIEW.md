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

## Runtime usage

```python
from recon_lite.macrograph import load_macrograph, describe_macrograph, instantiate_macrograph
from recon_lite.macro_engine import MacroEngine

spec = load_macrograph("specs/macrograph_v0.json")
print(describe_macrograph(spec))

graph = instantiate_macrograph("specs/macrograph_v0.json")
engine = MacroEngine()
frame = engine.capture_macro_frame({"board": None})
```

## Macro telemetry (preview)

`MacroEngine.capture_macro_frame(env)` generates the per-step payload used by the
visualization layer. Keys include:

- `goal_vector`: Goal projections (Material, KingSafety, Initiative, Structure, PhaseProgress, RiskBudget, TacticWindow) in `[0,1]`.
- `phase_mix`: Soft phase distribution (`Opening`, `Middlegame`, `Endgame`).
- `plan_groups`: Each top-level plan group with activation and plan IDs.
- `feature_groups`: Feature bundles with confidence and constituent feature IDs.
- `bindings`: High-level bindings (e.g., rook/king squares in macro/endgame contexts).
- `move_synth`: Weights, per-move component scores, and the currently chosen move.

The viewer consumes this frame in addition to standard KRK snapshots, enabling
side-by-side macro + micro reasoning traces.
