# ReCoN-lite Continuous Activations & KRK Strategic Layer Guide

This note documents the M1 upgrade that added continuous activations, micro-ticks, strategic latents, binding tables, and visualization support to the KRK persistent demo. It explains what changed, how the pieces fit together, and gives you a quick walk-through so you (or any new collaborator) can try the features hands-on.

---

## 1. What changed?

- **New core utilities** live under `src/recon_lite/core`, `src/recon_lite/time`, and `src/recon_lite/binding`. They provide sigmoid/softmax helpers, the `ActivationState` container, a micro-tick runner, and a binding table that tracks feature-to-square assignments.
- **Engine hook** (`ReConEngine.step`) now looks for `env["microticks"]` before every discrete tick. You opt in by passing a `MicrotickConfig`; legacy behaviour (no micro-ticks) stays intact.
- **Strategic layer** (`src/recon_lite_chess/strategy.py`) computes logits for KRK phases and turns them into continuous latents. Optionally, it scaffolds future OutcomeMode/StyleBias vectors.
- **Blended actuator** (`src/recon_lite_chess/actuators_blend.py`) scores phase move proposals using the current latents plus a tiny evaluation term.
- **Persistent demo** (`demos/persistent/krk_persistent_demo.py`) maintains activation state, binding tables, and phase logits between plies. You control micro-ticks, logging cadence, and the blended chooser via new CLI flags.
- **Visualization** upgrades pulse network nodes according to activation levels, show binding overlays on the board, surface a rationale ribbon, and expose a toggle to collapse wrapper nodes.
- **Tests** (`tests/test_continuous.py`) lock in binding conflict detection, micro-tick convergence, and monotonically increasing phase logits when key predicates hold.

---

## 2. Implementation strategy (design intent)

1. **Keep changes backwards compatible.** Defaults keep micro-ticks disabled, blended actuators off, and logging cadence unchanged so existing demos run as before.
2. **Add stateful, incremental upgrades.** Activations, bindings, and latents are stored on long-lived objects (`ActivationState`, `BindingTable`) so the persistent loop can settle before emitting a discrete tick.
3. **Expose everything through `env`.** The engine and logger already pass an `env` dict; new features piggy-back on it (`phase_latents`, `binding`, `microtick_history`). Visualizers read the same payload.
4. **Guard features with flags.** CLI controls (`--phase-microticks`, `--use-blended-actuator`, etc.) let you test in isolation.
5. **Document pipeline → tests → viz.** Regression coverage ensures the new modules behave; documentation (this file, `updates_continuous.md`) explains the interplay.

Module mapping:

| Responsibility | Module(s) |
| --- | --- |
| Activation math & container | `src/recon_lite/core/activations.py` |
| Micro-tick integration loop | `src/recon_lite/time/microtick.py` |
| Binding manager | `src/recon_lite/binding/manager.py` |
| Engine hook | `src/recon_lite/engine.py` |
| KRK strategy logits | `src/recon_lite_chess/strategy.py` |
| Blended move chooser | `src/recon_lite_chess/actuators_blend.py` |
| Persistent orchestration | `demos/persistent/krk_persistent_demo.py` |
| Visualization upgrades | `demos/visualization/*.js`, `styles.css` |

---

## 3. How to use the new features

### 3.1 Run the persistent KRK demo with micro-ticks & bindings

```bash
python demos/persistent/krk_persistent_demo.py \
  --phase-microticks 5 \
  --phase-eta 0.3 \
  --phase-temperature 1.4 \
  --latent-log-stride 1 \
  --use-blended-actuator \
  --output-basename krk_microtick_demo
```

Key flags:

- `--phase-microticks`: number of micro-iterations before each engine tick (0 = legacy).
- `--phase-eta`: smoothing factor for each micro-tick settle step.
- `--phase-temperature`: softmax temperature used to turn logits into latents.
- `--latent-log-stride`: log latents/binding snapshots every N ticks.
- `--use-blended-actuator`: enable the soft move scorer; fallbacks still keep behaviour safe.

Outputs land in `demos/outputs/persistent/*.json`. The visualization consumes the same logs.

### 3.2 Inspect bindings and latents programmatically

```python
import chess

from recon_lite_chess.strategy import compute_phase_logits, phase_latents_from_logits
from recon_lite.binding.manager import BindingTable, BindingInstance

board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")
logits = compute_phase_logits(board)
latents = phase_latents_from_logits(logits, temperature=1.2)
print("Phase latents:", latents)

table = BindingTable()
with table.begin_tick("krk/p1/drive") as session:
    rook_square = "square:a1"
    claimed = session.reserve(BindingInstance("rook_anchor", {rook_square}))
    print("Reserve rook anchor:", claimed)
print(table.snapshot())
```

### 3.3 Enable the viz upgrades

1. Run a demo with `--output-basename my_run`.
2. Open `demos/visualization/chessboard_view.html` in a browser.
3. Use “Load JSON” to point to `demos/outputs/persistent/my_run_visualization.json`.
4. Toggle “Collapse wrappers” to hide intermediate nodes, watch node circles pulse with activation, and observe square overlays (colour-coded by binding namespace). The rationale ribbon shows `env["last_reason"]` if the demo populated it.

---

## 4. Crash course: build a tiny script

This example demonstrates the new primitives end-to-end without invoking the full demo.

```python
import chess

from recon_lite.core.activations import ActivationState
from recon_lite.time.microtick import MicrotickConfig, run_microticks
from recon_lite.binding.manager import BindingTable, BindingInstance
from recon_lite_chess.strategy import compute_phase_logits, phase_latents_from_logits

# 1) Seed activations for the five KRK phases
phase_states = {
    phase: ActivationState()
    for phase in [
        "phase0_establish_cut",
        "phase1_drive_to_edge",
        "phase2_shrink_box",
        "phase3_take_opposition",
        "phase4_deliver_mate",
    ]
}

board = chess.Board("4k3/6K1/8/8/8/8/R7/8 w - - 0 1")

def compute_targets(_):
    logits = compute_phase_logits(board)
    return phase_latents_from_logits(logits, temperature=1.3)

# 2) Run five micro-ticks to settle activations
cfg = MicrotickConfig(states=phase_states, compute_targets=compute_targets, steps=5, eta=0.3)
run_microticks(cfg)
print("Settled activations:", {k: st.value for k, st in phase_states.items()})

# 3) Bind squares for the current hypothesis
table = BindingTable()
with table.begin_tick("krk/p1/drive") as session:
    rook_sq = board.king(board.turn)  # quick example: use our king square
    session.reserve(BindingInstance("our_king", {f"square:{chess.square_name(rook_sq)}"}))
print("Binding snapshot:", table.snapshot())
```

You can drop this into a `python` shell; the activations converge toward the softmax target, and the binding table records the square token.

---

## 5. What to look for when extending

- **Micro-ticks elsewhere?** Reuse `MicrotickConfig` and the engine’s hook. Ensure you push activation state into `env["microticks"]` before calling `engine.step`.
- **Additional bindings?** Create new namespaces (e.g., `kpk/p2/push`) and feed `BindingInstance` objects in `BindingTable.begin_tick`. Namespaces stay isolated so collisions remain local.
- **More latents or strategies?** Add detectors to `compute_phase_logits`, adjust temperature, and log them via `env["phase_latents"]`. Visualization auto-pulses any node that appears in the `latents` payload.
- **Testing:** extend `tests/test_continuous.py` or mimic its pattern when new detectors need monotonicity checks.

---

## 6. Next steps

1. **Run the persistent demo** with the new flags to generate logs.
2. **Inspect logs** with `jq` or the visualization to watch activations settle.
3. **Iterate on strategy heuristics** by adjusting `compute_phase_logits` and rerunning tests (`python3 -m pytest`) once `pytest` is installed.
4. **Plan M2 (KPK / rook techniques)** with the same scaffolding—bind new feature instances, feed them into the strategic layer, and reuse the micro-tick loop.

Feel free to append questions or TODOs to `updates_continuous.md`; this guide will remain the canonical entry-point for the continuous activation stack. Happy debugging!
