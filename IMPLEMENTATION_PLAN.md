Here’s an updated **IMPLEMENTATION\_PLAN.md** that reflects your current state (static evaluator works), my last review, and the next milestones toward an interactive chess-playing ReCoN.

---

## 🎯 CURRENT STATUS SUMMARY

| Phase | Status | Description |
|-------|--------|-------------|
| **0-2** | ✅ Complete | Core ReCoN engine + logging + visualization schema |
| **3-4** | ✅ Complete | Domain separation + chess substrate ready |
| **5** | ✅ Complete | KRK evaluators + move generators implemented |
| **6** | ✅ Complete | Interactive gameplay demo with "dumb" moves |
| **7** | ✅ Advanced | Enhanced dashboard with move replay |
| **8** | ❌ Future | Persistent super-ReCoN + learning |

**🎮 WORKING NOW**: Complete KRK chess player (static eval + interactive games)
**🚀 NEXT**: Persistent super-ReCoN with internal state + learning

---

# Implementation Plan (ReCoN-lite → KRK Interactive Demo)

## Phase 0 — Repo + Run (✅ done)

* Install:

  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv pip install -e .
  ```
* Run toy demo:

  ```bash
  uv run python -m demos.sequence_demo
  ```
* Confirms that SUB/POR execution, confirmation logic, and logging work.

## Phase 1 — Engine + Logging (✅ done)

* `Graph`: nodes (SCRIPT/TERMINAL), links (SUB/POR).
* `Engine.step(env)`:

  * Terminals advance via `predicate(node, env)`.
  * Scripts request children when POR-predecessors TRUE.
  * Scripts confirm when last nodes of POR chains confirm.
* `RunLogger.snapshot(...)`: per-tick frame with
  `tick, nodes, new_requests, env, thoughts, latents` → JSON.

## Phase 2 — Visualization JSON schema

* Each frame example:

```json
{
  "type": "snapshot",
  "tick": 7,
  "note": "tick 7",
  "nodes": { "ROOT": "WAITING", "PHASE1": "CONFIRMED" },
  "new_requests": ["PHASE2"],
  "env": { "fen": "8/8/8/8/8/8/8/8 w - - 0 1" },
  "thoughts": "Driving BK north; chosen move: Rh8+; box=4x3; opposition=false",
  "latents": { "PHASE2": [0.12, -0.7, 0.1] }
}
```

## Phase 3 — Domain Separation & Plugins

* Keep `src/recon_lite/` domain-agnostic (engine only).
* Add `src/recon_lite_chess/` for chess logic.

  * `predicates.py`: sensors (is\_mate, on\_edge, rook\_safe…).
  * `actuators.py`: choose\_move\_\* terminals writing `env["chosen_move"]`.
  * `krk_graph.py`: builds PHASE1→PHASE4 ReCoN graph.
* Plugins API (optional):

  * `TerminalPlugin` Protocol with `.step(node, env) -> (done, success)`.
  * Wraps rule-based checks and actuators.

## Phase 4 — Chess substrate

* Dependency already added: `python-chess`.
* `env` holds:

  * `board: chess.Board`
  * `chosen_move: str` (set by actuators)
* After each `engine.step`:

  * If `chosen_move` exists → push move on board → clear field.
  * Log `fen` for visualization.

## Phase 5 — KRK ReCoN graph (🔄 Partially Complete)

* Scripts: `PHASE1 → PHASE2 → PHASE3 → PHASE4` via POR. ✅ DONE
* Terminals per phase (evaluators working, move generators missing):

  * **Phase 1**: `on_edge(BK)`, `rook_safe` ✅ | `choose_move_p1` ❌
  * **Phase 2**: `box_size <= target`, `rook_safe` ✅ | `choose_move_p2` ❌
  * **Phase 3**: `has_opposition`, `rook_safe` ✅ | `choose_move_p3` ❌
  * **Phase 4**: `is_mate` ✅ | `choose_mate` ❌
* **Current State**: Static position evaluator - analyzes but doesn't generate moves
* **Next**: Add move selection actuators (`choose_move_*`) for interactive play

## Phase 6 — Interactive Gameplay Demo (🚀 Ready to Implement)

### **Two Implementation Approaches:**

#### **Option A: Per-Move "Dumb" Loop (Recommended First)**
* **Strategy**: Rebuild KRK graph from scratch each move (stateless)
* **Pros**: Simple, fast to implement, clean logging per move
* **Implementation**:
  * Outer loop: `env = {"board": chess.Board(...), "chosen_move": None}`
  * Build fresh KRK ReCoN graph each turn
  * Tick engine until actuator sets `env["chosen_move"]`
  * Apply move, opponent plays random legal move
  * Repeat with fresh graph
* **File**: `demos/krk_play_demo.py`
* **Milestone**: Working player vs random opponent

#### **Option B: Persistent "Super-ReCoN" (Advanced)**
* **Strategy**: Keep internal state between moves
* **Structure**:
  * Root: `WIN_GAME` (persistent across moves)
  * Children: `SELECT_STRATEGY`, `EXECUTE_STRATEGY`, `MONITOR_SAFETY`
  * KRK sub-ReCoN as subgraph under `EXECUTE_STRATEGY`
  * `WAIT_FOR_BOARD_CHANGE` sensor detects opponent moves
* **Pros**: More sophisticated, maintains reasoning state
* **File**: `demos/krk_super_play_demo.py`
* **Prerequisites**: Working Option A first

### **Prerequisites for Both:**
* Phase 5 move generators: `choose_move_p1`, `choose_move_p2`, etc.
* Actuator pattern: Terminals that write `env["chosen_move"]`
* Random opponent: Simple legal move selector

## Phase 7 — Visualization (✅ Advanced Implementation)

* **BEYOND described**: Full interactive dashboard at `demos/visualization/enhanced_visualization.html`
* **Features implemented**:
  * **Chess Board**: Renders FEN with move replay support
  * **AI Selfie + Thoughts**: Dynamic commentary system
  * **2D Network Graph**: Node states with color coding + edges
  * **Phase Schematic**: Interactive KRK phase visualization
  * **Controls**: Play/pause/step through execution timeline
* **JSON Support**: Reads `krk_visualization_data.json` with move-based format
* **Fallback**: Works with demo data when live data unavailable

## Phase 8 — Extras (optional)

* `RET` links; explicit OR-groups.
* Latent view: record per-node feature vectors.
* “Super-ReCoN”: orchestrator that dispatches to subgraphs (KRK, K+P, etc.).
* Export `run.json` + HTML viz for sharing.

---

Would you like me to also **draft the Phase 6 demo file (`krk_play_demo.py`)** now, so you can run a first end-to-end interactive game loop with random opponent right away?

---

## 📁 FILES READY

**Core ReCoN Implementation:**
- `src/recon_lite/graph.py` - Node/Edge definitions, core logic
- `src/recon_lite/engine.py` - ReCoN execution engine
- `src/recon_lite/logger.py` - Structured logging for visualization
- `VIS_SPEC.md` - Visualization JSON schema specification

**Chess Domain Implementation:**
- `src/recon_lite_chess/krk_nodes.py` - Current KRK evaluators
- `demos/krk_checkmate_demo.py` - Working static evaluator demo

**Visualization:**
- `demos/visualization/enhanced_visualization.html` - Advanced interactive dashboard
- `demos/krk_visualization_data.json` - Sample logged execution data

**Documentation:**
- `ARCHITECTURE.md` - Design decisions and rationale
- `IMPLEMENTATION_PLAN.md` - Current status and next steps
