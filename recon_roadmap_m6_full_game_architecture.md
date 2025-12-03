# M6 Roadmap: Full-Game Architecture & Multi-Scale Dynamics

**Status**: ✅ Implemented (December 2025)

## Goal

Restructure ReCoN around a time-scale goal hierarchy where Ultimate/Strategic/Tactical/Sensor layers blend together, sensor terminals are shared services (fan-in), plans have persistence (via activation), and the system can play a complete game from opening to checkmate.

## Design Principles

### ReCoN Paper Adherence
- **Four link types only**: `sub`, `sur`, `por`, `ret`
- **State machine**: `{inactive, requested, active, suppressed, waiting, true, confirmed, failed}`
- **Request/confirm flow**: top-down `sub` request → bottom-up `sur` confirmation

### Minimal Extension
- **Fan-in for sensor terminals**: Multiple scripts can `sub`-link to the same terminal (read-only sensors)
- Terminal sends `sur` confirmation to ALL requesting parents
- Actuators remain 1:1 (only one script can own an action)

### Already Extended (M3)
- Continuous activation values (paper allows real-valued activation)
- Weight plasticity (paper mentions learning)
- Activation stores persistence state (paper: "activation used to store additional state")

---

## Phase 1: Fan-In Terminal Support (M6.1)

### Deliverables
- [x] Terminals can have multiple incoming `sub` links
- [x] When terminal confirms, send `sur` to ALL parents
- [x] `Graph.all_parents()` method for fan-in queries
- [x] `Graph.is_fanin_terminal()` check

### Files Modified
- `src/recon_lite/graph.py`

---

## Phase 2: Goal Hierarchy Restructure (M6.2)

### Deliverables
- [x] `UltimateGoal` assessment (WIN/DRAW/SURVIVE)
- [x] `MaterialSensor` terminal (fan-in)
- [x] `PhaseSensor` terminal with soft weights
- [x] Strategic plan definitions with phase boosts

### Files Created
- `src/recon_lite_chess/goals/ultimate.py`
- `src/recon_lite_chess/goals/strategic.py`
- `src/recon_lite_chess/sensors/material.py`
- `src/recon_lite_chess/sensors/phase.py`

---

## Phase 3: Plan Persistence (M6.3)

### Deliverables
- [x] `PersistenceState` dataclass for activation storage
- [x] `update_persistence()` with inertia + decay
- [x] Interrupt mechanism for urgent tactics
- [x] `compute_plan_competition()` soft-max over activations

### Files Created
- `src/recon_lite/dynamics/persistence.py`

---

## Phase 4: Opening & Middlegame Scripts (M6.4)

### Deliverables
- [x] Opening sensors (Development, Castling, CenterControl)
- [x] Opening plans (DevelopMinorPieces, CastleEarly, ControlCenter)
- [x] Middlegame sensors (KingSafety, PieceActivity, Structure)
- [x] Middlegame plans (AttackKing, ImproveWorstPiece, CreateWeakness, Simplify)
- [x] Move candidate generation for each phase

### Files Created
- `src/recon_lite_chess/scripts/opening.py`
- `src/recon_lite_chess/scripts/middlegame.py`

---

## Phase 5: Full Game Integration (M6.5)

### Deliverables
- [x] Full game demo from starting position
- [x] Phase-aware plan activation
- [x] Goal-based strategy selection
- [x] Visualization data output

### Files Created
- `demos/persistent/full_game_demo.py`
- `specs/macrograph_v1.json`

---

## Testing

All tests passing:
- `tests/test_fanin_terminals.py` (11 tests)
- `tests/test_goal_hierarchy.py` (9 tests)
- `tests/test_persistence.py` (13 tests)
- `tests/test_opening_middlegame.py` (14 tests)

---

## Usage

```bash
# Play full game
uv run python demos/persistent/full_game_demo.py --max-moves 200

# Play against random opponent
uv run python demos/persistent/full_game_demo.py --vs-random --output game.json
```

---

## Acceptance Criteria

- [x] Fan-in terminals support multiple sub links
- [x] Actuators remain 1:1
- [x] Goal hierarchy: Ultimate → Strategic → Tactical → Sensor
- [x] Plans have persistence via activation
- [x] Standard request/confirm flow (no separate phases)
- [x] Full game playable from starting position
- [x] Only sub/sur/por/ret link types used

