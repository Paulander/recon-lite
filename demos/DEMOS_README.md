# ReCoN Demos - Learning Progression

This directory contains demos that showcase ReCoN's capabilities, building from basic concepts to advanced interactive applications.

## 🎯 Demo Hierarchy & Learning Path

### 1. **`sequence_demo.py`** - Core ReCoN Fundamentals
**What it shows:** Basic ReCoN execution with SUB/POR links
- ✅ Node types: SCRIPT, TERMINAL
- ✅ Link types: SUB (hierarchy), POR (sequence)
- ✅ Execution flow: Requests → Confirmation → Propagation
- ✅ Logging: JSON snapshots with `tick`, `nodes`, `thoughts`
- **Purpose:** Validate ReCoN engine works correctly
- **Skills learned:** Basic graph structure, execution semantics

---

### 2. **`krk_checkmate_demo.py`** - Static Chess Position Analysis
**What it shows:** ReCoN analyzing chess positions (evaluation mode)
- ✅ Chess domain integration: `python-chess` library
- ✅ Terminal evaluators: `on_edge()`, `box_size()`, `opposition()`, `is_mate()`
- ✅ KRK strategy phases: Drive to edge → Shrink box → Opposition → Mate
- ✅ Static evaluation: Same position analyzed repeatedly
- **Purpose:** Chess position analysis without making moves
- **Skills learned:** Domain-specific terminals, multi-phase strategies
- **Current state:** ✅ Working - evaluates but doesn't play

---

### 3. **`krk_play_demo.py`** - Interactive Chess Player ("Dumb" Mode)
**What it shows:** ReCoN playing full chess games with alternating moves
- ✅ **Move generators:** `KingDriveMoves`, `RandomLegalMoves` (actuators)
- ✅ **Game loop:** ReCoN move → Opponent move → Repeat
- ✅ **Fresh graphs:** Rebuild KRK graph each move (stateless)
- ✅ **Random opponent:** Simple legal move selection
- ✅ **Complete games:** Play until checkmate/stalemate
- **Purpose:** First working chess-playing ReCoN
- **Skills learned:** Actuators, game loops, interactive execution
- **Approach:** "Dumb" - stateless between moves

---

### 4. **`krk_super_play_demo.py`** (Future) - Persistent Chess Player
**What it shows:** Advanced ReCoN with internal state between moves
- 🔄 **Persistent graphs:** Maintain state across moves
- 🔄 **Super-ReCoN:** Top-level orchestrator with KRK subgraph
- 🔄 **WAIT sensors:** Detect opponent moves without rebuild
- 🔄 **Learning ready:** Weight updates between moves
- **Purpose:** Sophisticated chess player with memory
- **Skills learned:** State persistence, multi-level graphs, learning integration

---

## 🔄 Evolution: From Evaluation to Playing

| Demo | Mode | Moves Made | Graph Persistence | Opponent | Learning |
|------|------|------------|-------------------|----------|----------|
| `sequence_demo` | Abstract | 0 | N/A | None | None |
| `krk_checkmate_demo` | Evaluation | 0 | Per-run | None | None |
| `krk_play_demo` | Interactive | ✅ Full games | Per-move | Random | None |
| `krk_super_play_demo` | Advanced | ✅ Full games | Persistent | Random/Self | ✅ Ready |

---

## 🎮 Current Status & What Works

### ✅ **Working Now:**
- **Static evaluation:** `krk_checkmate_demo.py` analyzes KRK positions
- **Interactive playing:** `krk_play_demo.py` plays complete chess games
- **Visualization:** `enhanced_visualization.html` shows execution
- **Logging:** JSON output with move-by-move data

### 🔄 **Next Steps:**
- **Persistent state:** Keep ReCoN reasoning between moves
- **Learning integration:** Update move selection weights
- **Opponent sophistication:** Beyond random moves

---

## 📊 Key Concepts Demonstrated

### **Terminal Types:**
- **Evaluators:** `on_edge()`, `is_mate()` - analyze board state
- **Actuators:** `KingDriveMoves()`, `RandomLegalMoves()` - choose moves

### **Execution Patterns:**
- **Static:** Same position, multiple evaluations
- **Interactive:** Alternating moves, fresh graphs
- **Persistent:** Maintain state across moves

### **Data Flow:**
- **Environment:** `{"board": chess.Board, "chosen_move": None}`
- **Actuators set:** `env["chosen_move"] = "e2e4"`
- **Engine applies:** `board.push_uci(env["chosen_move"])`

---

## 🚀 Running the Demos

```bash
# 1. Basic ReCoN validation
uv run python demos/sequence_demo.py

# 2. Chess position evaluation
uv run python demos/krk_checkmate_demo.py

# 3. Interactive chess playing
uv run python demos/krk_play_demo.py

# 4. View results
# Open demos/visualization/enhanced_visualization.html
```

---

## 🎓 Learning Outcomes

By working through these demos in order, you'll learn:

1. **ReCoN Fundamentals:** Node types, links, execution flow
2. **Domain Integration:** Chess-specific evaluators and actuators
3. **Interactive Systems:** Game loops, alternating execution
4. **State Management:** Stateless vs persistent approaches
5. **Visualization:** Real-time execution monitoring

Each demo builds on the previous one, creating a complete learning path from basic ReCoN concepts to advanced interactive chess playing! 🧠♟️
