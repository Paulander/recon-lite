# ReCoN Visualization Suite

This directory contains multiple visualization approaches for ReCoN networks:

## 📊 Visualization Types

### 1. **Parent Dashboard View** (`enhanced_visualization.html`)
**Purpose**: Comprehensive dashboard with multiple sub-views per VIS_SPEC.md
- **Chess Board**: Interactive board from FEN/move data
- **AI Portrait**: Dynamic thoughts and commentary
- **2D Network Graph**: Node states with color coding
- **Phase Schematic**: KRK strategy phase visualization
- **Standalone**: Works without server, double-click to run
- **Controls**: Play/Pause/Step through execution

### 2. **3D Network Visualization** (`index.html`)
**Purpose**: Focused 3D visualization of network structure and state changes
- **3D Network**: Interactive 3D graph with Three.js
- **State Colors**: Real-time node state visualization
- **Modular**: Separate JS/CSS files
- **Server Required**: HTTP server needed for ES6 modules

### 3. **Standalone 3D Demo** (`standalone_html_example.html`)
**Purpose**: Self-contained 3D network demo
- **All-in-one**: CSS/JS inline, no external dependencies
- **Same 3D features**: As modular version but standalone
- **Perfect for sharing**: Single file, works without server

### 4. **Neuro MRI View** (`neuro_mri/index.html`)
**Purpose**: Futuristic “brain” rendering of node activity
- **Three.js brainspace**: Nodes grouped into lobes with glowing pulses
- **Per-frame fades**: Requires logs generated with `--log-full-state`
- **Minimal UI**: Load JSON, play/pause, scrub speed
- **Best for showcase**: Works offline (double-click)

### 5. **Macrograph Viewer** (`macrograph_view.html`)
**Purpose**: Inspect the top-level macrograph skeleton (`specs/macrograph_v0.json`)
- **Static layout**: Displays control/phase/plan/feature hubs with edge labels
- **Toggle spec**: Load default or supply a custom JSON spec
- **Zero dependencies**: Works offline; ideal for design reviews

### 6. **Consolidation Dashboard** (`consolidation_dashboard.html`)
**Purpose**: Monitor slow consolidation training progress (M4)
- **Load/Compare**: Load consolidation state JSON files, compare two states
- **Metrics Dashboard**: Total episodes, edges tracked, configuration
- **Weight Histogram**: Distribution of w_base values with statistics
- **Top Changes**: Table showing largest weight drifts from initial
- **No Server Required**: Works offline (double-click to open)

### 7. **Full Game View** (`full_game_view.html`) - NEW (M8)
**Purpose**: Visualize M6 goal hierarchy during full game play
- **Goal Hierarchy**: Shows Ultimate → Strategic → Tactical layers
- **Phase Weights**: Live Opening/Middlegame/Endgame progress bar
- **Active Plans**: Bar chart of currently active strategic plans
- **Material Balance**: White/Black material tracking
- **Move History**: Clickable move history with navigation
- **No Server Required**: Works offline, load JSON via button


## 🚀 Quick Start Guide

### Option 1: Parent Dashboard (Recommended for KRK Demo)
**Best for seeing the complete chess + network visualization**
```bash
# Just double-click this file:
enhanced_visualization.html

# It will:
# ✅ Load immediately (no server needed)
# ✅ Show chess board + network + phases
# ✅ Fall back to demo data if JSON not found
# ✅ Work with live KRK data when available
```

### Option 2: 3D Network Visualization (Requires Server)
**Best for detailed network structure exploration**
```bash
# Start server in this directory:
uv run python -m http.server 8000

# Open in browser:
http://localhost:8000/index.html
```

### Option 3: Standalone 3D Demo (No Server)
**Best for quick sharing/demos**
```bash
# Just double-click:
standalone_html_example.html
```

### Option 4: Neuro MRI View (No Server)
**Best for cinematic presentations**
```bash
# Generate log with full node states
uv run python demos/persistent/krk_persistent_demo.py \
  --max-plies 40 --seed 0 --log-full-state \
  --output-basename krk_fulltrace

# Double-click to open
neuro_mri/index.html
```

### Option 5: Macrograph Viewer (No Server)
**Best for validating macrograph specs**
```bash
# Double-click to open
macrograph_view.html

# Loads ../../specs/macrograph_v0.json by default; drag-drop a custom spec if needed.
```

### Option 6: Consolidation Dashboard (No Server)
**Best for monitoring training progress**
```bash
# 1. Run training with consolidation
uv run python demos/persistent/krk_persistent_demo.py \
  --batch 20 --plasticity --consolidate \
  --consolidate-pack weights/nightly/krk_consol.json

# 2. Double-click to open dashboard
consolidation_dashboard.html

# 3. Click "Load State" and select weights/nightly/krk_consol.json
# 4. Optionally, click "Compare With..." to compare against a baseline
```

### Option 7: Full Game View (No Server) - NEW (M8)
**Best for visualizing M6 goal hierarchy and full game play**
```bash
# 1. Run full game demo with visualization output
uv run python demos/persistent/full_game_demo.py \
  --max-moves 100 --vs-random --output game_viz.json

# 2. Double-click to open
full_game_view.html

# 3. Click "Load JSON" and select game_viz.json
# 4. Use arrow keys or buttons to navigate through frames
# 5. Press Space to auto-play
```

## 📋 Prerequisites

- **uv**: Fast Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Internet**: For loading Three.js/chess.js from CDN
- **Browser**: Modern browser with ES6 support


## 📄 Data Formats

### JSON Schema (VIS_SPEC.md Compliant)
The parent dashboard uses JSON with this structure:
```json
{
  "type": "snapshot",
  "tick": 1,
  "note": "Step 1",
  "nodes": { "node_id": "STATE_NAME", ... },
  "env": {
    "initial_fen": "4k3/6K1/8/8/8/8/R7/8 w - - 0 1",
    "moves": ["a1a2", "g7f6"],
    "fen": "4k3/5K2/R7/5k2/8/8/8/8 w - - 0 3"
  },
  "thoughts": "AI commentary",
  "new_requests": ["node_id1", "node_id2"]
}
```

### Generating Data
```bash
# Run the KRK demo to generate visualization data:
uv run python demos/krk_checkmate_demo.py
```

This creates `demos/krk_visualization_data.json` with the ReCoN execution frames.

## 🎯 Key Differences

| Feature | Parent Dashboard | 3D Network | Standalone 3D | Consolidation Dashboard |
|---------|------------------|------------|---------------|------------------------|
| **Chess Board** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **AI Thoughts** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Phase Schematic** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Training Metrics** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Weight Histograms** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **3D Network** | ❌ No (2D) | ✅ Yes | ✅ Yes | ❌ No |
| **Server Required** | ❌ No | ✅ Yes | ❌ No | ❌ No |
| **Data Source** | viz JSON | JS array | JS array | consol JSON |

## 🛠️ Development

### File Structure
- **Parent Dashboard**: `enhanced_visualization.html` (all-in-one file)
- **3D Network**: Modular files:
  - `index.html` - Main HTML structure
  - `main.js` - Three.js setup and controls
  - `visualization.js` - Scene rendering and updates
  - `network-data.js` - Node data, positions, edges
  - `utils.js` - Helper functions
  - `styles.css` - UI styling
- **Standalone 3D**: `standalone_html_example.html` (all-in-one)
- **Testing**: `test_visualization.html` (JSON loading verification)

### Data Sources
- **Parent Dashboard**: Loads `../krk_visualization_data.json`
- **3D Visualizations**: Use hardcoded data in `network-data.js`

## 🔮 Future Enhancements

- **Dynamic Layout**: Support for arbitrary ReCoN graphs
- **Live Updates**: Real-time visualization during ReCoN execution
- **Multiple Data Sources**: Support for different JSON formats
- **Export Features**: GIF/video export of visualization sequences
- **Advanced Interactions**: Node selection, detail panels, filtering
