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

| Feature | Parent Dashboard | 3D Network | Standalone 3D |
|---------|------------------|------------|---------------|
| **Chess Board** | ✅ Yes | ❌ No | ❌ No |
| **AI Thoughts** | ✅ Yes | ❌ No | ❌ No |
| **Phase Schematic** | ✅ Yes | ❌ No | ❌ No |
| **3D Network** | ❌ No (2D) | ✅ Yes | ✅ Yes |
| **Server Required** | ❌ No | ✅ Yes | ❌ No |
| **Modular Code** | ❌ Inline | ✅ Yes | ❌ Inline |
| **Data Source** | JSON file | JS array | JS array |

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
