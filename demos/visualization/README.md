# ReCoN Enhanced Visualization

This directory contains the enhanced visualization system that implements the VIS_SPEC.md layout for ReCoN KRK checkmate demonstrations.

## Features

### Layout (per VIS_SPEC.md)
- **Left (main)**: Interactive chess board rendered from FEN positions
- **Bottom-left**: AI "portrait" with dynamic thoughts/comments
- **Right-top**: 2D ReCoN network graph with colored nodes by state
- **Right-bottom**: Phase schematic showing KRK strategy phases

### Interactive Controls
- Play/Pause/Step through the ReCoN execution
- Scrub through execution steps
- Real-time state visualization
- Phase progression tracking

## Files

- `enhanced_visualization.html` - Main visualization application
- `test_visualization.html` - Simple test page to verify JSON data loading

## Usage

### 1. Generate Visualization Data
First, run the KRK demo to generate the JSON data:

```bash
cd /path/to/recon-lite
uv run python demos/krk_checkmate_demo.py
```

This creates `demos/krk_visualization_data.json` with all the ReCoN execution frames.

### 2. View the Visualization
Open `enhanced_visualization.html` in a web browser. For best results, serve it via HTTP:

```bash
cd demos/visualization
python3 -m http.server 8000
# Then open http://localhost:8000/enhanced_visualization.html
```

Or simply double-click the HTML file if your browser supports ES6 modules locally.

### 3. Test Data Loading
Open `test_visualization.html` to verify the JSON data is properly formatted and loadable.

## JSON Schema (VIS_SPEC.md Compliant)

Each frame in the visualization data follows this structure:

```json
{
  "type": "snapshot",
  "tick": 1,
  "note": "Step 1",
  "nodes": {
    "node_id": "STATE_NAME"
  },
  "new_requests": ["node_id1", "node_id2"],
  "env": {
    "fen": "4k3/6K1/8/8/8/8/R7/8 w - - 0 1"
  },
  "thoughts": "AI commentary text"
}
```

## Node States & Colors

- **INACTIVE** (gray) - Node not yet activated
- **REQUESTED** (blue) - Node requested for evaluation
- **WAITING** (orange) - Node waiting for children to complete
- **TRUE** (lime) - Terminal node evaluation succeeded
- **CONFIRMED** (green) - Script node confirmed all requirements met
- **FAILED** (red) - Node evaluation failed

## KRK Phases

The visualization tracks these phases:
1. **ROOT**: KRK Checkmate (overall strategy)
2. **PHASE1**: Drive to Edge (force king to board edge)
3. **PHASE2**: Shrink Box (reduce king's mobility)
4. **PHASE3**: Take Opposition (proper king alignment)
5. **PHASE4**: Deliver Mate (final checkmate execution)

## Technical Notes

- Uses chess.js library for board rendering
- Three.js ready for future 3D enhancements
- Responsive design works on different screen sizes
- ES6 modules for clean code organization
- CORS-friendly for local file access

## Future Enhancements

- Dynamic thought generation using LLM API
- 3D network visualization with Three.js
- Move animations on chess board
- Multiple visualization layouts
- Export capabilities (GIF, video)