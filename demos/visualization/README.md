# ReCoN Visualization Demo

This folder provides an interactive 3D visualization of a Request Confirmation Network (ReCoN) using Three.js.
It visualizes the step-by-step execution of a ReCoN in a virtual environment showing node state transitions 
(e.g., REQUESTED, CONFIRMED) and hierarchical relationships.

## Features

- **Visualization**: Renders nodes as spheres and edges as lines in 3D, color-coded by state (e.g., gray for INACTIVE, green for CONFIRMED).
- **Dual Views**:
  - **Advanced View**: 3D with particle effects for state changes, subtle glow, and auto-rotation.
  - **Simple View**: 2D, shadow-free, minimalistic.
- **Interaction**:
  - Mouse drag to rotate the 3D scene (using OrbitControls, with auto-rotation in advanced mode).
  - Slider to scrub through ticks (0-14).
  - Play/pause button to animate progression (0.6s per tick).
  - File input to load custom JSON logs.
- **Toggles**:
  - Switch between advanced and simple views.
  - Show/hide node and edge labels.
  - Toggle legend for state color guide.
  - (Event log display is planned but not implemented.)
- **Responsive Design**: Adjusts to window size.

## Setup and Usage

### (alternative) Quick Viewing Option
For a quick preview without setting up a server, open `standalone_html_example.html` by double-clicking it in your file explorer.
This single HTML file contains all code (no ES modules) and works directly in a browser but lacks modularity.

### Prerequisites
- **WSL2** (Windows Subsystem for Linux) with a Linux distribution (e.g., Ubuntu).
- **uv**: A fast Python package manager. Install it in WSL2:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Internet connection to load Three.js from CDN.

### Running the Visualization
1. **Navigate to the Project Directory**:
   ```bash
   cd ~/path/to/recon-lite/demos/visualization
   ```
   Ensure the following files are present:
   - `index.html`
   - `styles.css`
   - `network-data.js`
   - `utils.js`
   - `visualization.js`
   - `main.js`
   - `standalone_html_example.html` (optional, for quick viewing)

2. **Start the Server**:
   Use `uv` to run a Python HTTP server:
   ```bash
   uv run python -m http.server 8000
   ```
   This serves the files at `http://localhost:8000`.

3. **Access the Visualization**:
   Open a browser (e.g., Chrome) on your Windows system and navigate to:
   ```
   http://localhost:8000
   ```

4. **Stop the Server**:
   Press `Ctrl+C` in the WSL2 terminal.


### JSON Import
- Use the file input in the UI to load custom JSON logs.
- The JSON must match the format of `frames` in `network-data.js` (with `tick`, `nodes`, `note`, `thoughts`, and optional `new_requests`).
- **Limitation**: The visualization currently supports only the fixed node set (`ROOT`, `A`, `B1`, `B2`, `C`, `A_done`, `B1_done`, `B2_done`, `C_done`) and their predefined positions/edges. Arbitrary node sets are not fully supported without modifying `nodePositions` and `edges` in `network-data.js`.

## Files
- `index.html`: Main HTML structure with UI.
- `styles.css`: CSS for UI and visualization.
- `network-data.js`: JSON data, node positions, edges, and state colors.
- `utils.js`: Utility functions (e.g., label texture creation).
- `visualization.js`: Scene element creation and visualization updates.
- `main.js`: Core logic for Three.js setup and event handling.
- `standalone_html_example.html`: Single-file version for quick viewing.

## Notes
- The advanced view uses more transparent labels (opacity 0.6 for nodes, 0.5 for edges) to avoid obscuring nodes.
- For arbitrary ReCoN graphs, the code would need a dynamic layout algorithm (future enhancement).
- Ensure files are in the correct directory and have proper permissions (`chmod +r *.js *.css *.html`).