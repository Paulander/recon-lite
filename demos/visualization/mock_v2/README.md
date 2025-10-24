# mock_v2 Synthetic Topology + Viewer

This mini-suite produces large synthetic datasets and a lightweight Three.js viewer capable of animating 1k+ nodes across 20–30 ticks with contextual metadata.

## Layout

- `generate_mock.py` &mdash; topology/timeline generator.
- `data/` &mdash; sample JSON exports.
- `viewer/` &mdash; standalone HTML/JS/CSS playback UI.
- `schema.md` &mdash; authoritative JSON contract.

## Generator

```
python3 demos/visualization/mock_v2/generate_mock.py \
  --tactic-instances 80 \
  --expert-clones 6 \
  --ticks 30 \
  --include-deltas \
  --out demos/visualization/mock_v2/data/demo_dataset_big.json
```

Key options:

- `--tactic-instances` controls how many copies of every tactic terminal are created.
- `--expert-clones` duplicates each expert archetype to enlarge the mid-layer.
- `--ticks` sets the number of timeline frames (20–30 recommended for playback demos).
- `--include-deltas` emits a `deltas` array for future streaming workflows.
- `--topology-out`, `--frames-out`, `--deltas-out` allow split exports alongside the combined dataset.
- `--compact` disables indentation for faster parsing when generating very large files.

### Sample exports

- `data/demo_topology_small.json` &mdash; trimmed topology with baked positions.
- `data/demo_frames_small.json` &mdash; 12 ticks (~15 KB) referencing the same node ids.
- `data/demo_dataset_small.json` &mdash; combined topology + frames for quick smoke tests.

## Viewer

1. Serve or open `viewer/index.html` in a modern browser.  
   - When loading from disk, most browsers allow local file access; if CORS blocks fetches, start a quick server:  
     `python3 -m http.server --directory demos/visualization/mock_v2/viewer`
2. By default the viewer attempts to load `../data/demo_dataset_small.json`. Use the file picker to supply any generator output (either combined dataset or frames-only JSON).
3. Controls:
   - `Play/Pause` toggles timeline autoplay (~1.4s per tick).
   - Slider scrubs to a specific tick.
   - `Show labels` overlays sprites (disabled by default for performance on 1k+ nodes).
   - `Advanced View` reveals notes, phase weights, and state distribution.
   - `Reset View` recentres the camera around the layout centroid.

Rendering details:

- Nodes are rendered via `InstancedMesh` spheres with per-state colours.
- New requests trigger a short pulse animation to aid visual tracking.
- Edges are omitted for clarity but the topology keeps them for future upgrades.
- Auto-layout falls back to layered rings when node positions are absent.

## Performance Tips

- Keep labels off for dense graphs; enable them only when inspecting smaller slices.
- Use `--compact` or gzip the JSON when shipping datasets to keep payloads light.
- Adjust `--tactic-instances` / `--expert-clones` to scale up or down without editing the viewer.
- For incremental streams, reuse a fixed `topology` and append `frames` (or `deltas`) over time.
