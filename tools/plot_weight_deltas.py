#!/usr/bin/env python3
"""Plot per-tick weight deltas from a viz/debug JSON file."""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: uv pip install matplotlib")
    sys.exit(1)


def load_frames(json_path: Path) -> list:
    with open(json_path) as f:
        return json.load(f)


def extract_deltas(frames: list) -> dict[str, list[tuple[int, float]]]:
    """Extract per-edge deltas over ticks."""
    edge_series = defaultdict(list)
    
    for i, frame in enumerate(frames):
        env = frame.get("env", {})
        tick = env.get("evaluation_tick", i)
        
        # M3 weight deltas (if present)
        deltas = env.get("m3_weight_deltas", {})
        for edge_key, delta in deltas.items():
            edge_series[edge_key].append((tick, delta))
        
        # Also check reward_tick
        reward = env.get("m3_reward_tick")
        if reward is not None:
            edge_series["__reward_tick__"].append((tick, reward))
    
    return dict(edge_series)


def extract_cumulative(frames: list) -> dict[str, list[tuple[int, float]]]:
    """Extract cumulative delta sums over ticks."""
    edge_cumulative = defaultdict(float)
    edge_series = defaultdict(list)
    
    for i, frame in enumerate(frames):
        env = frame.get("env", {})
        tick = env.get("evaluation_tick", i)
        
        deltas = env.get("m3_weight_deltas", {})
        for edge_key, delta in deltas.items():
            edge_cumulative[edge_key] += delta
            edge_series[edge_key].append((tick, edge_cumulative[edge_key]))
    
    return dict(edge_series)


def plot_series(series: dict[str, list], title: str, top_n: int = 8):
    """Plot top N edges by total movement."""
    if not series:
        print("No delta data found in JSON. Run with --plasticity to generate.")
        return
    
    # Sort by total absolute movement
    totals = {k: sum(abs(d) for _, d in v) for k, v in series.items()}
    top_edges = sorted(totals, key=lambda k: totals[k], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for edge in top_edges:
        if edge == "__reward_tick__":
            continue  # Plot separately
        ticks, deltas = zip(*series[edge])
        label = edge.replace("->", "→").replace(":POR", "")
        ax.plot(ticks, deltas, label=label, linewidth=1.5, marker='o', markersize=3)
    
    ax.set_xlabel("Tick")
    ax.set_ylabel("Delta (cumulative)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python tools/plot_weight_deltas.py <json_file> [--cumulative]")
        print("Example: uv run python tools/plot_weight_deltas.py demos/outputs/persistent/krk_fullgame_debug.json")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    cumulative = "--cumulative" in sys.argv
    
    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)
    
    frames = load_frames(json_path)
    print(f"Loaded {len(frames)} frames from {json_path.name}")
    
    if cumulative:
        series = extract_cumulative(frames)
        title = f"Cumulative Weight Deltas - {json_path.name}"
    else:
        series = extract_deltas(frames)
        title = f"Per-Tick Weight Deltas - {json_path.name}"
    
    if not series:
        print("\n⚠️  No m3_weight_deltas found in JSON.")
        print("   Make sure you ran with --plasticity flag.")
        print("   The deltas are computed but may not be logged to viz output yet.")
        return
    
    print(f"Found {len(series)} edges with delta data")
    
    fig = plot_series(series, title)
    if fig:
        out_path = json_path.with_suffix(".png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        # plt.show()  # Commented out for headless environments


if __name__ == "__main__":
    main()