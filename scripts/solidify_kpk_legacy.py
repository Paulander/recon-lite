#!/usr/bin/env python3
"""
Solidify KPK Legacy Sensors

This script force-solidifies the top 15 TRIAL nodes from KPK training
to MATURE status, creating "Universal King Sensors" for transfer to KRK.

Part of Bach-Integrated KRK Transition Plan.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recon_lite.nodes.stem_cell import StemCellManager, StemCellState


def solidify_top_sensors(
    source_path: Path,
    output_path: Path,
    top_n: int = 15,
    min_xp: int = 0
) -> dict:
    """
    Force-solidify top TRIAL nodes to MATURE.
    
    Selection criteria (in order):
    1. Most recent confirmation (last_confirm_cycle)
    2. Highest XP
    3. Most samples
    
    Args:
        source_path: Path to source stem_cells.json
        output_path: Path to save solidified stem_cells.json
        top_n: Number of cells to solidify
        min_xp: Minimum XP required for consideration
        
    Returns:
        Summary dict with solidification results
    """
    # Load stem cells
    with open(source_path) as f:
        data = json.load(f)
    
    cells = data.get("cells", {})
    
    # Find TRIAL cells
    trial_cells = []
    for cell_id, cell_data in cells.items():
        if cell_data.get("state") == "TRIAL":
            trial_cells.append({
                "id": cell_id,
                "data": cell_data,
                "xp": cell_data.get("xp", 0),
                "last_confirm": cell_data.get("last_confirm_cycle") or 0,
                "samples": len(cell_data.get("samples", []))
            })
    
    print(f"Found {len(trial_cells)} TRIAL cells")
    
    # Sort by: last_confirm (desc), xp (desc), samples (desc)
    trial_cells.sort(
        key=lambda c: (c["last_confirm"], c["xp"], c["samples"]),
        reverse=True
    )
    
    # Filter by min_xp
    if min_xp > 0:
        trial_cells = [c for c in trial_cells if c["xp"] >= min_xp]
        print(f"After XP filter (>={min_xp}): {len(trial_cells)} cells")
    
    # Select top N
    to_solidify = trial_cells[:top_n]
    
    solidified = []
    for cell_info in to_solidify:
        cell_id = cell_info["id"]
        cell_data = cells[cell_id]
        
        # Force to MATURE
        cell_data["state"] = "MATURE"
        cell_data["xp"] = 100  # Lock at solidify threshold
        
        # Add legacy metadata
        if "metadata" not in cell_data:
            cell_data["metadata"] = {}
        cell_data["metadata"]["legacy"] = "kpk_universal"
        cell_data["metadata"]["safe_from_pruning"] = True
        cell_data["metadata"]["solidified_by"] = "solidify_kpk_legacy.py"
        
        solidified.append({
            "id": cell_id,
            "prev_xp": cell_info["xp"],
            "last_confirm": cell_info["last_confirm"],
            "samples": cell_info["samples"]
        })
        
        print(f"  Solidified: {cell_id} (xp={cell_info['xp']}, confirm={cell_info['last_confirm']})")
    
    # Save updated stem cells
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    return {
        "total_trial": len(trial_cells),
        "solidified_count": len(solidified),
        "solidified_cells": solidified
    }


def create_universal_king_sensors_list(source_path: Path) -> list:
    """
    Extract the top king-related sensors for manual seeding.
    
    Returns list of sensor IDs that are likely "universal" across endgames.
    """
    with open(source_path) as f:
        data = json.load(f)
    
    cells = data.get("cells", {})
    
    # Find MATURE cells (after solidification)
    mature_cells = []
    for cell_id, cell_data in cells.items():
        if cell_data.get("state") == "MATURE":
            # Check if it's marked as legacy
            meta = cell_data.get("metadata", {})
            if meta.get("legacy") == "kpk_universal":
                mature_cells.append(cell_id)
    
    return mature_cells


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Solidify KPK legacy sensors")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("backups/kpk_legacy_20260106/stem_cells.json"),
        help="Source stem_cells.json path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backups/kpk_legacy_20260106/stem_cells_legacy.json"),
        help="Output path for solidified cells"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of cells to solidify (default: 15)"
    )
    parser.add_argument(
        "--min-xp",
        type=int,
        default=0,
        help="Minimum XP for consideration (default: 0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOLIDIFY KPK LEGACY SENSORS")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Top N: {args.top_n}")
    print(f"Min XP: {args.min_xp}")
    print("=" * 60)
    
    if not args.source.exists():
        print(f"ERROR: Source file not found: {args.source}")
        sys.exit(1)
    
    # Solidify
    result = solidify_top_sensors(
        args.source,
        args.output,
        args.top_n,
        args.min_xp
    )
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total TRIAL cells found: {result['total_trial']}")
    print(f"Cells solidified: {result['solidified_count']}")
    
    # List universal sensors
    print("\nUniversal King Sensors (for KRK seeding):")
    universal = create_universal_king_sensors_list(args.output)
    for sensor_id in universal:
        print(f"  - {sensor_id}")
    
    print("\n✓ Legacy solidification complete!")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()

