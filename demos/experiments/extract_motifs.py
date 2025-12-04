#!/usr/bin/env python3
"""
M5.1: Extract motifs from trace JSONL files.

Usage:
    uv run python demos/experiments/extract_motifs.py \
        --traces reports/krk_trace.jsonl \
        --out reports/motifs/krk_motifs.jsonl \
        --reward-threshold 0.3

Filters:
    - |reward_tick| > threshold (significant eval swings)
    - High-activity ticks (many nodes firing)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import chess
except ImportError:
    chess = None

from recon_lite.motifs.descriptors import (
    BindingDescriptor,
    MotifDataset,
    MotifType,
)
from recon_lite.motifs.extractors import (
    extract_all_features,
    extract_hanging_pieces,
    extract_tactical_features,
)


def classify_motif_type(features: Dict[str, Any], reward: float) -> str:
    """Classify the motif type based on extracted features."""
    tactical = features.get("tactical", {})
    hanging = features.get("hanging_pieces", {})
    
    # Check for tactical patterns
    if tactical.get("potential_forks"):
        return MotifType.TACTICAL.value
    if tactical.get("pins"):
        return MotifType.TACTICAL.value
    if hanging.get("white_en_prise") or hanging.get("black_en_prise"):
        return MotifType.TACTICAL.value
    
    # Check for endgame patterns
    material = features.get("material", {})
    total_material = material.get("white", 0) + material.get("black", 0)
    if total_material <= 10:  # Endgame threshold
        return MotifType.ENDGAME.value
    
    # Check for structural patterns
    pawn = features.get("pawn_structure", {})
    if pawn.get("passed_pawns") or pawn.get("isolated_pawns"):
        return MotifType.STRUCTURAL.value
    
    return MotifType.POSITIONAL.value


def generate_pattern_key(features: Dict[str, Any], dtype: str) -> str:
    """Generate a pattern key based on features."""
    tactical = features.get("tactical", {})
    hanging = features.get("hanging_pieces", {})
    pawn = features.get("pawn_structure", {})
    
    if dtype == MotifType.TACTICAL.value:
        if tactical.get("potential_forks"):
            return "fork_opportunity"
        if tactical.get("pins"):
            return "pin_detected"
        if hanging.get("white_en_prise") or hanging.get("black_en_prise"):
            return "hanging_piece"
        return "tactical_other"
    
    if dtype == MotifType.ENDGAME.value:
        material = features.get("material", {})
        if material.get("white", 0) <= 5 or material.get("black", 0) <= 5:
            return "endgame_simple"
        return "endgame_complex"
    
    if dtype == MotifType.STRUCTURAL.value:
        if pawn.get("passed_pawns"):
            return "passed_pawn"
        if pawn.get("isolated_pawns"):
            return "isolated_pawn"
        return "structural_other"
    
    return "positional_general"


def extract_from_tick(
    tick_data: Dict[str, Any],
    episode_id: str,
    reward_threshold: float,
    activity_threshold: int,
) -> Optional[BindingDescriptor]:
    """
    Extract a motif from a single tick record.
    
    Returns:
        BindingDescriptor if tick meets criteria, None otherwise.
    """
    env = tick_data.get("env", {})
    reward_tick = env.get("m3_reward_tick") or env.get("reward_tick")
    active_nodes = tick_data.get("nodes", {})
    fen = env.get("fen")
    
    # Skip if no FEN
    if not fen or not chess:
        return None
    
    # Check reward threshold
    reward_value = reward_tick if reward_tick is not None else 0.0
    if abs(reward_value) < reward_threshold:
        # Also check activity threshold
        active_count = sum(1 for state in active_nodes.values() 
                         if state in ("TRUE", "CONFIRMED", "WAITING"))
        if active_count < activity_threshold:
            return None
    
    # Extract features from board
    try:
        board = chess.Board(fen)
        features = extract_all_features(board)
    except Exception:
        return None
    
    # Classify and generate pattern key
    dtype = classify_motif_type(features, reward_value)
    pattern_key = generate_pattern_key(features, dtype)
    
    # Calculate confidence based on reward magnitude and activity
    confidence = min(1.0, abs(reward_value) / 2.0 + 0.3)
    
    # Get active node list
    active_node_list = [
        node_id for node_id, state in active_nodes.items()
        if state in ("TRUE", "CONFIRMED", "WAITING", "REQUESTED")
    ]
    
    return BindingDescriptor(
        dtype=dtype,
        pattern_key=pattern_key,
        context=features,
        active_nodes=active_node_list,
        outcome_score=reward_value,
        source_tick=tick_data.get("tick", 0),
        source_episode=episode_id,
        fen=fen,
        move=env.get("chosen_move"),
        reward_tick=reward_tick,
        confidence=confidence,
        metadata={
            "note": tick_data.get("note", ""),
        },
    )


def process_trace_file(
    trace_path: Path,
    reward_threshold: float = 0.3,
    activity_threshold: int = 3,
) -> MotifDataset:
    """
    Process a trace JSONL file and extract motifs.
    
    Args:
        trace_path: Path to trace JSONL file
        reward_threshold: Minimum |reward_tick| to consider
        activity_threshold: Minimum active nodes to consider
    
    Returns:
        MotifDataset with extracted motifs
    """
    dataset = MotifDataset()
    dataset.metadata["source_file"] = str(trace_path)
    dataset.metadata["reward_threshold"] = reward_threshold
    dataset.metadata["activity_threshold"] = activity_threshold
    
    current_episode = "unknown"
    
    with open(trace_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Check if this is an episode record
            if "episode_id" in data:
                current_episode = data["episode_id"]
                continue
            
            # Try to extract motif from tick record
            motif = extract_from_tick(
                data,
                current_episode,
                reward_threshold,
                activity_threshold,
            )
            
            if motif:
                dataset.add(motif)
    
    return dataset


def process_viz_json(
    viz_path: Path,
    reward_threshold: float = 0.3,
    activity_threshold: int = 3,
) -> MotifDataset:
    """
    Process a visualization JSON file (array of frames) and extract motifs.
    """
    dataset = MotifDataset()
    dataset.metadata["source_file"] = str(viz_path)
    dataset.metadata["reward_threshold"] = reward_threshold
    dataset.metadata["activity_threshold"] = activity_threshold
    
    with open(viz_path) as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        return dataset
    
    episode_id = viz_path.stem
    
    for i, frame in enumerate(data):
        # Add tick number if not present
        if "tick" not in frame:
            frame["tick"] = i
        
        motif = extract_from_tick(
            frame,
            episode_id,
            reward_threshold,
            activity_threshold,
        )
        
        if motif:
            dataset.add(motif)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Extract motifs from trace files for M5 structure discovery."
    )
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to trace JSONL or viz JSON files",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("reports/motifs/extracted.jsonl"),
        help="Output path for motif dataset",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.3,
        help="Minimum |reward_tick| to extract (default: 0.3)",
    )
    parser.add_argument(
        "--activity-threshold",
        type=int,
        default=3,
        help="Minimum active nodes to extract (default: 3)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics about extracted motifs",
    )
    
    args = parser.parse_args()
    
    if chess is None:
        print("Error: python-chess is required. Install with: uv pip install chess")
        sys.exit(1)
    
    # Combine motifs from all input files
    combined = MotifDataset()
    combined.metadata["sources"] = []
    
    for trace_path in args.traces:
        if not trace_path.exists():
            print(f"Warning: File not found: {trace_path}")
            continue
        
        print(f"Processing: {trace_path}")
        
        # Determine file type and process
        if trace_path.suffix == ".jsonl":
            dataset = process_trace_file(
                trace_path,
                args.reward_threshold,
                args.activity_threshold,
            )
        elif trace_path.suffix == ".json":
            dataset = process_viz_json(
                trace_path,
                args.reward_threshold,
                args.activity_threshold,
            )
        else:
            print(f"  Skipping unknown file type: {trace_path.suffix}")
            continue
        
        print(f"  Extracted {len(dataset)} motifs")
        combined.metadata["sources"].append(str(trace_path))
        
        for motif in dataset:
            combined.add(motif)
    
    # Save combined dataset
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.save(args.out)
    print(f"\nSaved {len(combined)} motifs to {args.out}")
    
    # Print statistics if requested
    if args.stats or len(combined) > 0:
        stats = combined.statistics()
        print("\n=== Motif Statistics ===")
        print(f"Total motifs: {stats['count']}")
        if stats['count'] > 0:
            print(f"Average outcome: {stats['avg_outcome']:.3f}")
            print(f"Positive ratio: {stats['positive_ratio']:.1%}")
            print("\nBy type:")
            for dtype, count in stats.get('type_counts', {}).items():
                print(f"  {dtype}: {count}")
            print("\nBy pattern:")
            for pattern, count in sorted(stats.get('pattern_counts', {}).items(), 
                                         key=lambda x: -x[1])[:10]:
                print(f"  {pattern}: {count}")


if __name__ == "__main__":
    main()

