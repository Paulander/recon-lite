#!/usr/bin/env python3
"""Tactical Pattern Training and Evaluation.

Loads positions from FEN files with known tactical solutions,
runs ReCoN's tactical detectors, and trains via plasticity/consolidation.

This is used for micro-tactic training before full game sessions.

Usage:
    # Evaluate fork detection accuracy
    uv run python demos/experiments/tactics_eval.py \
        --fen-file data/puzzles/fork/lichess_fork.fen \
        --tactic-type fork
    
    # Train with consolidation
    uv run python demos/experiments/tactics_eval.py \
        --fen-file data/puzzles/fork/lichess_fork.fen \
        --tactic-type fork \
        --consolidate \
        --consolidate-pack weights/tactics/fork_consol.json
    
    # Batch evaluation across all tactics
    uv run python demos/experiments/tactics_eval.py \
        --fen-dir data/puzzles \
        --all-tactics
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import chess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recon_lite.graph import Graph, Node, NodeType
from recon_lite.plasticity.consolidate import ConsolidationEngine, ConsolidationConfig
from recon_lite_chess.scripts.tactics import (
    detect_forks,
    detect_pins,
    detect_skewers,
    detect_hanging_pieces,
    detect_back_rank_weakness,
    detect_discovered_attacks,
    get_fork_moves,
    get_pin_exploit_moves,
    get_skewer_moves,
    get_capture_hanging_moves,
    get_back_rank_moves,
    get_discovered_attack_moves,
    build_tactics_network,
)


@dataclass
class TacticsResult:
    """Result from evaluating a single tactic position."""
    fen: str
    expected_move: str
    detected: bool
    proposed_moves: List[str]
    correct: bool
    tactic_type: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TacticsStats:
    """Aggregate statistics for tactic evaluation."""
    total: int = 0
    detected: int = 0
    correct: int = 0
    by_tactic: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    @property
    def detection_rate(self) -> float:
        return self.detected / self.total if self.total > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "detected": self.detected,
            "correct": self.correct,
            "detection_rate": self.detection_rate,
            "accuracy": self.accuracy,
            "by_tactic": self.by_tactic,
        }


# Tactic type to detection/move functions mapping
TACTIC_HANDLERS = {
    "fork": {
        "detect": detect_forks,
        "get_moves": get_fork_moves,
        "check_detected": lambda result: len(result) > 0,
    },
    "pin": {
        "detect": detect_pins,
        "get_moves": get_pin_exploit_moves,
        "check_detected": lambda result: len(result) > 0,
    },
    "skewer": {
        "detect": detect_skewers,
        "get_moves": get_skewer_moves,
        "check_detected": lambda result: len(result) > 0,
    },
    "hanging": {
        "detect": detect_hanging_pieces,
        "get_moves": get_capture_hanging_moves,
        "check_detected": lambda result: len(result.get("enemy_hanging", [])) > 0,
    },
    "hangingPiece": {
        "detect": detect_hanging_pieces,
        "get_moves": get_capture_hanging_moves,
        "check_detected": lambda result: len(result.get("enemy_hanging", [])) > 0,
    },
    "backRankMate": {
        "detect": detect_back_rank_weakness,
        "get_moves": get_back_rank_moves,
        "check_detected": lambda result: result.get("has_weakness", False),
    },
    "back_rank": {
        "detect": detect_back_rank_weakness,
        "get_moves": get_back_rank_moves,
        "check_detected": lambda result: result.get("has_weakness", False),
    },
    "discoveredAttack": {
        "detect": detect_discovered_attacks,
        "get_moves": get_discovered_attack_moves,
        "check_detected": lambda result: len(result) > 0,
    },
    "discovered": {
        "detect": detect_discovered_attacks,
        "get_moves": get_discovered_attack_moves,
        "check_detected": lambda result: len(result) > 0,
    },
}


def parse_fen_file(path: Path) -> List[Tuple[str, str, str]]:
    """
    Parse a FEN file with format: FEN ; best_move ; description
    
    Returns list of (fen, best_move, description) tuples.
    """
    positions = []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split(";")
            if len(parts) >= 2:
                fen = parts[0].strip()
                best_move = parts[1].strip()
                description = parts[2].strip() if len(parts) > 2 else ""
                positions.append((fen, best_move, description))
    
    return positions


def evaluate_tactic(
    fen: str,
    expected_move: str,
    tactic_type: str,
) -> TacticsResult:
    """
    Evaluate ReCoN's tactic detection on a single position.
    
    Returns TacticsResult with detection status and proposed moves.
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        return TacticsResult(
            fen=fen,
            expected_move=expected_move,
            detected=False,
            proposed_moves=[],
            correct=False,
            tactic_type=tactic_type,
            details={"error": str(e)},
        )
    
    # Get handler for this tactic type
    handler = TACTIC_HANDLERS.get(tactic_type)
    if handler is None:
        # Try generic approach - run all detectors
        handler = TACTIC_HANDLERS.get("fork")  # Default
    
    # Run detection
    detect_func = handler["detect"]
    detection_result = detect_func(board)
    detected = handler["check_detected"](detection_result)
    
    # Get proposed moves
    get_moves_func = handler["get_moves"]
    proposed_moves = [m.uci() for m in get_moves_func(board)]
    
    # Check if expected move is among proposed
    # Normalize expected move (could be SAN or UCI)
    expected_uci = expected_move
    try:
        # Try parsing as SAN
        move = board.parse_san(expected_move)
        expected_uci = move.uci()
    except Exception:
        pass  # Already UCI or invalid
    
    correct = expected_uci in proposed_moves
    
    return TacticsResult(
        fen=fen,
        expected_move=expected_move,
        detected=detected,
        proposed_moves=proposed_moves,
        correct=correct,
        tactic_type=tactic_type,
        details={
            "detection_result": str(detection_result)[:200],
            "expected_uci": expected_uci,
        },
    )


def evaluate_file(
    path: Path,
    tactic_type: str,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[TacticsResult], TacticsStats]:
    """
    Evaluate all positions in a FEN file.
    
    Returns list of results and aggregate statistics.
    """
    positions = parse_fen_file(path)
    if limit:
        positions = positions[:limit]
    
    results = []
    stats = TacticsStats()
    stats.by_tactic[tactic_type] = {"total": 0, "detected": 0, "correct": 0}
    
    for i, (fen, best_move, desc) in enumerate(positions):
        result = evaluate_tactic(fen, best_move, tactic_type)
        results.append(result)
        
        stats.total += 1
        stats.by_tactic[tactic_type]["total"] += 1
        
        if result.detected:
            stats.detected += 1
            stats.by_tactic[tactic_type]["detected"] += 1
        
        if result.correct:
            stats.correct += 1
            stats.by_tactic[tactic_type]["correct"] += 1
        
        if verbose and (i + 1) % 50 == 0:
            print(f"\rProcessed {i + 1}/{len(positions)} positions...", end="")
    
    if verbose:
        print()
    
    return results, stats


def apply_training_reward(
    result: TacticsResult,
    consol_engine: ConsolidationEngine,
    graph: Graph,
) -> float:
    """
    Apply plasticity reward based on tactic result.
    
    Returns reward value applied.
    """
    # Determine reward
    if result.correct:
        reward = 1.0  # Found the correct move
    elif result.detected:
        reward = 0.3  # Detected pattern but wrong move
    else:
        reward = -0.3  # Failed to detect pattern
    
    # Map tactic type to graph nodes
    tactic_node_map = {
        "fork": ["detect_fork", "exploit_fork"],
        "pin": ["detect_pin", "exploit_pin"],
        "skewer": ["detect_skewer", "exploit_skewer"],
        "hanging": ["detect_hanging", "capture_hanging"],
        "hangingPiece": ["detect_hanging", "capture_hanging"],
        "backRankMate": ["detect_back_rank", "exploit_back_rank"],
        "back_rank": ["detect_back_rank", "exploit_back_rank"],
        "discoveredAttack": ["detect_discovered", "exploit_discovered"],
        "discovered": ["detect_discovered", "exploit_discovered"],
    }
    
    nodes = tactic_node_map.get(result.tactic_type, [])
    
    # Apply reward to relevant edges
    for node_id in nodes:
        if node_id in graph.nodes:
            consol_engine.update_edge_weight(
                f"tactics_root->{node_id}",
                reward,
                signal_type="reward",
            )
    
    return reward


def main():
    parser = argparse.ArgumentParser(description="Tactical Pattern Training and Evaluation")
    parser.add_argument("--fen-file", type=str, help="Path to FEN file with tactics positions")
    parser.add_argument("--fen-dir", type=str, help="Directory containing tactic FEN files")
    parser.add_argument("--tactic-type", type=str, help="Type of tactic to evaluate")
    parser.add_argument("--all-tactics", action="store_true", help="Evaluate all available tactics")
    parser.add_argument("--limit", type=int, help="Limit positions to evaluate per file")
    parser.add_argument("--consolidate", action="store_true", help="Apply consolidation training")
    parser.add_argument("--consolidate-pack", type=str, help="Path to consolidation state file")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Collect files to evaluate
    files_to_eval: List[Tuple[Path, str]] = []
    
    if args.fen_file:
        path = Path(args.fen_file)
        tactic_type = args.tactic_type or "unknown"
        files_to_eval.append((path, tactic_type))
    
    if args.fen_dir:
        fen_dir = Path(args.fen_dir)
        for fen_path in fen_dir.rglob("*.fen"):
            # Infer tactic type from path
            parts = fen_path.parts
            tactic_type = "unknown"
            for part in parts:
                if part in TACTIC_HANDLERS:
                    tactic_type = part
                    break
                # Check directory name
                for known_tactic in TACTIC_HANDLERS:
                    if known_tactic.lower() in part.lower():
                        tactic_type = known_tactic
                        break
            
            if args.all_tactics or tactic_type != "unknown":
                files_to_eval.append((fen_path, tactic_type))
    
    if not files_to_eval:
        print("No files to evaluate. Use --fen-file or --fen-dir")
        sys.exit(1)
    
    # Initialize consolidation if requested
    consol_engine = None
    graph = None
    if args.consolidate:
        graph = build_tactics_network()
        consol_engine = ConsolidationEngine(ConsolidationConfig(enabled=True))
        consol_engine.init_from_graph(graph)
        
        if args.consolidate_pack:
            pack_path = Path(args.consolidate_pack)
            if pack_path.exists():
                consol_engine.load_state(pack_path)
                if verbose:
                    print(f"Loaded consolidation state from {pack_path}")
    
    # Evaluate files
    all_results = []
    total_stats = TacticsStats()
    
    if verbose:
        print(f"\n=== Tactical Evaluation ===")
        print(f"Files to evaluate: {len(files_to_eval)}")
        if args.consolidate:
            print("Training mode: ENABLED")
    
    for path, tactic_type in files_to_eval:
        if not path.exists():
            if verbose:
                print(f"Skipping (not found): {path}")
            continue
        
        if verbose:
            print(f"\n--- {path.name} ({tactic_type}) ---")
        
        results, stats = evaluate_file(path, tactic_type, args.limit, verbose)
        all_results.extend(results)
        
        # Aggregate stats
        total_stats.total += stats.total
        total_stats.detected += stats.detected
        total_stats.correct += stats.correct
        for tactic, tactic_stats in stats.by_tactic.items():
            if tactic not in total_stats.by_tactic:
                total_stats.by_tactic[tactic] = {"total": 0, "detected": 0, "correct": 0}
            total_stats.by_tactic[tactic]["total"] += tactic_stats["total"]
            total_stats.by_tactic[tactic]["detected"] += tactic_stats["detected"]
            total_stats.by_tactic[tactic]["correct"] += tactic_stats["correct"]
        
        if verbose:
            print(f"Detection rate: {stats.detection_rate:.1%}")
            print(f"Accuracy: {stats.accuracy:.1%}")
        
        # Apply training if enabled
        if consol_engine and graph:
            if verbose:
                print("Applying training rewards...")
            
            for result in results:
                apply_training_reward(result, consol_engine, graph)
    
    # Save consolidation state
    if consol_engine and args.consolidate_pack:
        pack_path = Path(args.consolidate_pack)
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        consol_engine.save_state(pack_path)
        if verbose:
            print(f"\nSaved consolidation state to {pack_path}")
    
    # Print summary
    if verbose:
        print(f"\n=== Summary ===")
        print(f"Total positions: {total_stats.total}")
        print(f"Detection rate: {total_stats.detection_rate:.1%}")
        print(f"Accuracy: {total_stats.accuracy:.1%}")
        print("\nBy tactic:")
        for tactic, tactic_stats in sorted(total_stats.by_tactic.items()):
            t = tactic_stats["total"]
            d = tactic_stats["detected"]
            c = tactic_stats["correct"]
            print(f"  {tactic}: {c}/{t} correct ({c/t:.1%}), {d}/{t} detected ({d/t:.1%})" if t > 0 else f"  {tactic}: no data")
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "stats": total_stats.to_dict(),
            "results": [
                {
                    "fen": r.fen,
                    "expected": r.expected_move,
                    "detected": r.detected,
                    "correct": r.correct,
                    "proposed": r.proposed_moves,
                    "tactic": r.tactic_type,
                }
                for r in all_results
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to {output_path}")
    
    # Print final stats for script parsing
    print(f"\nTACTICS_EVAL_RESULT: total={total_stats.total} detected={total_stats.detected} correct={total_stats.correct} accuracy={total_stats.accuracy:.3f}")


if __name__ == "__main__":
    main()

