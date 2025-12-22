#!/usr/bin/env python3
"""
KPK Epoch Training with Weight Snapshots.

Trains KPK endgame positions using the FULL UNIFIED GRAPH (not standalone KPK).
Saves weight snapshots after each epoch for visualization/animation.

Usage:
    uv run python scripts/kpk_epoch_train.py --epochs 10 --games-per-epoch 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite import Graph, ReConEngine, NodeState
from recon_lite_chess.graph.unified_builder import build_unified_graph
from recon_lite.plasticity.consolidate import ConsolidationEngine
from recon_lite.trace_db import EpisodeSummary


def create_random_kpk_board(white_to_move: bool = True) -> chess.Board:
    """Create a random valid KPK position."""
    import random
    
    while True:
        defender_king_sq = random.randint(0, 63)
        attacker_king_sq = random.randint(0, 63)
        if attacker_king_sq == defender_king_sq:
            continue
        
        dk_file = chess.square_file(defender_king_sq)
        dk_rank = chess.square_rank(defender_king_sq)
        ak_file = chess.square_file(attacker_king_sq)
        ak_rank = chess.square_rank(attacker_king_sq)
        
        if abs(dk_file - ak_file) <= 1 and abs(dk_rank - ak_rank) <= 1:
            continue
        
        pawn_rank = random.randint(1, 6)
        pawn_file = random.randint(0, 7)
        pawn_sq = chess.square(pawn_file, pawn_rank)
        
        if pawn_sq in (defender_king_sq, attacker_king_sq):
            continue
        
        board = chess.Board.empty()
        
        if white_to_move:
            board.set_piece_at(attacker_king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(defender_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.turn = chess.WHITE
        else:
            board.set_piece_at(attacker_king_sq, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.BLACK))
            board.set_piece_at(defender_king_sq, chess.Piece(chess.KING, chess.WHITE))
            board.turn = chess.BLACK
        
        if not board.is_valid():
            continue
        if board.is_game_over():
            continue
        
        return board


def play_kpk_game(
    graph: Graph,
    engine: ReConEngine,
    max_moves: int = 100,
) -> Dict[str, Any]:
    """Play a single KPK game using the full unified graph."""
    board = create_random_kpk_board()
    starting_fen = board.fen()
    moves = []
    promoted = False
    edge_activations: Dict[str, float] = {}
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        
        # Reset graph state
        for node in graph.nodes.values():
            node.state = NodeState.INACTIVE
        graph.nodes["GameRoot"].state = NodeState.REQUESTED
        
        # Run engine
        env = {"board": board}
        engine.step(env)
        
        # Track which edges fired
        for e in graph.edges:
            src_node = graph.nodes.get(e.src)
            dst_node = graph.nodes.get(e.dst)
            if src_node and dst_node:
                if src_node.state in (NodeState.TRUE, NodeState.CONFIRMED):
                    if dst_node.state in (NodeState.TRUE, NodeState.CONFIRMED, NodeState.WAITING):
                        key = f"{e.src}->{e.dst}:{e.ltype.name}"
                        edge_activations[key] = edge_activations.get(key, 0) + 1
        
        # Get suggested move from KPK policy
        suggested = env.get("kpk", {}).get("policy", {}).get("suggested_move")
        
        if suggested:
            try:
                move = chess.Move.from_uci(suggested)
                if move in board.legal_moves:
                    if move.promotion:
                        promoted = True
                    board.push(move)
                    moves.append(move.uci())
                    continue
            except:
                pass
        
        # Fallback: first legal move
        legal = list(board.legal_moves)
        if legal:
            move = legal[0]
            board.push(move)
            moves.append(move.uci())
    
    # Determine outcome
    if board.is_checkmate():
        outcome = "checkmate"
        reward = 1.0
    elif promoted:
        outcome = "promoted"
        reward = 0.8
    elif board.is_stalemate():
        outcome = "stalemate"
        reward = -0.5
    else:
        outcome = "timeout"
        reward = -0.2
    
    return {
        "starting_fen": starting_fen,
        "outcome": outcome,
        "moves": len(moves),
        "promoted": promoted,
        "reward": reward,
        "edge_activations": edge_activations,
    }


def save_snapshot(
    consolidation: ConsolidationEngine,
    output_dir: Path,
    epoch: int,
    stats: Dict[str, Any],
) -> Path:
    """Save a weight snapshot for this epoch."""
    snapshot_path = output_dir / f"epoch_{epoch:03d}.json"
    
    weights = consolidation.get_all_w_base()
    
    snapshot = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "w_base": weights,
    }
    
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    
    return snapshot_path


def run_epoch(
    graph: Graph,
    engine: ReConEngine,
    consolidation: ConsolidationEngine,
    games_per_epoch: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one training epoch."""
    results = []
    
    for i in range(games_per_epoch):
        result = play_kpk_game(graph, engine)
        results.append(result)
        
        # Create episode summary for consolidation
        summary = EpisodeSummary(
            edge_delta_sums={k: v * result["reward"] for k, v in result["edge_activations"].items()},
            outcome_score=result["reward"],
            avg_reward_tick=result["reward"],
            total_reward_tick=result["reward"],
            reward_tick_count=1,
        )
        consolidation.accumulate_episode(summary)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"    Game {i+1}/{games_per_epoch}: {result['outcome']}")
    
    # Aggregate stats
    promotions = sum(1 for r in results if r["promoted"])
    checkmates = sum(1 for r in results if r["outcome"] == "checkmate")
    stalemates = sum(1 for r in results if r["outcome"] == "stalemate")
    timeouts = sum(1 for r in results if r["outcome"] == "timeout")
    avg_moves = sum(r["moves"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    
    # Apply consolidation
    if consolidation.should_apply():
        consolidation.apply_to_graph(graph)
    
    return {
        "games": games_per_epoch,
        "promotions": promotions,
        "promotion_rate": promotions / games_per_epoch,
        "checkmates": checkmates,
        "stalemates": stalemates,
        "timeouts": timeouts,
        "avg_moves": avg_moves,
        "avg_reward": avg_reward,
    }


def main():
    parser = argparse.ArgumentParser(description="KPK Epoch Training with Snapshots")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--games-per-epoch", type=int, default=50, help="Games per epoch")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--base-weights", type=str, default=None, help="Starting weights")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"weights/runs/kpk_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Build unified graph
    print("Building unified graph...")
    graph = build_unified_graph(include_endgames=True, include_tactics=False)
    engine = ReConEngine(graph)
    
    # Initialize consolidation
    consolidation = ConsolidationEngine()
    
    # Get KPK edge keys
    kpk_edges = [
        f"{e.src}->{e.dst}:{e.ltype.name}"
        for e in graph.edges
        if "kpk_" in e.src or "kpk_" in e.dst
    ]
    consolidation.init_from_graph(graph, edge_whitelist=kpk_edges)
    
    # Load base weights if provided
    if args.base_weights and Path(args.base_weights).exists():
        print(f"Loading base weights from {args.base_weights}")
        consolidation.load_state(Path(args.base_weights))
    
    # Save initial snapshot
    initial_stats = {"games": 0, "promotions": 0, "promotion_rate": 0.0}
    save_snapshot(consolidation, output_dir, 0, initial_stats)
    print("Saved initial snapshot (epoch 0)")
    
    # Training loop
    print(f"\n=== Training {args.epochs} epochs, {args.games_per_epoch} games each ===\n")
    
    all_stats = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}...")
        
        stats = run_epoch(
            graph, engine, consolidation,
            games_per_epoch=args.games_per_epoch,
            verbose=args.verbose,
        )
        all_stats.append(stats)
        
        snapshot_path = save_snapshot(consolidation, output_dir, epoch, stats)
        
        print(f"  Promotion rate: {stats['promotion_rate']*100:.1f}%")
        print(f"  Avg reward: {stats['avg_reward']:.3f}")
        print(f"  Saved: {snapshot_path.name}")
    
    # Save final weights
    final_path = output_dir / "final_weights.json"
    consolidation.save_state(final_path)
    
    # Copy to latest
    latest_path = Path("weights/latest/kpk_epoch_trained.json")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(final_path, latest_path)
    
    # Save summary
    summary = {
        "epochs": args.epochs,
        "games_per_epoch": args.games_per_epoch,
        "total_games": args.epochs * args.games_per_epoch,
        "output_dir": str(output_dir),
        "epoch_stats": all_stats,
        "final_promotion_rate": all_stats[-1]["promotion_rate"] if all_stats else 0,
    }
    
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Final promotion rate: {summary['final_promotion_rate']*100:.1f}%")
    print(f"Snapshots saved to: {output_dir}")
    print(f"Final weights: {latest_path}")


if __name__ == "__main__":
    main()
