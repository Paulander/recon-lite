"""
Test spawn points with verified mate-in-1 positions.

Uses the checkmate actuator as anchor and tests stem cell exploration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite.graph import Graph
from recon_lite_chess.krk_checkmate_actuator import create_checkmate_actuator
from recon_lite_chess.spawn_point import SpawnPointManager, SpawnPointConfig

# Verified mate-in-1 positions
VERIFIED_POSITIONS = [
    "4k3/7R/4K3/8/8/8/8/8 w - - 0 1",
    "3k4/7R/3K4/8/8/8/8/8 w - - 0 1",
    "3k4/6R1/3K4/8/8/8/8/8 w - - 0 1",
    "8/8/8/k1K5/8/8/1R6/8 w - - 0 1",
    "8/8/8/k1K5/8/8/8/1R6 w - - 0 1",
    "8/8/8/8/8/6K1/R7/6k1 w - - 0 1",
    "8/8/8/8/8/4K3/7R/4k3 w - - 0 1",
    "8/8/8/5K1k/8/8/6R1/8 w - - 0 1",
    "8/8/5K1k/8/8/8/6R1/8 w - - 0 1",
    "5K1k/8/8/8/8/8/6R1/8 w - - 0 1",
    "k7/3R4/1K6/8/8/8/8/8 w - - 0 1",
]


def main():
    print("=" * 70)
    print("Phase 3: Spawn Point Test")
    print("=" * 70)
    
    # Create graph with checkmate actuator
    graph = Graph()
    actuator = create_checkmate_actuator("krk_checkmate")
    graph.add_node(actuator)
    
    # Add fake leg nodes for spawn points
    from recon_lite.graph import Node, NodeType
    for i in range(3):
        leg = Node(nid=f"leg_{i}", ntype=NodeType.SCRIPT)
        graph.add_node(leg)
    
    # Create spawn point manager
    config = SpawnPointConfig(
        spawn_probability=0.5,  # Higher for testing
        max_trials=3,
        trial_lifetime=20
    )
    manager = SpawnPointManager(config)
    manager.attach_to_legs(graph)
    
    print(f"\nProcessing {len(VERIFIED_POSITIONS)} verified positions...")
    print("-" * 70)
    
    # Process each position multiple times to simulate training
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}:")
        
        for i, fen in enumerate(VERIFIED_POSITIONS):
            board = chess.Board(fen)
            
            # Get move from actuator
            env = {"board": board}
            actuator.predicate(actuator, graph, env)
            selected_move = env.get("suggested_move")
            
            if selected_move is None:
                continue
            
            # Check if it achieved checkmate
            board_after = board.copy()
            board_after.push(selected_move)
            is_mate = board_after.is_checkmate()
            
            # Process through spawn points
            manager.process_position(board, selected_move, is_mate)
    
    # Print final stats
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)
    
    stats = manager.get_stats()
    print(f"Total ticks: {stats['tick']}")
    print(f"Active trials: {stats['total_active_trials']}")
    print(f"Total promotions: {stats['total_promotions']}")
    print(f"Total prunes: {stats['total_prunes']}")
    
    # Show trial details
    print("\nTrial Details:")
    for sp_id, sp in manager.spawn_points.items():
        if sp.active_trials:
            print(f"\n  {sp_id}:")
            for trial_id, trial in sp.active_trials.items():
                print(f"    {trial_id}:")
                print(f"      Sensors: {trial.sensor_ids}")
                print(f"      Samples: {trial.samples}")
                print(f"      Mate hits: {trial.checkmate_hits}/{trial.samples}")
                print(f"      XP: {trial.xp:.3f}")
    
    print("\n" + "=" * 70)
    if stats['total_promotions'] > 0:
        print("✓ SUCCESS: Some trials were promoted!")
    else:
        print("No promotions yet - need more training data")
    print("=" * 70)


if __name__ == "__main__":
    main()
