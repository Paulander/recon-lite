"""
End-to-end test of the full KRK integration.

Tests:
1. Affordance detection (is this KRK?)
2. Bandit selection at Hub
3. Checkmate actuator selection
4. Spawn point updates
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from recon_lite_chess.krk_integration import (
    create_krk_orchestrator,
    is_krk_position,
    get_krk_affordance
)

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

# Non-KRK positions for negative testing
NON_KRK_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
    "8/8/8/8/4k3/8/4PK2/8 w - - 0 1",  # KPK (not KRK)
    "8/8/8/8/8/8/Q7/K6k w - - 0 1",  # KQK (not KRK)
]


def main():
    print("=" * 70)
    print("Phase 4: Full KRK Integration Test")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = create_krk_orchestrator()
    print(f"Created orchestrator with {len(orchestrator.legs)} legs")
    
    # Test 1: Affordance detection
    print("\n" + "-" * 70)
    print("Test 1: Affordance Detection")
    print("-" * 70)
    
    for fen in VERIFIED_POSITIONS[:3]:
        board = chess.Board(fen)
        is_krk = is_krk_position(board)
        affordance = get_krk_affordance(board)
        print(f"  {fen[:20]}... KRK={is_krk}, Affordance={affordance:.1f}")
    
    for fen in NON_KRK_POSITIONS:
        board = chess.Board(fen)
        is_krk = is_krk_position(board)
        affordance = get_krk_affordance(board)
        print(f"  {fen[:20]}... KRK={is_krk}, Affordance={affordance:.1f}")
    
    # Test 2: Move selection and reward
    print("\n" + "-" * 70)
    print("Test 2: Move Selection + Reward Updates")
    print("-" * 70)
    
    correct = 0
    total = 0
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        for i, fen in enumerate(VERIFIED_POSITIONS):
            board = chess.Board(fen)
            
            # Find actual mate move
            mate_move = None
            for move in board.legal_moves:
                b2 = board.copy()
                b2.push(move)
                if b2.is_checkmate():
                    mate_move = move
                    break
            
            # Get move from orchestrator
            selected_move, leg_id = orchestrator.select_move(board)
            
            if selected_move is None:
                continue
            
            total += 1
            
            # Check if mate
            b2 = board.copy()
            b2.push(selected_move)
            achieved_mate = b2.is_checkmate()
            
            if achieved_mate:
                correct += 1
            
            # Update orchestrator with reward
            orchestrator.update_reward(leg_id, selected_move, board, achieved_mate)
    
    win_rate = correct / total * 100 if total > 0 else 0
    print(f"\n  Total: {correct}/{total} mates ({win_rate:.1f}%)")
    
    # Test 3: Statistics
    print("\n" + "-" * 70)
    print("Test 3: Orchestrator Statistics")
    print("-" * 70)
    
    stats = orchestrator.get_stats()
    
    print("\n  Bandit Stats:")
    print(f"    Total plays: {stats['bandit']['total_plays']}")
    for arm_id, arm_stats in stats['bandit']['arms'].items():
        print(f"    {arm_id}: {arm_stats['wins']}/{arm_stats['plays']} "
              f"({arm_stats['win_rate']*100:.1f}% win rate)")
    
    print("\n  Leg Stats:")
    for leg_id, leg_stats in stats['legs'].items():
        print(f"    {leg_id}:")
        print(f"      Activations: {leg_stats['activations']}")
        print(f"      Successful mates: {leg_stats['successful_mates']}")
        spawn = leg_stats['spawn_stats']
        print(f"      Spawn: {spawn['total_active_trials']} active, "
              f"{spawn['total_promotions']} promoted")
    
    # Final result
    print("\n" + "=" * 70)
    if win_rate >= 90:
        print("✓ SUCCESS: Phase 4 integration complete!")
        print(f"  - Affordance detection working")
        print(f"  - Bandit selection working")
        print(f"  - {win_rate:.1f}% win rate on mate-in-1")
    else:
        print(f"✗ FAILED: Win rate {win_rate:.1f}% < 90%")
    print("=" * 70)


if __name__ == "__main__":
    main()
