#!/usr/bin/env python3
"""Quick test of Bach-Integrated components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_gating_schedule():
    """Test GatingSchedule implementation."""
    from recon_lite.engine import GatingSchedule
    
    gs = GatingSchedule()
    print("GatingSchedule Test:")
    print(f"  Game 0: {gs.get_strictness(0)*100:.0f}% strictness")
    print(f"  Game 50: {gs.get_strictness(50)*100:.0f}% strictness")
    print(f"  Game 100: {gs.get_strictness(100)*100:.0f}% strictness")
    
    # Test win-based mode
    gs_win = GatingSchedule(win_based=True)
    print("\nWin-Based Mode:")
    print(f"  0 wins: {gs_win.get_strictness(0)*100:.0f}% strictness")
    gs_win.record_win()
    gs_win.record_win()
    print(f"  After 2 wins: {gs_win.get_strictness(0)*100:.0f}% strictness")
    return True

def test_binding_space():
    """Test BindingNodeSpace implementation."""
    from recon_lite.nodes.binding_space import BindingNodeSpace, BindingSpaceRegistry
    
    print("\nBindingNodeSpace Test:")
    
    # Create a space
    space = BindingNodeSpace(
        space_id="test_king_binding",
        member_nodes=["node_a", "node_b"],
        required_bindings={"piece_type", "role"}
    )
    
    print(f"  Initial state: {space.state.name}")
    
    # Bind variables
    space.bind("piece_type", "king", var_type="piece")
    print(f"  After binding piece_type: {space.state.name}")
    
    space.bind("role", "approach", var_type="role")
    print(f"  After binding role: {space.state.name}")
    
    # Test context
    context = space.get_binding_context()
    print(f"  Binding context: {context}")
    
    # Test registry
    registry = BindingSpaceRegistry()
    registry.create_space("krk_king", member_nodes=["krk_king_leg"])
    registry.add_node_to_space("universal_king_logic", "krk_king")
    
    print(f"  Registry spaces: {list(registry.spaces.keys())}")
    return True

def test_think_harder_config():
    """Test Think Harder configuration in EvolutionConfig."""
    from scripts.evolution_driver import EvolutionConfig
    
    print("\nEvolutionConfig Think Harder Test:")
    
    config = EvolutionConfig(
        enable_think_harder=True,
        pure_cognitive_mode=True,
        enable_progressive_gating=True,
        gating_initial_strictness=0.30,
    )
    
    print(f"  Think Harder: {config.enable_think_harder}")
    print(f"  Pure Cognitive: {config.pure_cognitive_mode}")
    print(f"  Progressive Gating: {config.enable_progressive_gating}")
    print(f"  Initial Strictness: {config.gating_initial_strictness}")
    return True

def test_krk_topology():
    """Test KRK topology has universal_king_logic."""
    import json
    
    print("\nKRK Topology Test:")
    
    topo_path = Path("topologies/krk_legs_topology.json")
    if not topo_path.exists():
        print(f"  ERROR: {topo_path} not found")
        return False
    
    with open(topo_path) as f:
        topo = json.load(f)
    
    nodes = {n["id"]: n for n in topo.get("nodes", [])}
    
    # Check for universal_king_logic
    if "universal_king_logic" in nodes:
        print(f"  universal_king_logic: PRESENT")
        ukl = nodes["universal_king_logic"]
        print(f"    origin: {ukl.get('meta', {}).get('origin')}")
    else:
        print(f"  universal_king_logic: MISSING")
        return False
    
    # Check gating on legs
    for leg in ["krk_rook_leg", "krk_king_leg"]:
        if leg in nodes:
            gated = nodes[leg].get("meta", {}).get("require_child_confirm", False)
            print(f"  {leg}: gating={'ON' if gated else 'OFF'}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("BACH-INTEGRATED COMPONENTS TEST")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_gating_schedule()
    except Exception as e:
        print(f"GatingSchedule ERROR: {e}")
        all_passed = False
    
    try:
        all_passed &= test_binding_space()
    except Exception as e:
        print(f"BindingSpace ERROR: {e}")
        all_passed = False
    
    try:
        all_passed &= test_think_harder_config()
    except Exception as e:
        print(f"ThinkHarder ERROR: {e}")
        all_passed = False
    
    try:
        all_passed &= test_krk_topology()
    except Exception as e:
        print(f"KRK Topology ERROR: {e}")
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

