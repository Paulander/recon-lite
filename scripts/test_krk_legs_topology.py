#!/usr/bin/env python3
"""Quick test to verify KRK legs topology loads and factories work."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from recon_lite_chess.graph.builder import build_graph_from_topology
from recon_lite_chess.krk_nodes import create_krk_rook_leg, create_krk_king_leg, create_krk_arbiter

def test_factories():
    """Test that factory functions can be called."""
    print("Testing factory functions...")
    
    rook_leg = create_krk_rook_leg("test_rook_leg")
    assert rook_leg.ntype.name == "SCRIPT", f"Expected SCRIPT, got {rook_leg.ntype.name}"
    assert rook_leg.predicate is not None, "Rook leg should have predicate"
    print("  ✅ create_krk_rook_leg works")
    
    king_leg = create_krk_king_leg("test_king_leg")
    assert king_leg.ntype.name == "SCRIPT", f"Expected SCRIPT, got {king_leg.ntype.name}"
    assert king_leg.predicate is not None, "King leg should have predicate"
    print("  ✅ create_krk_king_leg works")
    
    arbiter = create_krk_arbiter("test_arbiter")
    assert arbiter.ntype.name == "SCRIPT", f"Expected SCRIPT, got {arbiter.ntype.name}"
    assert arbiter.predicate is not None, "Arbiter should have predicate"
    print("  ✅ create_krk_arbiter works")


def test_topology_load():
    """Test that topology can be loaded."""
    print("\nTesting topology load...")
    
    topology_path = Path("topologies/krk_legs_topology.json")
    if not topology_path.exists():
        print(f"  ❌ Topology file not found: {topology_path}")
        return False
    
    try:
        graph = build_graph_from_topology(topology_path)
        print(f"  ✅ Topology loaded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Check key nodes exist
        required_nodes = ["krk_root", "krk_rook_leg", "krk_king_leg", "krk_arbiter"]
        for node_id in required_nodes:
            if node_id not in graph.nodes:
                print(f"  ❌ Missing node: {node_id}")
                return False
            node = graph.nodes[node_id]
            print(f"    - {node_id}: {node.ntype.name}")
        
        # Verify arbiter is SCRIPT
        arbiter = graph.nodes["krk_arbiter"]
        if arbiter.ntype.name != "SCRIPT":
            print(f"  ❌ Arbiter should be SCRIPT, got {arbiter.ntype.name}")
            return False
        
        # Verify legs have predicates
        rook_leg = graph.nodes["krk_rook_leg"]
        king_leg = graph.nodes["krk_king_leg"]
        if rook_leg.predicate is None:
            print("  ❌ Rook leg missing predicate")
            return False
        if king_leg.predicate is None:
            print("  ❌ King leg missing predicate")
            return False
        
        print("  ✅ All nodes have correct types and predicates")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading topology: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("KRK Legs Topology Test")
    print("=" * 60)
    
    test_factories()
    success = test_topology_load()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed")
        sys.exit(1)

