#!/usr/bin/env python3
"""Test that TRIAL nodes are now wired to legs for gating support."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import tempfile

from recon_lite.nodes.stem_cell import StemCellTerminal, StemCellState
from recon_lite.models.registry import TopologyRegistry


def test_trial_leg_wiring():
    """Test that promote_to_trial wires TRIAL nodes to legs."""
    print("=" * 60)
    print("TEST: TRIAL Nodes Wired to Legs for Gating")
    print("=" * 60)
    
    # Create a temporary topology file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        base_topo = {
            "version": "2.0",
            "network": "test",
            "nodes": [
                {"id": "kpk_detect", "type": "SCRIPT", "group": "backbone", "meta": {}},
                {"id": "kpk_pawn_leg", "type": "SCRIPT", "group": "actuator", "meta": {}},
                {"id": "kpk_king_leg", "type": "SCRIPT", "group": "actuator", "meta": {}},
            ],
            "edges": []
        }
        json.dump(base_topo, f)
        topo_path = Path(f.name)
    
    try:
        # Create registry
        registry = TopologyRegistry(topo_path)
        
        # Create a stem cell with some samples
        from recon_lite.nodes.stem_cell import StemCellConfig, StemCellSample
        
        config = StemCellConfig()
        cell = StemCellTerminal("test_cell", config=config)
        cell.state = StemCellState.CANDIDATE
        
        # Add some fake samples for pattern analysis
        for i in range(5):
            sample = StemCellSample(
                fen="8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",  # Simple KPK position
                features=[0.5, 0.3, 0.7, 0.2, 0.8],
                reward=1.0,
                tick=i,
            )
            cell.samples.append(sample)
        
        # Promote to TRIAL
        print("\n1. Promoting cell to TRIAL...")
        success = cell.promote_to_trial(
            registry=registry,
            parent_id="kpk_detect",
            current_tick=100,
            min_consistency=0.0,  # Allow promotion even with low consistency
            wire_to_legs=True,    # This is the key!
        )
        
        print(f"   Promotion success: {success}")
        print(f"   TRIAL node ID: {cell.trial_node_id}")
        
        if not success:
            print("\n❌ FAIL: Promotion failed")
            return False
        
        # Check edges in registry
        print("\n2. Checking edges in registry...")
        # Edges were already saved, get them directly
        
        edges = list(registry.get_all_edges())
        trial_edges = [e for e in edges if cell.trial_node_id in (e.src, e.dst)]
        
        print(f"   Total edges: {len(edges)}")
        print(f"   Edges involving TRIAL: {len(trial_edges)}")
        
        for e in trial_edges:
            print(f"   - {e.src} --{e.type}--> {e.dst}")
        
        # Verify wiring
        detect_edge = any(e.src == "kpk_detect" and e.dst == cell.trial_node_id for e in trial_edges)
        pawn_leg_edge = any(e.src == "kpk_pawn_leg" and e.dst == cell.trial_node_id for e in trial_edges)
        king_leg_edge = any(e.src == "kpk_king_leg" and e.dst == cell.trial_node_id for e in trial_edges)
        
        print(f"\n3. Wiring verification:")
        print(f"   kpk_detect -> TRIAL: {'✓' if detect_edge else '✗'}")
        print(f"   kpk_pawn_leg -> TRIAL: {'✓' if pawn_leg_edge else '✗'}")
        print(f"   kpk_king_leg -> TRIAL: {'✓' if king_leg_edge else '✗'}")
        
        all_wired = detect_edge and pawn_leg_edge and king_leg_edge
        
        if all_wired:
            print("\n✅ SUCCESS: TRIAL node is wired to all required parents!")
            print("   Gating (require_child_confirm) should now work on legs.")
        else:
            print("\n⚠️ PARTIAL: Some wiring is missing")
        
        return all_wired
        
    finally:
        # Cleanup
        topo_path.unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_trial_leg_wiring()
    sys.exit(0 if success else 1)

