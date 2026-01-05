#!/usr/bin/env python3
"""Quick sanity test for KRK knowledge transfer implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test all required imports."""
    from recon_lite.nodes.stem_cell import StemCellManager
    from recon_lite_chess.features.krk_features import extract_krk_features, KRKFeatures
    from recon_lite.learning.m5_structure import discover_krk_box_method_por, compute_branching_metrics
    print("✅ All imports successful!")
    return True


def test_krk_features():
    """Test KRK feature extraction."""
    import chess
    from recon_lite_chess.features.krk_features import extract_krk_features, KRKFeatures
    
    board = chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")  # KRK position
    features = extract_krk_features(board)
    
    print(f"KRK Features: king_dist={features.king_distance}, box_area={features.box_area}")
    print(f"Universal indices: {KRKFeatures.universal_feature_indices()}")
    
    assert features.king_distance >= 0
    assert features.box_area >= 0
    print("✅ KRK features working!")
    return True


def test_transfer_api():
    """Test knowledge transfer API exists."""
    from recon_lite.nodes.stem_cell import StemCellManager
    
    has_load = hasattr(StemCellManager, "load_with_transfer")
    has_reuse = hasattr(StemCellManager, "compute_sensor_reuse_ratio")
    
    print(f"load_with_transfer exists: {has_load}")
    print(f"compute_sensor_reuse_ratio exists: {has_reuse}")
    
    assert has_load
    assert has_reuse
    print("✅ Knowledge transfer API ready!")
    return True


def test_topologies():
    """Test topology files exist."""
    topologies_dir = Path(__file__).parent.parent / "topologies"
    
    required = ["krk_topology.json", "krk_legs_topology.json"]
    for fname in required:
        path = topologies_dir / fname
        if not path.exists():
            print(f"❌ Missing: {path}")
            return False
        print(f"✅ Found: {fname}")
    
    return True


if __name__ == "__main__":
    all_passed = True
    
    print("\n=== KRK Knowledge Transfer Implementation Tests ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("KRK Features", test_krk_features),
        ("Transfer API", test_transfer_api),
        ("Topologies", test_topologies),
    ]
    
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            if not test_fn():
                all_passed = False
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

