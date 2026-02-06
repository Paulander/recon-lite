#!/usr/bin/env python3
"""Test the pure ReCoN learned topology."""

from recon_lite_chess.graph.builder import build_graph_from_topology

# Test topology loading
try:
    g = build_graph_from_topology("topologies/kpk_learned_topology.json")
    print(f"✓ Loaded {len(g.nodes)} nodes")
    print(f"  Nodes: {list(g.nodes.keys())}")
except Exception as e:
    print(f"✗ Failed to load topology: {e}")
    import traceback
    traceback.print_exc()

# Test feature vector
try:
    import chess
    from recon_lite.nodes.stem_cell import StemCellManager
    
    mgr = StemCellManager()
    board = chess.Board()
    board.clear()
    # Set up a KPK position
    board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.D6, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.D8, chess.Piece(chess.KING, chess.BLACK))
    
    features = mgr._default_board_features(board)
    print(f"✓ Feature vector: {len(features)} features")
    print(f"  Last 8 features (opposition-related):")
    for i, f in enumerate(features[-8:]):
        print(f"    [{len(features)-8+i}] = {f:.3f}")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()
