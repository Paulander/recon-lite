"""Signature Visualization for Promoted Nodes.

Generates visual signatures showing where and how patterns were detected.

Usage:
    from recon_lite.viz.signature_viz import generate_signature_heatmap
    
    path = generate_signature_heatmap(
        samples=stem_cell.samples,
        output_path=Path("signatures/SC_001.png")
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import chess
    HAS_CHESS = True
except ImportError:
    chess = None
    HAS_CHESS = False


def generate_signature_heatmap(
    samples: List[Any],  # List[StemCellSample]
    output_path: Path,
    board_size: int = 8,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Create averaged heatmap showing where the pattern was active.
    
    For each sample FEN:
    - Extract piece positions
    - Accumulate into 8x8 grid
    - Average and colorize
    
    Args:
        samples: List of StemCellSample with FEN positions
        output_path: Where to save the PNG
        board_size: Chess board size (8)
        title: Optional title for the plot
        
    Returns:
        Path to saved PNG, or None if failed
    """
    if not HAS_NUMPY or not HAS_MATPLOTLIB or not HAS_CHESS:
        return None
    
    if not samples:
        return None
    
    # Initialize heatmap grids
    piece_heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    king_white_heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    king_black_heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    pawn_heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    
    sample_count = 0
    
    for sample in samples:
        fen = getattr(sample, 'fen', '') if hasattr(sample, 'fen') else sample.get('fen', '')
        if not fen:
            continue
        
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        
        sample_count += 1
        reward = getattr(sample, 'reward', 1.0) if hasattr(sample, 'reward') else sample.get('reward', 1.0)
        weight = abs(reward) + 0.1  # Ensure some weight even for zero reward
        
        # Scan all squares
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            
            # General piece presence
            piece_heatmap[rank, file] += weight
            
            # Specific piece types
            if piece.piece_type == chess.KING:
                if piece.color == chess.WHITE:
                    king_white_heatmap[rank, file] += weight
                else:
                    king_black_heatmap[rank, file] += weight
            elif piece.piece_type == chess.PAWN:
                pawn_heatmap[rank, file] += weight
    
    if sample_count == 0:
        return None
    
    # Normalize
    piece_heatmap /= sample_count
    king_white_heatmap /= sample_count
    king_black_heatmap /= sample_count
    pawn_heatmap /= sample_count
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Configure color maps
    cmap_pieces = plt.cm.YlOrRd
    cmap_white = plt.cm.Blues
    cmap_black = plt.cm.Greys
    cmap_pawn = plt.cm.Greens
    
    # Plot each heatmap
    _plot_board_heatmap(axes[0, 0], piece_heatmap, cmap_pieces, "All Pieces")
    _plot_board_heatmap(axes[0, 1], king_white_heatmap, cmap_white, "White King")
    _plot_board_heatmap(axes[1, 0], king_black_heatmap, cmap_black, "Black King")
    _plot_board_heatmap(axes[1, 1], pawn_heatmap, cmap_pawn, "Pawns")
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"Pattern Signature ({sample_count} samples)", fontsize=14)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _plot_board_heatmap(ax, heatmap: "np.ndarray", cmap, title: str):
    """Plot a single board heatmap."""
    # Flip vertically so rank 8 is at top
    display_map = np.flipud(heatmap)
    
    im = ax.imshow(display_map, cmap=cmap, aspect='equal')
    
    # Add grid
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Labels
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    
    ax.set_title(title)
    
    # Colorbar
    plt.colorbar(im, ax=ax, shrink=0.7)


def generate_activation_signature(
    pattern_signature: List[float],
    feature_names: List[str],
    output_path: Path,
    title: Optional[str] = None,
    top_n: int = 15,
) -> Optional[Path]:
    """
    Create bar chart showing which features define this pattern.
    
    Args:
        pattern_signature: Feature vector
        feature_names: Names for each feature
        output_path: Where to save the PNG
        title: Optional title
        top_n: Number of top features to show
        
    Returns:
        Path to saved PNG, or None if failed
    """
    if not HAS_NUMPY or not HAS_MATPLOTLIB:
        return None
    
    if not pattern_signature or not feature_names:
        return None
    
    sig = np.array(pattern_signature)
    
    # Get top features by absolute value
    indices = np.argsort(np.abs(sig))[::-1][:top_n]
    
    values = sig[indices]
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in indices]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if v >= 0 else 'red' for v in values]
    bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel("Feature Weight")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Top {top_n} Features in Pattern")
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def generate_combined_signature(
    samples: List[Any],
    pattern_signature: Optional[List[float]],
    feature_names: Optional[List[str]],
    output_path: Path,
    node_id: str,
) -> Optional[Path]:
    """
    Generate a combined signature visualization with both heatmap and feature chart.
    
    Args:
        samples: Stem cell samples
        pattern_signature: Feature vector (optional)
        feature_names: Feature names (optional)
        output_path: Where to save
        node_id: Node ID for title
        
    Returns:
        Path to saved PNG
    """
    if not HAS_MATPLOTLIB:
        return None
    
    has_signature = pattern_signature is not None and feature_names is not None
    
    if has_signature:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # Generate heatmap in first subplot
    if HAS_NUMPY and HAS_CHESS and samples:
        heatmap = _compute_piece_heatmap(samples)
        _plot_board_heatmap(axes[0], heatmap, plt.cm.YlOrRd, "Position Heatmap")
    
    # Generate feature chart in second subplot if available
    if has_signature and len(axes) > 1:
        _plot_feature_bars(axes[1], pattern_signature, feature_names)
    
    fig.suptitle(f"Signature: {node_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _compute_piece_heatmap(samples: List[Any], board_size: int = 8) -> "np.ndarray":
    """Compute piece heatmap from samples."""
    heatmap = np.zeros((board_size, board_size), dtype=np.float32)
    count = 0
    
    for sample in samples:
        fen = getattr(sample, 'fen', '') if hasattr(sample, 'fen') else sample.get('fen', '')
        if not fen:
            continue
        
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        
        count += 1
        for sq in chess.SQUARES:
            if board.piece_at(sq):
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                heatmap[rank, file] += 1
    
    if count > 0:
        heatmap /= count
    
    return heatmap


def _plot_feature_bars(ax, signature: List[float], names: List[str], top_n: int = 10):
    """Plot feature importance bars."""
    sig = np.array(signature)
    indices = np.argsort(np.abs(sig))[::-1][:top_n]
    
    values = sig[indices]
    labels = [names[i] if i < len(names) else f"f{i}" for i in indices]
    
    colors = ['green' if v >= 0 else 'red' for v in values]
    ax.barh(range(len(values)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title("Top Features")
