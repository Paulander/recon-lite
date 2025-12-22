"""
Position Generators for Curriculum Training.

Provides functions to generate training positions for each curriculum phase:
- Anchor: Random valid KRK/KPK/KQK endgame positions
- Bridge: Simplified middlegames with material advantage
- Wilderness: Complex tactical positions from real games
- Integration: Standard starting position
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import chess


# ============================================================================
# Phase 1: Anchor Position Generators (Endgames)
# ============================================================================

def generate_krk_position(
    ensure_winning: bool = True,
    max_attempts: int = 100,
) -> chess.Board:
    """
    Generate a random valid KRK (King + Rook vs King) position.
    
    Args:
        ensure_winning: If True, ensure White is winning (not stalemate)
        max_attempts: Maximum attempts to find valid position
        
    Returns:
        A valid KRK position where White can win
    """
    for _ in range(max_attempts):
        board = chess.Board(None)
        board.clear()
        
        # Place kings ensuring they don't attack each other
        wk_sq = random.choice(chess.SQUARES)
        
        # Black king must be at least 2 squares away
        valid_bk_squares = [
            sq for sq in chess.SQUARES
            if chess.square_distance(sq, wk_sq) >= 2
        ]
        if not valid_bk_squares:
            continue
        bk_sq = random.choice(valid_bk_squares)
        
        # Place rook ensuring it's not on king squares
        valid_rook_squares = [
            sq for sq in chess.SQUARES
            if sq not in (wk_sq, bk_sq)
        ]
        rook_sq = random.choice(valid_rook_squares)
        
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(rook_sq, chess.Piece(chess.ROOK, chess.WHITE))
        
        # Always white to move for training consistency
        board.turn = chess.WHITE
        
        # Validate position has both kings
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            continue
        
        # Validate position
        if not board.is_valid():
            continue
        
        if ensure_winning:
            # Check if rook is immediately capturable by black king
            if rook_sq in board.attacks(bk_sq):
                continue
            
            # Check if ANY white move creates immediate stalemate for black
            has_safe_move = False
            for move in board.legal_moves:
                board.push(move)
                is_stale = board.is_stalemate()
                board.pop()
                if not is_stale:
                    has_safe_move = True
                    break
            
            if not has_safe_move:
                # All white moves lead to stalemate - reject this position
                continue
        
        return board
    
    # Fallback to known position
    return chess.Board("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")


def generate_kpk_position(
    ensure_winning: bool = True,
    max_attempts: int = 100,
) -> chess.Board:
    """
    Generate a random valid KPK (King + Pawn vs King) position.
    
    Avoids theoretical draws where possible.
    
    Args:
        ensure_winning: If True, attempt to generate won positions
        max_attempts: Maximum attempts
        
    Returns:
        A valid KPK position
    """
    for _ in range(max_attempts):
        board = chess.Board(None)
        board.clear()
        
        # Place pawn (not on rank 1 or 8)
        pawn_file = random.randint(0, 7)
        pawn_rank = random.randint(1, 6)  # Ranks 2-7
        pawn_sq = chess.square(pawn_file, pawn_rank)
        
        # White king near the pawn
        wk_candidates = [
            sq for sq in chess.SQUARES
            if chess.square_distance(sq, pawn_sq) <= 3
            and sq != pawn_sq
        ]
        if not wk_candidates:
            continue
        wk_sq = random.choice(wk_candidates)
        
        # Black king away from pawn but not too far
        bk_candidates = [
            sq for sq in chess.SQUARES
            if chess.square_distance(sq, wk_sq) >= 2
            and sq not in (pawn_sq, wk_sq)
        ]
        if not bk_candidates:
            continue
        bk_sq = random.choice(bk_candidates)
        
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
        
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        if not board.is_valid():
            continue
        
        # Basic validation
        if board.turn == chess.BLACK and board.is_stalemate():
            continue
        
        return board
    
    # Fallback
    return chess.Board("8/8/8/8/4k3/8/4P3/4K3 w - - 0 1")


def generate_kpk_near_promotion(
    max_attempts: int = 50,
    white_to_move: bool = True,
    allow_rook_pawn: bool = False,
) -> chess.Board:
    """
    Generate a simple KPK position 1-2 plies from promotion.
    
    Places the pawn on the 7th (or 2nd) rank with minimal clutter so
    training can focus on the KPK -> promotion -> KQK handoff.
    """
    for _ in range(max_attempts):
        board = chess.Board(None)
        board.clear()
        
        # Prefer central files to avoid rook-pawn corner draws unless allowed
        candidate_files = list(range(0, 8)) if allow_rook_pawn else list(range(1, 7))
        pawn_file = random.choice(candidate_files)
        pawn_rank = 6 if white_to_move else 1  # 7th for White, 2nd for Black
        pawn_sq = chess.square(pawn_file, pawn_rank)
        
        # Place attacking king behind/near the pawn
        wk_candidates = [
            sq for sq in chess.SQUARES
            if chess.square_distance(sq, pawn_sq) <= 2 and sq != pawn_sq
        ]
        if not wk_candidates:
            continue
        wk_sq = random.choice(wk_candidates)
        
        # Defender king a couple of squares away from promotion square
        promo_sq = chess.square(pawn_file, pawn_rank + (1 if white_to_move else -1))
        bk_candidates = [
            sq for sq in chess.SQUARES
            if chess.square_distance(sq, promo_sq) >= 2
            and sq not in (pawn_sq, wk_sq)
        ]
        if not bk_candidates:
            continue
        bk_sq = random.choice(bk_candidates)
        
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
        board.turn = chess.WHITE if white_to_move else chess.BLACK
        
        if not board.is_valid():
            continue
        
        # Avoid immediate stalemate or illegal promotion issues
        if board.turn != chess.WHITE and board.is_stalemate():
            continue
        
        return board
    
    # Fallback: simple near-promotion
    return chess.Board("8/8/8/3k4/8/8/4P3/4K3 w - - 0 1")


def generate_kqk_position(
    ensure_winning: bool = True,
    max_attempts: int = 100,
) -> chess.Board:
    """
    Generate a random valid KQK (King + Queen vs King) position.
    
    Args:
        ensure_winning: If True, avoid stalemates
        max_attempts: Maximum attempts
        
    Returns:
        A valid KQK position
    """
    for _ in range(max_attempts):
        board = chess.Board(None)
        board.clear()
        
        # Place kings
        wk_sq = random.choice(chess.SQUARES)
        valid_bk = [
            sq for sq in chess.SQUARES
            if chess.square_distance(sq, wk_sq) >= 2
        ]
        if not valid_bk:
            continue
        bk_sq = random.choice(valid_bk)
        
        # Place queen
        valid_queen = [
            sq for sq in chess.SQUARES
            if sq not in (wk_sq, bk_sq)
        ]
        queen_sq = random.choice(valid_queen)
        
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(queen_sq, chess.Piece(chess.QUEEN, chess.WHITE))
        
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        if not board.is_valid():
            continue
        
        if ensure_winning:
            if board.turn == chess.BLACK and board.is_stalemate():
                continue
            # Avoid queen en prise
            if board.turn == chess.BLACK and queen_sq in board.attacks(bk_sq):
                continue
        
        return board
    
    # Fallback
    return chess.Board("8/8/8/4k3/8/8/8/Q3K3 w - - 0 1")


def generate_anchor_position(
    endgame_type: Optional[str] = None,
) -> chess.Board:
    """
    Generate an anchor phase position.
    
    Args:
        endgame_type: Specific type ("KRK", "KPK", "KQK") or None for random
        
    Returns:
        A valid endgame position
    """
    if endgame_type is None:
        endgame_type = random.choice(["KRK", "KPK", "KQK"])
    
    if endgame_type == "KRK":
        return generate_krk_position()
    elif endgame_type == "KPK":
        return generate_kpk_position()
    elif endgame_type == "KQK":
        return generate_kqk_position()
    else:
        return generate_krk_position()


# ============================================================================
# Phase 2: Bridge Position Generators (Simplified Middlegame)
# ============================================================================

def generate_bridge_position(
    piece_count_range: Tuple[int, int] = (8, 12),
    material_advantage: float = 3.0,
) -> chess.Board:
    """
    Generate a simplified middlegame position with material advantage.
    
    These positions should be winnable but require transition to endgame.
    
    Args:
        piece_count_range: Min/max total pieces (excluding kings)
        material_advantage: Desired material advantage for White
        
    Returns:
        A bridge position
    """
    piece_values = {
        chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3,
        chess.KNIGHT: 3, chess.PAWN: 1,
    }
    
    max_attempts = 50
    for _ in range(max_attempts):
        board = chess.Board(None)
        board.clear()
        
        # Place kings first
        wk_sq = random.choice([chess.E1, chess.G1, chess.C1])
        bk_sq = random.choice([chess.E8, chess.G8, chess.C8])
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        
        # Determine piece counts
        min_pieces, max_pieces = piece_count_range
        target_pieces = random.randint(min_pieces, max_pieces)
        
        # Generate piece lists
        white_material = 0.0
        black_material = 0.0
        placed_squares = {wk_sq, bk_sq}
        
        # Add White pieces (more material)
        white_pieces = []
        piece_choices = [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN, chess.PAWN]
        
        for _ in range(target_pieces // 2 + 1):
            piece_type = random.choice(piece_choices)
            white_pieces.append(piece_type)
            white_material += piece_values[piece_type]
        
        # Add Black pieces (less material)
        target_black = white_material - material_advantage
        black_pieces = []
        
        while sum(piece_values[p] for p in black_pieces) < target_black - 2:
            piece_type = random.choice([chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN])
            if sum(piece_values[p] for p in black_pieces) + piece_values[piece_type] <= target_black + 1:
                black_pieces.append(piece_type)
        
        # Place pieces
        def place_piece(piece_type: chess.PieceType, color: chess.Color) -> bool:
            # Get valid squares
            if piece_type == chess.PAWN:
                if color == chess.WHITE:
                    ranks = [1, 2, 3, 4, 5]  # Ranks 2-6
                else:
                    ranks = [2, 3, 4, 5, 6]  # Ranks 3-7
            else:
                ranks = list(range(8))
            
            candidates = [
                sq for sq in chess.SQUARES
                if sq not in placed_squares
                and chess.square_rank(sq) in ranks
            ]
            
            if not candidates:
                return False
            
            sq = random.choice(candidates)
            board.set_piece_at(sq, chess.Piece(piece_type, color))
            placed_squares.add(sq)
            return True
        
        # Place all pieces
        for piece_type in white_pieces:
            place_piece(piece_type, chess.WHITE)
        for piece_type in black_pieces:
            place_piece(piece_type, chess.BLACK)
        
        board.turn = chess.WHITE
        
        if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
            return board
    
    # Fallback: a simple winning position
    return chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")


# ============================================================================
# Phase 3: Wilderness Position Generators (Complex Tactical)
# ============================================================================

# Sample tactical positions from common openings
WILDERNESS_POSITIONS = [
    # Italian Game positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5",
    # Sicilian positions
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkb1r/pp2pppp/2np1n2/6B1/3NP3/8/PPP2PPP/RN1QKB1R w KQkq - 0 6",
    # Queen's Gambit positions
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    # Ruy Lopez positions
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 w kq - 0 7",
    # Complex middlegame
    "r2qkb1r/pb1n1ppp/1p2pn2/2pp4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQkq - 0 8",
    "r1bq1rk1/ppp2ppp/2n2n2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 4 7",
]


def generate_wilderness_position(
    randomize_moves: int = 0,
) -> chess.Board:
    """
    Generate a complex tactical position.
    
    Args:
        randomize_moves: Number of random moves to play from base position
        
    Returns:
        A complex position for tactical training
    """
    # Start from a known position
    base_fen = random.choice(WILDERNESS_POSITIONS)
    board = chess.Board(base_fen)
    
    # Optionally play some random moves
    for _ in range(randomize_moves):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            break
        board.push(random.choice(legal))
    
    return board


def generate_wilderness_from_opening(
    opening_moves: int = 6,
) -> chess.Board:
    """
    Generate a wilderness position by playing opening moves.
    
    Args:
        opening_moves: Number of half-moves to play from start
        
    Returns:
        A position after some opening moves
    """
    board = chess.Board()
    
    for _ in range(opening_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        # Slightly bias toward center moves
        center_files = {3, 4}  # d and e files
        center_moves = [m for m in legal if chess.square_file(m.to_square) in center_files]
        if center_moves and random.random() < 0.6:
            board.push(random.choice(center_moves))
        else:
            board.push(random.choice(legal))
    
    return board


# ============================================================================
# Phase 4: Integration Position Generators
# ============================================================================

def generate_integration_position() -> chess.Board:
    """
    Generate a position for full game integration training.
    
    Always returns the starting position.
    
    Returns:
        Standard starting position
    """
    return chess.Board()


# ============================================================================
# Utility Functions
# ============================================================================

def validate_position(board: chess.Board) -> bool:
    """Check if a position is valid for training."""
    if not board.is_valid():
        return False
    if board.is_game_over():
        return False
    return True


def estimate_theoretical_moves(board: chess.Board) -> int:
    """
    Estimate theoretical minimum moves to win.
    
    Very rough estimate based on piece count and position.
    """
    pieces = board.piece_map()
    piece_count = len([p for p in pieces.values() if p.piece_type != chess.KING])
    
    # KRK: ~16 moves average
    # KQK: ~10 moves average
    # KPK: ~20-30 moves (depends heavily on position)
    
    if piece_count == 1:
        piece = next(p for p in pieces.values() if p.piece_type != chess.KING)
        if piece.piece_type == chess.QUEEN:
            return 10
        elif piece.piece_type == chess.ROOK:
            return 16
        elif piece.piece_type == chess.PAWN:
            return 25
    
    # More complex positions
    return 30 + piece_count * 2

