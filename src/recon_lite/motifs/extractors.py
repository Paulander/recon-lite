"""M5.1: Feature extractors for motif discovery from board positions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import chess
except ImportError:
    chess = None  # type: ignore


def _require_chess():
    """Raise ImportError if chess is not available."""
    if chess is None:
        raise ImportError("python-chess is required for board feature extraction")


def extract_3x3_patch(board: "chess.Board", center_square: int) -> Dict[str, Any]:
    """
    Extract a 3x3 patch of pieces around a center square.
    
    Returns:
        Dictionary with:
        - center: center square name
        - pieces: dict mapping relative position to piece symbol
        - our_pieces: count of friendly pieces in patch
        - enemy_pieces: count of enemy pieces in patch
    """
    _require_chess()
    
    center_file = chess.square_file(center_square)
    center_rank = chess.square_rank(center_square)
    turn = board.turn
    
    pieces = {}
    our_count = 0
    enemy_count = 0
    
    for df in range(-1, 2):
        for dr in range(-1, 2):
            f = center_file + df
            r = center_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                key = f"{df},{dr}"
                if piece:
                    pieces[key] = piece.symbol()
                    if piece.color == turn:
                        our_count += 1
                    else:
                        enemy_count += 1
                else:
                    pieces[key] = "."
    
    return {
        "center": chess.square_name(center_square),
        "pieces": pieces,
        "our_pieces": our_count,
        "enemy_pieces": enemy_count,
    }


def extract_king_zone(board: "chess.Board", color: bool) -> Dict[str, Any]:
    """
    Extract king safety zone features for a given color.
    
    Returns:
        Dictionary with:
        - king_square: king position
        - zone_squares: list of squares in king zone
        - defenders: count of defending pieces
        - attackers: count of attacking pieces
        - pawn_shield: count of pawns shielding king
        - open_files: list of open files near king
    """
    _require_chess()
    
    king_sq = board.king(color)
    if king_sq is None:
        return {"king_square": None, "zone_squares": [], "defenders": 0, "attackers": 0}
    
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    
    zone_squares = []
    defenders = 0
    attackers = 0
    pawn_shield = 0
    open_files: List[int] = []
    
    # King zone is 3x3 around king, extended forward
    forward = 1 if color == chess.WHITE else -1
    
    for df in range(-1, 2):
        for dr in range(-1, 3):  # Extended forward
            f = king_file + df
            r = king_rank + dr * forward
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                zone_squares.append(chess.square_name(sq))
                piece = board.piece_at(sq)
                if piece:
                    if piece.color == color:
                        defenders += 1
                        if piece.piece_type == chess.PAWN:
                            pawn_shield += 1
                    else:
                        attackers += 1
    
    # Check for open files near king
    for df in range(-1, 2):
        f = king_file + df
        if 0 <= f <= 7:
            has_pawn = False
            for r in range(8):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                open_files.append(f)
    
    return {
        "king_square": chess.square_name(king_sq),
        "zone_squares": zone_squares,
        "defenders": defenders,
        "attackers": attackers,
        "pawn_shield": pawn_shield,
        "open_files": [chess.FILE_NAMES[f] for f in open_files],
    }


def extract_pawn_chain(board: "chess.Board") -> Dict[str, Any]:
    """
    Extract pawn structure features.
    
    Returns:
        Dictionary with:
        - white_pawns: list of white pawn squares
        - black_pawns: list of black pawn squares
        - isolated_pawns: list of isolated pawn squares
        - doubled_pawns: list of doubled pawn squares
        - passed_pawns: list of passed pawn squares
        - pawn_islands: count of pawn islands per side
    """
    _require_chess()
    
    white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
    
    isolated_pawns = []
    doubled_pawns = []
    passed_pawns = []
    
    # Helper to check if a pawn is isolated
    def is_isolated(sq: int, color: bool) -> bool:
        f = chess.square_file(sq)
        pawns = white_pawns if color == chess.WHITE else black_pawns
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7:
                for p in pawns:
                    if chess.square_file(p) == adj_f:
                        return False
        return True
    
    # Helper to check if a pawn is passed
    def is_passed(sq: int, color: bool) -> bool:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        enemy_pawns = black_pawns if color == chess.WHITE else white_pawns
        direction = 1 if color == chess.WHITE else -1
        
        for enemy_sq in enemy_pawns:
            ef = chess.square_file(enemy_sq)
            er = chess.square_rank(enemy_sq)
            # Check if enemy pawn can block
            if abs(ef - f) <= 1:
                if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                    return False
        return True
    
    # Check white pawns
    white_files: Set[int] = set()
    for sq in white_pawns:
        f = chess.square_file(sq)
        if f in white_files:
            doubled_pawns.append(chess.square_name(sq))
        white_files.add(f)
        
        if is_isolated(sq, chess.WHITE):
            isolated_pawns.append(chess.square_name(sq))
        if is_passed(sq, chess.WHITE):
            passed_pawns.append(chess.square_name(sq))
    
    # Check black pawns
    black_files: Set[int] = set()
    for sq in black_pawns:
        f = chess.square_file(sq)
        if f in black_files:
            doubled_pawns.append(chess.square_name(sq))
        black_files.add(f)
        
        if is_isolated(sq, chess.BLACK):
            isolated_pawns.append(chess.square_name(sq))
        if is_passed(sq, chess.BLACK):
            passed_pawns.append(chess.square_name(sq))
    
    # Count pawn islands
    def count_islands(files: Set[int]) -> int:
        if not files:
            return 0
        sorted_files = sorted(files)
        islands = 1
        for i in range(1, len(sorted_files)):
            if sorted_files[i] - sorted_files[i-1] > 1:
                islands += 1
        return islands
    
    return {
        "white_pawns": [chess.square_name(sq) for sq in white_pawns],
        "black_pawns": [chess.square_name(sq) for sq in black_pawns],
        "isolated_pawns": isolated_pawns,
        "doubled_pawns": doubled_pawns,
        "passed_pawns": passed_pawns,
        "white_pawn_islands": count_islands(white_files),
        "black_pawn_islands": count_islands(black_files),
    }


def extract_hanging_pieces(board: "chess.Board") -> Dict[str, Any]:
    """
    Extract information about hanging (undefended) pieces.
    
    Returns:
        Dictionary with:
        - white_hanging: list of undefended white pieces
        - black_hanging: list of undefended black pieces
        - white_en_prise: list of white pieces attacked and not defended
        - black_en_prise: list of black pieces attacked and not defended
    """
    _require_chess()
    
    white_hanging = []
    black_hanging = []
    white_en_prise = []
    black_en_prise = []
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        
        color = piece.color
        sq_name = chess.square_name(sq)
        
        # Check if defended by own pieces
        defenders = board.attackers(color, sq)
        is_defended = len(defenders) > 0
        
        # Check if attacked by enemy pieces
        attackers = board.attackers(not color, sq)
        is_attacked = len(attackers) > 0
        
        if not is_defended:
            if color == chess.WHITE:
                white_hanging.append(sq_name)
            else:
                black_hanging.append(sq_name)
        
        if is_attacked and not is_defended:
            if color == chess.WHITE:
                white_en_prise.append(sq_name)
            else:
                black_en_prise.append(sq_name)
    
    return {
        "white_hanging": white_hanging,
        "black_hanging": black_hanging,
        "white_en_prise": white_en_prise,
        "black_en_prise": black_en_prise,
    }


def extract_tactical_features(board: "chess.Board") -> Dict[str, Any]:
    """
    Extract tactical pattern features (forks, pins, skewers).
    
    Returns:
        Dictionary with:
        - potential_forks: list of squares where a piece forks multiple targets
        - pins: list of pinned pieces
        - discovered_attacks: potential discovered attack opportunities
        - checks_available: number of checking moves available
    """
    _require_chess()
    
    turn = board.turn
    enemy = not turn
    
    potential_forks: List[Dict[str, Any]] = []
    pins: List[str] = []
    checks_available = 0
    
    # Find potential knight forks
    for sq in board.pieces(chess.KNIGHT, turn):
        attacks = board.attacks(sq)
        valuable_targets = []
        for target_sq in attacks:
            target = board.piece_at(target_sq)
            if target and target.color == enemy:
                if target.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    valuable_targets.append(chess.square_name(target_sq))
        if len(valuable_targets) >= 2:
            potential_forks.append({
                "piece": "N",
                "from": chess.square_name(sq),
                "targets": valuable_targets,
            })
    
    # Find pinned pieces (simplified)
    enemy_king_sq = board.king(enemy)
    if enemy_king_sq:
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == enemy and piece.piece_type != chess.KING:
                # Check if removing this piece would expose king to attack
                if board.is_pinned(enemy, sq):
                    pins.append(chess.square_name(sq))
    
    # Count available checks
    for move in board.legal_moves:
        board.push(move)
        if board.is_check():
            checks_available += 1
        board.pop()
    
    return {
        "potential_forks": potential_forks,
        "pins": pins,
        "checks_available": checks_available,
    }


def extract_all_features(board: "chess.Board") -> Dict[str, Any]:
    """
    Extract all features from a board position.
    
    Returns:
        Combined dictionary with all feature categories.
    """
    _require_chess()
    
    return {
        "king_zone_white": extract_king_zone(board, chess.WHITE),
        "king_zone_black": extract_king_zone(board, chess.BLACK),
        "pawn_structure": extract_pawn_chain(board),
        "hanging_pieces": extract_hanging_pieces(board),
        "tactical": extract_tactical_features(board),
        "material": _count_material(board),
        "turn": "white" if board.turn else "black",
    }


def _count_material(board: "chess.Board") -> Dict[str, int]:
    """Count material for both sides."""
    _require_chess()
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    
    white_material = 0
    black_material = 0
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    return {
        "white": white_material,
        "black": black_material,
        "diff": white_material - black_material,
    }

