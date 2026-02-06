"""
Sensor Flooding Wave 1: 20 New Sensors for Reverse Curriculum Training.

This module adds tactical, geometric, positional, and dynamic sensors
to expand the FeatureHub's pattern detection capabilities.

Wave 1 Strategy:
- Conservative start with 20 sensors
- Focus on atomic building blocks for emergent strategy discovery
- Each sensor returns continuous [0.0, 1.0] (or [-1.0, 1.0] for symmetric)
- Enable stem cell instantiation and M5 structure learning

Future Waves:
- Wave 2: +20 (40 total) - Positional latents, Zugzwang precursors
- Wave 3: +30 (70 total) - Stem cell promotions, bridge patterns
- Wave 4: +30 (100 total) - Advanced latents (prophylaxis, tempo)
- Wave 5+: 100-200 - M5-discovered patterns, auto-promoted
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import chess

from .hub import FeatureHub, FeatureDefinition, FeatureCategory


# ============================================================================
# Tactical Sensors (8 new)
# ============================================================================

def compute_battery_present(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect aligned rooks/queen on files/ranks (battery formation).
    
    A battery is when heavy pieces (R, Q) are aligned on an open or semi-open
    file or rank, multiplying their attacking power.
    
    Returns: [0.0, 1.0] - strength of battery formation
    """
    score = 0.0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1.0 if color == board.turn else -0.5
        
        # Find heavy pieces
        queens = list(board.pieces(chess.QUEEN, color))
        rooks = list(board.pieces(chess.ROOK, color))
        heavy_pieces = queens + rooks
        
        if len(heavy_pieces) < 2:
            continue
        
        # Check file alignment
        files_occupied: Dict[int, List[chess.Square]] = {}
        for sq in heavy_pieces:
            f = chess.square_file(sq)
            if f not in files_occupied:
                files_occupied[f] = []
            files_occupied[f].append(sq)
        
        for file_sqs in files_occupied.values():
            if len(file_sqs) >= 2:
                score += sign * 0.3 * len(file_sqs)
        
        # Check rank alignment
        ranks_occupied: Dict[int, List[chess.Square]] = {}
        for sq in heavy_pieces:
            r = chess.square_rank(sq)
            if r not in ranks_occupied:
                ranks_occupied[r] = []
            ranks_occupied[r].append(sq)
        
        for rank_sqs in ranks_occupied.values():
            if len(rank_sqs) >= 2:
                score += sign * 0.2 * len(rank_sqs)
    
    return max(0.0, min(1.0, score))


def compute_discovered_attack_possible(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if a discovered attack is possible.
    
    A discovered attack occurs when moving one piece reveals an attack
    from another piece behind it.
    
    Returns: [0.0, 1.0] - probability of discovered attack opportunity
    """
    score = 0.0
    color = board.turn
    enemy_king = board.king(not color)
    
    if enemy_king is None:
        return 0.0
    
    # Check each piece that could move to reveal an attack
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.color != color:
            continue
        
        # Check if there's a sliding piece behind this one
        for direction in [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # Orthogonal
            (1, 1), (1, -1), (-1, 1), (-1, -1),  # Diagonal
        ]:
            behind_sq = sq
            df, dr = direction
            
            # Look behind
            behind_file = chess.square_file(sq) - df
            behind_rank = chess.square_rank(sq) - dr
            
            if 0 <= behind_file <= 7 and 0 <= behind_rank <= 7:
                behind_sq = chess.square(behind_file, behind_rank)
                behind_piece = board.piece_at(behind_sq)
                
                if behind_piece and behind_piece.color == color:
                    # Check if behind piece attacks enemy king through current square
                    if behind_piece.piece_type in [chess.ROOK, chess.QUEEN]:
                        if df == 0 or dr == 0:  # Orthogonal
                            # Check if king is in that direction
                            king_file = chess.square_file(enemy_king)
                            king_rank = chess.square_rank(enemy_king)
                            if (df == 0 and king_file == behind_file) or \
                               (dr == 0 and king_rank == behind_rank):
                                score += 0.5
                    
                    if behind_piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                        if df != 0 and dr != 0:  # Diagonal
                            score += 0.3
    
    return min(1.0, score)


def compute_overloaded_defender(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect pieces that are defending multiple attacked pieces.
    
    An overloaded defender cannot adequately protect all its charges,
    creating tactical opportunities.
    
    Returns: [0.0, 1.0] - degree of overloaded defenders in opponent's position
    """
    score = 0.0
    enemy_color = not board.turn
    
    # Find enemy pieces and what they defend
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.color != enemy_color:
            continue
        
        # Count what this piece defends
        defends: List[chess.Square] = []
        attacks_on_defended: List[chess.Square] = []
        
        for target_sq in chess.SQUARES:
            target_piece = board.piece_at(target_sq)
            if target_piece and target_piece.color == enemy_color and target_sq != sq:
                # Check if this piece defends target
                if board.is_attacked_by(enemy_color, target_sq):
                    # Rough check: is our piece a defender?
                    attackers = board.attackers(enemy_color, target_sq)
                    if sq in attackers:
                        defends.append(target_sq)
                        # Is the defended piece attacked by us?
                        if board.is_attacked_by(board.turn, target_sq):
                            attacks_on_defended.append(target_sq)
        
        # Overloaded if defending 2+ attacked pieces
        if len(attacks_on_defended) >= 2:
            score += 0.3 * len(attacks_on_defended)
    
    return min(1.0, score)


def compute_trapped_piece(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect high-value pieces with limited escape squares.
    
    A trapped piece (especially Q, R, B, N) that cannot move safely
    may be captured with advantage.
    
    Returns: [0.0, 1.0] - presence of trapped enemy pieces
    """
    score = 0.0
    enemy_color = not board.turn
    
    piece_values = {
        chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3
    }
    
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.color != enemy_color:
            continue
        if piece.piece_type not in piece_values:
            continue
        
        # Count safe squares this piece can move to
        safe_moves = 0
        total_moves = 0
        
        for move in board.legal_moves:
            if move.from_square == sq:
                total_moves += 1
                # Check if destination is safe
                board.push(move)
                if not board.is_attacked_by(board.turn, move.to_square):
                    safe_moves += 1
                board.pop()
        
        if total_moves > 0 and safe_moves == 0:
            # Completely trapped
            score += 0.3 * (piece_values[piece.piece_type] / 9.0)
        elif total_moves > 0 and safe_moves <= 1:
            # Nearly trapped
            score += 0.15 * (piece_values[piece.piece_type] / 9.0)
    
    return min(1.0, score)


def compute_desperado_opportunity(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect doomed pieces that can capture before being captured.
    
    A desperado is a piece that is going to be lost anyway,
    so it might as well capture something first.
    
    Returns: [0.0, 1.0] - desperado opportunity present
    """
    score = 0.0
    color = board.turn
    
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.color != color:
            continue
        
        # Is this piece attacked and undefended (or insufficiently defended)?
        if board.is_attacked_by(not color, sq):
            our_defenders = len(board.attackers(color, sq))
            their_attackers = len(board.attackers(not color, sq))
            
            if their_attackers > our_defenders:
                # This piece is likely lost - check captures
                for move in board.legal_moves:
                    if move.from_square == sq and board.is_capture(move):
                        captured = board.piece_at(move.to_square)
                        if captured:
                            capture_value = piece_values.get(captured.piece_type, 0)
                            our_value = piece_values.get(piece.piece_type, 0)
                            if capture_value > 0:
                                score += 0.2 * (capture_value / 9.0)
    
    return min(1.0, score)


def compute_zwischenzug_possible(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if an intermediate move (zwischenzug) is possible.
    
    A zwischenzug is an unexpected intermediate move that changes the
    evaluation before the expected recapture.
    
    Returns: [0.0, 1.0] - zwischenzug opportunity
    """
    score = 0.0
    color = board.turn
    
    # Look for check-giving captures or attacks
    for move in board.legal_moves:
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        
        if gives_check:
            # Check with a purpose - might be zwischenzug
            if board.is_capture(move):
                score += 0.4
            else:
                # Non-capturing check - classic zwischenzug
                score += 0.3
    
    # Also check for attacks on undefended high-value pieces
    enemy_color = not color
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece and piece.color == enemy_color and piece.piece_type in [chess.QUEEN, chess.ROOK]:
            if not board.is_attacked_by(enemy_color, sq):
                # Undefended heavy piece
                if board.is_attacked_by(color, sq):
                    score += 0.2
    
    return min(1.0, score)


def compute_x_ray_attack(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect X-ray attacks (attacks through another piece).
    
    X-ray attacks occur when a sliding piece attacks a square
    through an intervening piece.
    
    Returns: [0.0, 1.0] - x-ray attack strength
    """
    score = 0.0
    color = board.turn
    enemy_king = board.king(not color)
    enemy_queen_sqs = list(board.pieces(chess.QUEEN, not color))
    
    high_value_targets = [enemy_king] if enemy_king else []
    high_value_targets.extend(enemy_queen_sqs)
    
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.color != color:
            continue
        
        if piece.piece_type not in [chess.ROOK, chess.QUEEN, chess.BISHOP]:
            continue
        
        # Check sliding attacks
        for target in high_value_targets:
            if target is None:
                continue
            
            # Check if there's exactly one piece between us and target
            between = list(chess.SquareSet.between(sq, target))
            if len(between) == 1:
                blocker = board.piece_at(between[0])
                if blocker:
                    # X-ray exists
                    score += 0.25
    
    return min(1.0, score)


def compute_deflection_possible(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if a defender can be deflected from its duty.
    
    Deflection forces a piece away from defending a key square or piece.
    
    Returns: [0.0, 1.0] - deflection opportunity
    """
    score = 0.0
    color = board.turn
    enemy_color = not color
    
    # Find enemy defenders of high-value squares/pieces
    enemy_king = board.king(enemy_color)
    if enemy_king is None:
        return 0.0
    
    # Find defenders of the enemy king's vicinity
    king_defenders = board.attackers(enemy_color, enemy_king)
    
    for defender_sq in king_defenders:
        defender = board.piece_at(defender_sq)
        if defender is None:
            continue
        
        # Can we attack this defender?
        if board.is_attacked_by(color, defender_sq):
            # Can we force it to move?
            for move in board.legal_moves:
                if move.to_square == defender_sq:
                    # Attacking the defender
                    board.push(move)
                    # After they respond, is the king weaker?
                    new_defenders = len(board.attackers(enemy_color, enemy_king))
                    board.pop()
                    
                    if new_defenders < len(king_defenders):
                        score += 0.3
    
    return min(1.0, score)


# ============================================================================
# Geometric Sensors (5 new)
# ============================================================================

def compute_mating_net_present(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if enemy king is restricted to corner/edge area (mating net).
    
    A mating net is when the enemy king has very limited escape squares,
    often a precursor to checkmate.
    
    Returns: [0.0, 1.0] - mating net strength
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.0
    
    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)
    
    # Count escape squares
    escape_squares = 0
    total_adjacent = 0
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            nf, nr = king_file + df, king_rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                total_adjacent += 1
                sq = chess.square(nf, nr)
                # Is this square a valid escape?
                occupant = board.piece_at(sq)
                if occupant and occupant.color == (not board.turn):
                    continue  # Blocked by own piece
                if not board.is_attacked_by(board.turn, sq):
                    escape_squares += 1
    
    if total_adjacent == 0:
        return 0.0
    
    restriction = 1.0 - (escape_squares / total_adjacent)
    
    # Bonus for corner/edge
    edge_bonus = 0.0
    if king_file in [0, 7] or king_rank in [0, 7]:
        edge_bonus = 0.2
    if (king_file in [0, 7]) and (king_rank in [0, 7]):
        edge_bonus = 0.4  # Corner
    
    return min(1.0, restriction * 0.6 + edge_bonus)


def compute_king_tropism(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Measure pieces aimed at enemy king (king tropism).
    
    High tropism indicates attacking potential against the enemy king.
    
    Returns: [0.0, 1.0] - king tropism score
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.0
    
    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)
    
    tropism = 0.0
    
    piece_weights = {
        chess.QUEEN: 4.0, chess.ROOK: 2.0, chess.BISHOP: 1.5,
        chess.KNIGHT: 2.0, chess.PAWN: 0.5
    }
    
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None or piece.color != board.turn:
            continue
        
        weight = piece_weights.get(piece.piece_type, 0)
        if weight == 0:
            continue
        
        # Manhattan distance to enemy king
        pf = chess.square_file(sq)
        pr = chess.square_rank(sq)
        distance = abs(pf - king_file) + abs(pr - king_rank)
        
        if distance > 0:
            tropism += weight / distance
    
    # Normalize to [0, 1]
    return min(1.0, tropism / 15.0)


def compute_square_control_imbalance(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Measure central square control difference.
    
    Central control is a key positional factor.
    
    Returns: [-1.0, 1.0] - positive means we control more center
    """
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    extended_center = [
        chess.C3, chess.C4, chess.C5, chess.C6,
        chess.D3, chess.D6, chess.E3, chess.E6,
        chess.F3, chess.F4, chess.F5, chess.F6,
    ]
    
    our_control = 0
    their_control = 0
    
    for sq in center_squares:
        if board.is_attacked_by(board.turn, sq):
            our_control += 2
        if board.is_attacked_by(not board.turn, sq):
            their_control += 2
    
    for sq in extended_center:
        if board.is_attacked_by(board.turn, sq):
            our_control += 1
        if board.is_attacked_by(not board.turn, sq):
            their_control += 1
    
    total = our_control + their_control
    if total == 0:
        return 0.0
    
    return (our_control - their_control) / max(total, 20)


def compute_outpost_available(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect available outpost squares for knights/bishops.
    
    An outpost is a square that cannot be attacked by enemy pawns.
    
    Returns: [0.0, 1.0] - outpost availability
    """
    score = 0.0
    color = board.turn
    enemy_pawns = board.pieces(chess.PAWN, not color)
    
    # Good outpost squares (central, in enemy territory)
    if color == chess.WHITE:
        outpost_ranks = [4, 5, 6]  # Ranks 5, 6, 7
    else:
        outpost_ranks = [1, 2, 3]  # Ranks 2, 3, 4
    
    for sq in chess.SQUARES:
        if chess.square_rank(sq) not in outpost_ranks:
            continue
        
        sq_file = chess.square_file(sq)
        
        # Check if enemy pawns can attack this square
        can_be_attacked = False
        for pawn_sq in enemy_pawns:
            pawn_file = chess.square_file(pawn_sq)
            pawn_rank = chess.square_rank(pawn_sq)
            
            # Can this pawn ever attack the square?
            if abs(pawn_file - sq_file) == 1:
                if color == chess.WHITE and pawn_rank < chess.square_rank(sq):
                    can_be_attacked = True
                elif color == chess.BLACK and pawn_rank > chess.square_rank(sq):
                    can_be_attacked = True
        
        if not can_be_attacked:
            # Is one of our minor pieces on or near this outpost?
            occupant = board.piece_at(sq)
            if occupant and occupant.color == color and occupant.piece_type in [chess.KNIGHT, chess.BISHOP]:
                score += 0.3
            elif board.is_attacked_by(color, sq):
                score += 0.1
    
    return min(1.0, score)


def compute_weak_square_complex(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect multiple weak squares on same color complex.
    
    Color complex weakness occurs when pawns are fixed on one color,
    weakening squares of the opposite color.
    
    Returns: [0.0, 1.0] - color complex weakness
    """
    enemy_color = not board.turn
    enemy_pawns = board.pieces(chess.PAWN, enemy_color)
    enemy_bishops = board.pieces(chess.BISHOP, enemy_color)
    
    # Analyze pawn structure
    light_square_pawns = 0
    dark_square_pawns = 0
    
    for pawn_sq in enemy_pawns:
        if (chess.square_file(pawn_sq) + chess.square_rank(pawn_sq)) % 2 == 0:
            dark_square_pawns += 1
        else:
            light_square_pawns += 1
    
    # Check if they have bishop of the weak color
    has_light_bishop = False
    has_dark_bishop = False
    
    for bishop_sq in enemy_bishops:
        if (chess.square_file(bishop_sq) + chess.square_rank(bishop_sq)) % 2 == 0:
            has_dark_bishop = True
        else:
            has_light_bishop = True
    
    score = 0.0
    
    # If many pawns on light squares and no light bishop
    if light_square_pawns >= 3 and not has_light_bishop:
        score += 0.4
    
    # If many pawns on dark squares and no dark bishop
    if dark_square_pawns >= 3 and not has_dark_bishop:
        score += 0.4
    
    return min(1.0, score)


# ============================================================================
# Positional Sensors (4 new)
# ============================================================================

def compute_pawn_majority(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect pawn count advantage on kingside/queenside.
    
    A pawn majority can create a passed pawn.
    
    Returns: [-1.0, 1.0] - positive means we have a useful majority
    """
    our_pawns = board.pieces(chess.PAWN, board.turn)
    their_pawns = board.pieces(chess.PAWN, not board.turn)
    
    # Kingside (files e-h)
    our_kingside = sum(1 for sq in our_pawns if chess.square_file(sq) >= 4)
    their_kingside = sum(1 for sq in their_pawns if chess.square_file(sq) >= 4)
    
    # Queenside (files a-d)
    our_queenside = sum(1 for sq in our_pawns if chess.square_file(sq) < 4)
    their_queenside = sum(1 for sq in their_pawns if chess.square_file(sq) < 4)
    
    kingside_advantage = our_kingside - their_kingside
    queenside_advantage = our_queenside - their_queenside
    
    # Use the larger advantage
    if abs(kingside_advantage) > abs(queenside_advantage):
        return max(-1.0, min(1.0, kingside_advantage / 3.0))
    else:
        return max(-1.0, min(1.0, queenside_advantage / 3.0))


def compute_passed_pawn_potential(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect pawns close to becoming passed or already passed.
    
    A passed pawn has no enemy pawns blocking or able to capture it.
    
    Returns: [0.0, 1.0] - passed pawn potential
    """
    score = 0.0
    our_pawns = board.pieces(chess.PAWN, board.turn)
    their_pawns = board.pieces(chess.PAWN, not board.turn)
    
    for pawn_sq in our_pawns:
        pawn_file = chess.square_file(pawn_sq)
        pawn_rank = chess.square_rank(pawn_sq)
        
        is_passed = True
        
        for enemy_pawn in their_pawns:
            enemy_file = chess.square_file(enemy_pawn)
            enemy_rank = chess.square_rank(enemy_pawn)
            
            # Check if enemy pawn blocks or can capture
            if abs(enemy_file - pawn_file) <= 1:
                if board.turn == chess.WHITE:
                    if enemy_rank > pawn_rank:
                        is_passed = False
                        break
                else:
                    if enemy_rank < pawn_rank:
                        is_passed = False
                        break
        
        if is_passed:
            # Score based on advancement
            if board.turn == chess.WHITE:
                advancement = pawn_rank / 7.0
            else:
                advancement = (7 - pawn_rank) / 7.0
            
            score += 0.2 + 0.3 * advancement
    
    return min(1.0, score)


def compute_blockade_present(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect pieces blocking passed pawns.
    
    A blockade stops a passed pawn from advancing.
    
    Returns: [0.0, 1.0] - blockade effectiveness
    """
    score = 0.0
    their_pawns = board.pieces(chess.PAWN, not board.turn)
    
    for pawn_sq in their_pawns:
        pawn_file = chess.square_file(pawn_sq)
        pawn_rank = chess.square_rank(pawn_sq)
        
        # Find square in front of pawn
        if board.turn == chess.WHITE:
            # Their pawns move down (from our perspective they're going down)
            front_rank = pawn_rank - 1
        else:
            front_rank = pawn_rank + 1
        
        if 0 <= front_rank <= 7:
            front_sq = chess.square(pawn_file, front_rank)
            blocker = board.piece_at(front_sq)
            
            if blocker and blocker.color == board.turn:
                # We're blockading
                # Knights are best blockaders
                if blocker.piece_type == chess.KNIGHT:
                    score += 0.4
                elif blocker.piece_type in [chess.BISHOP, chess.KING]:
                    score += 0.3
                else:
                    score += 0.2
    
    return min(1.0, score)


def compute_bishop_pair_bonus(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect bishop pair advantage.
    
    Two bishops of opposite colors are strong, especially in open positions.
    
    Returns: [0.0, 1.0] - bishop pair advantage
    """
    our_bishops = list(board.pieces(chess.BISHOP, board.turn))
    their_bishops = list(board.pieces(chess.BISHOP, not board.turn))
    
    our_pair = len(our_bishops) >= 2
    their_pair = len(their_bishops) >= 2
    
    if our_pair and not their_pair:
        return 0.8
    elif our_pair and their_pair:
        return 0.4
    elif not our_pair and len(our_bishops) == 1 and len(their_bishops) == 0:
        return 0.3
    
    return 0.0


# ============================================================================
# Dynamic Sensors (3 new)
# ============================================================================

def compute_mobility_restriction(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect low mobility (pre-Zugzwang indicator).
    
    Low mobility often indicates positional problems or zugzwang potential.
    
    Returns: [0.0, 1.0] - mobility restriction (higher = more restricted)
    """
    legal_moves = len(list(board.legal_moves))
    
    # Typical midgame has ~35 legal moves
    max_expected = 35.0
    
    # Very low mobility is concerning
    if legal_moves < 5:
        return 1.0
    elif legal_moves < 10:
        return 0.7
    elif legal_moves < 20:
        return 0.4
    else:
        return max(0.0, 1.0 - (legal_moves / max_expected))


def compute_tempo_advantage(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Assess tempo/initiative advantage.
    
    Tempo advantage means having threats that force the opponent to respond.
    
    Returns: [-1.0, 1.0] - positive means we have initiative
    """
    score = 0.0
    
    # Checks give tempo
    if board.is_check():
        score -= 0.5  # We're in check (bad)
    
    # Count our threats vs their threats
    our_attacks = 0
    their_attacks = 0
    
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece is None:
            continue
        
        if board.is_attacked_by(board.turn, sq) and piece.color != board.turn:
            # We attack an enemy piece
            our_attacks += 1
        
        if board.is_attacked_by(not board.turn, sq) and piece.color == board.turn:
            # They attack our piece
            their_attacks += 1
    
    # Undefended attacks are stronger
    for move in board.legal_moves:
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured and not board.is_attacked_by(not board.turn, move.to_square):
                our_attacks += 0.5
    
    score += (our_attacks - their_attacks) * 0.1
    
    return max(-1.0, min(1.0, score))


def compute_piece_coordination(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Measure how well pieces support each other.
    
    Well-coordinated pieces defend each other and create combined threats.
    
    Returns: [0.0, 1.0] - piece coordination score
    """
    score = 0.0
    color = board.turn
    
    pieces = []
    for sq in board.piece_map():
        piece = board.piece_at(sq)
        if piece and piece.color == color and piece.piece_type != chess.KING:
            pieces.append(sq)
    
    if len(pieces) < 2:
        return 0.0
    
    # Count mutual defense
    defending_pairs = 0
    for i, sq1 in enumerate(pieces):
        for sq2 in pieces[i+1:]:
            # Does sq1 defend sq2?
            if board.is_attacked_by(color, sq2):
                attackers = board.attackers(color, sq2)
                if sq1 in attackers:
                    defending_pairs += 1
            
            # Does sq2 defend sq1?
            if board.is_attacked_by(color, sq1):
                attackers = board.attackers(color, sq1)
                if sq2 in attackers:
                    defending_pairs += 1
    
    # Normalize
    max_pairs = len(pieces) * (len(pieces) - 1)
    if max_pairs > 0:
        score = defending_pairs / max_pairs
    
    return min(1.0, score)


# ============================================================================
# Registration Function
# ============================================================================

def register_v2_sensors(hub: FeatureHub) -> None:
    """
    Register all Wave 1 sensors (20 total) with the FeatureHub.
    
    Call this after creating the hub to add the new sensors.
    """
    # Tactical sensors (8)
    hub.register(FeatureDefinition(
        name="battery_present",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_battery_present,
        description="Rooks/Queen aligned on file/rank",
    ))
    
    hub.register(FeatureDefinition(
        name="discovered_attack_possible",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_discovered_attack_possible,
        description="Piece can reveal attack by moving",
    ))
    
    hub.register(FeatureDefinition(
        name="overloaded_defender",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_overloaded_defender,
        description="Piece defending 2+ attacked pieces",
    ))
    
    hub.register(FeatureDefinition(
        name="trapped_piece",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_trapped_piece,
        description="High-value piece with limited moves",
    ))
    
    hub.register(FeatureDefinition(
        name="desperado_opportunity",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_desperado_opportunity,
        description="Doomed piece can capture before dying",
    ))
    
    hub.register(FeatureDefinition(
        name="zwischenzug_possible",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_zwischenzug_possible,
        description="Intermediate move changes evaluation",
    ))
    
    hub.register(FeatureDefinition(
        name="x_ray_attack",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_x_ray_attack,
        description="Attack through another piece",
    ))
    
    hub.register(FeatureDefinition(
        name="deflection_possible",
        category=FeatureCategory.TACTICAL,
        compute_fn=compute_deflection_possible,
        description="Can force defender away",
    ))
    
    # Geometric sensors (5)
    hub.register(FeatureDefinition(
        name="mating_net_present",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_mating_net_present,
        description="King restricted to corner/edge area",
    ))
    
    hub.register(FeatureDefinition(
        name="king_tropism",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_king_tropism,
        description="Pieces aimed at enemy king",
    ))
    
    hub.register(FeatureDefinition(
        name="square_control_imbalance",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_square_control_imbalance,
        description="Central square control difference",
    ))
    
    hub.register(FeatureDefinition(
        name="outpost_available",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_outpost_available,
        description="Knight/Bishop outpost squares",
    ))
    
    hub.register(FeatureDefinition(
        name="weak_square_complex",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_weak_square_complex,
        description="Multiple weak squares same color",
    ))
    
    # Positional sensors (4)
    hub.register(FeatureDefinition(
        name="pawn_majority",
        category=FeatureCategory.POSITIONAL,
        compute_fn=compute_pawn_majority,
        description="Pawn count advantage on wing",
    ))
    
    hub.register(FeatureDefinition(
        name="passed_pawn_potential",
        category=FeatureCategory.POSITIONAL,
        compute_fn=compute_passed_pawn_potential,
        description="Pawn close to promotion",
    ))
    
    hub.register(FeatureDefinition(
        name="blockade_present",
        category=FeatureCategory.POSITIONAL,
        compute_fn=compute_blockade_present,
        description="Piece blocking passed pawn",
    ))
    
    hub.register(FeatureDefinition(
        name="bishop_pair_bonus",
        category=FeatureCategory.POSITIONAL,
        compute_fn=compute_bishop_pair_bonus,
        description="Both bishops vs one/none",
    ))
    
    # Dynamic sensors (3)
    hub.register(FeatureDefinition(
        name="mobility_restriction",
        category=FeatureCategory.DYNAMIC,
        compute_fn=compute_mobility_restriction,
        description="Low mobility indicator (pre-Zugzwang)",
    ))
    
    hub.register(FeatureDefinition(
        name="tempo_advantage",
        category=FeatureCategory.DYNAMIC,
        compute_fn=compute_tempo_advantage,
        description="Initiative/tempo assessment",
    ))
    
    hub.register(FeatureDefinition(
        name="piece_coordination",
        category=FeatureCategory.DYNAMIC,
        compute_fn=compute_piece_coordination,
        description="Pieces supporting each other",
    ))
    
    # Stalemate detection sensors (shared across endgames)
    hub.register(FeatureDefinition(
        name="enemy_king_rank",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_enemy_king_rank,
        description="Enemy king rank position (0-1)",
    ))
    
    hub.register(FeatureDefinition(
        name="enemy_king_file",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_enemy_king_file,
        description="Enemy king file position (0-1)",
    ))
    
    hub.register(FeatureDefinition(
        name="enemy_king_at_edge",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_enemy_king_at_edge,
        description="Enemy king on edge rank/file (0/1)",
    ))
    
    hub.register(FeatureDefinition(
        name="enemy_king_in_corner",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_enemy_king_in_corner,
        description="Enemy king in/near corner (0/0.5/1)",
    ))
    
    hub.register(FeatureDefinition(
        name="enemy_king_mobility",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_enemy_king_mobility,
        description="Enemy king escape squares (0-1 normalized)",
    ))
    
    hub.register(FeatureDefinition(
        name="enemy_king_mobility_raw",
        category=FeatureCategory.GEOMETRIC,
        compute_fn=compute_enemy_king_mobility_raw,
        description="Enemy king escape squares (0-8 raw)",
    ))
    
    hub.register(FeatureDefinition(
        name="stalemate_danger",
        category=FeatureCategory.DYNAMIC,
        compute_fn=compute_stalemate_danger,
        description="Composite stalemate risk signal (0-1)",
    ))


# ============================================================================
# Stalemate Detection Sensors (Shared across endgames)
# ============================================================================

def compute_enemy_king_rank(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Get enemy king's rank position.
    
    Returns: [0.0, 1.0] - normalized rank (0=rank 1, 1=rank 8)
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.5
    return chess.square_rank(enemy_king) / 7.0


def compute_enemy_king_file(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Get enemy king's file position.
    
    Returns: [0.0, 1.0] - normalized file (0=a-file, 1=h-file)
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.5
    return chess.square_file(enemy_king) / 7.0


def compute_enemy_king_at_edge(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if enemy king is on an edge rank or file.
    
    Returns: [0.0, 1.0] - 1.0 if on edge, 0.0 if not
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.0
    
    rank = chess.square_rank(enemy_king)
    file = chess.square_file(enemy_king)
    
    # On edge if rank is 0 or 7, or file is 0 or 7
    if rank in (0, 7) or file in (0, 7):
        return 1.0
    return 0.0


def compute_enemy_king_in_corner(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Detect if enemy king is in a corner (a1, a8, h1, h8).
    
    Returns: [0.0, 1.0] - 1.0 if in corner, 0.5 if near corner, 0.0 otherwise
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.0
    
    corners = [chess.A1, chess.A8, chess.H1, chess.H8]
    if enemy_king in corners:
        return 1.0
    
    # Near corner (within 1 square of corner)
    rank = chess.square_rank(enemy_king)
    file = chess.square_file(enemy_king)
    
    near_corner = (
        (rank <= 1 and file <= 1) or  # Near a1
        (rank <= 1 and file >= 6) or  # Near h1
        (rank >= 6 and file <= 1) or  # Near a8
        (rank >= 6 and file >= 6)     # Near h8
    )
    
    return 0.5 if near_corner else 0.0


def compute_enemy_king_mobility(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Count enemy king's legal escape squares (normalized).
    
    Returns: [0.0, 1.0] - 0.0 = no escapes (trapped), 1.0 = all 8 escapes free
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 1.0
    
    escape_count = 0
    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)
    
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            nf, nr = king_file + df, king_rank + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                sq = chess.square(nf, nr)
                occupant = board.piece_at(sq)
                # Blocked by enemy's own piece
                if occupant and occupant.color == (not board.turn):
                    continue
                # Attacked by us
                if board.is_attacked_by(board.turn, sq):
                    continue
                escape_count += 1
    
    return escape_count / 8.0


def compute_enemy_king_mobility_raw(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Raw count of enemy king escape squares (0-8).
    
    Useful for threshold checks (e.g., <= 2 means danger).
    
    Returns: [0, 8] as float
    """
    return compute_enemy_king_mobility(board, computed) * 8.0


def compute_stalemate_danger(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Composite sensor: High when enemy king is nearly trapped (stalemate risk).
    
    Combines:
    - King at edge
    - King mobility <= 2-3 squares
    - King in corner (higher danger)
    
    This is the key "scent" signal for stalemate avoidance.
    
    Returns: [0.0, 1.0] - 0.0 = safe, 1.0 = extreme stalemate danger
    """
    # Get component values (use computed cache if available)
    at_edge = computed.get("enemy_king_at_edge")
    if at_edge is None:
        at_edge = compute_enemy_king_at_edge(board, computed)
    
    mobility = computed.get("enemy_king_mobility")
    if mobility is None:
        mobility = compute_enemy_king_mobility(board, computed)
    
    in_corner = computed.get("enemy_king_in_corner")
    if in_corner is None:
        in_corner = compute_enemy_king_in_corner(board, computed)
    
    # Stalemate danger formula:
    # - High if on edge AND low mobility
    # - Extra high if in corner
    
    mobility_danger = 0.0
    raw_mobility = mobility * 8.0
    
    if raw_mobility <= 1:
        mobility_danger = 1.0  # Extreme: only 0-1 escape
    elif raw_mobility <= 2:
        mobility_danger = 0.8  # High: only 2 escapes
    elif raw_mobility <= 3:
        mobility_danger = 0.5  # Medium: only 3 escapes
    elif raw_mobility <= 4:
        mobility_danger = 0.2  # Low: 4 escapes
    
    # Combine with position
    danger = mobility_danger
    
    # On edge increases danger
    if at_edge > 0.5:
        danger = min(1.0, danger + 0.2)
    
    # In corner maximizes danger
    if in_corner >= 1.0:
        danger = min(1.0, danger + 0.3)
    elif in_corner >= 0.5:
        danger = min(1.0, danger + 0.1)
    
    return danger


# ============================================================================
# Convenience function for testing
# ============================================================================

def test_sensors():
    """Quick test of all sensors on starting position."""
    board = chess.Board()
    hub = FeatureHub()
    register_v2_sensors(hub)
    
    values = hub.compute_all(board)
    
    print("Wave 1 Sensor Values (starting position):")
    print("=" * 50)
    for name, value in sorted(values.items()):
        print(f"  {name}: {value:.3f}")
    
    return values


if __name__ == "__main__":
    test_sensors()

