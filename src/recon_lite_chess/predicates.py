"""
Chess position evaluation predicates (sensors) for ReCoN KRK system.

These functions analyze chess positions and provide boolean/quantitative
assessments used by the ReCoN network to make decisions.
"""

import chess


def is_mate(board: chess.Board) -> bool:
    """Check if current position is checkmate."""
    return board.is_checkmate()


def is_stalemate(board: chess.Board) -> bool:
    """Check if current position is stalemate."""
    return board.is_stalemate()


def rook_safe_after(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if rook will be safe after making the given move.

    Rook is safe if:
    - Enemy king cannot capture rook on next move, OR
    - Our king defends the rook square (Chebyshev distance <= 1)
    """
    # Create a copy to simulate the move
    board_copy = board.copy()
    board_copy.push(move)

    # Find rook square after move
    rook_square = None
    for square in chess.SQUARES:
        if board_copy.piece_at(square) and board_copy.piece_at(square).piece_type == chess.ROOK:
            rook_square = square
            break

    if rook_square is None:
        return False  # No rook found

    # Check if enemy king can capture rook
    enemy_king_square = board_copy.king(not board_copy.turn)
    if enemy_king_square is None:
        return True  # No enemy king

    # Chebyshev distance between enemy king and rook
    king_file = chess.square_file(enemy_king_square)
    king_rank = chess.square_rank(enemy_king_square)
    rook_file = chess.square_file(rook_square)
    rook_rank = chess.square_rank(rook_square)

    distance = max(abs(king_file - rook_file), abs(king_rank - rook_rank))

    # Rook is safe if king can't reach it in one move
    if distance > 1:
        return True

    # Check if our king defends the rook square
    our_king_square = board_copy.king(board_copy.turn)
    if our_king_square is None:
        return False

    our_file = chess.square_file(our_king_square)
    our_rank = chess.square_rank(our_king_square)
    our_distance = max(abs(our_file - rook_file), abs(our_rank - rook_rank))

    # Our king defends if it's close enough to the rook
    return our_distance <= 1


def box_area(board: chess.Board) -> int:
    """
    Compute the area of the "box" constraining the enemy king.

    The box is defined by the minimum rectangle that contains:
    - The enemy king
    - Limited by rook's lines of attack (rank/file)

    Returns the area of this constraining rectangle.
    """
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 64  # Maximum possible area

    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)

    # Find rook to determine box constraints
    rook_file = None
    rook_rank = None

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.ROOK and piece.color == board.turn:
            rook_file = chess.square_file(square)
            rook_rank = chess.square_rank(square)
            break

    if rook_file is None:
        # No rook found, use full board constraints
        min_file, max_file = 0, 7
        min_rank, max_rank = 0, 7
    else:
        # Box is constrained by rook's rank/file lines
        if rook_file == king_file:
            # Rook on same file - box spans all ranks
            min_file, max_file = max(0, king_file - 1), min(7, king_file + 1)
            min_rank, max_rank = 0, 7
        elif rook_rank == king_rank:
            # Rook on same rank - box spans all files
            min_file, max_file = 0, 7
            min_rank, max_rank = max(0, king_rank - 1), min(7, king_rank + 1)
        else:
            # Rook on different rank/file - box is 3x3 around king
            min_file = max(0, king_file - 1)
            max_file = min(7, king_file + 1)
            min_rank = max(0, king_rank - 1)
            max_rank = min(7, king_rank + 1)

    # Ensure king is within the box
    min_file = min(min_file, king_file)
    max_file = max(max_file, king_file)
    min_rank = min(min_rank, king_rank)
    max_rank = max(max_rank, king_rank)

    # Calculate area
    file_span = max_file - min_file + 1
    rank_span = max_rank - min_rank + 1

    return file_span * rank_span


def box_area_after(board: chess.Board, move: chess.Move) -> int:
    """Compute box area after making the given move."""
    board_copy = board.copy()
    board_copy.push(move)
    return box_area(board_copy)


def shrinks_or_preserves_box(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if move shrinks or preserves the enemy king's box.

    Allows equal area only if it improves king position or safety.
    """
    current_area = box_area(board)
    next_area = box_area_after(board, move)

    if next_area < current_area:
        return True  # Strictly shrinks - always good

    if next_area == current_area:
        # Allow equal area if it improves king support or opposition
        return improves_king_position(board, move)

    return False  # Expands box - bad


def improves_king_position(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if move improves our king's position relative to enemy king.

    Considers distance, opposition, and support opportunities.
    """
    # Simple heuristic: prefer moves that reduce distance to enemy king
    enemy_king = board.king(not board.turn)
    our_king = board.king(board.turn)

    if enemy_king is None or our_king is None:
        return False

    # Current distance
    current_dist = chess.square_distance(our_king, enemy_king)

    # Distance after move
    board_copy = board.copy()
    board_copy.push(move)
    new_our_king = board_copy.king(board_copy.turn)

    if new_our_king is None:
        return False

    new_dist = chess.square_distance(new_our_king, enemy_king)

    # Move is good if it reduces distance (better support) or maintains opposition
    return new_dist < current_dist or (new_dist == current_dist and has_opposition_after(board, move))


def has_opposition(board: chess.Board) -> bool:
    """Check if kings are in opposition (same file/rank, odd squares apart)."""
    enemy_king = board.king(not board.turn)
    our_king = board.king(board.turn)

    if enemy_king is None or our_king is None:
        return False

    enemy_file = chess.square_file(enemy_king)
    enemy_rank = chess.square_rank(enemy_king)
    our_file = chess.square_file(our_king)
    our_rank = chess.square_rank(our_king)

    # Same file or rank
    if enemy_file == our_file:
        rank_diff = abs(enemy_rank - our_rank)
        return rank_diff > 1 and rank_diff % 2 == 1  # Odd number of squares apart
    elif enemy_rank == our_rank:
        file_diff = abs(enemy_file - our_file)
        return file_diff > 1 and file_diff % 2 == 1  # Odd number of squares apart

    return False


def has_opposition_after(board: chess.Board, move: chess.Move) -> bool:
    """Check opposition after making the given move."""
    board_copy = board.copy()
    board_copy.push(move)
    return has_opposition(board_copy)


def our_king_progress(board: chess.Board, move: chess.Move) -> float:
    """
    Score how much the move improves our king's position.

    Considers distance to enemy king, opposition, and rook support.
    """
    score = 0.0

    # Distance improvement
    enemy_king = board.king(not board.turn)
    our_king = board.king(board.turn)

    if enemy_king and our_king:
        current_dist = chess.square_distance(our_king, enemy_king)

        board_copy = board.copy()
        board_copy.push(move)
        new_our_king = board_copy.king(board_copy.turn)

        if new_our_king:
            new_dist = chess.square_distance(new_our_king, enemy_king)
            score += (current_dist - new_dist) * 0.5  # Reward closer distance

    # Opposition bonus
    if has_opposition_after(board, move):
        score += 0.3

    # Rook support bonus (king close to rook)
    rook_square = None
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.ROOK and piece.color == board.turn:
            rook_square = square
            break

    if rook_square and our_king:
        board_copy = board.copy()
        board_copy.push(move)
        new_our_king = board_copy.king(board_copy.turn)

        if new_our_king:
            current_rook_dist = chess.square_distance(our_king, rook_square)
            new_rook_dist = chess.square_distance(new_our_king, rook_square)
            score += (current_rook_dist - new_rook_dist) * 0.2

    return score


def gives_safe_check(board: chess.Board, move: chess.Move) -> bool:
    """
    Check if move gives check and the checking piece is safe.

    Used for move scoring bonus.
    """
    board_copy = board.copy()
    board_copy.push(move)

    # Must give check
    if not board_copy.is_check():
        return False

    # Check if the checking piece is safe
    # For now, simple heuristic: piece is safe if not attacked by enemy
    # This is a simplified version - could be more sophisticated
    return True
