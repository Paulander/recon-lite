import chess

# ---------- basic helpers ----------

def chebyshev(a: chess.Square, b: chess.Square) -> int:
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))

def dist_to_edge(sq: chess.Square) -> int:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return min(f, 7 - f, r, 7 - r)

# ---------- state predicates ----------

def is_mate(board: chess.Board) -> bool:
    return board.is_checkmate()

def is_stalemate(board: chess.Board) -> bool:
    return board.is_stalemate()

def is_stalemate_after(board: chess.Board, move: chess.Move) -> bool:
    b = board.copy(stack=False)
    b.push(move)
    return b.is_stalemate()

# ---------- “box” proxy ----------
# We use a simple, robust proxy: area shrinks as the enemy king approaches any edge.
# area(d) = (2d+1)^2, where d = dist_to_edge(enemy king). This is monotone for “drive to edge.”

def _enemy_king_square(board: chess.Board) -> chess.Square:
    # Always “the other side” relative to side-to-move.
    return board.king(not board.turn)


def box_area(board: chess.Board) -> int:
    ek = _enemy_king_square(board)
    d = dist_to_edge(ek)
    return (2 * d + 1) ** 2

def box_area_after(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    return box_area(b)

# ---------- checks & safety ----------

def gives_safe_check(board: chess.Board, move: chess.Move) -> bool:
    b = board.copy(stack=False)
    b.push(move)
    if not b.is_check():
        return False
    if b.is_stalemate():
        return False
    return rook_safe_after(board, move)


def _find_our_rook_square_after(b: chess.Board, our_color: bool) -> chess.Square | None:
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p and p.color == our_color and p.piece_type == chess.ROOK:
            return sq
    return None

def rook_safe_after(board: chess.Board, move: chess.Move) -> bool:
    """
    Rook is safe right after our move if the enemy king cannot *legally* capture it in one move,
    unless our king can immediately recapture.
    """
    color = board.turn
    b = board.copy(stack=False)
    b.push(move)

    rook_sq = _find_our_rook_square_after(b, color)
    if rook_sq is None:
        return False  # we lost the rook in our own move…

    enemy = not color
    enemy_k = b.king(enemy)
    our_k   = b.king(color)

    # Fast adjacency test (cheap reject)
    if chebyshev(enemy_k, rook_sq) > 1:
        return True

    # Construct the capture and test if it is legal in the new position
    cap = chess.Move(enemy_k, rook_sq)
    if cap in b.legal_moves:
        # If our king is adjacent too, we will recapture and remain winning -> allow
        return chebyshev(our_k, rook_sq) <= 1
    return True
    """
    After our move, the enemy king must not be able to capture our rook on the next move
    *unless* our king can immediately recapture (i.e., rook square is defended by our king).
    This captures the common 'don't hang the rook' rule in KRK.
    """
    color = board.turn
    b = board.copy(stack=False)
    b.push(move)
    rook_sq = _find_our_rook_square_after(b, color)
    if rook_sq is None:
        # Rook should exist; treat as unsafe
        return False

    enemy_k = b.king(not color)
    our_k = b.king(color)

    # if enemy king adjacent to rook (could capture)
    if chebyshev(enemy_k, rook_sq) <= 1:
        # if our king also adjacent → defended, allowed
        if chebyshev(our_k, rook_sq) <= 1:
            return True
        return False
    return True

# ---------- king progress & stable cut ----------

def our_king_progress(board: chess.Board, move: chess.Move) -> float:
    """
    Heuristic: progress if our king reduces Chebyshev distance to the enemy king.
    (Good enough for Phase 0 rendezvous.)
    """
    color = board.turn
    ek = _enemy_king_square(board)
    ok_before = board.king(color)

    b = board.copy(stack=False)
    b.push(move)
    ok_after = b.king(color)

    before = chebyshev(ok_before, ek)
    after = chebyshev(ok_after, ek)
    return float(before - after)  # positive = closer

def creates_stable_cut(board: chess.Board, move: chess.Move) -> bool:
    """
    Approximate 'cut' condition after the move:
    - Our rook is on same file or rank as enemy king, at distance >= 2
    - Rook is safe (per rook_safe_after)
    This loosely encodes 'you shall not pass' with a rook line, supported later by the king.
    """
    color = board.turn
    b = board.copy(stack=False)
    b.push(move)

    ek = b.king(not color)
    rook_sq = _find_our_rook_square_after(b, color)
    if rook_sq is None:
        return False

    same_file = chess.square_file(rook_sq) == chess.square_file(ek)
    same_rank = chess.square_rank(rook_sq) == chess.square_rank(ek)
    aligned = same_file or same_rank
    far_enough = chebyshev(rook_sq, ek) >= 2

    return aligned and far_enough and rook_safe_after(board, move)

# --- KRK Phase-1/2 monotonicity helper ---------------------------------------

def shrinks_or_preserves_box(board, move, *, allow_equal_if_progress=True):
    """
    Return True if the 'box' around the enemy king strictly shrinks after 'move'.
    If it stays equal, allow it only when our king meaningfully progresses or we give a safe check.

    Relies on existing helpers in this module:
      - box_area(board)              -> numeric proxy for box size
      - box_area_after(board, move)  -> same, after move
      - our_king_progress(board, move) -> positive if our K gets closer to the key region/enemy K
      - gives_safe_check(board, move)  -> True if we safely check the enemy K
    """
    before = box_area(board)
    after  = box_area_after(board, move)

    if after < before:
        return True

    if not allow_equal_if_progress:
        return False

    # Equal box is acceptable iff we make progress (king steps) or a safe check that corrals.
    try:
        kp = our_king_progress(board, move)
    except Exception:
        kp = 0

    try:
        sc = gives_safe_check(board, move)
    except Exception:
        sc = False

    return (after == before) and (kp > 0 or sc)


# --- Feature logger used by the demo -----------------------------------------

def move_features(board, move):
    """
    Compact feature bundle for logging/diagnostics.
    Safe to call on any legal move. Uses the same predicates used by actuators.
    """
    b_before = box_area(board)
    try:
        b_after = box_area_after(board, move)
    except Exception:
        # If your box_area_after expects legality and something went wrong, fall back.
        b_after = b_before

    try:
        kp = our_king_progress(board, move)
    except Exception:
        kp = 0

    try:
        rs = rook_safe_after(board, move)
    except Exception:
        rs = False

    try:
        sc = gives_safe_check(board, move)
    except Exception:
        sc = False

    return {
        "box_area_before": b_before,
        "box_area_after":  b_after,
        "king_progress":   kp,
        "rook_safe_after": rs,
        "gives_safe_check": sc,
    }

def on_rim(sq: chess.Square) -> bool:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    return f in (0, 7) or r in (0, 7)

def is_cornered(sq: chess.Square) -> bool:
    return sq in (chess.A1, chess.A8, chess.H1, chess.H8)

def king_to_rook_distance(board: chess.Board) -> int:
    """Chebyshev distance between our king and our rook (after *current* board)."""
    color = board.turn
    ok = board.king(color)
    rsq = _find_our_rook_square_after(board, color)
    if rsq is None:
        return 8  # “far” / missing
    return chebyshev(ok, rsq)

# ---------- opposition helpers ----------

def kings_in_opposition(a: chess.Square, b: chess.Square) -> bool:
    """
    Geometric opposition: kings on same file or rank with exactly one square between.
    (Adjacent-diagonal is NOT 'opposition' in this simple KRK sense.)
    """
    fa, ra = chess.square_file(a), chess.square_rank(a)
    fb, rb = chess.square_file(b), chess.square_rank(b)
    same_file = fa == fb
    same_rank = ra == rb
    dx = abs(fa - fb)
    dy = abs(ra - rb)
    return (same_file and dy == 2) or (same_rank and dx == 2)


def has_opposition(board: chess.Board) -> bool:
    """
    Returns True iff the SIDE-NOT-TO-MOVE currently has opposition.
    In endgames, 'having the opposition' means: kings in opposition AND it's your opponent's turn.
    """
    color = board.turn                  # side to move now
    ok = board.king(color)
    ek = board.king(not color)
    # If kings are in opposition and it's *opponent* to move, then WE (not color) have it.
    return kings_in_opposition(ok, ek) and (board.turn == color)


def has_opposition_after(board: chess.Board, move: chess.Move) -> bool:
    """
    After we play 'move' (color = board.turn), return True iff WE (the mover) will have opposition.
    That means: after the move, kings are in opposition and it's now opponent's turn.
    """
    color = board.turn
    b = board.copy(stack=False)
    b.push(move)
    ok = b.king(color)
    ek = b.king(not color)
    # After our move, it's opponent's turn (b.turn != color), and kings must be in opposition.
    return kings_in_opposition(ok, ek) and (b.turn != color)
