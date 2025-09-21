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


def _find_our_rook_square_now(board: chess.Board, color: bool) -> chess.Square | None:
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type == chess.ROOK:
            return sq
    return None

def _find_our_rooks_now(board: chess.Board, color: bool):
    rooks = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type == chess.ROOK:
            rooks.append(sq)
    return rooks

def _box_dims_with_rook(board: chess.Board) -> tuple[int, int]:
    """
    Compute a rectangle proxy that considers both board edges and our rook line(s) as boundaries.
    width = left_gap + right_gap + 1, height = down_gap + up_gap + 1.
    A rook on same rank/file provides an interior boundary on that side: gap up to the rook minus one.
    """
    color = board.turn
    ek = _enemy_king_square(board)
    ef, er = chess.square_file(ek), chess.square_rank(ek)

    left_gap = ef
    right_gap = 7 - ef
    down_gap = er
    up_gap = 7 - er

    for rsq in _find_our_rooks_now(board, color):
        rf, rr = chess.square_file(rsq), chess.square_rank(rsq)
        # Treat a rook as a 'wall' along its whole rank and file (no blockers in KRK)
        # Horizontal influence (rank): reduces vertical gaps toward rr
        if rr < er:
            down_gap = min(down_gap, er - rr - 1)
        elif rr > er:
            up_gap = min(up_gap, rr - er - 1)
        # Vertical influence (file): reduces horizontal gaps toward rf
        if rf < ef:
            left_gap = min(left_gap, ef - rf - 1)
        elif rf > ef:
            right_gap = min(right_gap, rf - ef - 1)

    width = left_gap + right_gap + 1
    height = down_gap + up_gap + 1
    return width, height

def enemy_at_edge(board: chess.Board) -> bool:
    ek = _enemy_king_square(board)
    return dist_to_edge(ek) == 0

def rook_safe_now(board: chess.Board) -> bool:
    """Is our rook safe right now (before we move)?"""
    color = board.turn
    rooks = _find_our_rooks_now(board, color)
    if not rooks:
        return False
    rsq = rooks[0]
    enemy_k = board.king(not color)
    our_k = board.king(color)
    if enemy_k is None or our_k is None:
        return True
    if chebyshev(enemy_k, rsq) > 1:
        return True
    # simulate enemy to move capturing rook
    b = board.copy(stack=False)
    b.turn = not color
    cap = chess.Move(enemy_k, rsq)
    if cap in b.legal_moves:
        return chebyshev(our_k, rsq) <= 1
    return True

def has_stable_cut(board: chess.Board) -> bool:
    """Whether our rook already forms a safe fence against the enemy king."""
    color = board.turn
    ek = _enemy_king_square(board)
    rsq = _find_our_rook_square_now(board, color)
    if rsq is None or ek is None:
        return False
    same_file = chess.square_file(rsq) == chess.square_file(ek)
    same_rank = chess.square_rank(rsq) == chess.square_rank(ek)
    aligned = same_file or same_rank
    far_enough = chebyshev(rsq, ek) >= 2
    return aligned and far_enough and rook_safe_now(board)


# ---------- nearest-edge & rook fence targeting ----------

def enemy_nearest_edge_info(board: chess.Board, enemy_king: chess.Square = None) -> dict:
    """
    Compute the enemy king's nearest board edge and the ideal rook 'fence' one square inside.

    Returns dict with keys:
      - axis: 'file' or 'rank' indicating which edge we are targeting
      - edge_index: 0 or 7 (edge coordinate)
      - distance: Chebyshev distance of enemy king to the chosen edge (0..3)
      - target_line: the file (if axis=='file') or rank (if axis=='rank') for the rook fence (edge±1)
      - ef, er: enemy king file and rank (0..7)
    """
    if enemy_king is None:
        ek = _enemy_king_square(board)
    else:
        ek = enemy_king
    ef, er = chess.square_file(ek), chess.square_rank(ek)
    d_file_left = ef
    d_file_right = 7 - ef
    d_rank_down = er
    d_rank_up = 7 - er
    # Best distances along file and rank axes
    d_file = min(d_file_left, d_file_right)
    d_rank = min(d_rank_down, d_rank_up)
    if d_file <= d_rank:
        # Target a vertical fence (same rank alignment) by placing rook on file edge±1
        edge_index = 0 if d_file_left <= d_file_right else 7
        target_line = 1 if edge_index == 0 else 6
        return {"axis": "file", "edge_index": edge_index, "distance": d_file, "target_line": target_line, "ef": ef, "er": er}
    else:
        edge_index = 0 if d_rank_down <= d_rank_up else 7
        target_line = 1 if edge_index == 0 else 6
        return {"axis": "rank", "edge_index": edge_index, "distance": d_rank, "target_line": target_line, "ef": ef, "er": er}


def rook_distance_to_target_fence(board: chess.Board) -> int:
    """
    Distance (in squares) from our rook to the ideal target fence line one square inside
    the enemy king's nearest edge. Returns a large value if rook is absent.
    """
    info = enemy_nearest_edge_info(board)
    rsq = _find_our_rook_square_now(board, board.turn)
    if rsq is None:
        return 8
    rf, rr = chess.square_file(rsq), chess.square_rank(rsq)
    if info["axis"] == "file":
        return abs(rf - info["target_line"])
    return abs(rr - info["target_line"])  # axis == 'rank'


def rook_distance_to_target_fence_after(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    # Use current side-to-move perspective doesn't matter for geometry
    return rook_distance_to_target_fence(b)


def rook_on_target_fence_after(board: chess.Board, move: chess.Move) -> bool:
    return rook_distance_to_target_fence_after(board, move) == 0


def rook_fence_ready_now(board: chess.Board) -> bool:
    """
    True iff enemy king is on some edge and our rook is on the target fence line
    one square inside that edge, and the rook is currently safe.
    """
    info = enemy_nearest_edge_info(board)
    if info["distance"] != 0:
        return False
    rsq = _find_our_rook_square_now(board, board.turn)
    if rsq is None:
        return False
    rf, rr = chess.square_file(rsq), chess.square_rank(rsq)
    target = info["target_line"]
    on_fence = (rr == target) if info["axis"] == "rank" else (rf == target)
    return on_fence and rook_safe_now(board)

def king_outside_rook_cut_after(board: chess.Board, move: chess.Move) -> bool:
    """
    After 'move', return True if our king is on the enemy side of our rook
    relative to the separation line (file or rank) when the rook and enemy are aligned.
    This encourages stepping OUTSIDE the fence to help deliver mate at the rim.
    """
    color = board.turn
    b = board.copy(stack=False)
    b.push(move)
    ok = b.king(color)
    ek = b.king(not color)
    rsq = _find_our_rook_square_now(b, color)
    if ok is None or ek is None or rsq is None:
        return False
    of, or_ = chess.square_file(ok), chess.square_rank(ok)
    ef, er = chess.square_file(ek), chess.square_rank(ek)
    rf, rr = chess.square_file(rsq), chess.square_rank(rsq)
    # Vertical cut (same file)
    if rf == ef:
        # our king must be on the same file and beyond rook toward enemy direction
        if of != rf:
            return False
        return (er - rr) * (or_ - rr) > 0
    # Horizontal cut (same rank)
    if rr == er:
        if or_ != rr:
            return False
        return (ef - rf) * (of - rf) > 0
    return False

def confinement_box(board: chess.Board) -> tuple[int, int]:
    """
    Calculate confinement box based on target fence and rook position.

    The box represents the space where the enemy king is confined,
    bounded by the target fence (where our rook should be) and the board edge.
    """
    # Find the color that has the rook (in KRK, it's the attacking side)
    rook_color = None
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.ROOK:
            rook_color = p.color
            break

    if rook_color is None:
        return 8, 8

    # Find the enemy king (the king not belonging to the rook's side)
    enemy_king = None
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.KING and p.color != rook_color:
            enemy_king = sq
            break

    if enemy_king is None:
        return 8, 8

    ek = enemy_king

    # Get target fence information
    edge_info = enemy_nearest_edge_info(board, ek)
    axis = edge_info["axis"]
    edge_index = edge_info["edge_index"]
    target_fence = edge_info["target_line"]

    # Find our rook
    rsq = _find_our_rook_square_now(board, rook_color)
    if rsq is None:
        # No rook, full board confinement
        return 8, 8

    rf, rr = chess.square_file(rsq), chess.square_rank(rsq)

    if axis == "rank":
        # King closest to rank edge, confinement determined by rank positions
        fence_rank = target_fence
        edge_rank = edge_index

        if rr == fence_rank:
            # Rook is on target fence - king confined to the edge side only
            height = abs(edge_index - fence_rank)  # Distance between fence and edge
        else:
            # Rook not on fence, full height
            height = 8

        # Width is full board (no file confinement yet)
        width = 8

    else:  # axis == "file"
        # King closest to file edge, confinement determined by file positions
        fence_file = target_fence
        edge_file = edge_index

        if rf == fence_file:
            # Rook is on target fence - king confined between fence and edge
            if edge_index == 0:
                # Driving toward left, king confined to files 0 to fence_file
                width = fence_file - edge_file + 1
            else:
                # Driving toward right, king confined to files fence_file to edge_file
                width = edge_file - fence_file + 1
        else:
            # Rook not on fence, full width
            width = 8

        # Height is full board (no rank confinement yet)
        height = 8

    return width, height

def box_area(board: chess.Board) -> int:
    w, h = confinement_box(board)
    return w * h

def box_min_side(board: chess.Board) -> int:
    """
    Minimum side length of the confinement rectangle.
    This is the critical dimension - the one we most want to minimize.
    """
    w, h = confinement_box(board)
    return min(w, h)

def box_min_side_after(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    return box_min_side(b)

def box_area_after(board: chess.Board, move: chess.Move) -> int:
    b = board.copy(stack=False)
    b.push(move)
    return box_area(b)


# Legacy function for backward compatibility - now uses confinement_box
def _box_dims_with_rook(board: chess.Board) -> tuple[int, int]:
    """
    Legacy function - use confinement_box instead.
    Kept for any external dependencies.
    """
    return confinement_box(board)

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

    # Robust guard: if kings are undefined (shouldn't happen in KRK), treat as safe
    if enemy_k is None or our_k is None or rook_sq is None:
        return True

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

    # Robust guards: if something is off with the board, avoid crashes and assume safe
    if enemy_k is None or our_k is None or rook_sq is None:
        return True

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
    If it stays equal, allow it only when our king meaningfully progresses.

    Relies on existing helpers in this module:
      - box_area(board)              -> numeric proxy for box size
      - box_area_after(board, move)  -> same, after move
      - our_king_progress(board, move) -> positive if our K gets closer to the key region/enemy K
    """
    before = box_area(board)
    after  = box_area_after(board, move)

    if after < before:
        return True

    if not allow_equal_if_progress:
        return False

    # Equal box is acceptable iff we make progress (king steps)
    try:
        kp = our_king_progress(board, move)
    except Exception:
        kp = 0

    return (after == before) and (kp > 0)


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

    ms_before = box_min_side(board)
    try:
        ms_after = box_min_side_after(board, move)
    except Exception:
        ms_after = ms_before

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
        "min_side_before": ms_before,
        "min_side_after":  ms_after,
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
    if a is None or b is None:
        return False
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

def enemy_king_mobility_after(board: chess.Board, move: chess.Move) -> int:
    """
    Number of legal king moves available to the enemy after we play 'move'.
    Lower is better (tighter confinement).
    """
    b = board.copy(stack=False)
    b.push(move)
    enemy = b.turn  # after our move, it's enemy to move
    king_sq = b.king(enemy)
    if king_sq is None:
        return 0
    cnt = 0
    for mv in b.legal_moves:
        piece = b.piece_at(mv.from_square)
        if piece and piece.piece_type == chess.KING and piece.color == enemy:
            cnt += 1
    return cnt

# ---------- repetition & 50-move awareness ----------

from collections import deque
from typing import Deque, Iterable

def _fen_after_move(board: chess.Board, move: chess.Move) -> str:
    b = board.copy(stack=False)
    b.push(move)
    # Normalize to piece placement + side to move only, matching fen_history entries
    return b.board_fen() + " " + ("w" if b.turn else "b")

def repetition_penalty(board: chess.Board, move: chess.Move, fen_history: Iterable[str] | None) -> float:
    """
    Return 1.0 iff the position after 'move' matches the most recent FEN-with-turn in history,
    else 0.0. This is a soft penalty used in scoring.
    Expects fen_history entries normalized with side-to-move.
    """
    if not fen_history:
        return 0.0
    try:
        last = next(iter(fen_history))  # most recent first if using deque.appendleft; but we don't know order
    except StopIteration:
        last = None
    # If history is append-right and we read from the rightmost as last, we can't know.
    # Robust: compare against the immediate previous env value if available in a list/deque.
    hist = list(fen_history)
    prev = hist[-1] if hist else None
    pos_after = _fen_after_move(board, move)
    return 1.0 if prev is not None and pos_after == prev else 0.0

def would_cause_threefold(board: chess.Board, move: chess.Move, fen_history: Iterable[str] | None) -> bool:
    """
    Heuristic detector: if the FEN-with-turn after 'move' already appears twice in history,
    then playing it would produce a 3rd occurrence -> treat as threefold risk.
    """
    if not fen_history:
        return False
    pos_after = _fen_after_move(board, move)
    count = 0
    for fen in fen_history:
        if fen == pos_after:
            count += 1
            if count >= 2:
                return True
    return False

def fifty_move_pressure(board: chess.Board) -> float:
    """
    Scale 0.0..1.0 as halfmove clock goes 40..48.
    Encourages decisive progress as the draw limit approaches.
    """
    hm = getattr(board, "halfmove_clock", 0)
    if hm <= 40:
        return 0.0
    if hm >= 48:
        return 1.0
    return (hm - 40) / 8.0