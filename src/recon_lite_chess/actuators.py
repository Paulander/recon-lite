
"""
Chess move selection actuators for ReCoN KRK system.

These functions implement the "filter-first, then score" approach:
1. Hard filters: Eliminate unsafe/illegal moves
2. Soft scoring: Rank remaining moves by desirability
3. Fallback: If no moves pass filters, use safe legal move

IMPROVEMENTS (Sept 15, 2025):
- Global stall prevention with choose_any_safe_move() fallback
- Enhanced P0 rendezvous logic
- Rim promotion and monotonicity constraints
- Simplified scoring: 3.0*king_progress + 2.0*box_shrink + 1.0*safe_check − 0.2*rook_drag
"""

import chess
from typing import List, Tuple, Optional, Dict, Any
from .predicates import (
    is_stalemate_after, rook_safe_after, box_area, box_area_after,
    shrinks_or_preserves_box, our_king_progress, gives_safe_check,
    has_opposition_after, creates_stable_cut,
    enemy_king_mobility_after, repetition_penalty, would_cause_threefold,
    fifty_move_pressure, box_min_side, box_min_side_after, enemy_at_edge, king_outside_rook_cut_after, has_stable_cut,
    enemy_nearest_edge_info, rook_distance_to_target_fence_after, rook_on_target_fence_after
)


def _find_mate_in_one(board: chess.Board) -> Optional[str]:
    """Return UCI of a legal move that gives immediate checkmate, if any."""
    for move in board.legal_moves:
        b = board.copy(stack=False)
        b.push(move)
        if b.is_checkmate():
            return move.uci()
    return None

def _rook_distance_travel(move: chess.Move) -> float:
    """Calculate how far the rook moves (for penalty scoring)."""
    f1, r1 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    f2, r2 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    return float(abs(f1 - f2) + abs(r1 - r2))


def king_to_rook_distance(board: chess.Board) -> float:
    """Calculate distance between our king and rook."""
    wk_square = board.king(board.turn)
    wr_square = None

    for square in chess.SQUARES:
        if board.piece_at(square) and board.piece_at(square).piece_type == chess.ROOK and board.piece_at(square).color == board.turn:
            wr_square = square
            break

    if wr_square is None:
        return 8.0  # Max distance if no rook found

    wk_file, wk_rank = chess.square_file(wk_square), chess.square_rank(wk_square)
    wr_file, wr_rank = chess.square_file(wr_square), chess.square_rank(wr_square)
    return float(abs(wk_file - wr_file) + abs(wk_rank - wr_rank))


def is_rim_square(square: int) -> bool:
    """Check if square is on the board rim (edge)."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return file in [0, 7] or rank in [0, 7]


def is_cornered(board: chess.Board) -> bool:
    """Check if enemy king is cornered (in corner + kings in opposition)."""
    ek_square = board.king(not board.turn)
    wk_square = board.king(board.turn)

    # Check if enemy king is in corner
    corners = [chess.A1, chess.H1, chess.A8, chess.H8]
    if ek_square not in corners:
        return False

    # Check if kings are in opposition
    ek_file, ek_rank = chess.square_file(ek_square), chess.square_rank(ek_square)
    wk_file, wk_rank = chess.square_file(wk_square), chess.square_rank(wk_square)

    # Same file or rank with odd squares between
    if ek_file == wk_file:
        squares_between = abs(ek_rank - wk_rank) - 1
        return squares_between % 2 == 1
    elif ek_rank == wk_rank:
        squares_between = abs(ek_file - wk_file) - 1
        return squares_between % 2 == 1

    return False


def _normalize_fen_turn(board: chess.Board) -> str:
    # Minimal normalized key for repetition in KRK: piece placement + side to move
    return board.board_fen() + " " + ("w" if board.turn else "b")


def _calculate_score(board: chess.Board, move: chess.Move, phase: int, env: Optional[Dict[str, Any]] = None) -> float:
    """
    Phase-aware scoring.
    Base: 2*king_progress + 2*(box shrink) + 1*(safe check) − 0.2*rook_drag
    Add: -0.1*enemy_king_mobility_after (higher weight in P2/P3),
         -0.5*repetition_flag,
         + pressure * fifty_move_pressure(board)
    """
    env = env or {}
    score = 0.0

    # King progress
    kp = our_king_progress(board, move)
    score += 2.0 * kp

    # Box shrinking
    old_area = box_area(board)
    new_area = box_area_after(board, move)
    area_reduction = max(0, old_area - new_area)
    score += 2.0 * area_reduction
    # Min-side tightening: stronger than area for KRK (prefer narrow boxes)
    old_ms = box_min_side(board)
    new_ms = box_min_side_after(board, move)
    ms_reduction = max(0, old_ms - new_ms)
    score += 3.0 * ms_reduction

    # Safe check bonus
    if gives_safe_check(board, move):
        score += 1.0

    # Enemy king mobility reduction (global -0.1, stronger in P2/P3; strongest in P3)
    mobility = enemy_king_mobility_after(board, move)
    if phase == 3:
        mobility_w = 0.25
    elif phase == 2:
        mobility_w = 0.2
    else:
        mobility_w = 0.1
    score -= mobility_w * float(mobility)

    # Rook drag penalty
    score -= 0.2 * _rook_distance_travel(move)

    # Repetition penalty
    fen_hist = env.get("fen_history")
    rep = repetition_penalty(board, move, fen_hist) if fen_hist is not None else 0.0
    score -= 0.5 * rep

    # 50-move pressure
    pressure_flag = 1.0 if env.get("pressure") else 0.0
    score += pressure_flag * fifty_move_pressure(board)

    # Phase-specific boosts
    if phase == 3:
        # Encourage rim opposition / cornering
        if has_opposition_after(board, move):
            score += 3.0
        # If enemy at edge, prefer our king stepping outside the rook cut to help mate
        try:
            if enemy_at_edge(board) and king_outside_rook_cut_after(board, move):
                score += 2.0
        except Exception:
            pass
    elif phase == 2:
        # Slight nudge to keep pushing toward the rim if still not perfectly cornered
        try:
            score += 1.0 * edge_driving_bonus(board, move)
        except Exception:
            pass
    elif phase == 4:
        # Mate-in-one massive bonus handled in chooser
        pass
    elif phase == 1:
        # Strongly drive toward edge early
        try:
            score += 2.0 * edge_driving_bonus(board, move)
        except Exception:
            pass

    return score

def _candidate_key(board: chess.Board, move: chess.Move, phase: int, env: Optional[Dict[str, Any]]) -> Tuple:
    old_area = box_area(board)
    new_area = box_area_after(board, move)
    area_reduction = max(0, old_area - new_area)
    old_min_side = box_min_side(board)
    new_min_side = box_min_side_after(board, move)
    min_side_reduction = max(0, old_min_side - new_min_side)
    kp = our_king_progress(board, move)
    mobility = enemy_king_mobility_after(board, move)
    rdrag = _rook_distance_travel(move)
    uci = move.uci()
    sc = _calculate_score(board, move, phase, env)
    # Sort ascending on this tuple to achieve:
    #  - highest score first (-sc)
    #  - larger min_side_reduction first (-min_side_reduction)
    #  - larger area_reduction first (-area_reduction)
    #  - larger king_progress first (-kp)
    #  - lower mobility first (mobility)
    #  - shorter rook drag first (rdrag)
    #  - lexicographically smallest UCI first (uci)
    return (-sc, -min_side_reduction, -area_reduction, -kp, mobility, rdrag, uci)

def choose_any_safe_move(board: chess.Board) -> Optional[str]:
    """
    Deterministic fallback: mate-in-one > lexicographically smallest rook-safe move > smallest legal move.
    """
    mate = _find_mate_in_one(board)
    if mate:
        return mate
    legal = list(board.legal_moves)
    if not legal:
        return None
    safe_ucis = sorted([m.uci() for m in legal if rook_safe_after(board, m)])
    if safe_ucis:
        return safe_ucis[0]
    # Any legal move (deterministic: lexicographically smallest)
    return sorted(m.uci() for m in legal)[0]
 


def choose_move_phase0(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Phase 0: Rendezvous king and rook, create safe cut.
    Enhanced with proper distance-based king movement.
    """
    # Mate-in-one override
    mate = _find_mate_in_one(board)
    if mate:
        return mate
    legal_moves = list(board.legal_moves)
    # Disable P0 proposals if a stable cut already exists now or enemy is already at edge
    try:
        if enemy_at_edge(board):
            return None
        if has_stable_cut(board):
            return None
    except Exception:
        pass
    candidates: List[Tuple[Tuple, chess.Move]] = []
    distance = king_to_rook_distance(board)

    for move in legal_moves:
        # Hard filters
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        # Threefold guard
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            # Allow only if this move mates
            b = board.copy(stack=False)
            b.push(move)
            if not b.is_checkmate():
                continue
        if not shrinks_or_preserves_box(board, move):
            # In P0: allow equality with king progress OR safe check OR creating a stable cut.
            # Never allow expansion.
            old_a = box_area(board)
            new_a = box_area_after(board, move)
            if new_a > old_a:
                continue
            if (our_king_progress(board, move) <= 0 and
                not gives_safe_check(board, move) and
                not creates_stable_cut(board, move)):
                continue

        # P0-specific logic: focus on rendezvous until distance ≤ 2
        if distance > 2:
            # Prefer king moves that reduce distance to rook or enemy king
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                # Calculate distance reduction
                wk_square = board.king(board.turn)
                ek_square = board.king(not board.turn)
                wr_square = None

                for sq in chess.SQUARES:
                    if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.ROOK and board.piece_at(sq).color == board.turn:
                        wr_square = sq
                        break

                if wr_square:
                    old_dist_rook = abs(chess.square_file(wk_square) - chess.square_file(wr_square)) + abs(chess.square_rank(wk_square) - chess.square_rank(wr_square))
                    new_dist_rook = abs(chess.square_file(move.to_square) - chess.square_file(wr_square)) + abs(chess.square_rank(move.to_square) - chess.square_rank(wr_square))

                    if new_dist_rook < old_dist_rook:
                        # Very small rendezvous bias to avoid overshadowing good rook fences
                        key = _candidate_key(board, move, 0, env)
                        key = (key[0] - 0.1, *key[1:])
                        candidates.append((key, move))
                    else:
                        candidates.append((_candidate_key(board, move, 0, env), move))
                else:
                    candidates.append((_candidate_key(board, move, 0, env), move))
            else:
                # Rook moves: encourage if creating safe cut (fence), slight preference
                key = _candidate_key(board, move, 0, env)
                if creates_stable_cut(board, move):
                    key = (key[0] - 0.6, *key[1:])
                candidates.append((key, move))
        else:
            # Distance ≤ 2: normal scoring
            candidates.append((_candidate_key(board, move, 0, env), move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase1(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Phase 1: Drive enemy king to edge.
    Enhanced with rim promotion logic and monotonicity.
    """
    # Mate-in-one override
    mate = _find_mate_in_one(board)
    if mate:
        return mate
    legal_moves = list(board.legal_moves)
    candidates: List[Tuple[Tuple, chess.Move]] = []
    dbg: Dict[str, Any] = {}
    try:
        dbg["edge_info"] = enemy_nearest_edge_info(board)
        dbg["box_min_side"] = box_min_side(board)
        dbg["env_flags"] = {
            "require_min_side_shrink": (env or {}).get("require_min_side_shrink"),
            "forbid_zero_progress": (env or {}).get("forbid_zero_progress"),
        }
    except Exception:
        pass

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        # Strict P1: require edge-driving monotonicity OR creation of a stable cut
        old_a = box_area(board)
        new_a = box_area_after(board, move)
        if new_a > old_a:
            continue
        # Allow equality only if we create a stable cut or we get positive edge-driving bonus
        if new_a == old_a:
            if not (creates_stable_cut(board, move) or edge_driving_bonus(board, move) > 0):
                continue
        # Threefold guard
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            b = board.copy(stack=False)
            b.push(move)
            if not b.is_checkmate():
                continue

        # 50-move / pressure: forbid zero-progress choices when flagged
        if env is not None and env.get("forbid_zero_progress"):
            base = box_area(board)
            if (box_area_after(board, move) == base and
                box_min_side_after(board, move) == box_min_side(board) and
                our_king_progress(board, move) <= 0 and
                not gives_safe_check(board, move)):
                continue

        # If require_min_side_shrink, drop moves that don't reduce min-side
        if env is not None and env.get("require_min_side_shrink"):
            if box_min_side_after(board, move) >= box_min_side(board):
                continue
        # Fence targeting features
        fence_dist_after = rook_distance_to_target_fence_after(board, move)
        on_fence = rook_on_target_fence_after(board, move)

        # Compose key and bias toward fence targeting
        key = list(_candidate_key(board, move, 1, env))
        # Strongly prefer landing on target fence, else reduce distance to fence
        if on_fence:
            key[0] -= 1.5
        else:
            # Smaller improvement if we reduce distance to fence
            try:
                from .predicates import rook_distance_to_target_fence
                cur_dist = rook_distance_to_target_fence(board)
                if fence_dist_after < cur_dist:
                    key[0] -= 0.3
            except Exception:
                pass
        candidates.append((tuple(key), move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        # Save top-3 diagnostics
        try:
            top = []
            for k, mv in candidates[:3]:
                top.append({
                    "uci": mv.uci(),
                    "key": list(k),
                    "ms_after": box_min_side_after(board, mv),
                    "area_after": box_area_after(board, mv),
                    "fence_dist_after": rook_distance_to_target_fence_after(board, mv),
                    "rook_safe_after": rook_safe_after(board, mv),
                })
            dbg["top3"] = top
            if env is not None:
                env["debug_phase1"] = dbg
        except Exception:
            pass
        return candidates[0][1].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase2(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Phase 2: Shrink the box.
    Enhanced with strict monotonicity requirements.
    """
    # Mate-in-one override
    mate = _find_mate_in_one(board)
    if mate:
        return mate
    legal_moves = list(board.legal_moves)
    candidates: List[Tuple[Tuple, chess.Move]] = []
    dbg: Dict[str, Any] = {}
    try:
        dbg["edge_info"] = enemy_nearest_edge_info(board)
        dbg["box_min_side"] = box_min_side(board)
        dbg["env_flags"] = {
            "require_min_side_shrink": (env or {}).get("require_min_side_shrink"),
            "forbid_zero_progress": (env or {}).get("forbid_zero_progress"),
        }
    except Exception:
        pass

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        # Strict P2: require min-side shrink; equality allowed only with king progress and mobility drop
        old_ms = box_min_side(board)
        new_ms = box_min_side_after(board, move)
        if new_ms > old_ms:
            continue
        if new_ms == old_ms:
            # Allow equality if we land the rook on the target fence (one inside the nearest edge) and it stays safe
            if not rook_on_target_fence_after(board, move):
                try:
                    bcur = board
                    cur_cnt = 0
                    enemy = not bcur.turn
                    for mv2 in bcur.legal_moves:
                        p = bcur.piece_at(mv2.from_square)
                        if p and p.piece_type == chess.KING and p.color == enemy:
                            cur_cnt += 1
                    reduces_mob = enemy_king_mobility_after(board, move) < cur_cnt
                except Exception:
                    reduces_mob = False
                if not (our_king_progress(board, move) > 0 and reduces_mob):
                    continue
        # Threefold guard
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            b = board.copy(stack=False)
            b.push(move)
            if not b.is_checkmate():
                continue
        # 50-move / pressure: forbid zero-progress choices when flagged
        if env is not None and env.get("forbid_zero_progress"):
            base = box_area(board)
            if (box_area_after(board, move) == base and
                box_min_side_after(board, move) == box_min_side(board) and
                our_king_progress(board, move) <= 0 and
                not gives_safe_check(board, move)):
                continue
        if env is not None and env.get("require_min_side_shrink"):
            if box_min_side_after(board, move) >= box_min_side(board):
                continue
        # add fence preference to key
        key2 = list(_candidate_key(board, move, 2, env))
        try:
            if rook_on_target_fence_after(board, move):
                key2[0] -= 1.2
            else:
                cur_fd = rook_distance_to_target_fence_after(board, move)  # reuse below for debug
                # prefer getting closer to fence slightly (smaller distance)
                # we don't know current distance cheaply; small static bias
                key2[0] -= 0.1 * max(0, 3 - cur_fd)
        except Exception:
            pass
        candidates.append((tuple(key2), move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        try:
            top = []
            for k, mv in candidates[:3]:
                top.append({
                    "uci": mv.uci(),
                    "key": list(k),
                    "ms_after": box_min_side_after(board, mv),
                    "area_after": box_area_after(board, mv),
                    "fence_dist_after": rook_distance_to_target_fence_after(board, mv),
                    "rook_safe_after": rook_safe_after(board, mv),
                })
            dbg["top3"] = top
            if env is not None:
                env["debug_phase2"] = dbg
        except Exception:
            pass
        return candidates[0][1].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase3(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Phase 3: Take opposition.
    Enhanced with opposition detection and P4 promotion.
    """
    # Mate-in-one override
    mate = _find_mate_in_one(board)
    if mate:
        return mate
    legal_moves = list(board.legal_moves)
    candidates: List[Tuple[Tuple, chess.Move]] = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        # Strict P3: if enemy on rim, require opposition; otherwise require strict mobility drop or safe check
        enemy_sq = board.king(not board.turn)
        on_rim_now = False
        try:
            file = chess.square_file(enemy_sq); rank = chess.square_rank(enemy_sq)
            on_rim_now = (file in (0,7) or rank in (0,7))
        except Exception:
            pass
        if on_rim_now:
            if not has_opposition_after(board, move):
                continue
        else:
            try:
                bcur = board
                cur_cnt = 0
                enemy = not bcur.turn
                for mv2 in bcur.legal_moves:
                    p = bcur.piece_at(mv2.from_square)
                    if p and p.piece_type == chess.KING and p.color == enemy:
                        cur_cnt += 1
                reduce_mob = enemy_king_mobility_after(board, move) < cur_cnt
            except Exception:
                reduce_mob = False
            if not (reduce_mob or gives_safe_check(board, move)):
                continue
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            b = board.copy(stack=False)
            b.push(move)
            if not b.is_checkmate():
                continue
        if env is not None and env.get("forbid_zero_progress"):
            base = box_area(board)
            if (box_area_after(board, move) == base and
                box_min_side_after(board, move) == box_min_side(board) and
                our_king_progress(board, move) <= 0 and
                not gives_safe_check(board, move)):
                continue
        if env is not None and env.get("require_min_side_shrink"):
            if box_min_side_after(board, move) >= box_min_side(board):
                continue
        candidates.append((_candidate_key(board, move, 3, env), move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)


def choose_move_phase4(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Phase 4: Deliver mate.
    Enhanced with mate detection.
    """
    legal_moves = list(board.legal_moves)
    candidates: List[Tuple[Tuple, chess.Move]] = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue
        # Threefold guard — irrelevant if mate
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            btmp = board.copy(stack=False)
            btmp.push(move)
            if not btmp.is_checkmate():
                continue

        # Mate bonus by skewing the key strongly to top
        b = board.copy(stack=False)
        b.push(move)
        key = _candidate_key(board, move, 4, env)
        if b.is_checkmate():
            key = (key[0] - 5.0, *key[1:])  # strong boost (since key[0] is -score)
        candidates.append((key, move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1].uci()

    # Fallback: any safe move
    return choose_any_safe_move(board)
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    base = box_area(board)
    best_mv, best_score = None, float("-inf")

    for mv in legal:
        if is_stalemate_after(board, mv):
            continue
        if not rook_safe_after(board, mv):
            continue

        area_next = box_area_after(board, mv)

        # Phase-0 rule: allow equal area if we get king progress OR a safe check
        preserves = (area_next == base)
        if area_next > base:
            # expanding the box is only OK in Phase 0 if you implement a stricter 'creates_stable_cut'
            # which we skip here to stay compatible with your current predicates.
            continue

        kprog = our_king_progress(board, mv)
        safe_chk = gives_safe_check(board, mv)

        if (area_next < base) or (preserves and (kprog > 0 or safe_chk)):
            score = 0.0
            score += 3.0 * (base - area_next)     # prefer shrinking
            score += 1.5 * kprog                  # bring king closer
            if safe_chk:
                score += 0.5
            score -= 0.1 * _rook_distance_travel(mv)

            if score > best_score:
                best_score, best_mv = score, mv

    return best_mv.uci() if best_mv else None
    """



def choose_move_with_filters(board: chess.Board, phase: str = "general") -> Optional[str]:
    """
    Core move selection with filter-first, then score approach.

    Args:
        board: Current chess position
        phase: Phase context ("phase1", "phase2", etc.) for scoring weights

    Returns:
        Best move in UCI format, or None if no good moves
    """
    legal_moves = list(board.legal_moves)

    # PHASE 1: HARD FILTERS (must pass all)
    candidates = []
    base_area = box_area(board)

    for move in legal_moves:
        # Filter 1: No stalemate
        if is_stalemate_after(board, move):
            continue

        # Filter 2: Rook safety (critical!)
        if not rook_safe_after(board, move):
            continue

        # Filter 3: No box expansion (monotonic constraint)
        if not shrinks_or_preserves_box(board, move):
            continue

        # Passed all filters - add to candidates with area info
        area_next = box_area_after(board, move)
        candidates.append((move, area_next))

    if not candidates:
        return None

    # PHASE 2: SCORING (rank remaining moves)
    scored_moves = []

    for move, area_next in candidates:
        score = 0.0

        # Base box shrinking score (always positive factor)
        area_reduction = base_area - area_next
        score += area_reduction * 3.0

        # King progress score
        king_score = our_king_progress(board, move)
        score += king_score

        # Phase-specific bonuses
        if phase == "phase1":
            score += edge_driving_bonus(board, move) * 2.0
        elif phase == "phase2":
            score += area_reduction * 5.0
        elif phase == "phase3":
            if has_opposition_after(board, move):
                score += 1.0
            score += king_score * 1.5
        elif phase == "phase4":
            if gives_safe_check(board, move):
                score += 1.0

        # Global: bonus for forming a stable rook cut
        if creates_stable_cut(board, move):
            score += 1.5

        # Penalize rook drag to avoid shuffling
        score -= 0.2 * _rook_distance_travel(move)

        scored_moves.append((score, move))

    # Sort by score (highest first) and return best move
    scored_moves.sort(reverse=True, key=lambda x: x[0])

    if scored_moves:
        best_move = scored_moves[0][1]
        return best_move.uci()

    return None


def is_stalemate_after(board: chess.Board, move: chess.Move) -> bool:
    """Check if move leads to stalemate."""
    board_copy = board.copy()
    board_copy.push(move)
    return board_copy.is_stalemate()


def edge_driving_bonus(board: chess.Board, move: chess.Move) -> float:
    "Calculate bonus for moves that drive enemy king toward edge."
    enemy_king = board.king(not board.turn)
    if enemy_king is None:
        return 0.0

    # Current edge distance
    king_file = chess.square_file(enemy_king)
    king_rank = chess.square_rank(enemy_king)
    current_dist = min(king_file, 7 - king_file, king_rank, 7 - king_rank)

    # Edge distance after move
    board_copy = board.copy()
    board_copy.push(move)
    new_enemy_king = board_copy.king(not board_copy.turn)

    if new_enemy_king is None:
        return 0.0

    new_file = chess.square_file(new_enemy_king)
    new_rank = chess.square_rank(new_enemy_king)
    new_dist = min(new_file, 7 - new_file, new_rank, 7 - new_rank)

    # Bonus for reducing edge distance
    distance_reduction = current_dist - new_dist
    return max(0, distance_reduction) * 0.5


# Legacy function for backward compatibility
def choose_confinement_move(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Choose move that reduces confinement box min-side.
    Prioritizes barrier creation and confinement reduction.
    """
    # Mate-in-one override
    mate = _find_mate_in_one(board)
    if mate:
        return mate

    legal_moves = list(board.legal_moves)
    candidates: List[Tuple[Tuple, chess.Move]] = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue

        # Threefold guard
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            b = board.copy(stack=False)
            b.push(move)
            if not b.is_checkmate():
                continue

        # Confinement-focused scoring
        key = _candidate_key(board, move, 1, env)  # Use phase 1 scoring as base

        # Extra boost for moves that create barriers or reduce confinement
        from .predicates import rook_on_target_fence_after, box_min_side_after
        if rook_on_target_fence_after(board, move):
            key = (key[0] - 2.0, *key[1:])  # Strong boost for barrier creation

        min_side_before = box_min_side(board)
        min_side_after = box_min_side_after(board, move)
        if min_side_after < min_side_before:
            # Boost proportional to confinement reduction
            confinement_boost = (min_side_before - min_side_after) * 1.5
            key = (key[0] - confinement_boost, *key[1:])

        candidates.append((key, move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1].uci()

    # Fallback
    return choose_any_safe_move(board)


def choose_barrier_move(board: chess.Board, env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Choose move that positions rook for barrier creation.
    Focuses on getting rook to target fence line.
    """
    # Mate-in-one override
    mate = _find_mate_in_one(board)
    if mate:
        return mate

    legal_moves = list(board.legal_moves)
    candidates: List[Tuple[Tuple, chess.Move]] = []

    for move in legal_moves:
        if is_stalemate_after(board, move):
            continue
        if not rook_safe_after(board, move):
            continue

        # Threefold guard
        if env is not None and would_cause_threefold(board, move, env.get("fen_history")):
            b = board.copy(stack=False)
            b.push(move)
            if not b.is_checkmate():
                continue

        key = _candidate_key(board, move, 1, env)  # Base scoring

        # Boost for barrier-related moves
        from .predicates import rook_distance_to_target_fence_after, rook_on_target_fence_after

        fence_dist_after = rook_distance_to_target_fence_after(board, move)
        fence_dist_before = rook_distance_to_target_fence(board)

        if rook_on_target_fence_after(board, move):
            key = (key[0] - 3.0, *key[1:])  # Massive boost for landing on fence
        elif fence_dist_after < fence_dist_before:
            # Boost for getting closer to fence
            distance_improvement = fence_dist_before - fence_dist_after
            key = (key[0] - distance_improvement * 0.8, *key[1:])

        candidates.append((key, move))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1].uci()

    # Fallback
    return choose_any_safe_move(board)


def choose_move_phase(board: chess.Board) -> Optional[str]:
    """Legacy function - use phase-specific functions instead."""
    return choose_move_with_filters(board)

 