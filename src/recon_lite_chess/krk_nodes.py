# src/recon_lite_chess/krk_nodes.py
"""
KRK (King+Rook vs King) chess-specific node implementations for ReCoN networks.

Strategy phases:
  PHASE0: establish first cut (box) and rendezvous our king with the rook
  PHASE1: drive enemy king to the edge
  PHASE2: shrink the box
  PHASE3: take opposition
  PHASE4: deliver mate
"""

from typing import Tuple, Any, Dict, List
from dataclasses import dataclass
import chess

from recon_lite.graph import Node, NodeType, NodeState, LinkType  # LinkType for wire helper
from .predicates import box_area, move_features
from .predicates import has_stable_cut


# ===== TERMINAL NODES (Leaf Operations) =====

@dataclass
class WaitForBoardChange(Node):
    """
    Terminal that waits until the board position changes.
    - On first evaluation, caches current FEN and returns TRUE immediately to allow the opening move.
    - On subsequent ticks, returns TRUE only when the FEN changes; otherwise keeps WAITING.
    """
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._wait_predicate)

    def _wait_predicate(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return True, False
        cur_fen = board.fen()
        last_fen = node.meta.get("last_fen")

        # First-time arming: allow pipeline to proceed, while caching baseline
        if last_fen is None:
            node.meta["last_fen"] = cur_fen
            # Optional strict mode: do not confirm immediately on first arm
            allow_initial_true = node.meta.get("allow_initial_true", True)
            if allow_initial_true:
                return True, True
            else:
                return False, False

        if cur_fen != last_fen:
            node.meta["last_fen"] = cur_fen
            return True, True

        return False, False

@dataclass
class KingAtEdgeDetector(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._king_at_edge)

    def _king_at_edge(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return False, False
        enemy_king = board.king(not board.turn)
        f, r = chess.square_file(enemy_king), chess.square_rank(enemy_king)
        at_edge = f in (0, 7) or r in (0, 7)
        if at_edge:
            return True, True
        return False, False


@dataclass
class BoxShrinkEvaluator(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._can_shrink_box)

    def _can_shrink_box(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return False, False

        enemy_king = board.king(not board.turn)
        our_king = board.king(board.turn)

        try:
            from .predicates import box_min_side
            if box_min_side(board) <= 1:
                return True, True
        except Exception:
            pass

        our_rook = None
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == board.turn and p.piece_type == chess.ROOK:
                our_rook = sq
                break
        if our_rook is None:
            return False, False

        kd = chess.square_distance(our_king, enemy_king)
        rd = chess.square_distance(our_rook, enemy_king)
        can_shrink = kd >= 2 and rd <= 3
        if can_shrink:
            return True, True
        return False, False


@dataclass
class ConfinementEvaluator(Node):
    """Evaluates if enemy king is in minimal confinement (box_min_side <= target)."""
    def __init__(self, nid: str, target_size: int = 2):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._king_confined)
        self.target_size = target_size

    def _king_confined(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        from .predicates import box_min_side
        board = env.get("board")
        if board is None:
            return False, False

        current_min_side = box_min_side(board)
        confined = current_min_side <= self.target_size
        return confined, confined


@dataclass
class BarrierReadyEvaluator(Node):
    """Evaluates if rook is positioned to create confinement barrier."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._barrier_ready)

    def _barrier_ready(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        from .predicates import enemy_nearest_edge_info, rook_distance_to_target_fence
        board = env.get("board")
        if board is None:
            return False, False

        # Rook is on or adjacent to target fence line
        distance = rook_distance_to_target_fence(board)
        ready = distance <= 1  # On fence (0) or one away (1)
        return ready, ready


@dataclass
class OppositionEvaluator(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._has_opposition)

    def _has_opposition(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return False, False
        ok = board.king(board.turn)
        ek = board.king(not board.turn)
        same_file = chess.square_file(ok) == chess.square_file(ek)
        same_rank = chess.square_rank(ok) == chess.square_rank(ek)
        enemy_at_edge = (chess.square_file(ek) in (0, 7)) or (chess.square_rank(ek) in (0, 7))
        cond = enemy_at_edge and (same_file or same_rank)
        if cond:
            return True, True
        return False, False


@dataclass
class MateDeliverEvaluator(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._can_deliver_mate)

    def _can_deliver_mate(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return False, False
        for mv in board.legal_moves:
            board.push(mv)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return True, True
        return False, False


@dataclass
class StalemateDetector(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._detect_stalemate)

    def _detect_stalemate(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return False, False
        if board.is_stalemate():
            return True, True
        return False, False


@dataclass
class CutEstablishedDetector(Node):
    """
    Terminal that detects whether a safe rook cut (fence) is already established.
    Uses predicates.has_stable_cut(board).
    """
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._cut_established)

    def _cut_established(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return False, False
        ok = has_stable_cut(board)
        return (True, True) if ok else (False, False)


@dataclass
class RookLostDetector(Node):
    """
    Terminal that confirms TRUE iff our rook is absent; otherwise confirms FAILED immediately.
    Designed for root-level sentinel wiring without blocking confirmation.
    """
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._rook_lost)

    def _rook_lost(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        board = env.get("board")
        if board is None:
            return True, False
        lost = not any(p.piece_type == chess.ROOK and p.color == board.turn for p in board.piece_map().values())
        # Expose sentinel flag for outer loop handling
        if lost:
            try:
                env["rook_lost"] = True
            except Exception:
                pass
        # Always resolve (done=True) to avoid blocking parent confirmation
        return True, bool(lost)


# ===== SCRIPT NODES =====

@dataclass
class Phase0EstablishCut(Node):
    """Script node for PHASE0: establish a first cut/box and rendezvous our king."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT)


@dataclass
class Phase1DriveToEdge(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT)


@dataclass
class Phase2ShrinkBox(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT)


@dataclass
class Phase3TakeOpposition(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT)


@dataclass
class Phase4DeliverMate(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT)


@dataclass
class KRKCheckmateRoot(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT)


# ===== MOVE-GENERATING TERMINALS (ACTUATORS) =====

@dataclass
class Phase0ChooseMoves(Node):
    """Leg/actuator that chooses a Phase-0 move and writes env['chosen_move'].
    
    Uses SCRIPT type to allow POR edges for leg sequencing.
    """
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT, predicate=self._choose)

    def _choose(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase0, choose_any_safe_move
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase0(board, env)
        if mv is None:
            reason = "P0 rendezvous satisfied (no move required)"
            node.meta["reason"] = reason
            env["last_reason"] = reason
            return True, True
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase0"
            try:
                mobj = chess.Move.from_uci(mv)
                feats = move_features(board, mobj)
                reason = f"P0 rendezvous: box {feats['box_area_before']}→{feats['box_area_after']}, +{feats['king_progress']} king, rook_safe={feats['rook_safe_after']}, safe_check={feats['gives_safe_check']}"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


@dataclass
class KingDriveMoves(Node):
    """Leg/actuator for king drive moves. Uses SCRIPT type for POR edge support."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase1
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase1(board, env)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase1"
            try:
                mobj = chess.Move.from_uci(mv)
                feats = move_features(board, mobj)
                reason = f"P1 drive: box {feats['box_area_before']}→{feats['box_area_after']}, +{feats['king_progress']} king, rook_safe={feats['rook_safe_after']}"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


@dataclass
class BoxShrinkMoves(Node):
    """Leg/actuator for box shrink moves. Uses SCRIPT type for POR edge support."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.SCRIPT, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase2
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase2(board, env)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase2"
            try:
                mobj = chess.Move.from_uci(mv)
                feats = move_features(board, mobj)
                reason = f"P2 shrink: box {feats['box_area_before']}→{feats['box_area_after']}, +{feats['king_progress']} king, rook_safe={feats['rook_safe_after']}"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


@dataclass
class OppositionMoves(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase3
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase3(board, env)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase3"
            try:
                mobj = chess.Move.from_uci(mv)
                feats = move_features(board, mobj)
                reason = f"P3 opposition: box {feats['box_area_before']}→{feats['box_area_after']}, +{feats['king_progress']} king, rook_safe={feats['rook_safe_after']}"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


@dataclass
class MateMoves(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase4
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase4(board, env)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase4"
            try:
                mobj = chess.Move.from_uci(mv)
                feats = move_features(board, mobj)
                reason = f"P4 mate: box {feats['box_area_before']}→{feats['box_area_after']}, +{feats['king_progress']} king, rook_safe={feats['rook_safe_after']}"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


@dataclass
class ConfinementMoves(Node):
    """Moves that prioritize reducing confinement box min-side."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_confinement_move
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_confinement_move(board, env)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase1"
            try:
                mobj = chess.Move.from_uci(mv)
                feats = move_features(board, mobj)
                reason = f"Confinement: min_side {feats.get('min_side_before', '?')}→{feats.get('min_side_after', '?')}, barrier creation"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


@dataclass
class BarrierPlacementMoves(Node):
    """Moves that position rook to create confinement barriers."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_barrier_move
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_barrier_move(board, env)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase1"
            try:
                mobj = chess.Move.from_uci(mv)
                from .predicates import rook_distance_to_target_fence_after
                fence_dist_before = rook_distance_to_target_fence(board)
                fence_dist_after = rook_distance_to_target_fence_after(board, mobj)
                reason = f"Barrier: fence_dist {fence_dist_before}→{fence_dist_after}"
                node.meta["reason"] = reason
                env["last_reason"] = reason
            except Exception:
                pass
            return True, [mv]
        return False, []


# ===== SUPERVISOR: NoProgressWatch =====

@dataclass
class NoProgressWatch(Node):
    """
    Tracks last N box areas. If no strict improvement in 6 plies, set env["pressure"] for a few plies.
    Also maintains env["fen_history"] for repetition checks.
    """
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._tick)

    def _tick(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        from collections import deque
        board = env.get("board")
        if board is None:
            return True, True

        # Init fen_history
        fen_hist = env.get("fen_history")
        if fen_hist is None:
            fen_hist = deque(maxlen=12)
            env["fen_history"] = fen_hist
        # Append normalized FEN-with-turn
        try:
            fen_key = board.board_fen() + " " + ("w" if board.turn else "b")
        except Exception:
            fen_key = board.fen()
        # Avoid duplicate spam if same as last
        if not fen_hist or fen_hist[-1] != fen_key:
            fen_hist.append(fen_key)

        # Track recent box areas
        areas = node.meta.get("recent_areas")
        if areas is None:
            areas = deque(maxlen=8)
            node.meta["recent_areas"] = areas
        areas.append(box_area(board))

        # Track recent mobility proxy: number of enemy king moves
        try:
            bcur = board
            cur_cnt = 0
            enemy = not bcur.turn
            for mv2 in bcur.legal_moves:
                p = bcur.piece_at(mv2.from_square)
                if p and p.piece_type == chess.KING and p.color == enemy:
                    cur_cnt += 1
        except Exception:
            cur_cnt = 0
        mobs = node.meta.get("recent_mobility")
        if mobs is None:
            mobs = deque(maxlen=8)
            node.meta["recent_mobility"] = mobs
        mobs.append(cur_cnt)

        # Pressure window management: trigger if no min-side or mobility improvement in 6 plies
        steps = int(env.get("pressure_steps", 0))
        # Trigger if last 6 entries show no strict decrease in area AND no decrease in mobility
        if len(areas) >= 6:
            window = list(areas)[-6:]
            mobs_w = list(mobs)[-6:]
            a_improved = any(window[i] > window[i+1] for i in range(len(window)-1))
            m_improved = any(mobs_w[i] > mobs_w[i+1] for i in range(len(mobs_w)-1))
            if not (a_improved or m_improved) and steps <= 0:
                steps = 6
        steps = max(0, steps - 1) if steps > 0 else steps
        env["pressure_steps"] = steps
        env["pressure"] = steps > 0
        # Require min-side shrink next ply if pressure is active and last two plies had no improvement
        require_shrink = False
        if len(areas) >= 3 and len(mobs) >= 3 and steps > 0:
            a = list(areas)
            m = list(mobs)
            a_impr = (a[-3] > a[-2]) or (a[-2] > a[-1])
            m_impr = (m[-3] > m[-2]) or (m[-2] > m[-1])
            if not (a_impr or m_impr):
                require_shrink = True
        env["require_min_side_shrink"] = require_shrink

        # Forbid zero-progress choices near 50-move: set a flag others can read
        hm = getattr(board, "halfmove_clock", 0)
        env["forbid_zero_progress"] = (hm >= 48)

        # This node is a side observer; do not gate
        return True, True


@dataclass
class RandomLegalMoves(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        board = env.get("board")
        if not board:
            return False, []
        legal = list(board.legal_moves)
        ucis = [m.uci() for m in legal]
        if ucis and env.get("chosen_move") is None:
            import random
            chosen = random.choice(ucis)
            env["chosen_move"] = chosen
            node.meta["suggested_moves"] = ucis
        return bool(ucis), ucis


# ===== FACTORY FUNCTIONS =====

def create_king_edge_detector(nid: str) -> KingAtEdgeDetector: return KingAtEdgeDetector(nid)
def create_box_shrink_evaluator(nid: str) -> BoxShrinkEvaluator: return BoxShrinkEvaluator(nid)
def create_opposition_evaluator(nid: str) -> OppositionEvaluator: return OppositionEvaluator(nid)
def create_mate_deliver_evaluator(nid: str) -> MateDeliverEvaluator: return MateDeliverEvaluator(nid)
def create_stalemate_detector(nid: str) -> StalemateDetector: return StalemateDetector(nid)
def create_cut_established_detector(nid: str) -> CutEstablishedDetector: return CutEstablishedDetector(nid)
def create_rook_lost_detector(nid: str) -> RookLostDetector: return RookLostDetector(nid)
def create_wait_for_board_change(nid: str) -> WaitForBoardChange: return WaitForBoardChange(nid)

def create_phase0_establish_cut(nid: str) -> Phase0EstablishCut: return Phase0EstablishCut(nid)
def create_phase1_drive_to_edge(nid: str) -> Phase1DriveToEdge: return Phase1DriveToEdge(nid)
def create_phase2_shrink_box(nid: str) -> Phase2ShrinkBox: return Phase2ShrinkBox(nid)
def create_phase3_take_opposition(nid: str) -> Phase3TakeOpposition: return Phase3TakeOpposition(nid)
def create_phase4_deliver_mate(nid: str) -> Phase4DeliverMate: return Phase4DeliverMate(nid)

def create_phase0_choose_moves(nid: str) -> Phase0ChooseMoves: return Phase0ChooseMoves(nid)
def create_king_drive_moves(nid: str) -> KingDriveMoves: return KingDriveMoves(nid)
def create_box_shrink_moves(nid: str) -> BoxShrinkMoves: return BoxShrinkMoves(nid)
def create_opposition_moves(nid: str) -> OppositionMoves: return OppositionMoves(nid)
def create_mate_moves(nid: str) -> MateMoves: return MateMoves(nid)
def create_random_legal_moves(nid: str) -> RandomLegalMoves: return RandomLegalMoves(nid)
def create_no_progress_watch(nid: str) -> NoProgressWatch: return NoProgressWatch(nid)

# New confinement-aware nodes
def create_confinement_evaluator(nid: str, target_size: int = 2) -> ConfinementEvaluator:
    return ConfinementEvaluator(nid, target_size)

def create_barrier_ready_evaluator(nid: str) -> BarrierReadyEvaluator:
    return BarrierReadyEvaluator(nid)

def create_confinement_moves(nid: str) -> ConfinementMoves:
    return ConfinementMoves(nid)

def create_barrier_placement_moves(nid: str) -> BarrierPlacementMoves:
    return BarrierPlacementMoves(nid)

def create_krk_root(nid: str) -> KRKCheckmateRoot: return KRKCheckmateRoot(nid)


# ===== WIRING HELPER =====

def wire_default_krk(g, root_id: str, ids: Dict[str, str]) -> None:
    """
    Convenience wiring:
      SUB: root -> P0,P1,P2,P3,P4 and P0 -> CHOOSE_P0
      POR: P0 -> P1 -> P2 -> P3 -> P4
    ids = {
      "root": "ROOT",
      "phase0": "PHASE0", "choose_p0": "CHOOSE_P0",
      "phase1": "PHASE1", "phase2": "PHASE2", "phase3": "PHASE3", "phase4": "PHASE4",
    }
    """
    g.add_edge(ids["root"], ids["phase0"], LinkType.SUB)
    g.add_edge(ids["root"], ids["phase1"], LinkType.SUB)
    g.add_edge(ids["root"], ids["phase2"], LinkType.SUB)
    g.add_edge(ids["root"], ids["phase3"], LinkType.SUB)
    g.add_edge(ids["root"], ids["phase4"], LinkType.SUB)
    # Add supervisor without POR edges
    if "watch" in ids:
        g.add_edge(ids["root"], ids["watch"], LinkType.SUB)

    g.add_edge(ids["phase0"], ids["choose_p0"], LinkType.SUB)

    g.add_edge(ids["phase0"], ids["phase1"], LinkType.POR)
    g.add_edge(ids["phase1"], ids["phase2"], LinkType.POR)
    g.add_edge(ids["phase2"], ids["phase3"], LinkType.POR)
    g.add_edge(ids["phase3"], ids["phase4"], LinkType.POR)
