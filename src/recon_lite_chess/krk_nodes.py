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
            return True, True

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
    """Terminal that chooses a Phase-0 move and writes env['chosen_move']."""
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._choose)

    def _choose(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase0, choose_any_safe_move
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase0(board)
        if mv is None:
            mv = choose_any_safe_move(board)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase0"
            return True, [mv]
        return False, []


@dataclass
class KingDriveMoves(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase1
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase1(board)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase1"
            return True, [mv]
        return False, []


@dataclass
class BoxShrinkMoves(Node):
    def __init__(self, nid: str):
        super().__init__(nid=nid, ntype=NodeType.TERMINAL, predicate=self._gen)

    def _gen(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        from .actuators import choose_move_phase2
        board = env.get("board")
        if not board:
            return False, []
        mv = choose_move_phase2(board)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase2"
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
        mv = choose_move_phase3(board)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase3"
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
        mv = choose_move_phase4(board)
        if mv:
            env["chosen_move"] = mv
            node.meta["suggested_moves"] = [mv]
            node.meta["phase"] = "phase4"
            return True, [mv]
        return False, []


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

    g.add_edge(ids["phase0"], ids["choose_p0"], LinkType.SUB)

    g.add_edge(ids["phase0"], ids["phase1"], LinkType.POR)
    g.add_edge(ids["phase1"], ids["phase2"], LinkType.POR)
    g.add_edge(ids["phase2"], ids["phase3"], LinkType.POR)
    g.add_edge(ids["phase3"], ids["phase4"], LinkType.POR)
