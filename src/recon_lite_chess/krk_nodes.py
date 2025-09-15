# src/recon_lite_chess/krk_nodes.py
"""
KRK (King+Rook vs King) chess-specific node implementations for ReCoN networks.

These nodes implement the hierarchical KRK checkmate strategy:
ROOT: "KRK mate procedure"
  ├─ PHASE1: drive black king to edge
  ├─ PHASE2: shrink the box (keep rook safe)
  ├─ PHASE3: take opposition (king alignment)
  └─ PHASE4: deliver mate

Extensible for future scenarios like pawn promotion and computer vision.
"""

from typing import Tuple, Any, Dict, List
from dataclasses import dataclass
from recon_lite.graph import Node, NodeType, NodeState
import chess


# ===== TERMINAL NODES (Leaf Operations) =====

@dataclass
class KingAtEdgeDetector(Node):
    """
    Terminal node that checks if enemy king is on the edge of the board.
    Used in PHASE1 to determine if we need to drive the king to edge.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._king_at_edge
        )

    def _king_at_edge(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if enemy king is on board edge (files a/h or ranks 1/8)."""
        board = env.get("board")
        if board is None:
            return True, False

        enemy_king = board.king(not board.turn)  # Enemy king square
        file_idx = chess.square_file(enemy_king)
        rank_idx = chess.square_rank(enemy_king)

        # King is at edge if on file a/h or rank 1/8
        at_edge = file_idx in [0, 7] or rank_idx in [0, 7]
        return True, at_edge


@dataclass
class BoxShrinkEvaluator(Node):
    """
    Terminal node that evaluates if we can shrink the "box" (safe zone for enemy king).
    Checks if current king position allows box shrinking while keeping rook safe.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._can_shrink_box
        )

    def _can_shrink_box(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """Evaluate if box shrinking is possible and safe."""
        board = env.get("board")
        if board is None:
            return True, False

        # Get positions
        enemy_king = board.king(not board.turn)
        our_king = board.king(board.turn)
        our_rook = None

        # Find our rook
        for square in chess.SQUARES:
            if board.piece_at(square) and board.piece_at(square).piece_type == chess.ROOK:
                if board.piece_at(square).color == board.turn:
                    our_rook = square
                    break

        if our_rook is None:
            return True, False  # No rook found

        # Simple heuristic: can shrink if enemy king is not too close to our king
        # and rook is reasonably positioned
        king_distance = chess.square_distance(our_king, enemy_king)
        rook_distance = chess.square_distance(our_rook, enemy_king)

        # Can shrink if kings are separated and rook is attacking
        can_shrink = king_distance >= 2 and rook_distance <= 3
        return True, can_shrink


@dataclass
class OppositionEvaluator(Node):
    """
    Terminal node that checks if we can take opposition (proper king alignment).
    Evaluates if kings are properly positioned for the final attack.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._has_opposition
        )

    def _has_opposition(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if kings have opposition for mating attack."""
        board = env.get("board")
        if board is None:
            return True, False

        our_king = board.king(board.turn)
        enemy_king = board.king(not board.turn)

        # Simple opposition check: kings on same file/rank with enemy king confined
        same_file = chess.square_file(our_king) == chess.square_file(enemy_king)
        same_rank = chess.square_rank(our_king) == chess.square_rank(enemy_king)

        # Enemy king should be at edge for opposition to be effective
        enemy_at_edge = (chess.square_file(enemy_king) in [0, 7] or
                        chess.square_rank(enemy_king) in [0, 7])

        has_opposition = (same_file or same_rank) and enemy_at_edge
        return True, has_opposition


@dataclass
class MateDeliverEvaluator(Node):
    """
    Terminal node that evaluates if mate can be delivered from current position.
    Checks if king and rook are positioned for immediate checkmate.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._can_deliver_mate
        )

    def _can_deliver_mate(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if mate can be delivered from current position."""
        board = env.get("board")
        if board is None:
            return True, False

        # Simple check: look for moves that would result in checkmate
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return True, True
            board.pop()

        return True, False


@dataclass
class StalemateDetector(Node):
    """
    Terminal node that detects if position is stalemate (draw).
    Important to avoid in KRK endgames.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._detect_stalemate
        )

    def _detect_stalemate(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if current position is stalemate."""
        board = env.get("board")
        if board is None:
            return True, False

        return True, board.is_stalemate()


# ===== SCRIPT NODES (Hierarchical Strategy) =====

@dataclass
class Phase1DriveToEdge(Node):
    """
    Script node for PHASE1: Drive enemy king to the edge of the board.
    Contains sub-nodes for different edge-driving strategies.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.SCRIPT
        )


@dataclass
class Phase2ShrinkBox(Node):
    """
    Script node for PHASE2: Shrink the safe "box" for enemy king while keeping rook safe.
    Coordinates king and rook movement to reduce enemy king's mobility.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.SCRIPT
        )


@dataclass
class Phase3TakeOpposition(Node):
    """
    Script node for PHASE3: Take opposition and align kings properly.
    Positions kings so enemy king is forced into vulnerable positions.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.SCRIPT
        )


@dataclass
class Phase4DeliverMate(Node):
    """
    Script node for PHASE4: Deliver the actual checkmate.
    Final coordination between king and rook for the mating attack.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.SCRIPT
        )


@dataclass
class KRKCheckmateRoot(Node):
    """
    Root script node for the complete KRK checkmate procedure.
    Contains all phases as sub-nodes and manages the overall strategy.
    """
    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.SCRIPT
        )


# ===== MOVE GENERATING TERMINALS (ACTUATORS) =====

@dataclass
class KingDriveMoves(Node):
    """Terminal that generates moves to drive enemy king toward edge using improved filtering."""

    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._generate_king_drive_moves
        )

    def _generate_king_drive_moves(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Generate moves using improved actuator logic."""
        from .actuators import choose_move_phase1

        board = env.get("board")
        if not board:
            return False, []

        # Use improved actuator for Phase 1 moves
        best_move = choose_move_phase1(board)

        if best_move:
            env["chosen_move"] = best_move
            node.meta["suggested_moves"] = [best_move]
            node.meta["phase"] = "phase1"
            return True, [best_move]

        return False, []


@dataclass
class BoxShrinkMoves(Node):
    """Terminal that generates moves to shrink enemy king's box."""

    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._generate_box_shrink_moves
        )

    def _generate_box_shrink_moves(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Generate moves using improved actuator logic for Phase 2."""
        from .actuators import choose_move_phase2

        board = env.get("board")
        if not board:
            return False, []

        best_move = choose_move_phase2(board)

        if best_move:
            env["chosen_move"] = best_move
            node.meta["suggested_moves"] = [best_move]
            node.meta["phase"] = "phase2"
            return True, [best_move]

        return False, []


@dataclass
class OppositionMoves(Node):
    """Terminal that generates moves to achieve opposition."""

    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._generate_opposition_moves
        )

    def _generate_opposition_moves(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Generate moves using improved actuator logic for Phase 3."""
        from .actuators import choose_move_phase3

        board = env.get("board")
        if not board:
            return False, []

        best_move = choose_move_phase3(board)

        if best_move:
            env["chosen_move"] = best_move
            node.meta["suggested_moves"] = [best_move]
            node.meta["phase"] = "phase3"
            return True, [best_move]

        return False, []


@dataclass
class MateMoves(Node):
    """Terminal that generates checkmate moves."""

    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._generate_mate_moves
        )

    def _generate_mate_moves(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Generate moves using improved actuator logic for Phase 4."""
        from .actuators import choose_move_phase4

        board = env.get("board")
        if not board:
            return False, []

        best_move = choose_move_phase4(board)

        if best_move:
            env["chosen_move"] = best_move
            node.meta["suggested_moves"] = [best_move]
            node.meta["phase"] = "phase4"
            return True, [best_move]

        return False, []


@dataclass
class RandomLegalMoves(Node):
    """Terminal that generates all legal moves (fallback when no strategic moves)."""

    def __init__(self, nid: str):
        super().__init__(
            nid=nid,
            ntype=NodeType.TERMINAL,
            predicate=self._generate_legal_moves
        )

    def _generate_legal_moves(self, node: Node, env: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Generate all legal moves as fallback."""
        board = env.get("board")
        if not board:
            return False, []

        legal_moves = list(board.legal_moves)
        move_ucis = [move.uci() for move in legal_moves]

        # If we have legal moves and no chosen_move yet, pick a random one
        if move_ucis and env.get("chosen_move") is None:
            import random
            chosen = random.choice(move_ucis)
            env["chosen_move"] = chosen
            # Store all suggestions in node metadata for debugging
            node.meta["suggested_moves"] = move_ucis

        return bool(move_ucis), move_ucis


# ===== FACTORY FUNCTIONS =====

def create_king_edge_detector(nid: str) -> KingAtEdgeDetector:
    """Factory for king-at-edge detector nodes."""
    return KingAtEdgeDetector(nid)

def create_box_shrink_evaluator(nid: str) -> BoxShrinkEvaluator:
    """Factory for box shrink evaluator nodes."""
    return BoxShrinkEvaluator(nid)

def create_opposition_evaluator(nid: str) -> OppositionEvaluator:
    """Factory for opposition evaluator nodes."""
    return OppositionEvaluator(nid)

def create_mate_deliver_evaluator(nid: str) -> MateDeliverEvaluator:
    """Factory for mate delivery evaluator nodes."""
    return MateDeliverEvaluator(nid)

def create_stalemate_detector(nid: str) -> StalemateDetector:
    """Factory for stalemate detector nodes."""
    return StalemateDetector(nid)

# Phase nodes
def create_phase1_drive_to_edge(nid: str) -> Phase1DriveToEdge:
    """Factory for phase 1 (drive to edge) nodes."""
    return Phase1DriveToEdge(nid)

def create_phase2_shrink_box(nid: str) -> Phase2ShrinkBox:
    """Factory for phase 2 (shrink box) nodes."""
    return Phase2ShrinkBox(nid)

def create_phase3_take_opposition(nid: str) -> Phase3TakeOpposition:
    """Factory for phase 3 (take opposition) nodes."""
    return Phase3TakeOpposition(nid)

def create_phase4_deliver_mate(nid: str) -> Phase4DeliverMate:
    """Factory for phase 4 (deliver mate) nodes."""
    return Phase4DeliverMate(nid)

def create_king_drive_moves(nid: str) -> KingDriveMoves:
    """Factory for king drive move generator nodes."""
    return KingDriveMoves(nid)

def create_box_shrink_moves(nid: str) -> BoxShrinkMoves:
    """Factory for box shrink move generator nodes."""
    return BoxShrinkMoves(nid)

def create_opposition_moves(nid: str) -> OppositionMoves:
    """Factory for opposition move generator nodes."""
    return OppositionMoves(nid)

def create_mate_moves(nid: str) -> MateMoves:
    """Factory for mate move generator nodes."""
    return MateMoves(nid)

def create_random_legal_moves(nid: str) -> RandomLegalMoves:
    """Factory for random legal move generator nodes."""
    return RandomLegalMoves(nid)

def create_krk_root(nid: str) -> KRKCheckmateRoot:
    """Factory for KRK checkmate root nodes."""
    return KRKCheckmateRoot(nid)
