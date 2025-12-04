"""Phase sensor terminal for M6 goal hierarchy.

This sensor provides game phase estimation that can be queried
by multiple strategic plans (fan-in terminal). Phase is "soft" -
it outputs continuous weights rather than discrete gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import chess

from .material import count_material


@dataclass
class PhaseWeights:
    """Soft phase weights (sum to ~1.0)."""
    opening: float
    middlegame: float
    endgame: float
    
    def dominant_phase(self) -> str:
        """Return the phase with highest weight."""
        phases = {"opening": self.opening, "middlegame": self.middlegame, "endgame": self.endgame}
        return max(phases.keys(), key=lambda k: phases[k])
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "opening": self.opening,
            "middlegame": self.middlegame,
            "endgame": self.endgame,
        }


def estimate_phase(board: chess.Board) -> PhaseWeights:
    """
    Estimate game phase as soft weights.
    
    Factors considered:
    - Material count (endgame indicator)
    - Queen presence (middlegame/endgame transition)
    - Developed pieces (opening indicator)
    - Move count / fullmove number
    - King position (central = opening, active = endgame)
    - Castling rights (opening indicator)
    """
    white_mat, white_counts = count_material(board, chess.WHITE)
    black_mat, black_counts = count_material(board, chess.BLACK)
    total_material = white_mat + black_mat
    
    # Base weights from material
    # Full material ~78 (Q=9, 2R=10, 2B=6.5, 2N=6, 8P=8, x2 = 79 per side, minus kings)
    # Actually: 9 + 10 + 6.5 + 6 + 8 = 39.5 per side, 79 total
    if total_material >= 60:
        mat_opening = 0.6
        mat_middle = 0.35
        mat_endgame = 0.05
    elif total_material >= 40:
        mat_opening = 0.2
        mat_middle = 0.6
        mat_endgame = 0.2
    elif total_material >= 20:
        mat_opening = 0.0
        mat_middle = 0.3
        mat_endgame = 0.7
    else:
        mat_opening = 0.0
        mat_middle = 0.1
        mat_endgame = 0.9
    
    # Queen factor
    has_white_queen = white_counts.get(chess.QUEEN, 0) > 0
    has_black_queen = black_counts.get(chess.QUEEN, 0) > 0
    if not has_white_queen and not has_black_queen:
        queen_endgame_boost = 0.2
    elif not has_white_queen or not has_black_queen:
        queen_endgame_boost = 0.1
    else:
        queen_endgame_boost = 0.0
    
    # Development factor (opening indicator)
    # Check if minor pieces are on starting squares
    undeveloped = 0
    starting_minors = [
        (chess.B1, chess.KNIGHT), (chess.G1, chess.KNIGHT),
        (chess.C1, chess.BISHOP), (chess.F1, chess.BISHOP),
        (chess.B8, chess.KNIGHT), (chess.G8, chess.KNIGHT),
        (chess.C8, chess.BISHOP), (chess.F8, chess.BISHOP),
    ]
    for sq, pt in starting_minors:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == pt:
            undeveloped += 1
    
    # High undeveloped = still opening
    dev_opening_boost = min(0.3, undeveloped * 0.05)
    
    # Castling rights factor
    castling_bonus = 0
    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        castling_bonus += 0.1
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        castling_bonus += 0.1
    
    # Move count factor
    fullmove = board.fullmove_number
    if fullmove <= 10:
        move_opening = 0.4
        move_middle = 0.4
        move_endgame = 0.2
    elif fullmove <= 30:
        move_opening = 0.1
        move_middle = 0.6
        move_endgame = 0.3
    else:
        move_opening = 0.0
        move_middle = 0.3
        move_endgame = 0.7
    
    # Combine factors
    opening = mat_opening + dev_opening_boost + castling_bonus * 0.5 + move_opening * 0.3
    middlegame = mat_middle + move_middle * 0.3
    endgame = mat_endgame + queen_endgame_boost + move_endgame * 0.3
    
    # Normalize to sum to 1.0
    total = opening + middlegame + endgame
    if total > 0:
        opening /= total
        middlegame /= total
        endgame /= total
    else:
        opening = middlegame = endgame = 1/3
    
    return PhaseWeights(
        opening=round(opening, 3),
        middlegame=round(middlegame, 3),
        endgame=round(endgame, 3),
    )


# Terminal predicate for use in ReCoN graphs
def phase_sensor_predicate(node: Any, env: Dict[str, Any]) -> Tuple[bool, bool]:
    """
    Terminal predicate that estimates game phase.
    
    Stores phase weights in node.activation.meta and env.
    Returns (done=True, success=True) immediately since this is a sensor.
    """
    board = env.get("board")
    if board is None:
        return True, False
    
    phase = estimate_phase(board)
    
    # Store in node activation meta
    node.activation.meta["phase"] = phase.as_dict()
    node.activation.meta["dominant"] = phase.dominant_phase()
    
    # Also store in env for other nodes
    env["phase_weights"] = phase.as_dict()
    env["dominant_phase"] = phase.dominant_phase()
    
    # Activation value encodes phase as a continuous value
    # 0.0 = opening, 0.5 = middlegame, 1.0 = endgame
    node.activation.value = phase.middlegame * 0.5 + phase.endgame * 1.0
    
    return True, True


def create_phase_sensor_node():
    """Factory to create a phase sensor terminal node."""
    from recon_lite.graph import Node, NodeType
    
    return Node(
        nid="PhaseSensor",
        ntype=NodeType.TERMINAL,
        predicate=phase_sensor_predicate,
        meta={"sensor_type": "phase", "fan_in_allowed": True},
    )

