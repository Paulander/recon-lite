"""
Goal-aware modulation of plasticity and exploration parameters.

This module derives risk and urgency scalars from the goal_vector and uses
them to modulate the effective learning rate and exploration coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModulationConfig:
    """
    Configuration for goal-based modulation.

    Attributes:
        alpha_risk: Scaling factor for risk modulation of learning rate
        alpha_urgency: Scaling factor for urgency modulation of exploration
        eta_base: Base learning rate (before modulation)
        c_explore_base: Base exploration coefficient (before modulation)
    """

    alpha_risk: float = 0.5
    alpha_urgency: float = 0.5
    eta_base: float = 0.05
    c_explore_base: float = 1.0


@dataclass
class Modulators:
    """
    Computed modulation values for a given goal_vector.

    Attributes:
        risk: Risk level in [0, 1]
        urgency: Urgency level in [0, 1]
        eta_tick_eff: Effective learning rate
        c_explore_eff: Effective exploration coefficient
    """

    risk: float
    urgency: float
    eta_tick_eff: float
    c_explore_eff: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "risk": round(self.risk, 4),
            "urgency": round(self.urgency, 4),
            "eta_tick_eff": round(self.eta_tick_eff, 4),
            "c_explore_eff": round(self.c_explore_eff, 4),
        }


def _extract_risk(goal_vector: Dict[str, Any]) -> float:
    """
    Derive risk scalar from goal_vector components.

    Risk is high when:
    - Material is unfavorable
    - King safety is poor
    - Defense pressure is high

    Returns value in [0, 1].
    """
    risk = 0.0

    # Material component (negative material = higher risk)
    material = goal_vector.get("material", 0.0)
    if isinstance(material, (int, float)):
        # Normalize: -5 pawns = max risk, +5 pawns = min risk
        material_risk = max(0.0, min(1.0, (5.0 - float(material)) / 10.0))
        risk += material_risk * 0.4

    # Defense pressure component
    defense_pressure = goal_vector.get("defense_pressure", 0.0)
    if isinstance(defense_pressure, (int, float)):
        risk += float(defense_pressure) * 0.3

    # King safety component (inverted: low safety = high risk)
    king_safety = goal_vector.get("king_safety", 1.0)
    if isinstance(king_safety, (int, float)):
        risk += (1.0 - float(king_safety)) * 0.3

    # Clamp to [0, 1]
    return max(0.0, min(1.0, risk))


def _extract_urgency(goal_vector: Dict[str, Any]) -> float:
    """
    Derive urgency scalar from goal_vector components.

    Urgency is high when:
    - We need to make progress (phase_progress is low)
    - The box is large (in KRK)
    - Attack pressure is high

    Returns value in [0, 1].
    """
    urgency = 0.0

    # Phase progress component (low progress = high urgency)
    phase_progress = goal_vector.get("phase_progress", 0.5)
    if isinstance(phase_progress, (int, float)):
        urgency += (1.0 - float(phase_progress)) * 0.4

    # Box size component (larger box = higher urgency in KRK)
    box_area = goal_vector.get("box_area", 0.0)
    if isinstance(box_area, (int, float)):
        # Normalize: 64 squares = max urgency, 4 squares = min
        box_urgency = max(0.0, min(1.0, (float(box_area) - 4.0) / 60.0))
        urgency += box_urgency * 0.3

    # Attack pressure component
    attack_pressure = goal_vector.get("attack_pressure", 0.0)
    if isinstance(attack_pressure, (int, float)):
        urgency += float(attack_pressure) * 0.3

    # Clamp to [0, 1]
    return max(0.0, min(1.0, urgency))


def compute_modulators(
    goal_vector: Optional[Dict[str, Any]],
    config: Optional[ModulationConfig] = None,
) -> Modulators:
    """
    Compute modulation values from goal_vector.

    Args:
        goal_vector: Dict with goal components (material, king_safety, etc.)
        config: Modulation configuration (uses defaults if None)

    Returns:
        Modulators with risk, urgency, and effective parameters
    """
    if config is None:
        config = ModulationConfig()

    if goal_vector is None:
        goal_vector = {}

    # Extract risk and urgency
    risk = _extract_risk(goal_vector)
    urgency = _extract_urgency(goal_vector)

    # Compute effective parameters
    # eta_tick_eff = eta_base * (1 + alpha_risk * risk)
    eta_tick_eff = config.eta_base * (1.0 + config.alpha_risk * risk)

    # c_explore_eff = c_explore_base * (1 + alpha_urgency * urgency)
    c_explore_eff = config.c_explore_base * (1.0 + config.alpha_urgency * urgency)

    return Modulators(
        risk=risk,
        urgency=urgency,
        eta_tick_eff=eta_tick_eff,
        c_explore_eff=c_explore_eff,
    )


def compute_modulators_from_board(
    board: Any,
    config: Optional[ModulationConfig] = None,
) -> Modulators:
    """
    Compute modulators directly from a chess board (convenience function).

    This builds a goal_vector from board features and then computes modulators.

    Args:
        board: A chess.Board instance
        config: Modulation configuration

    Returns:
        Modulators with risk, urgency, and effective parameters
    """
    import chess

    if not isinstance(board, chess.Board):
        return compute_modulators(None, config)

    goal_vector: Dict[str, Any] = {}

    # Material score
    material = 0.0
    for square, piece in board.piece_map().items():
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        value = piece_values.get(piece.piece_type, 0.0)
        if piece.color == chess.WHITE:
            material += value
        else:
            material -= value
    goal_vector["material"] = material

    # King safety (simple: are there attackers near king?)
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is not None:
            attackers = len(board.attackers(not color, king_sq))
            safety = max(0.0, 1.0 - attackers * 0.3)
            if color == chess.WHITE:
                goal_vector["king_safety"] = safety

    # Box area for KRK
    enemy_king = board.king(chess.BLACK)
    if enemy_king is not None:
        file = chess.square_file(enemy_king)
        rank = chess.square_rank(enemy_king)
        # Rough box area: distance from edges
        box_width = min(file + 1, 8 - file)
        box_height = min(rank + 1, 8 - rank)
        goal_vector["box_area"] = box_width * box_height * 4  # Approximate

    # Phase progress (rough estimate based on piece count)
    total_pieces = len(board.piece_map())
    if total_pieces <= 3:
        goal_vector["phase_progress"] = 0.8  # Endgame, close to done
    elif total_pieces <= 10:
        goal_vector["phase_progress"] = 0.5
    else:
        goal_vector["phase_progress"] = 0.2

    return compute_modulators(goal_vector, config)

