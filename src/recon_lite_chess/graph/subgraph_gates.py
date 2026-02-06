"""
Subgraph gating logic for ReCoN chess.

Computes continuous affordance-based activation weights for subgraphs.

This module provides the "distance to applicability" signal for each subgraph,
enabling implicit lookahead where the planning layer can sense valid strategies
before they are fully executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import chess

# Import affordance sensors
from recon_lite_chess.affordance import (
    compute_all_affordances,
    compute_krk_affordance,
    compute_kpk_affordance,
    compute_kqk_affordance,
    AffordanceSignal,
    AffordanceConfig,
)


@dataclass
class MaterialInfo:
    """Material information from the board."""
    white_material: float
    black_material: float
    balance: float
    is_endgame: bool
    pattern: Optional[str] = None  # "KRK", "KPK", etc.
    white_pawns: int = 0
    white_knights: int = 0
    white_bishops: int = 0
    white_rooks: int = 0
    white_queens: int = 0
    black_pawns: int = 0
    black_knights: int = 0
    black_bishops: int = 0
    black_rooks: int = 0
    black_queens: int = 0


def analyze_material(board: chess.Board) -> MaterialInfo:
    """Analyze material on the board."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    
    white_material = 0.0
    black_material = 0.0
    counts = {
        "white_pawns": 0, "white_knights": 0, "white_bishops": 0,
        "white_rooks": 0, "white_queens": 0,
        "black_pawns": 0, "black_knights": 0, "black_bishops": 0,
        "black_rooks": 0, "black_queens": 0,
    }
    
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        
        value = piece_values.get(piece.piece_type, 0)
        color_prefix = "white" if piece.color == chess.WHITE else "black"
        
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
        
        # Count pieces
        piece_name = {
            chess.PAWN: "pawns", chess.KNIGHT: "knights", chess.BISHOP: "bishops",
            chess.ROOK: "rooks", chess.QUEEN: "queens",
        }.get(piece.piece_type)
        if piece_name:
            counts[f"{color_prefix}_{piece_name}"] += 1
    
    # Detect endgame patterns
    pattern = None
    total = white_material + black_material
    is_endgame = total <= 15  # Roughly queens off + limited material
    
    # KRK pattern: White K+R, Black K only
    if (counts["white_rooks"] == 1 and counts["white_queens"] == 0 and
        counts["white_bishops"] == 0 and counts["white_knights"] == 0 and
        counts["white_pawns"] == 0 and
        black_material == 0):
        pattern = "KRK"
    
    # KPK pattern: White K+P, Black K only
    elif (counts["white_pawns"] >= 1 and counts["white_queens"] == 0 and
          counts["white_rooks"] == 0 and counts["white_bishops"] == 0 and
          counts["white_knights"] == 0 and black_material == 0):
        pattern = "KPK"
    
    # KQK pattern: White K+Q, Black K only
    elif (counts["white_queens"] == 1 and counts["white_rooks"] == 0 and
          counts["white_bishops"] == 0 and counts["white_knights"] == 0 and
          counts["white_pawns"] == 0 and black_material == 0):
        pattern = "KQK"
    
    # Mirror patterns (Black has material, White has only King)
    elif (counts["black_rooks"] == 1 and counts["black_queens"] == 0 and
          counts["black_bishops"] == 0 and counts["black_knights"] == 0 and
          counts["black_pawns"] == 0 and white_material == 0):
        pattern = "KRK"  # From Black's perspective
    
    elif (counts["black_queens"] == 1 and counts["black_rooks"] == 0 and
          counts["black_bishops"] == 0 and counts["black_knights"] == 0 and
          counts["black_pawns"] == 0 and white_material == 0):
        pattern = "KQK"  # From Black's perspective
    
    elif (counts["black_pawns"] >= 1 and counts["black_queens"] == 0 and
          counts["black_rooks"] == 0 and counts["black_bishops"] == 0 and
          counts["black_knights"] == 0 and white_material == 0):
        pattern = "KPK"  # From Black's perspective
    
    return MaterialInfo(
        white_material=white_material,
        black_material=black_material,
        balance=white_material - black_material,
        is_endgame=is_endgame,
        pattern=pattern,
        **counts,
    )


def compute_subgraph_gates(
    board: chess.Board,
    phase: Optional[Dict[str, float]] = None,
    use_affordance: bool = True,
) -> Dict[str, float]:
    """
    Compute continuous activation weights for endgame subgraphs.
    
    Uses the new affordance-based system for continuous [0.0, 1.0] signals
    instead of binary gates. This enables implicit lookahead.
    
    Args:
        board: Current board position
        phase: Optional phase weights {"opening": 0.x, "middlegame": 0.x, "endgame": 0.x}
               Opening phase will dampen (but not fully gate) endgame affordances.
        use_affordance: If True (default), use continuous affordance signals.
                       If False, fall back to legacy binary gates.
    
    Returns:
        Dict mapping subgraph name to gate weight (0.0 to 1.0)
    """
    if use_affordance:
        return _compute_subgraph_gates_affordance(board, phase)
    else:
        return _compute_subgraph_gates_legacy(board, phase)


def _compute_subgraph_gates_affordance(
    board: chess.Board,
    phase: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute subgraph gates using continuous affordance signals.
    
    The affordance system provides smooth gradients instead of binary gates,
    enabling the Bandit to learn strategies that "climb the hill" toward
    endgame positions.
    """
    # Compute raw affordance signals
    affordances = compute_all_affordances(board)
    
    gates = {
        "krk": affordances["krk"].value,
        "kpk": affordances["kpk"].value,
        "kqk": affordances["kqk"].value,
    }
    
    # Apply phase-based modulation (softer than hard gates)
    if phase:
        opening_weight = phase.get("opening", 0)
        middlegame_weight = phase.get("middlegame", 0)
        endgame_weight = phase.get("endgame", 0)
        
        # In opening, dampen endgame affordances but don't zero them
        # This preserves the gradient for learning liquidation strategies
        if opening_weight > 0.5:
            dampening = 0.3 * (1.0 - opening_weight)  # 0.15 at opening=0.5, 0.0 at opening=1.0
            gates = {k: v * dampening for k, v in gates.items()}
        
        # In middlegame, moderate dampening
        elif middlegame_weight > 0.5:
            # Scale by how much we're into middlegame vs endgame
            endgame_factor = 0.3 + 0.7 * endgame_weight
            gates = {k: v * endgame_factor for k, v in gates.items()}
    
    return gates


def _compute_subgraph_gates_legacy(
    board: chess.Board,
    phase: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Legacy binary gating system (preserved for backward compatibility).
    
    This uses the original hard-gate logic with 0.0/0.3/1.0 values.
    """
    material = analyze_material(board)
    
    gates = {
        "krk": 0.0,
        "kpk": 0.0,
        "kqk": 0.0,
    }
    
    # Hard gate in opening
    if phase:
        opening_weight = phase.get("opening", 0)
        if opening_weight > 0.5:
            return gates
        
        middlegame_weight = phase.get("middlegame", 0)
        if middlegame_weight > 0.6 and not material.is_endgame:
            return gates
    
    # KRK gating
    if material.pattern == "KRK":
        gates["krk"] = 1.0
    elif material.is_endgame and (material.white_rooks > 0 or material.black_rooks > 0):
        gates["krk"] = 0.3
    
    # KPK gating
    if material.pattern == "KPK":
        gates["kpk"] = 1.0
    elif material.is_endgame and (material.white_pawns <= 2 or material.black_pawns <= 2):
        total_pieces = (material.white_knights + material.white_bishops + 
                       material.black_knights + material.black_bishops +
                       material.white_rooks + material.black_rooks +
                       material.white_queens + material.black_queens)
        if total_pieces <= 2:
            gates["kpk"] = 0.3
    
    # KQK gating
    if material.pattern == "KQK":
        gates["kqk"] = 1.0
    elif material.is_endgame and (material.white_queens > 0 or material.black_queens > 0):
        total_non_queen = (material.white_knights + material.white_bishops +
                          material.black_knights + material.black_bishops +
                          material.white_rooks + material.black_rooks +
                          material.white_pawns + material.black_pawns)
        if total_non_queen <= 2:
            gates["kqk"] = 0.3
    
    return gates


def compute_subgraph_affordances(
    board: chess.Board,
    phase: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, AffordanceSignal]]:
    """
    Compute both gate values and full affordance signals.
    
    This is useful when you need both the simplified gate values
    and the detailed affordance breakdown (components, exact match, etc.).
    
    Args:
        board: Current board position
        phase: Optional phase weights
        
    Returns:
        Tuple of (gates dict, full affordances dict)
    """
    affordances = compute_all_affordances(board)
    gates = {name: signal.value for name, signal in affordances.items()}
    
    # Apply phase modulation to gates
    if phase:
        opening_weight = phase.get("opening", 0)
        middlegame_weight = phase.get("middlegame", 0)
        endgame_weight = phase.get("endgame", 0)
        
        if opening_weight > 0.5:
            dampening = 0.3 * (1.0 - opening_weight)
            gates = {k: v * dampening for k, v in gates.items()}
        elif middlegame_weight > 0.5:
            endgame_factor = 0.3 + 0.7 * endgame_weight
            gates = {k: v * endgame_factor for k, v in gates.items()}
    
    return gates, affordances


def compute_tactics_context_weights(
    ultimate_goal: str,
    phase: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute context-dependent weights for tactics based on strategic situation.
    
    These multiply with the learned edge weights to produce final tactic influence.
    
    Args:
        ultimate_goal: "WIN", "DRAW", or "SURVIVE"
        phase: Dict with "opening", "middlegame", "endgame" weights
        
    Returns:
        Dict mapping tactic type to context weight multiplier
    """
    context = {
        "base": 1.0,
        # Individual tactic adjustments
        "fork": 1.0,
        "pin": 1.0,
        "skewer": 1.0,
        "hangingPiece": 1.0,
        "backRankMate": 1.0,
        "discoveredAttack": 1.0,
        "doubleCheck": 1.0,
        "smotheredMate": 1.0,
        "attraction": 1.0,
        "deflection": 1.0,
        "interference": 1.0,
        "sacrifice": 1.0,
        "quietMove": 1.0,
        "exposedKing": 1.0,
        "trappedPiece": 1.0,
        "zugzwang": 1.0,
    }
    
    # Adjust based on ultimate goal
    if ultimate_goal == "WIN":
        context["base"] = 1.3  # More aggressive tactics overall
        context["fork"] = 1.5  # Forks are excellent for winning material
        context["sacrifice"] = 1.3  # Sacrifices for attack
        context["backRankMate"] = 1.5  # Mate threats
        context["discoveredAttack"] = 1.3
        context["doubleCheck"] = 1.4
        context["smotheredMate"] = 1.5
        context["attraction"] = 1.2
        context["exposedKing"] = 1.3
    
    elif ultimate_goal == "SURVIVE":
        context["base"] = 0.8  # More conservative
        context["pin"] = 1.3  # Defensive pins valuable
        context["deflection"] = 1.2  # Deflect attacks
        context["quietMove"] = 1.4  # Quiet defensive moves
        context["trappedPiece"] = 0.8  # Less relevant when defending
        context["sacrifice"] = 0.5  # Avoid sacrificing when behind
    
    else:  # DRAW
        context["base"] = 1.0
        context["quietMove"] = 1.2  # Solid play
        context["zugzwang"] = 1.3  # Can force draws
    
    # Phase adjustments
    opening_weight = phase.get("opening", 0)
    middlegame_weight = phase.get("middlegame", 0)
    endgame_weight = phase.get("endgame", 0)
    
    if opening_weight > 0.5:
        # In opening, tactics less likely, focus on development
        context["base"] *= 0.7
        context["fork"] *= 0.8
        context["sacrifice"] *= 0.5  # Don't sacrifice in opening
    
    if endgame_weight > 0.5:
        # Endgame adjustments
        context["fork"] *= 0.8  # Fewer pieces to fork
        context["zugzwang"] *= 1.5  # Zugzwang very relevant
        context["quietMove"] *= 1.3
        context["backRankMate"] *= 0.7  # Less relevant
        context["smotheredMate"] *= 0.7
    
    return context
