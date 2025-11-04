"""
Thin wrapper around the macrograph-instantiated network.

This lets us spin up a ReCoN engine with the top-level macro skeleton while
delegating endgame behaviour to the mounted KRK subgraph when requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chess

from recon_lite.engine import ReConEngine
from recon_lite.graph import NodeState
from recon_lite.macrograph import instantiate_macrograph

# Default spec location (relative to project root).
DEFAULT_SPEC_PATH = Path("specs/macrograph_v0.json")


class MacroEngine(ReConEngine):
    """
    Engine that activates the KRK subgraph when the environment signals an
    endgame position. The activation logic is intentionally minimal: if the
    board is recognised as a KRK-like configuration (white to move with rook+king
    vs lone king by default) or `env["activate_endgame"]` is truthy, the
    `KRKSubgraph` node is requested, thereby kicking off the mounted KRK plan.
    """

    def __init__(self, spec_path: Path | str = DEFAULT_SPEC_PATH):
        from demos.shared.krk_network import build_krk_network  # lazy import

        graph = instantiate_macrograph(spec_path, krk_builder=build_krk_network)
        super().__init__(graph)

    @staticmethod
    def _is_krk_board(board: chess.Board) -> bool:
        pieces = board.piece_map()
        white_kings = sum(1 for p in pieces.values() if p.color and p.piece_type == chess.KING)
        black_kings = sum(1 for p in pieces.values() if (not p.color) and p.piece_type == chess.KING)
        white_rooks = sum(1 for p in pieces.values() if p.color and p.piece_type == chess.ROOK)
        return white_kings == 1 and black_kings == 1 and white_rooks == 1 and len(pieces) == 3

    PLAN_GROUPS = {
        "PlanOpening": ["Develop", "CastleSafety", "CenterControl", "AvoidTraps"],
        "PlanMiddlegame": ["CreateWeakness", "KingSafety", "ActivatePieces", "Simplify", "AttackKing"],
        "PlanEndgame": ["KRK", "KPK", "RookTechniques", "Conversion"],
    }

    FEATURE_GROUPS = {
        "FeatureTactics": ["HangingPiece", "MateThreat", "Fork", "Pin", "Skewer", "DiscoveredCheck"],
        "FeatureStructure": ["OpenFile", "PassedPawn", "IsolatedPawn", "DoubledPawn", "Outpost", "KingShelter"],
        "FeatureEndgame": ["KRK", "KPK", "RookCutoff", "BoxShrink", "Opposition"],
    }

    def _maybe_activate_endgame(self, env: Dict[str, Any]) -> None:
        node = self.g.nodes.get("KRKSubgraph")
        if not node or node.state not in (NodeState.INACTIVE, NodeState.CONFIRMED):
            return
        activate = bool(env.get("activate_endgame"))
        board = env.get("board")
        if isinstance(board, chess.Board):
            activate = activate or self._is_krk_board(board)
        if activate:
            node.state = NodeState.REQUESTED
            node.tick_entered = self.tick

    @staticmethod
    def _material_score(board: chess.Board) -> float:
        values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        score = 0.0
        for sq, piece in board.piece_map().items():
            val = values.get(piece.piece_type, 0.0)
            score += val if piece.color else -val
        return max(0.0, min(1.0, (score + 10.0) / 20.0))

    @staticmethod
    def _rook_square(board: chess.Board) -> Optional[chess.Square]:
        for sq, piece in board.piece_map().items():
            if piece.color == chess.WHITE and piece.piece_type == chess.ROOK:
                return sq
        return None

    def _goal_vector(self, board: Optional[chess.Board]) -> Dict[str, float]:
        if board is None:
            return {
                "Material": 0.5,
                "KingSafety": 0.5,
                "Initiative": 0.5,
                "Structure": 0.5,
                "PhaseProgress": 0.5,
                "RiskBudget": 0.5,
                "TacticWindow": 0.5,
            }

        material = self._material_score(board)
        is_endgame = self._is_krk_board(board)
        return {
            "Material": material,
            "KingSafety": 0.65 if is_endgame else 0.55,
            "Initiative": 0.58,
            "Structure": 0.6 if is_endgame else 0.5,
            "PhaseProgress": 0.92 if is_endgame else 0.33,
            "RiskBudget": 0.28 if is_endgame else 0.45,
            "TacticWindow": 0.37 if is_endgame else 0.52,
        }

    def _phase_mix(self, board: Optional[chess.Board]) -> Dict[str, float]:
        if board is None:
            return {"Opening": 0.33, "Middlegame": 0.33, "Endgame": 0.34}
        if self._is_krk_board(board):
            return {"Opening": 0.05, "Middlegame": 0.15, "Endgame": 0.80}
        return {"Opening": 0.4, "Middlegame": 0.4, "Endgame": 0.2}

    def _plan_groups(self, board: Optional[chess.Board]) -> List[Dict[str, Any]]:
        is_endgame = board is not None and self._is_krk_board(board)
        activations = {
            "PlanOpening": 0.2,
            "PlanMiddlegame": 0.35,
            "PlanEndgame": 0.85 if is_endgame else 0.3,
        }
        plans = []
        for pid, children in self.PLAN_GROUPS.items():
            plans.append(
                {
                    "id": pid,
                    "activation": activations.get(pid, 0.2),
                    "plans": children,
                }
            )
        return plans

    def _feature_groups(self, board: Optional[chess.Board]) -> List[Dict[str, Any]]:
        is_endgame = board is not None and self._is_krk_board(board)
        confidences = {
            "FeatureTactics": 0.45,
            "FeatureStructure": 0.52,
            "FeatureEndgame": 0.88 if is_endgame else 0.25,
        }
        features = []
        for fid, children in self.FEATURE_GROUPS.items():
            features.append(
                {
                    "id": fid,
                    "confidence": confidences.get(fid, 0.3),
                    "features": children,
                }
            )
        return features

    def _macro_bindings(self, board: Optional[chess.Board]) -> Dict[str, Any]:
        if board is None:
            return {}
        rook_sq = self._rook_square(board)
        king_sq = board.king(True)
        opp_sq = board.king(False)
        bindings: Dict[str, List[Dict[str, Any]]] = {}
        items = []
        if rook_sq is not None:
            items.append({"feature": "rook_anchor", "items": [f"square:{chess.square_name(rook_sq)}"]})
        if king_sq is not None:
            items.append({"feature": "own_king", "items": [f"square:{chess.square_name(king_sq)}"]})
        if opp_sq is not None:
            items.append({"feature": "enemy_king", "items": [f"square:{chess.square_name(opp_sq)}"]})
        if items:
            bindings["macro/endgame/krk"] = items
        return bindings

    def _move_synth_preview(self, board: Optional[chess.Board]) -> Dict[str, Any]:
        weights = {"plan": 0.4, "features": 0.3, "eval": 0.2, "goal": 0.1}
        proposals: List[Dict[str, Any]] = []
        chosen = None
        if board is not None:
            legal_moves = list(board.legal_moves)[:3]
            for idx, move in enumerate(legal_moves):
                components = {
                    "plan": round(0.45 + 0.03 * idx, 3),
                    "features": round(0.28 + 0.02 * idx, 3),
                    "eval": round(0.18 + 0.01 * idx, 3),
                    "goal": round(0.09 + 0.01 * idx, 3),
                }
                total = round(sum(components.values()), 3)
                proposals.append(
                    {
                        "uci": move.uci(),
                        "score": total,
                        "components": components,
                    }
                )
            if proposals:
                chosen = proposals[0]["uci"]
        return {
            "weights": weights,
            "proposals": proposals,
            "chosen": chosen,
        }

    def capture_macro_frame(self, env: Dict[str, Any]) -> Dict[str, Any]:
        board = env.get("board")
        return {
            "version": "0.1",
            "goal_vector": self._goal_vector(board),
            "phase_mix": self._phase_mix(board),
            "plan_groups": self._plan_groups(board),
            "feature_groups": self._feature_groups(board),
            "bindings": self._macro_bindings(board),
            "move_synth": self._move_synth_preview(board),
        }

    def step(self, env: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        env = env or {}
        self._maybe_activate_endgame(env)
        macro_frame = self.capture_macro_frame(env)
        result = super().step(env)
        env["macro_frame"] = macro_frame
        return result


def build_macro_engine(spec_path: Path | str = DEFAULT_SPEC_PATH) -> MacroEngine:
    """Convenience factory mirroring the KRK builders used in demos."""
    return MacroEngine(spec_path=spec_path)
