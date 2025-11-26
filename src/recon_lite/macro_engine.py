"""
Thin wrapper around the macrograph-instantiated network.

This lets us spin up a ReCoN engine with the top-level macro skeleton while
delegating endgame behaviour to the mounted KRK subgraph when requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chess

from recon_lite.engine import ReConEngine
from recon_lite.graph import NodeState
from recon_lite.macrograph import instantiate_macrograph
from recon_lite.trace_db import EpisodeRecord, TickRecord, TraceDB, pack_fingerprint

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
        from recon_lite_chess.scripts.kpk import build_kpk_network  # lazy import
        from recon_lite_chess.scripts.rook_endings import build_rook_techniques_network  # lazy import

        graph = instantiate_macrograph(
            spec_path,
            krk_builder=build_krk_network,
            kpk_builder=build_kpk_network,
            mount_builders={"rook_techniques": build_rook_techniques_network},
        )
        super().__init__(graph)

    @staticmethod
    def _is_krk_board(board: chess.Board) -> bool:
        pieces = board.piece_map()
        white_kings = sum(1 for p in pieces.values() if p.color and p.piece_type == chess.KING)
        black_kings = sum(1 for p in pieces.values() if (not p.color) and p.piece_type == chess.KING)
        white_rooks = sum(1 for p in pieces.values() if p.color and p.piece_type == chess.ROOK)
        return white_kings == 1 and black_kings == 1 and white_rooks == 1 and len(pieces) == 3

    @staticmethod
    def _is_kpk_board(board: chess.Board) -> bool:
        if board is None:
            return False
        try:
            from recon_lite_chess.sensors.structure import summarize_kpk_material
        except ImportError:
            return False
        summary = summarize_kpk_material(board)
        return bool(summary.get("is_kpk"))

    @staticmethod
    def _is_endgame_board(board: chess.Board) -> bool:
        return board is not None and (MacroEngine._is_krk_board(board) or MacroEngine._is_kpk_board(board))

    @staticmethod
    def _looks_like_rook_endgame(board: chess.Board) -> bool:
        if board is None:
            return False
        pieces = board.piece_map()
        has_rook = any(p.color and p.piece_type == chess.ROOK for p in pieces.values())
        return has_rook and MacroEngine._is_endgame_board(board)

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
        board = env.get("board")
        mounts = [
            ("KRKSubgraph", self._is_krk_board),
            ("KPKSubgraph", self._is_kpk_board),
            ("RookTechniquesSubgraph", self._is_endgame_board),
        ]
        for node_id, detector in mounts:
            node = self.g.nodes.get(node_id)
            if not node or node.state not in (NodeState.INACTIVE, NodeState.CONFIRMED):
                continue
            activate = bool(env.get("activate_endgame"))
            if isinstance(board, chess.Board):
                activate = activate or detector(board)
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
        is_endgame = self._is_endgame_board(board)
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
        if self._is_endgame_board(board):
            return {"Opening": 0.05, "Middlegame": 0.15, "Endgame": 0.80}
        return {"Opening": 0.4, "Middlegame": 0.4, "Endgame": 0.2}

    def _plan_groups(self, board: Optional[chess.Board], features: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        feat = features or {}
        is_endgame = bool(feat.get("is_endgame")) if features is not None else (board is not None and self._is_endgame_board(board))
        activations = {
            "PlanOpening": 0.2,
            "PlanMiddlegame": 0.35,
            "PlanEndgame": 0.85 if is_endgame else 0.3,
        }
        highlights: Set[str] = set()
        if features is not None:
            if feat.get("is_krk"):
                highlights.add("KRK")
            if feat.get("is_kpk"):
                highlights.add("KPK")
        elif board is not None:
            if self._is_krk_board(board):
                highlights.add("KRK")
            if self._is_kpk_board(board):
                highlights.add("KPK")
            if self._looks_like_rook_endgame(board):
                highlights.add("RookTechniques")
        plans = []
        for pid, children in self.PLAN_GROUPS.items():
            plan_details = [{"name": child, "highlight": child in highlights} for child in children]
            plans.append(
                {
                    "id": pid,
                    "activation": activations.get(pid, 0.2),
                    "plans": children,
                    "details": plan_details,
                }
            )
        return plans

    def _feature_groups(self, board: Optional[chess.Board], features: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        feat = features or {}
        is_endgame = bool(feat.get("is_endgame")) if features is not None else (board is not None and self._is_endgame_board(board))
        confidences = {
            "FeatureTactics": 0.45,
            "FeatureStructure": 0.52,
            "FeatureEndgame": 0.88 if is_endgame else 0.25,
        }
        highlight_lookup: Dict[str, bool] = {}
        if features is not None:
            if feat.get("is_krk"):
                highlight_lookup.update({"KRK": True, "BoxShrink": True, "Opposition": True})
            if feat.get("is_kpk"):
                highlight_lookup.update({"KPK": True})
        elif board is not None:
            if self._is_krk_board(board):
                highlight_lookup.update({"KRK": True, "BoxShrink": True, "Opposition": True})
            if self._is_kpk_board(board):
                highlight_lookup.update({"KPK": True})
            if self._looks_like_rook_endgame(board):
                highlight_lookup.update({"RookCutoff": True})
        features_out = []
        for fid, children in self.FEATURE_GROUPS.items():
            feature_details = [{"name": child, "highlight": bool(highlight_lookup.get(child))} for child in children]
            features_out.append(
                {
                    "id": fid,
                    "confidence": confidences.get(fid, 0.3),
                    "features": children,
                    "details": feature_details,
                }
            )
        return features_out

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
        if self._is_kpk_board(board):
            try:
                from recon_lite_chess.sensors.structure import summarize_kpk_material
            except ImportError:
                pass
            else:
                summary = summarize_kpk_material(board)
                kpk_items = []
                pawn_sq = summary.get("pawn_square")
                attacker_king = summary.get("attacker_king")
                defender_king = summary.get("defender_king")
                attacker_color = summary.get("attacker_color")
                if pawn_sq is not None:
                    kpk_items.append({"feature": "pawn", "items": [f"square:{chess.square_name(pawn_sq)}"]})
                if attacker_king is not None:
                    label = "white" if attacker_color else "black"
                    kpk_items.append({"feature": f"attacker_king_{label}", "items": [f"square:{chess.square_name(attacker_king)}"]})
                if defender_king is not None:
                    kpk_items.append({"feature": "defender_king", "items": [f"square:{chess.square_name(defender_king)}"]})
                if kpk_items:
                    bindings["macro/endgame/kpk"] = kpk_items
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

    def _feature_bus(self, board: Optional[chess.Board]) -> Dict[str, Any]:
        """
        Aggregate raw features per tick for macro decisions. This is a "bus"
        that downstream nodes (plan selection, viz, trainers) can consume.
        """
        if board is None:
            return {}
        features: Dict[str, Any] = {}
        # Material breakdown (white-centric)
        material_score = self._material_score(board)
        pieces = board.piece_map()
        counts: Dict[str, int] = {"white": 0, "black": 0}
        piece_counts: Dict[str, Dict[str, int]] = {
            "white": {k: 0 for k in ["P", "N", "B", "R", "Q"]},
            "black": {k: 0 for k in ["p", "n", "b", "r", "q"]},
        }
        for sq, piece in pieces.items():
            color_key = "white" if piece.color else "black"
            counts[color_key] += 1
            symbol = piece.symbol()
            if symbol in piece_counts[color_key]:
                piece_counts[color_key][symbol] += 1
        features["material_score"] = material_score
        features["piece_counts"] = piece_counts
        features["total_pieces"] = counts

        # Phase / endgame detectors
        features["is_endgame"] = self._is_endgame_board(board)
        features["is_krk"] = self._is_krk_board(board)
        features["is_kpk"] = self._is_kpk_board(board)
        features["phase_mix"] = self._phase_mix(board)
        features["goal_vector"] = self._goal_vector(board)

        # Simple king distances
        wk = board.king(True)
        bk = board.king(False)
        if wk is not None and bk is not None:
            dx = abs(chess.square_file(wk) - chess.square_file(bk))
            dy = abs(chess.square_rank(wk) - chess.square_rank(bk))
            features["king_distance_chebyshev"] = max(dx, dy)
        # Bindings snapshot
        features["bindings"] = self._macro_bindings(board)
        return features

    def capture_macro_frame(self, env: Dict[str, Any]) -> Dict[str, Any]:
        board = env.get("board")
        feat_bus = self._feature_bus(board)
        return {
            "version": "0.1",
            "goal_vector": feat_bus.get("goal_vector") or self._goal_vector(board),
            "phase_mix": feat_bus.get("phase_mix") or self._phase_mix(board),
            "plan_groups": self._plan_groups(board, feat_bus),
            "feature_groups": self._feature_groups(board, feat_bus),
            "bindings": feat_bus.get("bindings") or self._macro_bindings(board),
            "move_synth": self._move_synth_preview(board),
            "features": feat_bus,
        }

    def step(self, env: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        env = env or {}
        self._maybe_activate_endgame(env)
        macro_frame = self.capture_macro_frame(env)
        env["macro_frame"] = macro_frame
        env["features"] = macro_frame.get("features")
        result = super().step(env)
        # Optional fallback: if no script proposed a move, provide one to keep play going.
        try:
            board = env.get("board")
            if board is not None and env.get("chosen_move") is None:
                if env.get("fallback_move", False) or env.get("stockfish_path"):
                    move = self._fallback_move(board, env)
                    if move is not None:
                        env["chosen_move"] = move.uci()
        except Exception:
            # Fallback must never crash the engine loop
            pass
        return result


def run_macro_episode(
    board: chess.Board,
    *,
    max_ticks: int = 200,
    trace_db: Optional[TraceDB] = None,
    episode_id: str = "macro-episode",
    pack_paths: Optional[list[Path]] = None,
) -> EpisodeRecord:
    """Utility to run a single macro-engine tick loop and emit an EpisodeRecord."""
    engine = MacroEngine("specs/macrograph_v0.json")
    env: Dict[str, Any] = {"board": board.copy()}
    ticks: list[TickRecord] = []
    pack_meta = pack_fingerprint(pack_paths or [])

    while len(ticks) < max_ticks:
        now_req = engine.step(env)
        ticks.append(
            TickRecord(
                tick_id=len(ticks) + 1,
                phase_estimate=None,
                goal_vector=env.get("macro_frame", {}).get("goal_vector"),
                board_fen=board.fen(),
                active_nodes=[nid for nid, node in engine.g.nodes.items() if node.state != NodeState.INACTIVE],
                fired_edges=[],
                action=env.get("chosen_move"),
                meta={
                    "macro_frame": env.get("macro_frame"),
                    "new_requests": list(now_req.keys()),
                },
            )
        )
        if env.get("chosen_move"):
            break

    ep = EpisodeRecord(
        episode_id=episode_id,
        result=None,
        ticks=ticks,
        pack_meta=pack_meta,
        notes={"ticks": len(ticks)},
    )
    if trace_db:
        trace_db.add_episode(ep)
    return ep


def build_macro_engine(spec_path: Path | str = DEFAULT_SPEC_PATH) -> MacroEngine:
    """Convenience factory mirroring the KRK builders used in demos."""
    return MacroEngine(spec_path=spec_path)
