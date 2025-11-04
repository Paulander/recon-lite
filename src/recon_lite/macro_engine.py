"""
Thin wrapper around the macrograph-instantiated network.

This lets us spin up a ReCoN engine with the top-level macro skeleton while
delegating endgame behaviour to the mounted KRK subgraph when requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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

    def step(self, env: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        env = env or {}
        self._maybe_activate_endgame(env)
        return super().step(env)


def build_macro_engine(spec_path: Path | str = DEFAULT_SPEC_PATH) -> MacroEngine:
    """Convenience factory mirroring the KRK builders used in demos."""
    return MacroEngine(spec_path=spec_path)
