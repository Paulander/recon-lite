"""
Evaluation Manager (M4)

Unified interface for chess position evaluation that supports multiple backends:
- Heuristic: Fast, CPU-only evaluation
- Stockfish: Accurate, requires external engine
- Distilled: Future ML-based evaluation (stub for now)

Provides caching and logging of evaluation sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, TYPE_CHECKING

import chess

from .heuristic import eval_position, eval_position_fast, eval_position_full

if TYPE_CHECKING:
    import chess.engine


class EvalMode(Enum):
    """Available evaluation modes."""

    HEURISTIC_FAST = auto()  # Material + king safety + mobility only
    HEURISTIC = auto()  # Default heuristic with pawn structure + activity
    HEURISTIC_FULL = auto()  # Full heuristic including tactical tension
    STOCKFISH = auto()  # External Stockfish engine
    DISTILLED = auto()  # ML-based distilled eval (future)
    HYBRID = auto()  # Heuristic with occasional Stockfish validation


@dataclass
class EvalConfig:
    """Configuration for the evaluation manager."""

    mode: EvalMode = EvalMode.HEURISTIC
    stockfish_path: Optional[str] = None
    stockfish_depth: int = 3
    stockfish_time_limit: float = 0.1
    cache_enabled: bool = True
    cache_max_size: int = 10000
    log_source: bool = False
    hybrid_sample_rate: float = 0.1  # Fraction of positions to validate with SF


@dataclass
class EvalResult:
    """Result of an evaluation with metadata."""

    score: float  # Evaluation in pawn units (positive = white advantage)
    source: EvalMode  # Which evaluator produced this result
    cached: bool = False  # Whether this came from cache
    confidence: float = 1.0  # Confidence in the evaluation (for future use)
    meta: Dict[str, Any] = field(default_factory=dict)


class EvalManager:
    """
    Unified evaluation manager supporting multiple backends.

    Usage:
        manager = EvalManager(EvalConfig(mode=EvalMode.HEURISTIC))
        result = manager.evaluate(board)
        print(f"Score: {result.score} (from {result.source.name})")
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self._cache: Dict[str, EvalResult] = {}
        self._sf_engine: Optional["chess.engine.SimpleEngine"] = None
        self._stats = {
            "total_evals": 0,
            "cache_hits": 0,
            "heuristic_evals": 0,
            "stockfish_evals": 0,
            # M4: Hybrid mode tracking
            "hybrid_validations": 0,
            "hybrid_error_sum": 0.0,
            "hybrid_error_abs_sum": 0.0,
        }

    def _get_cache_key(self, board: chess.Board) -> str:
        """Generate cache key from board position."""
        return board.fen()

    def _init_stockfish(self) -> bool:
        """Initialize Stockfish engine if needed."""
        if self._sf_engine is not None:
            return True

        if not self.config.stockfish_path:
            return False

        try:
            import chess.engine

            self._sf_engine = chess.engine.SimpleEngine.popen_uci(
                self.config.stockfish_path
            )
            return True
        except Exception:
            return False

    def _eval_heuristic(self, board: chess.Board) -> EvalResult:
        """Evaluate using heuristic."""
        if self.config.mode == EvalMode.HEURISTIC_FAST:
            score = eval_position_fast(board)
        elif self.config.mode == EvalMode.HEURISTIC_FULL:
            score = eval_position_full(board)
        else:
            score = eval_position(board)

        self._stats["heuristic_evals"] += 1
        return EvalResult(score=score, source=self.config.mode)

    def _eval_stockfish(self, board: chess.Board) -> EvalResult:
        """Evaluate using Stockfish."""
        if not self._init_stockfish():
            # Fallback to heuristic
            result = self._eval_heuristic(board)
            result.meta["fallback"] = "stockfish_unavailable"
            return result

        try:
            import chess.engine

            info = self._sf_engine.analyse(
                board,
                chess.engine.Limit(
                    depth=self.config.stockfish_depth,
                    time=self.config.stockfish_time_limit,
                ),
            )
            score_obj = info.get("score")
            if score_obj is None:
                return self._eval_heuristic(board)

            pov_score = score_obj.white()

            if pov_score.is_mate():
                mate_in = pov_score.mate()
                if mate_in is not None:
                    if mate_in > 0:
                        score = 100.0 - mate_in * 0.1
                    else:
                        score = -100.0 - mate_in * 0.1
                else:
                    score = 0.0
            else:
                cp = pov_score.score()
                score = cp / 100.0 if cp is not None else 0.0

            self._stats["stockfish_evals"] += 1
            return EvalResult(
                score=score,
                source=EvalMode.STOCKFISH,
                meta={"depth": info.get("depth"), "nodes": info.get("nodes")},
            )

        except Exception:
            return self._eval_heuristic(board)

    def _eval_hybrid(self, board: chess.Board) -> EvalResult:
        """Evaluate using heuristic with occasional Stockfish validation."""
        import random

        heuristic_result = self._eval_heuristic(board)

        # Occasionally validate with Stockfish
        if random.random() < self.config.hybrid_sample_rate:
            sf_result = self._eval_stockfish(board)
            diff = sf_result.score - heuristic_result.score
            heuristic_result.meta["sf_validation"] = sf_result.score
            heuristic_result.meta["sf_diff"] = diff

            # M4: Track hybrid validation statistics
            self._stats["hybrid_validations"] += 1
            self._stats["hybrid_error_sum"] += diff
            self._stats["hybrid_error_abs_sum"] += abs(diff)

        return heuristic_result

    def evaluate(self, board: chess.Board) -> EvalResult:
        """
        Evaluate a chess position.

        Args:
            board: The chess board to evaluate

        Returns:
            EvalResult with score and metadata
        """
        self._stats["total_evals"] += 1

        # Check cache
        if self.config.cache_enabled:
            cache_key = self._get_cache_key(board)
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                result = self._cache[cache_key]
                result.cached = True
                return result

        # Evaluate based on mode
        if self.config.mode in (
            EvalMode.HEURISTIC,
            EvalMode.HEURISTIC_FAST,
            EvalMode.HEURISTIC_FULL,
        ):
            result = self._eval_heuristic(board)
        elif self.config.mode == EvalMode.STOCKFISH:
            result = self._eval_stockfish(board)
        elif self.config.mode == EvalMode.HYBRID:
            result = self._eval_hybrid(board)
        elif self.config.mode == EvalMode.DISTILLED:
            # Stub: fall back to heuristic
            result = self._eval_heuristic(board)
            result.meta["distilled_stub"] = True
        else:
            result = self._eval_heuristic(board)

        # Cache result
        if self.config.cache_enabled:
            if len(self._cache) >= self.config.cache_max_size:
                # Simple eviction: clear half the cache
                keys = list(self._cache.keys())
                for key in keys[: len(keys) // 2]:
                    del self._cache[key]
            self._cache[cache_key] = result

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        stats = dict(self._stats)
        if stats["total_evals"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_evals"]
        # M4: Compute hybrid error metrics
        if stats["hybrid_validations"] > 0:
            stats["hybrid_mean_error"] = stats["hybrid_error_sum"] / stats["hybrid_validations"]
            stats["hybrid_mean_abs_error"] = stats["hybrid_error_abs_sum"] / stats["hybrid_validations"]
        return stats

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()

    def close(self) -> None:
        """Clean up resources."""
        if self._sf_engine is not None:
            try:
                self._sf_engine.quit()
            except Exception:
                pass
            self._sf_engine = None

    def __enter__(self) -> "EvalManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

