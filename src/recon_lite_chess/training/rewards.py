"""Reward Providers for M5-Evolution Training.

Implements reward signals for stem cell training:
- Stockfish eval delta (dense feedback)
- Game outcome (sparse but definitive)
- Progress-based (pawn advancement)

Usage:
    from recon_lite_chess.training.rewards import StockfishRewardProvider
    
    provider = StockfishRewardProvider()
    reward = provider.compute_reward(board_before, board_after)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import chess


@dataclass
class MoveReward:
    """Reward signal for a single move."""
    eval_delta: float = 0.0      # Stockfish eval change (normalized)
    progress: float = 0.0        # Pawn advancement reward
    outcome: float = 0.0         # Game outcome (+1 win, -1 loss, 0 draw/ongoing)
    inertia_penalty: float = 0.0 # Penalty for repetition/draws/slow play
    is_blunder: bool = False     # Large negative eval swing
    raw_eval: float = 0.0        # Raw eval for logging
    
    @property
    def total(self) -> float:
        """Combined reward signal."""
        return self.eval_delta + self.progress + self.outcome + self.inertia_penalty


class StockfishRewardProvider:
    """
    Provides dense reward signals using Stockfish evaluation.
    
    Falls back gracefully if Stockfish is unavailable.
    """
    
    def __init__(self, depth: int = 8, blunder_threshold: float = 2.0):
        """
        Args:
            depth: Stockfish search depth (lower = faster)
            blunder_threshold: Eval drop (in pawns) to flag as blunder
        """
        self.depth = depth
        self.blunder_threshold = blunder_threshold
        self._engine = None
        self._available = None  # None = not checked yet
    
    def _ensure_engine(self) -> bool:
        """Lazily initialize Stockfish engine."""
        if self._available is False:
            return False
        
        if self._engine is not None:
            return True
        
        try:
            import chess.engine
            # Try common Stockfish paths
            for path in ["stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]:
                try:
                    self._engine = chess.engine.SimpleEngine.popen_uci(path)
                    self._available = True
                    return True
                except Exception:
                    continue
            self._available = False
            return False
        except ImportError:
            self._available = False
            return False
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Get Stockfish evaluation for position.
        
        Returns:
            Eval in pawns (positive = white advantage)
        """
        if not self._ensure_engine():
            return 0.0
        
        try:
            info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].white()
            
            # Handle mate scores
            if score.is_mate():
                mate_in = score.mate()
                if mate_in > 0:
                    return 100.0  # Winning
                else:
                    return -100.0  # Losing
            
            # Convert centipawns to pawns
            return score.score() / 100.0
        except Exception:
            return 0.0
    
    def compute_reward(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        move: Optional[chess.Move] = None,
    ) -> MoveReward:
        """
        Compute reward for a move.
        
        Args:
            board_before: Position before the move
            board_after: Position after the move
            move: The move played (optional, for logging)
            
        Returns:
            MoveReward with all signal components
        """
        reward = MoveReward()
        
        # Check for game-ending conditions first
        if board_after.is_checkmate():
            # Opponent is in checkmate = we won
            reward.outcome = 1.0
            return reward
        elif board_after.is_stalemate() or board_after.is_insufficient_material():
            reward.outcome = 0.0
            return reward
        
        # Stockfish eval delta
        if self._ensure_engine():
            eval_before = self.evaluate(board_before)
            eval_after = self.evaluate(board_after)
            
            # Flip sign since we moved and now it's opponent's turn
            # After our move, positive eval still means we're better
            reward.eval_delta = (eval_after - eval_before) / 10.0  # Normalize
            reward.raw_eval = eval_after
            
            # Detect blunders
            if reward.eval_delta < -self.blunder_threshold:
                reward.is_blunder = True
        
        # Pawn progress reward (for KPK)
        reward.progress = self._compute_progress(board_before, board_after)
        
        return reward
    
    def _compute_progress(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
    ) -> float:
        """Compute pawn advancement reward."""
        # Find pawns
        pawns_before = list(board_before.pieces(chess.PAWN, chess.WHITE))
        pawns_after = list(board_after.pieces(chess.PAWN, chess.WHITE))
        
        if not pawns_before:
            return 0.0
        
        # Check for promotion
        if len(pawns_after) < len(pawns_before):
            # Pawn disappeared - likely promoted
            return 0.5
        
        # Check rank advancement
        if pawns_before and pawns_after:
            rank_before = chess.square_rank(pawns_before[0])
            rank_after = chess.square_rank(pawns_after[0])
            
            if rank_after > rank_before:
                return 0.1 * (rank_after - rank_before)
        
        return 0.0
    
    def close(self):
        """Clean up engine resources."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
    
    def __del__(self):
        self.close()


class SimpleRewardProvider:
    """
    Simple reward provider using only game outcomes.
    
    Use when Stockfish is not available.
    Includes inertia penalties for draws/repetition/slow play.
    """
    
    def compute_reward(
        self,
        board_before: chess.Board,
        board_after: chess.Board,
        move: Optional[chess.Move] = None,
    ) -> MoveReward:
        """Compute reward based on game state only."""
        reward = MoveReward()
        
        if board_after.is_checkmate():
            reward.outcome = 1.0
        elif board_after.is_stalemate():
            # Stalemate is bad in KPK - we should be winning!
            reward.outcome = -0.3
            reward.inertia_penalty = -0.2  # Additional penalty
        elif board_after.is_insufficient_material():
            # Lost the pawn - very bad!
            reward.outcome = -0.5
        elif board_after.can_claim_threefold_repetition() or board_after.is_repetition(2):
            # Repetition detection - punish "dancing"
            reward.inertia_penalty = -0.5
        elif board_after.can_claim_fifty_moves():
            # 50-move rule draw imminent
            reward.inertia_penalty = -0.3
        else:
            # Check for inertia (slow play) penalty
            move_count = board_after.fullmove_number
            if move_count > 30:
                # -0.01 per move after move 30 (inertia tax)
                reward.inertia_penalty = -0.01 * (move_count - 30)
            
            # Small progress reward for pawn advancement
            pawns_before = list(board_before.pieces(chess.PAWN, chess.WHITE))
            pawns_after = list(board_after.pieces(chess.PAWN, chess.WHITE))
            
            if pawns_before and pawns_after:
                rank_before = chess.square_rank(pawns_before[0])
                rank_after = chess.square_rank(pawns_after[0])
                if rank_after > rank_before:
                    reward.progress = 0.1
        
        return reward


def get_reward_provider(use_stockfish: bool = True) -> Any:
    """
    Get appropriate reward provider.
    
    Args:
        use_stockfish: Try to use Stockfish if available
        
    Returns:
        StockfishRewardProvider or SimpleRewardProvider
    """
    if use_stockfish:
        provider = StockfishRewardProvider()
        if provider._ensure_engine():
            return provider
    
    return SimpleRewardProvider()
