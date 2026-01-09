#!/usr/bin/env python3
"""KPK Gym Environment for PPO baseline comparison.

Provides a standard Gymnasium interface for KPK endgame training,
allowing comparison with ReCoN hierarchical approach.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    import gym
    from gym import spaces
    HAS_GYM = False

import chess


class KPKEnv(gym.Env):
    """
    KPK Endgame Environment.
    
    State: 64-dimensional vector (piece positions encoded)
    Action: Legal move index (variable action space per step)
    Reward: +1 win, -1 loss, 0 draw, +0.5 promotion
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        stage: int = 7,
        max_moves: int = 100,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.stage = stage
        self.max_moves = max_moves
        self.render_mode = render_mode
        
        # State: 12 channels x 64 squares (one-hot piece encoding)
        # Simplified: just 64 values for piece types
        self.observation_space = spaces.Box(
            low=-6, high=6, shape=(64,), dtype=np.float32
        )
        
        # Action: max 218 legal moves in chess (we'll mask illegal)
        self.action_space = spaces.Discrete(218)
        
        self.board: Optional[chess.Board] = None
        self.move_count = 0
        self.legal_moves_list = []
        
    def _get_obs(self) -> np.ndarray:
        """Convert board to observation vector."""
        obs = np.zeros(64, dtype=np.float32)
        
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                # Encode: positive for white, negative for black
                value = piece.piece_type
                if piece.color == chess.BLACK:
                    value = -value
                obs[sq] = value
                
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "fen": self.board.fen(),
            "legal_moves": len(self.legal_moves_list),
            "move_count": self.move_count,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Generate random KPK position based on stage
        self.board = self._generate_position()
        self.move_count = 0
        self.legal_moves_list = list(self.board.legal_moves)
        
        return self._get_obs(), self._get_info()
    
    def _generate_position(self) -> chess.Board:
        """Generate KPK position based on stage."""
        # Stage 7 = Full opposition - random valid KPK positions
        board = chess.Board()
        board.clear()
        
        # Random pawn location (not on rank 1 or 8)
        pawn_file = self.np_random.integers(0, 8)
        pawn_rank = self.np_random.integers(1, 7)  # Ranks 2-7
        pawn_sq = chess.square(pawn_file, pawn_rank)
        
        # White king near pawn
        king_offset = self.np_random.integers(-2, 3, size=2)
        wk_file = max(0, min(7, pawn_file + king_offset[0]))
        wk_rank = max(0, min(7, pawn_rank + king_offset[1]))
        wk_sq = chess.square(wk_file, wk_rank)
        
        # Black king somewhere else
        while True:
            bk_file = self.np_random.integers(0, 8)
            bk_rank = self.np_random.integers(0, 8)
            bk_sq = chess.square(bk_file, bk_rank)
            # Must be different from other pieces and not adjacent to white king
            if bk_sq not in (pawn_sq, wk_sq):
                if chess.square_distance(bk_sq, wk_sq) > 1:
                    break
        
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        
        # Validate position is legal
        if not board.is_valid():
            return self._generate_position()
            
        return board
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        
        # Handle empty legal moves (shouldn't happen, but safety check)
        if not self.legal_moves_list:
            return self._get_obs(), -1.0, True, False, self._get_info()
        
        # Map action to legal move
        if action >= len(self.legal_moves_list):
            # Invalid action - pick random legal move
            move = self.np_random.choice(self.legal_moves_list)
        else:
            move = self.legal_moves_list[action]
        
        # Check for promotion
        promoted = move.promotion is not None
        
        # Make move
        self.board.push(move)
        self.move_count += 1
        
        # Check terminal conditions
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.board.is_checkmate():
            # We won (black is checkmated)
            reward = 1.0
            terminated = True
        elif self.board.is_stalemate():
            reward = -0.5  # Stalemate is bad for attacker
            terminated = True
        elif self.board.is_insufficient_material():
            reward = -1.0  # We lost the pawn somehow
            terminated = True
        elif promoted:
            reward = 0.5  # Promotion bonus
            terminated = True  # End episode on promotion
        elif self.move_count >= self.max_moves:
            reward = -0.2  # Timeout penalty
            truncated = True
        else:
            # Black's move (random for now)
            black_moves = list(self.board.legal_moves)
            if black_moves:
                black_move = self.np_random.choice(black_moves)
                self.board.push(black_move)
                self.move_count += 1
                
                # Check if black caused terminal
                if self.board.is_checkmate():
                    reward = -1.0  # We got checkmated
                    terminated = True
                elif self.board.is_stalemate():
                    reward = -0.5  # Stalemate
                    terminated = True
                elif self.board.is_game_over():
                    reward = -0.5  # Other game over
                    terminated = True
            else:
                # No black moves = stalemate or checkmate
                if self.board.is_checkmate():
                    reward = 1.0  # We won
                else:
                    reward = -0.5  # Draw
                terminated = True
        
        # Update legal moves for next action
        if not terminated and not truncated:
            self.legal_moves_list = list(self.board.legal_moves)
            # Safety check: if no legal moves, it's a terminal state
            if not self.legal_moves_list:
                terminated = True
                if self.board.is_checkmate():
                    reward = -1.0
                else:
                    reward = -0.5
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        if self.render_mode == "human":
            print(self.board)


if __name__ == "__main__":
    # Quick test
    env = KPKEnv(stage=7)
    obs, info = env.reset()
    print(f"Initial: {info['fen']}")
    print(f"Legal moves: {info['legal_moves']}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Move {info['move_count']}: reward={reward:.2f}, {info['fen'][:30]}...")
        if term or trunc:
            print(f"Episode ended: reward={reward}")
            break
