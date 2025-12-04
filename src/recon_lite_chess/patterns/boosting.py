"""M9.3-4: Context-Aware Plan Boosting and Episodic Retrieval.

Integrates pattern memory with plan persistence to boost plans
based on similar positions seen before.

Usage:
    from recon_lite_chess.patterns import PatternBooster, EpisodicMemory
    
    booster = PatternBooster(memory)
    boost_factors = booster.get_plan_boosts(board, current_plans)
    
    episodic = EpisodicMemory()
    episodic.record_episode(positions, moves, outcome)
    similar_episodes = episodic.recall(current_position)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import chess

from .embeddings import encode_position, PositionEmbedding
from .memory import PatternMemory, MemoryConfig


@dataclass
class PlanBoost:
    """Boost factor for a plan based on pattern memory."""
    plan_id: str
    boost: float  # Multiplier for plan activation (1.0 = no change)
    confidence: float  # How confident we are in this boost
    source_count: int  # How many similar patterns contributed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "boost": self.boost,
            "confidence": self.confidence,
            "source_count": self.source_count,
        }


class PatternBooster:
    """
    Boosts plan activations based on pattern memory.
    
    When a position is similar to ones we've seen before,
    we boost plans that worked well in those positions.
    """
    
    def __init__(
        self,
        memory: PatternMemory,
        min_similarity: float = 0.6,
        max_boost: float = 2.0,
        min_patterns: int = 3,  # Require at least this many patterns for boosting
    ):
        self.memory = memory
        self.min_similarity = min_similarity
        self.max_boost = max_boost
        self.min_patterns = min_patterns
    
    def get_plan_boosts(
        self,
        board: chess.Board,
        current_plans: Optional[List[str]] = None,
        k: int = 20,
    ) -> Dict[str, PlanBoost]:
        """
        Get boost factors for plans based on pattern memory.
        
        Args:
            board: Current board position
            current_plans: Optional list of plan IDs to consider
            k: Number of similar patterns to retrieve
            
        Returns:
            Dict mapping plan_id to PlanBoost
        """
        # Encode current position
        embedding = encode_position(board)
        
        # Get plan recommendations from memory
        recommendations = self.memory.get_plan_recommendations(
            embedding,
            k=k,
            min_similarity=self.min_similarity,
        )
        
        if not recommendations:
            return {}
        
        # Convert to boost factors
        boosts = {}
        for plan_id, score in recommendations:
            if current_plans and plan_id not in current_plans:
                continue
            
            # Normalize score to boost factor
            # Positive score = boost > 1, negative = boost < 1
            boost = 1.0 + min(score, self.max_boost - 1.0)
            boost = max(0.5, min(self.max_boost, boost))  # Clamp
            
            # Count patterns for this plan
            matches = self.memory.retrieve(
                embedding, k=k, min_similarity=self.min_similarity, plan_filter=plan_id
            )
            source_count = len(matches)
            
            if source_count < self.min_patterns:
                continue  # Not enough evidence
            
            # Confidence based on number of patterns and their similarity
            avg_sim = sum(m.similarity for m in matches) / len(matches) if matches else 0
            confidence = min(1.0, source_count / 10) * avg_sim
            
            boosts[plan_id] = PlanBoost(
                plan_id=plan_id,
                boost=boost,
                confidence=confidence,
                source_count=source_count,
            )
        
        return boosts
    
    def apply_boosts(
        self,
        plan_activations: Dict[str, float],
        board: chess.Board,
    ) -> Dict[str, float]:
        """
        Apply boosts to plan activations.
        
        Args:
            plan_activations: Current plan activation levels
            board: Current board position
            
        Returns:
            Boosted plan activations
        """
        boosts = self.get_plan_boosts(board, list(plan_activations.keys()))
        
        boosted = {}
        for plan_id, activation in plan_activations.items():
            if plan_id in boosts:
                boost = boosts[plan_id]
                # Weight boost by confidence
                effective_boost = 1.0 + (boost.boost - 1.0) * boost.confidence
                boosted[plan_id] = activation * effective_boost
            else:
                boosted[plan_id] = activation
        
        return boosted


@dataclass
class Episode:
    """A recorded game episode for episodic memory."""
    positions: List[str]  # FENs
    moves: List[str]  # UCI moves
    outcome: float  # 1.0 = win, 0.0 = draw, -1.0 = loss
    plans_used: List[str]  # Plans that were active during the game
    embeddings: List[np.ndarray] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "moves": self.moves,
            "outcome": self.outcome,
            "plans_used": self.plans_used,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            positions=data["positions"],
            moves=data["moves"],
            outcome=data["outcome"],
            plans_used=data.get("plans_used", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EpisodeMatch:
    """A match from episodic memory."""
    episode: Episode
    position_idx: int  # Which position in the episode matched
    similarity: float
    
    @property
    def next_move(self) -> Optional[str]:
        """Get the move played after this position."""
        if self.position_idx < len(self.episode.moves):
            return self.episode.moves[self.position_idx]
        return None
    
    @property
    def remaining_moves(self) -> List[str]:
        """Get all moves after this position."""
        return self.episode.moves[self.position_idx:]


# Import numpy here to avoid issues if not installed
try:
    import numpy as np
except ImportError:
    np = None


class EpisodicMemory:
    """
    Memory for full game episodes.
    
    Enables "last time I saw this, I played X and it worked/failed" reasoning.
    """
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        self._position_index: Dict[str, List[Tuple[int, int]]] = {}  # fen -> [(episode_idx, position_idx)]
    
    def __len__(self) -> int:
        return len(self._episodes)
    
    def record_episode(
        self,
        positions: List[str],
        moves: List[str],
        outcome: float,
        plans_used: Optional[List[str]] = None,
        **metadata,
    ) -> None:
        """
        Record a game episode.
        
        Args:
            positions: List of FEN strings (board states)
            moves: List of UCI moves
            outcome: Game outcome (1.0 = win, 0.0 = draw, -1.0 = loss)
            plans_used: Optional list of plan IDs used during the game
            **metadata: Additional metadata
        """
        episode = Episode(
            positions=positions,
            moves=moves,
            outcome=outcome,
            plans_used=plans_used or [],
            metadata=metadata,
        )
        
        episode_idx = len(self._episodes)
        self._episodes.append(episode)
        
        # Index positions
        for pos_idx, fen in enumerate(positions):
            key = fen.split()[0]  # Just the piece placement part
            if key not in self._position_index:
                self._position_index[key] = []
            self._position_index[key].append((episode_idx, pos_idx))
        
        # Prune if needed
        if len(self._episodes) > self.max_episodes:
            self._prune()
    
    def recall(
        self,
        board: chess.Board,
        k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[EpisodeMatch]:
        """
        Recall episodes with similar positions.
        
        Args:
            board: Current board position
            k: Number of matches to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of EpisodeMatch objects
        """
        current_fen = board.fen()
        key = current_fen.split()[0]
        
        # Fast exact match check
        matches = []
        if key in self._position_index:
            for episode_idx, pos_idx in self._position_index[key]:
                matches.append(EpisodeMatch(
                    episode=self._episodes[episode_idx],
                    position_idx=pos_idx,
                    similarity=1.0,  # Exact match
                ))
        
        # If we have enough exact matches, return them
        if len(matches) >= k:
            return sorted(matches, key=lambda x: x.episode.outcome, reverse=True)[:k]
        
        # Otherwise, do embedding-based search
        if np is not None:
            current_embedding = encode_position(board)
            
            # Search all episodes
            for episode_idx, episode in enumerate(self._episodes):
                for pos_idx, fen in enumerate(episode.positions):
                    if (episode_idx, pos_idx) in [(m.episode, m.position_idx) for m in matches]:
                        continue
                    
                    try:
                        pos_board = chess.Board(fen)
                        pos_embedding = encode_position(pos_board)
                        similarity = current_embedding.cosine_similarity(pos_embedding)
                        
                        if similarity >= min_similarity:
                            matches.append(EpisodeMatch(
                                episode=episode,
                                position_idx=pos_idx,
                                similarity=similarity,
                            ))
                    except Exception:
                        continue
        
        # Sort by similarity and outcome, return top k
        matches.sort(key=lambda x: (x.similarity, x.episode.outcome), reverse=True)
        return matches[:k]
    
    def get_move_suggestions(
        self,
        board: chess.Board,
        k: int = 10,
    ) -> List[Tuple[str, float, float]]:
        """
        Get move suggestions from similar episodes.
        
        Returns:
            List of (move_uci, avg_outcome, count) tuples
        """
        matches = self.recall(board, k=k)
        
        move_stats: Dict[str, Tuple[float, int]] = {}  # move -> (outcome_sum, count)
        for match in matches:
            move = match.next_move
            if move:
                if move not in move_stats:
                    move_stats[move] = (0.0, 0)
                outcome_sum, count = move_stats[move]
                move_stats[move] = (
                    outcome_sum + match.episode.outcome * match.similarity,
                    count + 1,
                )
        
        results = []
        for move, (outcome_sum, count) in move_stats.items():
            avg_outcome = outcome_sum / count
            results.append((move, avg_outcome, count))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _prune(self) -> None:
        """Remove oldest episodes to stay under max size."""
        # Keep most recent episodes
        self._episodes = self._episodes[-self.max_episodes:]
        
        # Rebuild index
        self._position_index.clear()
        for episode_idx, episode in enumerate(self._episodes):
            for pos_idx, fen in enumerate(episode.positions):
                key = fen.split()[0]
                if key not in self._position_index:
                    self._position_index[key] = []
                self._position_index[key].append((episode_idx, pos_idx))
    
    def save(self, path: Path) -> None:
        """Save episodic memory to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_episodes": self.max_episodes,
            "episodes": [e.to_dict() for e in self._episodes],
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "EpisodicMemory":
        """Load episodic memory from disk."""
        with open(path) as f:
            data = json.load(f)
        
        memory = cls(max_episodes=data.get("max_episodes", 1000))
        for ep_data in data.get("episodes", []):
            episode = Episode.from_dict(ep_data)
            memory._episodes.append(episode)
            
            # Rebuild index
            episode_idx = len(memory._episodes) - 1
            for pos_idx, fen in enumerate(episode.positions):
                key = fen.split()[0]
                if key not in memory._position_index:
                    memory._position_index[key] = []
                memory._position_index[key].append((episode_idx, pos_idx))
        
        return memory

