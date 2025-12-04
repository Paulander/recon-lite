"""M9.2: Pattern Memory Storage and Retrieval.

Stores position embeddings with associated plans and outcomes.
Enables "I've seen this before" reasoning.

Usage:
    from recon_lite_chess.patterns import PatternMemory, MemoryConfig
    
    memory = PatternMemory(MemoryConfig(max_patterns=10000))
    
    # Store a pattern
    memory.store(embedding, plan_id="AttackKing", outcome=1.0)
    
    # Retrieve similar patterns
    matches = memory.retrieve(current_embedding, k=5)
    for match in matches:
        print(f"Similar pattern: {match.plan_id}, similarity: {match.similarity}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import numpy as np

from .embeddings import PositionEmbedding


@dataclass
class MemoryConfig:
    """Configuration for pattern memory."""
    max_patterns: int = 10000
    similarity_threshold: float = 0.7  # Min similarity for retrieval
    decay_rate: float = 0.99  # How fast old patterns fade
    outcome_weight: float = 0.3  # Weight of outcome in relevance scoring


@dataclass
class PatternEntry:
    """A single entry in pattern memory."""
    embedding: PositionEmbedding
    plan_id: str
    outcome: float  # 1.0 = win, 0.0 = draw, -1.0 = loss
    count: int = 1  # How many times seen
    confidence: float = 1.0  # Confidence in this pattern
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding.to_dict(),
            "plan_id": self.plan_id,
            "outcome": self.outcome,
            "count": self.count,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternEntry":
        return cls(
            embedding=PositionEmbedding.from_dict(data["embedding"]),
            plan_id=data["plan_id"],
            outcome=data["outcome"],
            count=data.get("count", 1),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PatternMatch:
    """Result of a pattern memory lookup."""
    entry: PatternEntry
    similarity: float  # Cosine similarity to query
    relevance: float  # Combined score (similarity + outcome)
    
    @property
    def plan_id(self) -> str:
        return self.entry.plan_id
    
    @property
    def outcome(self) -> float:
        return self.entry.outcome
    
    @property
    def fen(self) -> str:
        return self.entry.embedding.fen


class PatternMemory:
    """
    Memory store for position patterns with plan associations.
    
    Supports:
    - Storing new patterns with outcomes
    - Retrieving similar patterns
    - Updating existing patterns with new outcomes
    - Persisting to/from disk
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._entries: List[PatternEntry] = []
        self._index_dirty = True
        self._embedding_matrix: Optional[np.ndarray] = None
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def _rebuild_index(self) -> None:
        """Rebuild the embedding matrix for fast similarity search."""
        if not self._entries:
            self._embedding_matrix = None
            return
        
        vectors = [e.embedding.vector for e in self._entries]
        self._embedding_matrix = np.stack(vectors)
        self._index_dirty = False
    
    def store(
        self,
        embedding: PositionEmbedding,
        plan_id: str,
        outcome: float,
        merge_threshold: float = 0.95,
        **metadata,
    ) -> None:
        """
        Store a new pattern or merge with existing similar pattern.
        
        Args:
            embedding: Position embedding
            plan_id: ID of the plan that was used
            outcome: Game outcome (1.0 = win, 0.0 = draw, -1.0 = loss)
            merge_threshold: If similarity > this, merge instead of adding
            **metadata: Additional metadata to store
        """
        # Check for similar existing pattern
        if self._entries:
            matches = self.retrieve(embedding, k=1, min_similarity=merge_threshold)
            if matches and matches[0].entry.plan_id == plan_id:
                # Merge with existing
                existing = matches[0].entry
                existing.count += 1
                # Running average of outcome
                existing.outcome = (
                    existing.outcome * (existing.count - 1) + outcome
                ) / existing.count
                existing.confidence = min(1.0, existing.confidence + 0.1)
                return
        
        # Add new entry
        entry = PatternEntry(
            embedding=embedding,
            plan_id=plan_id,
            outcome=outcome,
            metadata=metadata,
        )
        self._entries.append(entry)
        self._index_dirty = True
        
        # Enforce max size
        if len(self._entries) > self.config.max_patterns:
            self._prune()
    
    def retrieve(
        self,
        query: PositionEmbedding,
        k: int = 5,
        min_similarity: float = 0.0,
        plan_filter: Optional[str] = None,
    ) -> List[PatternMatch]:
        """
        Retrieve k most similar patterns.
        
        Args:
            query: Query embedding
            k: Number of results to return
            min_similarity: Minimum cosine similarity threshold
            plan_filter: Only return patterns with this plan_id
            
        Returns:
            List of PatternMatch objects sorted by relevance
        """
        if not self._entries:
            return []
        
        if self._index_dirty:
            self._rebuild_index()
        
        if self._embedding_matrix is None:
            return []
        
        # Compute similarities
        query_vec = query.vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        
        query_normalized = query_vec / query_norm
        matrix_norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
        matrix_normalized = self._embedding_matrix / np.maximum(matrix_norms, 1e-8)
        
        similarities = matrix_normalized @ query_normalized
        
        # Build results
        results = []
        for idx, sim in enumerate(similarities):
            if sim < min_similarity:
                continue
            
            entry = self._entries[idx]
            if plan_filter and entry.plan_id != plan_filter:
                continue
            
            # Relevance combines similarity with outcome-weighted confidence
            relevance = sim + self.config.outcome_weight * entry.outcome * entry.confidence
            
            results.append(PatternMatch(
                entry=entry,
                similarity=float(sim),
                relevance=float(relevance),
            ))
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x.relevance, reverse=True)
        return results[:k]
    
    def get_plan_recommendations(
        self,
        query: PositionEmbedding,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Get plan recommendations based on similar positions.
        
        Returns list of (plan_id, boost_score) tuples.
        """
        matches = self.retrieve(query, k=k, min_similarity=min_similarity)
        
        # Aggregate by plan
        plan_scores: Dict[str, float] = {}
        for match in matches:
            plan_id = match.plan_id
            # Score = similarity * outcome * confidence
            score = match.similarity * (1 + match.outcome) * match.entry.confidence
            plan_scores[plan_id] = plan_scores.get(plan_id, 0) + score
        
        # Sort by score
        sorted_plans = sorted(plan_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_plans
    
    def _prune(self) -> None:
        """Remove old/low-confidence entries to stay under max size."""
        # Apply decay
        for entry in self._entries:
            entry.confidence *= self.config.decay_rate
        
        # Sort by confidence and keep top entries
        self._entries.sort(key=lambda e: e.confidence * e.count, reverse=True)
        self._entries = self._entries[:self.config.max_patterns]
        self._index_dirty = True
    
    def save(self, path: Path) -> None:
        """Save memory to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": {
                "max_patterns": self.config.max_patterns,
                "similarity_threshold": self.config.similarity_threshold,
                "decay_rate": self.config.decay_rate,
                "outcome_weight": self.config.outcome_weight,
            },
            "entries": [e.to_dict() for e in self._entries],
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "PatternMemory":
        """Load memory from disk."""
        with open(path) as f:
            data = json.load(f)
        
        config = MemoryConfig(**data.get("config", {}))
        memory = cls(config)
        memory._entries = [PatternEntry.from_dict(e) for e in data.get("entries", [])]
        memory._index_dirty = True
        return memory
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._entries:
            return {"count": 0}
        
        outcomes = [e.outcome for e in self._entries]
        plans = [e.plan_id for e in self._entries]
        
        return {
            "count": len(self._entries),
            "avg_outcome": sum(outcomes) / len(outcomes),
            "unique_plans": len(set(plans)),
            "avg_count": sum(e.count for e in self._entries) / len(self._entries),
            "plan_distribution": {
                p: plans.count(p) for p in set(plans)
            },
        }

