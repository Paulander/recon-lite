"""M10.1: Stem Cell Terminal Implementation.

Stem cells are special terminal nodes that can discover new patterns
and potentially specialize into full sensors.

Lifecycle: EXPLORING → CANDIDATE → SPECIALIZED

Usage:
    from recon_lite.nodes import StemCellTerminal, StemCellState
    
    stem = StemCellTerminal("stem_001", exploration_budget=100)
    
    # During game play, stem cell collects samples
    stem.observe(board, reward_tick)
    
    # When enough samples collected, attempt specialization
    if stem.should_specialize():
        sensor = stem.specialize()
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import chess
    HAS_CHESS = True
except ImportError:
    chess = None
    HAS_CHESS = False


class StemCellState(Enum):
    """Lifecycle states for a stem cell."""
    DORMANT = auto()      # Not yet activated
    EXPLORING = auto()    # Actively collecting samples
    CANDIDATE = auto()    # Has enough samples, awaiting evaluation
    SPECIALIZED = auto()  # Promoted to full sensor
    PRUNED = auto()       # Removed due to low value


@dataclass
class StemCellConfig:
    """Configuration for stem cell behavior."""
    min_samples: int = 50  # Minimum samples before specialization attempt
    max_samples: int = 500  # Maximum samples to collect
    reward_threshold: float = 0.3  # Only store samples with |reward| > this
    specialization_threshold: float = 0.7  # Min pattern consistency for specialization
    exploration_budget: int = 100  # Ticks allowed in exploring state
    decay_rate: float = 0.99  # How fast samples decay in importance


@dataclass
class StemCellSample:
    """A sample collected by a stem cell."""
    fen: str
    features: Optional[List[float]] = None
    reward: float = 0.0
    tick: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fen": self.fen,
            "features": self.features,
            "reward": self.reward,
            "tick": self.tick,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StemCellSample":
        return cls(
            fen=data["fen"],
            features=data.get("features"),
            reward=data.get("reward", 0.0),
            tick=data.get("tick", 0),
            metadata=data.get("metadata", {}),
        )


class StemCellTerminal:
    """
    A stem cell terminal that can discover patterns and specialize.
    
    Stem cells are exploratory nodes that:
    1. Collect samples during high-reward moments
    2. Analyze patterns in collected samples
    3. Potentially specialize into a full sensor
    """
    
    def __init__(
        self,
        cell_id: str,
        config: Optional[StemCellConfig] = None,
        feature_extractor: Optional[Callable] = None,
    ):
        self.cell_id = cell_id
        self.config = config or StemCellConfig()
        self.feature_extractor = feature_extractor
        
        self.state = StemCellState.DORMANT
        self.samples: List[StemCellSample] = []
        self.exploration_ticks = 0
        self.pattern_signature: Optional[List[float]] = None
        self.trust_score = 0.0
        self.metadata: Dict[str, Any] = {}
    
    def activate(self) -> None:
        """Start exploration."""
        if self.state == StemCellState.DORMANT:
            self.state = StemCellState.EXPLORING
            self.exploration_ticks = 0
    
    def observe(
        self,
        board: Any,  # chess.Board
        reward: float,
        tick: int = 0,
        **metadata,
    ) -> bool:
        """
        Observe a position during exploration.
        
        Args:
            board: Chess board (or any domain object)
            reward: Reward signal at this tick
            tick: Current tick number
            **metadata: Additional data to store
            
        Returns:
            True if sample was stored, False if filtered out
        """
        if self.state not in (StemCellState.EXPLORING, StemCellState.CANDIDATE):
            return False
        
        # Filter by reward threshold
        if abs(reward) < self.config.reward_threshold:
            return False
        
        # Extract features if extractor available
        features = None
        if self.feature_extractor and board is not None:
            try:
                features = self.feature_extractor(board)
                if hasattr(features, 'tolist'):
                    features = features.tolist()
            except Exception:
                features = None
        
        # Get FEN if it's a chess board
        fen = ""
        if HAS_CHESS and hasattr(board, 'fen'):
            fen = board.fen()
        elif isinstance(board, str):
            fen = board
        
        sample = StemCellSample(
            fen=fen,
            features=features,
            reward=reward,
            tick=tick,
            metadata=metadata,
        )
        
        self.samples.append(sample)
        self.exploration_ticks += 1
        
        # Check state transitions
        if len(self.samples) >= self.config.min_samples:
            if self.state == StemCellState.EXPLORING:
                self.state = StemCellState.CANDIDATE
        
        # Prune if too many samples
        if len(self.samples) > self.config.max_samples:
            self._prune_samples()
        
        # Check if exploration budget exhausted
        if self.exploration_ticks >= self.config.exploration_budget:
            if len(self.samples) < self.config.min_samples:
                self.state = StemCellState.PRUNED
        
        return True
    
    def should_specialize(self) -> bool:
        """Check if the stem cell should attempt specialization."""
        if self.state != StemCellState.CANDIDATE:
            return False
        if len(self.samples) < self.config.min_samples:
            return False
        return True
    
    def analyze_pattern(self) -> Tuple[float, Optional[List[float]]]:
        """
        Analyze collected samples for patterns.
        
        Returns:
            (consistency_score, pattern_signature)
            consistency_score: How consistent the pattern is (0-1)
            pattern_signature: Representative feature vector
        """
        if not self.samples or not HAS_NUMPY:
            return 0.0, None
        
        # Get samples with features
        featured = [s for s in self.samples if s.features is not None]
        if len(featured) < 10:
            return 0.0, None
        
        # Stack features
        try:
            features = np.array([s.features for s in featured])
        except Exception:
            return 0.0, None
        
        # Compute centroid
        centroid = np.mean(features, axis=0)
        
        # Compute consistency (inverse of average distance to centroid)
        distances = np.linalg.norm(features - centroid, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        # Consistency: 1.0 if all samples are identical, lower if spread out
        consistency = 1.0 - min(1.0, avg_distance / (max_distance + 1e-6))
        
        self.pattern_signature = centroid.tolist()
        return float(consistency), self.pattern_signature
    
    def specialize(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to specialize into a full sensor.
        
        Returns:
            Sensor specification dict if successful, None if failed
        """
        if self.state != StemCellState.CANDIDATE:
            return None
        
        consistency, signature = self.analyze_pattern()
        
        if consistency < self.config.specialization_threshold:
            # Not consistent enough
            self.state = StemCellState.PRUNED
            return None
        
        # Successful specialization
        self.state = StemCellState.SPECIALIZED
        self.trust_score = consistency
        
        # Create sensor specification
        sensor_spec = {
            "sensor_id": f"sensor_{self.cell_id}",
            "type": "pattern_detector",
            "pattern_signature": signature,
            "threshold": self.config.specialization_threshold,
            "sample_count": len(self.samples),
            "consistency": consistency,
            "source_cell": self.cell_id,
            "metadata": {
                "avg_reward": sum(s.reward for s in self.samples) / len(self.samples),
                "sample_fens": [s.fen for s in self.samples[:10]],  # Example positions
            },
        }
        
        return sensor_spec
    
    def _prune_samples(self) -> None:
        """Remove low-value samples to stay under max."""
        if len(self.samples) <= self.config.max_samples:
            return
        
        # Keep samples with highest absolute reward
        self.samples.sort(key=lambda s: abs(s.reward), reverse=True)
        self.samples = self.samples[:self.config.max_samples]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize stem cell to dict."""
        return {
            "cell_id": self.cell_id,
            "state": self.state.name,
            "samples": [s.to_dict() for s in self.samples],
            "exploration_ticks": self.exploration_ticks,
            "pattern_signature": self.pattern_signature,
            "trust_score": self.trust_score,
            "config": {
                "min_samples": self.config.min_samples,
                "max_samples": self.config.max_samples,
                "reward_threshold": self.config.reward_threshold,
                "specialization_threshold": self.config.specialization_threshold,
                "exploration_budget": self.config.exploration_budget,
            },
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StemCellTerminal":
        """Deserialize stem cell from dict."""
        config_data = data.get("config", {})
        config = StemCellConfig(
            min_samples=config_data.get("min_samples", 50),
            max_samples=config_data.get("max_samples", 500),
            reward_threshold=config_data.get("reward_threshold", 0.3),
            specialization_threshold=config_data.get("specialization_threshold", 0.7),
            exploration_budget=config_data.get("exploration_budget", 100),
        )
        
        cell = cls(cell_id=data["cell_id"], config=config)
        cell.state = StemCellState[data.get("state", "DORMANT")]
        cell.samples = [StemCellSample.from_dict(s) for s in data.get("samples", [])]
        cell.exploration_ticks = data.get("exploration_ticks", 0)
        cell.pattern_signature = data.get("pattern_signature")
        cell.trust_score = data.get("trust_score", 0.0)
        cell.metadata = data.get("metadata", {})
        return cell


class StemCellManager:
    """
    Manages a pool of stem cells for pattern discovery.
    """
    
    def __init__(
        self,
        max_cells: int = 20,
        spawn_rate: float = 0.1,  # Probability of spawning new cell per tick
        config: Optional[StemCellConfig] = None,
    ):
        self.max_cells = max_cells
        self.spawn_rate = spawn_rate
        self.default_config = config or StemCellConfig()
        self.cells: Dict[str, StemCellTerminal] = {}
        self._next_id = 0
    
    def spawn_cell(self, config: Optional[StemCellConfig] = None) -> Optional[StemCellTerminal]:
        """Spawn a new stem cell if under limit."""
        active = [c for c in self.cells.values() if c.state in (
            StemCellState.DORMANT, StemCellState.EXPLORING, StemCellState.CANDIDATE
        )]
        
        if len(active) >= self.max_cells:
            return None
        
        cell_id = f"stem_{self._next_id:04d}"
        self._next_id += 1
        
        cell = StemCellTerminal(
            cell_id=cell_id,
            config=config or self.default_config,
        )
        cell.activate()
        self.cells[cell_id] = cell
        return cell
    
    def tick(
        self,
        board: Any,
        reward: float,
        tick: int = 0,
    ) -> List[str]:
        """
        Process one tick for all active stem cells.
        
        Returns:
            List of cell IDs that stored samples
        """
        stored = []
        
        # Maybe spawn new cell
        if random.random() < self.spawn_rate:
            self.spawn_cell()
        
        # Update all active cells
        for cell_id, cell in list(self.cells.items()):
            if cell.state in (StemCellState.EXPLORING, StemCellState.CANDIDATE):
                if cell.observe(board, reward, tick):
                    stored.append(cell_id)
        
        return stored
    
    def get_specialization_candidates(self) -> List[StemCellTerminal]:
        """Get cells ready for specialization."""
        return [c for c in self.cells.values() if c.should_specialize()]
    
    def specialize_all_ready(self) -> List[Dict[str, Any]]:
        """Attempt to specialize all ready cells."""
        specs = []
        for cell in self.get_specialization_candidates():
            spec = cell.specialize()
            if spec:
                specs.append(spec)
        return specs
    
    def prune_failed(self) -> int:
        """Remove pruned cells from pool."""
        pruned = [cid for cid, c in self.cells.items() if c.state == StemCellState.PRUNED]
        for cid in pruned:
            del self.cells[cid]
        return len(pruned)
    
    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        states = {}
        for cell in self.cells.values():
            state = cell.state.name
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_cells": len(self.cells),
            "by_state": states,
            "next_id": self._next_id,
        }
    
    def save(self, path: Path) -> None:
        """Save manager state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_cells": self.max_cells,
            "spawn_rate": self.spawn_rate,
            "next_id": self._next_id,
            "cells": {cid: c.to_dict() for cid, c in self.cells.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "StemCellManager":
        """Load manager state."""
        with open(path) as f:
            data = json.load(f)
        
        manager = cls(
            max_cells=data.get("max_cells", 20),
            spawn_rate=data.get("spawn_rate", 0.1),
        )
        manager._next_id = data.get("next_id", 0)
        manager.cells = {
            cid: StemCellTerminal.from_dict(cdata)
            for cid, cdata in data.get("cells", {}).items()
        }
        return manager

