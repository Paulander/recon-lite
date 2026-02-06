"""M10.2-3: Pattern Induction Pipeline and Promotion Logic.

Integrates stem cells with the motif extraction system to discover
new patterns and promote them to sensors.

Usage:
    from recon_lite.motifs import PatternInduction, PromotionConfig
    
    pipeline = PatternInduction(stem_manager, motif_memory)
    
    # During gameplay
    pipeline.tick(board, reward)
    
    # After episodes
    promotions = pipeline.evaluate_and_promote()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

try:
    from ..nodes.stem_cell import (
        StemCellTerminal,
        StemCellState,
        StemCellConfig,
        StemCellManager,
    )
    HAS_STEM_CELL = True
except ImportError:
    HAS_STEM_CELL = False

try:
    from ..trust.scoring import compute_node_trust, TrustConfig
    HAS_TRUST = True
except ImportError:
    HAS_TRUST = False


@dataclass
class PromotionConfig:
    """Configuration for sensor promotion."""
    min_consistency: float = 0.7  # Minimum pattern consistency
    min_samples: int = 30  # Minimum samples required
    min_avg_reward: float = 0.1  # Minimum average reward
    promotion_threshold: float = 0.6  # Overall score threshold
    cooldown_ticks: int = 100  # Ticks before reconsidering rejected candidates


@dataclass
class PromotionCandidate:
    """A candidate for promotion to full sensor."""
    cell_id: str
    sensor_spec: Dict[str, Any]
    score: float
    consistency: float
    avg_reward: float
    sample_count: int
    created_tick: int = 0
    status: str = "pending"  # pending, approved, rejected, promoted
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "sensor_spec": self.sensor_spec,
            "score": self.score,
            "consistency": self.consistency,
            "avg_reward": self.avg_reward,
            "sample_count": self.sample_count,
            "created_tick": self.created_tick,
            "status": self.status,
        }


@dataclass
class PromotedSensor:
    """A sensor that was promoted from a stem cell."""
    sensor_id: str
    source_cell: str
    pattern_signature: List[float]
    trust_score: float
    promoted_tick: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "source_cell": self.source_cell,
            "pattern_signature": self.pattern_signature,
            "trust_score": self.trust_score,
            "promoted_tick": self.promoted_tick,
            "metadata": self.metadata,
        }


class PatternInduction:
    """
    Pipeline for discovering patterns via stem cells and promoting them.
    
    The pipeline:
    1. Manages stem cells during gameplay
    2. Evaluates candidate patterns
    3. Promotes successful candidates to sensors
    4. Prunes unsuccessful candidates
    """
    
    def __init__(
        self,
        stem_manager: Optional["StemCellManager"] = None,
        promo_config: Optional[PromotionConfig] = None,
    ):
        if HAS_STEM_CELL:
            from ..nodes.stem_cell import StemCellManager
            self.stem_manager = stem_manager or StemCellManager()
        else:
            self.stem_manager = stem_manager
        
        self.promo_config = promo_config or PromotionConfig()
        
        self.candidates: Dict[str, PromotionCandidate] = {}
        self.promoted: Dict[str, PromotedSensor] = {}
        self.rejected_ids: List[str] = []
        
        self._current_tick = 0
    
    def tick(
        self,
        board: Any,
        reward: float,
        feature_extractor: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process one tick of the pipeline.
        
        Args:
            board: Current board state
            reward: Reward signal
            feature_extractor: Optional function to extract features
            
        Returns:
            Dict with pipeline statistics
        """
        self._current_tick += 1
        stats = {"tick": self._current_tick, "samples_stored": 0, "new_candidates": 0}
        
        if not self.stem_manager:
            return stats
        
        # Let stem cells observe
        stored = self.stem_manager.tick(board, reward, self._current_tick)
        stats["samples_stored"] = len(stored)
        
        # Check for new candidates
        for cell in self.stem_manager.get_specialization_candidates():
            if cell.cell_id not in self.candidates:
                spec = cell.specialize()
                if spec:
                    candidate = self._create_candidate(cell, spec)
                    self.candidates[cell.cell_id] = candidate
                    stats["new_candidates"] += 1
        
        return stats
    
    def _create_candidate(
        self,
        cell: "StemCellTerminal",
        spec: Dict[str, Any],
    ) -> PromotionCandidate:
        """Create a promotion candidate from a specialized stem cell."""
        consistency = spec.get("consistency", 0.0)
        avg_reward = spec.get("metadata", {}).get("avg_reward", 0.0)
        sample_count = spec.get("sample_count", 0)
        
        # Compute overall score
        score = (
            0.4 * consistency +
            0.3 * min(1.0, abs(avg_reward)) +
            0.3 * min(1.0, sample_count / 100)
        )
        
        return PromotionCandidate(
            cell_id=cell.cell_id,
            sensor_spec=spec,
            score=score,
            consistency=consistency,
            avg_reward=avg_reward,
            sample_count=sample_count,
            created_tick=self._current_tick,
        )
    
    def evaluate_candidates(self) -> List[PromotionCandidate]:
        """
        Evaluate all pending candidates for promotion.
        
        Returns:
            List of candidates ready for promotion
        """
        ready = []
        for candidate in self.candidates.values():
            if candidate.status != "pending":
                continue
            
            # Check thresholds
            if candidate.consistency < self.promo_config.min_consistency:
                candidate.status = "rejected"
                continue
            if candidate.sample_count < self.promo_config.min_samples:
                continue  # Wait for more samples
            if abs(candidate.avg_reward) < self.promo_config.min_avg_reward:
                candidate.status = "rejected"
                continue
            if candidate.score < self.promo_config.promotion_threshold:
                candidate.status = "rejected"
                continue
            
            candidate.status = "approved"
            ready.append(candidate)
        
        return ready
    
    def promote(self, candidate: PromotionCandidate) -> PromotedSensor:
        """
        Promote a candidate to a full sensor.
        
        Args:
            candidate: Approved promotion candidate
            
        Returns:
            PromotedSensor object
        """
        sensor = PromotedSensor(
            sensor_id=candidate.sensor_spec.get("sensor_id", f"sensor_{candidate.cell_id}"),
            source_cell=candidate.cell_id,
            pattern_signature=candidate.sensor_spec.get("pattern_signature", []),
            trust_score=candidate.consistency,
            promoted_tick=self._current_tick,
            metadata={
                "score": candidate.score,
                "avg_reward": candidate.avg_reward,
                "sample_count": candidate.sample_count,
            },
        )
        
        self.promoted[sensor.sensor_id] = sensor
        candidate.status = "promoted"
        
        return sensor
    
    def evaluate_and_promote(self) -> List[PromotedSensor]:
        """
        Evaluate candidates and promote approved ones.
        
        Returns:
            List of newly promoted sensors
        """
        approved = self.evaluate_candidates()
        promoted = []
        for candidate in approved:
            sensor = self.promote(candidate)
            promoted.append(sensor)
        return promoted
    
    def prune_rejected(self) -> int:
        """Remove rejected candidates from tracking."""
        rejected = [cid for cid, c in self.candidates.items() if c.status == "rejected"]
        for cid in rejected:
            del self.candidates[cid]
            self.rejected_ids.append(cid)
        return len(rejected)
    
    def get_pending_count(self) -> int:
        """Get count of pending candidates."""
        return sum(1 for c in self.candidates.values() if c.status == "pending")
    
    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stem_stats = self.stem_manager.stats() if self.stem_manager else {}
        
        return {
            "current_tick": self._current_tick,
            "stem_cells": stem_stats,
            "candidates": {
                "total": len(self.candidates),
                "pending": sum(1 for c in self.candidates.values() if c.status == "pending"),
                "approved": sum(1 for c in self.candidates.values() if c.status == "approved"),
                "rejected": sum(1 for c in self.candidates.values() if c.status == "rejected"),
                "promoted": sum(1 for c in self.candidates.values() if c.status == "promoted"),
            },
            "promoted_sensors": len(self.promoted),
            "total_rejected": len(self.rejected_ids),
        }
    
    def save(self, path: Path) -> None:
        """Save pipeline state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "current_tick": self._current_tick,
            "candidates": {cid: c.to_dict() for cid, c in self.candidates.items()},
            "promoted": {sid: s.to_dict() for sid, s in self.promoted.items()},
            "rejected_ids": self.rejected_ids,
            "config": {
                "min_consistency": self.promo_config.min_consistency,
                "min_samples": self.promo_config.min_samples,
                "min_avg_reward": self.promo_config.min_avg_reward,
                "promotion_threshold": self.promo_config.promotion_threshold,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "PatternInduction":
        """Load pipeline state."""
        with open(path) as f:
            data = json.load(f)
        
        config_data = data.get("config", {})
        config = PromotionConfig(
            min_consistency=config_data.get("min_consistency", 0.7),
            min_samples=config_data.get("min_samples", 30),
            min_avg_reward=config_data.get("min_avg_reward", 0.1),
            promotion_threshold=config_data.get("promotion_threshold", 0.6),
        )
        
        pipeline = cls(promo_config=config)
        pipeline._current_tick = data.get("current_tick", 0)
        pipeline.rejected_ids = data.get("rejected_ids", [])
        
        # Restore candidates
        for cid, cdata in data.get("candidates", {}).items():
            pipeline.candidates[cid] = PromotionCandidate(
                cell_id=cdata["cell_id"],
                sensor_spec=cdata["sensor_spec"],
                score=cdata["score"],
                consistency=cdata["consistency"],
                avg_reward=cdata["avg_reward"],
                sample_count=cdata["sample_count"],
                created_tick=cdata.get("created_tick", 0),
                status=cdata.get("status", "pending"),
            )
        
        # Restore promoted sensors
        for sid, sdata in data.get("promoted", {}).items():
            pipeline.promoted[sid] = PromotedSensor(
                sensor_id=sdata["sensor_id"],
                source_cell=sdata["source_cell"],
                pattern_signature=sdata.get("pattern_signature", []),
                trust_score=sdata.get("trust_score", 0.0),
                promoted_tick=sdata.get("promoted_tick", 0),
                metadata=sdata.get("metadata", {}),
            )
        
        return pipeline

