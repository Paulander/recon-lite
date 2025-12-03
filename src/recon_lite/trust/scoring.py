"""M5.3: Trust scoring for nodes and edges based on trace analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrustAction(str, Enum):
    """Recommended actions based on trust scores."""
    NONE = "none"           # No action needed
    FREEZE = "freeze"       # Disable plasticity for this component
    REMOVE = "remove"       # Candidate for removal (requires review)
    PROMOTE = "promote"     # Increase baseline weight


@dataclass
class TrustConfig:
    """Configuration for trust scoring."""
    # Trust score weights
    alpha_activation: float = 0.3    # Weight for activation count
    beta_reward: float = 0.4         # Weight for mean reward
    gamma_variance: float = 0.3      # Weight for variance penalty
    
    # Thresholds
    freeze_threshold: float = 0.3    # Trust below this -> freeze
    remove_threshold: float = 0.1    # Trust below this -> candidate removal
    promote_threshold: float = 0.8   # Trust above this -> promote
    
    # Minimum observations
    min_activations: int = 10        # Min activations before scoring
    min_generations: int = 3         # Min generations for actions
    
    # Promotion factor
    promotion_weight_factor: float = 1.2  # Multiply w_base by this


@dataclass
class NodeTrustScore:
    """Trust score for a single node."""
    node_id: str
    activation_count: int = 0
    total_reward: float = 0.0
    reward_samples: int = 0
    reward_variance: float = 0.0
    context_count: int = 0       # Number of unique contexts
    generations_tracked: int = 0
    
    # Computed
    trust_score: float = 0.0
    recommended_action: str = TrustAction.NONE.value
    
    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.reward_samples if self.reward_samples > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean_reward"] = self.mean_reward
        return d


@dataclass
class EdgeTrustScore:
    """Trust score for a single edge."""
    edge_key: str
    fire_count: int = 0
    contribution_total: float = 0.0
    contribution_samples: int = 0
    weight_history: List[float] = field(default_factory=list)
    generations_tracked: int = 0
    
    # Computed
    trust_score: float = 0.0
    recommended_action: str = TrustAction.NONE.value
    
    @property
    def mean_contribution(self) -> float:
        return self.contribution_total / self.contribution_samples if self.contribution_samples > 0 else 0.0
    
    @property
    def weight_stability(self) -> float:
        """Compute stability (1 - normalized variance of weight history)."""
        if len(self.weight_history) < 2:
            return 1.0
        mean = sum(self.weight_history) / len(self.weight_history)
        if mean == 0:
            return 1.0
        variance = sum((w - mean) ** 2 for w in self.weight_history) / len(self.weight_history)
        std = math.sqrt(variance)
        # Normalize by mean and invert
        cv = std / abs(mean) if mean != 0 else 0
        return max(0, 1 - cv)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean_contribution"] = self.mean_contribution
        d["weight_stability"] = self.weight_stability
        return d


@dataclass
class TrustReport:
    """Complete trust report for a graph."""
    node_scores: Dict[str, NodeTrustScore] = field(default_factory=dict)
    edge_scores: Dict[str, EdgeTrustScore] = field(default_factory=dict)
    generation: int = 0
    config: Optional[TrustConfig] = None
    
    def add_node_score(self, score: NodeTrustScore) -> None:
        self.node_scores[score.node_id] = score
    
    def add_edge_score(self, score: EdgeTrustScore) -> None:
        self.edge_scores[score.edge_key] = score
    
    def get_freeze_candidates(self) -> List[str]:
        """Get nodes/edges recommended for freezing."""
        candidates = []
        for node_id, score in self.node_scores.items():
            if score.recommended_action == TrustAction.FREEZE.value:
                candidates.append(f"node:{node_id}")
        for edge_key, score in self.edge_scores.items():
            if score.recommended_action == TrustAction.FREEZE.value:
                candidates.append(f"edge:{edge_key}")
        return candidates
    
    def get_remove_candidates(self) -> List[str]:
        """Get nodes/edges recommended for removal review."""
        candidates = []
        for node_id, score in self.node_scores.items():
            if score.recommended_action == TrustAction.REMOVE.value:
                candidates.append(f"node:{node_id}")
        for edge_key, score in self.edge_scores.items():
            if score.recommended_action == TrustAction.REMOVE.value:
                candidates.append(f"edge:{edge_key}")
        return candidates
    
    def get_promote_candidates(self) -> List[str]:
        """Get nodes/edges recommended for promotion."""
        candidates = []
        for node_id, score in self.node_scores.items():
            if score.recommended_action == TrustAction.PROMOTE.value:
                candidates.append(f"node:{node_id}")
        for edge_key, score in self.edge_scores.items():
            if score.recommended_action == TrustAction.PROMOTE.value:
                candidates.append(f"edge:{edge_key}")
        return candidates
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "config": asdict(self.config) if self.config else None,
            "node_scores": {k: v.to_dict() for k, v in self.node_scores.items()},
            "edge_scores": {k: v.to_dict() for k, v in self.edge_scores.items()},
            "summary": {
                "total_nodes": len(self.node_scores),
                "total_edges": len(self.edge_scores),
                "freeze_candidates": len(self.get_freeze_candidates()),
                "remove_candidates": len(self.get_remove_candidates()),
                "promote_candidates": len(self.get_promote_candidates()),
            },
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrustReport":
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        report = cls(generation=data.get("generation", 0))
        
        if data.get("config"):
            report.config = TrustConfig(**data["config"])
        
        for node_id, score_data in data.get("node_scores", {}).items():
            # Remove computed fields
            score_data.pop("mean_reward", None)
            score = NodeTrustScore(**score_data)
            report.add_node_score(score)
        
        for edge_key, score_data in data.get("edge_scores", {}).items():
            # Remove computed fields
            score_data.pop("mean_contribution", None)
            score_data.pop("weight_stability", None)
            score = EdgeTrustScore(**score_data)
            report.add_edge_score(score)
        
        return report


def _normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a value to [0, 1] range."""
    if max_val <= min_val:
        return 0.5
    return max(0, min(1, (value - min_val) / (max_val - min_val)))


def compute_node_trust(
    score: NodeTrustScore,
    config: TrustConfig,
    max_activations: int = 100,
    max_reward: float = 2.0,
) -> float:
    """
    Compute trust score for a node.
    
    Formula:
        trust = α * norm(activation_count) + β * norm(mean_reward) - γ * norm(variance)
    """
    if score.activation_count < config.min_activations:
        return 0.5  # Insufficient data, neutral trust
    
    # Normalize components
    activation_norm = _normalize(score.activation_count, 0, max_activations)
    reward_norm = _normalize(score.mean_reward, -max_reward, max_reward)
    variance_norm = _normalize(score.reward_variance, 0, max_reward ** 2)
    
    # Compute trust score
    trust = (
        config.alpha_activation * activation_norm +
        config.beta_reward * reward_norm -
        config.gamma_variance * variance_norm
    )
    
    # Clamp to [0, 1]
    return max(0, min(1, trust))


def compute_edge_trust(
    score: EdgeTrustScore,
    config: TrustConfig,
    max_fires: int = 100,
    max_contribution: float = 2.0,
) -> float:
    """
    Compute trust score for an edge.
    
    Uses fire count, mean contribution, and weight stability.
    """
    if score.fire_count < config.min_activations:
        return 0.5  # Insufficient data, neutral trust
    
    # Normalize components
    fire_norm = _normalize(score.fire_count, 0, max_fires)
    contrib_norm = _normalize(score.mean_contribution, -max_contribution, max_contribution)
    stability = score.weight_stability
    
    # Compute trust score (stability is already [0, 1])
    trust = (
        config.alpha_activation * fire_norm +
        config.beta_reward * contrib_norm +
        config.gamma_variance * (stability - 0.5)  # Stability bonus/penalty
    )
    
    # Clamp to [0, 1]
    return max(0, min(1, trust))


def recommend_action(
    trust_score: float,
    generations_tracked: int,
    config: TrustConfig,
) -> str:
    """
    Recommend an action based on trust score.
    
    Returns TrustAction value.
    """
    if generations_tracked < config.min_generations:
        return TrustAction.NONE.value
    
    if trust_score < config.remove_threshold:
        return TrustAction.REMOVE.value
    
    if trust_score < config.freeze_threshold:
        return TrustAction.FREEZE.value
    
    if trust_score > config.promote_threshold:
        return TrustAction.PROMOTE.value
    
    return TrustAction.NONE.value


def compute_trust_report(
    trace_data: List[Dict[str, Any]],
    consolidation_state: Optional[Dict[str, Any]] = None,
    config: Optional[TrustConfig] = None,
    previous_report: Optional[TrustReport] = None,
) -> TrustReport:
    """
    Compute trust scores from trace data and consolidation state.
    
    Args:
        trace_data: List of trace frames (from viz/debug JSON)
        consolidation_state: Optional consolidation state for w_base history
        config: Trust configuration
        previous_report: Previous report for generation tracking
    
    Returns:
        TrustReport with computed scores
    """
    config = config or TrustConfig()
    generation = (previous_report.generation + 1) if previous_report else 1
    
    report = TrustReport(generation=generation, config=config)
    
    # Initialize from previous report if available
    if previous_report:
        for node_id, prev_score in previous_report.node_scores.items():
            score = NodeTrustScore(
                node_id=node_id,
                activation_count=prev_score.activation_count,
                total_reward=prev_score.total_reward,
                reward_samples=prev_score.reward_samples,
                reward_variance=prev_score.reward_variance,
                context_count=prev_score.context_count,
                generations_tracked=prev_score.generations_tracked + 1,
            )
            report.add_node_score(score)
        
        for edge_key, prev_score in previous_report.edge_scores.items():
            score = EdgeTrustScore(
                edge_key=edge_key,
                fire_count=prev_score.fire_count,
                contribution_total=prev_score.contribution_total,
                contribution_samples=prev_score.contribution_samples,
                weight_history=prev_score.weight_history.copy(),
                generations_tracked=prev_score.generations_tracked + 1,
            )
            report.add_edge_score(score)
    
    # Process trace data
    for frame in trace_data:
        nodes = frame.get("nodes", {})
        env = frame.get("env", {})
        reward_tick = env.get("m3_reward_tick") or env.get("reward_tick") or 0.0
        
        # Update node scores
        for node_id, state in nodes.items():
            if state in ("TRUE", "CONFIRMED", "WAITING"):
                if node_id not in report.node_scores:
                    report.node_scores[node_id] = NodeTrustScore(
                        node_id=node_id,
                        generations_tracked=1,
                    )
                score = report.node_scores[node_id]
                score.activation_count += 1
                score.total_reward += reward_tick
                score.reward_samples += 1
        
        # Update edge scores from weight deltas
        deltas = env.get("m3_weight_deltas", {})
        for edge_key, delta in deltas.items():
            if edge_key not in report.edge_scores:
                report.edge_scores[edge_key] = EdgeTrustScore(
                    edge_key=edge_key,
                    generations_tracked=1,
                )
            score = report.edge_scores[edge_key]
            score.fire_count += 1
            score.contribution_total += delta
            score.contribution_samples += 1
    
    # Add w_base history from consolidation state
    if consolidation_state:
        w_base = consolidation_state.get("w_base", {})
        for edge_key, weight in w_base.items():
            if edge_key not in report.edge_scores:
                report.edge_scores[edge_key] = EdgeTrustScore(
                    edge_key=edge_key,
                    generations_tracked=1,
                )
            score = report.edge_scores[edge_key]
            score.weight_history.append(weight)
    
    # Compute trust scores and recommendations
    for node_id, score in report.node_scores.items():
        score.trust_score = compute_node_trust(score, config)
        score.recommended_action = recommend_action(
            score.trust_score,
            score.generations_tracked,
            config,
        )
    
    for edge_key, score in report.edge_scores.items():
        score.trust_score = compute_edge_trust(score, config)
        score.recommended_action = recommend_action(
            score.trust_score,
            score.generations_tracked,
            config,
        )
    
    return report

