"""M5.1: Binding descriptors for motif extraction and pattern discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class MotifType(str, Enum):
    """Categories of extracted motifs."""
    TACTICAL = "tactical"      # Fork, pin, skewer, hanging piece
    ENDGAME = "endgame"        # Opposition, cutoff, passed pawn
    STRUCTURAL = "structural"  # Pawn chains, outposts, open files
    POSITIONAL = "positional"  # King safety, piece activity
    UNKNOWN = "unknown"


@dataclass
class BindingDescriptor:
    """
    Compact descriptor for an 'interesting patch' extracted from a trace.
    
    These are used to:
    - Identify recurring patterns across games
    - Cluster similar situations for script proposal
    - Track which patterns lead to positive/negative outcomes
    """
    dtype: str                          # MotifType value
    pattern_key: str                    # e.g., "fork_knight", "king_opposition"
    context: Dict[str, Any]             # Board features at extraction
    active_nodes: List[str]             # Nodes active when extracted
    outcome_score: float                # Reward contribution
    source_tick: int                    # Tick ID from trace
    source_episode: str                 # Episode ID from trace
    fen: Optional[str] = None           # Board position (optional)
    move: Optional[str] = None          # Move played (optional)
    reward_tick: Optional[float] = None # Raw reward signal
    confidence: float = 0.0             # Extraction confidence (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BindingDescriptor":
        """Create from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "BindingDescriptor":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class MotifDataset:
    """
    Collection of extracted motifs with indexing and filtering.
    """
    motifs: List[BindingDescriptor] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add(self, motif: BindingDescriptor) -> None:
        """Add a motif to the dataset."""
        self.motifs.append(motif)
    
    def filter_by_type(self, dtype: str) -> List[BindingDescriptor]:
        """Filter motifs by type."""
        return [m for m in self.motifs if m.dtype == dtype]
    
    def filter_by_pattern(self, pattern_key: str) -> List[BindingDescriptor]:
        """Filter motifs by pattern key."""
        return [m for m in self.motifs if m.pattern_key == pattern_key]
    
    def filter_by_outcome(self, min_score: float = 0.0) -> List[BindingDescriptor]:
        """Filter motifs by minimum outcome score."""
        return [m for m in self.motifs if m.outcome_score >= min_score]
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> List[BindingDescriptor]:
        """Filter motifs by minimum confidence."""
        return [m for m in self.motifs if m.confidence >= min_confidence]
    
    def group_by_pattern(self) -> Dict[str, List[BindingDescriptor]]:
        """Group motifs by pattern key."""
        groups: Dict[str, List[BindingDescriptor]] = {}
        for m in self.motifs:
            if m.pattern_key not in groups:
                groups[m.pattern_key] = []
            groups[m.pattern_key].append(m)
        return groups
    
    def group_by_type(self) -> Dict[str, List[BindingDescriptor]]:
        """Group motifs by type."""
        groups: Dict[str, List[BindingDescriptor]] = {}
        for m in self.motifs:
            if m.dtype not in groups:
                groups[m.dtype] = []
            groups[m.dtype].append(m)
        return groups
    
    def statistics(self) -> Dict[str, Any]:
        """Compute basic statistics about the dataset."""
        if not self.motifs:
            return {"count": 0}
        
        type_counts = {}
        pattern_counts = {}
        total_reward = 0.0
        positive_count = 0
        
        for m in self.motifs:
            type_counts[m.dtype] = type_counts.get(m.dtype, 0) + 1
            pattern_counts[m.pattern_key] = pattern_counts.get(m.pattern_key, 0) + 1
            total_reward += m.outcome_score
            if m.outcome_score > 0:
                positive_count += 1
        
        return {
            "count": len(self.motifs),
            "type_counts": type_counts,
            "pattern_counts": pattern_counts,
            "avg_outcome": total_reward / len(self.motifs),
            "positive_ratio": positive_count / len(self.motifs),
        }
    
    def save(self, path: Path) -> None:
        """Save dataset to JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Write metadata as first line
            f.write(json.dumps({"_metadata": self.metadata}) + "\n")
            for m in self.motifs:
                f.write(m.to_json() + "\n")
    
    @classmethod
    def load(cls, path: Path) -> "MotifDataset":
        """Load dataset from JSONL file."""
        dataset = cls()
        path = Path(path)
        if not path.exists():
            return dataset
        
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "_metadata" in data:
                    dataset.metadata = data["_metadata"]
                else:
                    dataset.add(BindingDescriptor.from_dict(data))
        
        return dataset
    
    def __len__(self) -> int:
        return len(self.motifs)
    
    def __iter__(self):
        return iter(self.motifs)

