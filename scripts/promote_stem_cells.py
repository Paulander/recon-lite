#!/usr/bin/env python3
"""
Stem Cell Promotion Pipeline.

Analyzes stem cell data from training traces and generates
FeatureHub registration code for approved sensor candidates.

Process:
1. Load stem cell states from JSONL traces
2. Cluster similar patterns via feature correlation
3. Evaluate candidate consistency and predictive value
4. Generate Python code for promoted sensors
5. Output review proposals with human-readable explanations

Usage:
    uv run python scripts/promote_stem_cells.py \
        --traces reports/curriculum/*/traces.jsonl \
        --output proposals/sensors/ \
        --min-consistency 0.6 \
        --min-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class StemCellCandidate:
    """A stem cell that may be promoted to a full sensor."""
    cell_id: str
    pattern_signature: List[float]
    consistency_score: float
    sample_count: int
    avg_reward: float
    source_phases: List[str]  # Training phases this cell was active in
    source_endgames: List[str]  # Endgame types it appeared in
    sample_fens: List[str]  # Example positions
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "pattern_signature": self.pattern_signature,
            "consistency_score": self.consistency_score,
            "sample_count": self.sample_count,
            "avg_reward": self.avg_reward,
            "source_phases": self.source_phases,
            "source_endgames": self.source_endgames,
            "sample_fens": self.sample_fens[:5],  # Top 5 examples
            "metadata": self.metadata,
        }


@dataclass
class PromotionProposal:
    """A proposed sensor promotion for human review."""
    candidate: StemCellCandidate
    proposed_name: str
    category: str
    description: str
    code_template: str
    confidence: float
    review_notes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed_name": self.proposed_name,
            "category": self.category,
            "description": self.description,
            "confidence": self.confidence,
            "review_notes": self.review_notes,
            "candidate": self.candidate.to_dict(),
        }
    
    def to_markdown(self) -> str:
        """Generate markdown for human review."""
        md = f"""## Proposal: `{self.proposed_name}`

**Category:** {self.category}
**Confidence:** {self.confidence:.1%}
**Based on:** {self.candidate.cell_id}

### Description
{self.description}

### Statistics
- Samples: {self.candidate.sample_count}
- Consistency: {self.candidate.consistency_score:.2f}
- Avg Reward: {self.candidate.avg_reward:.3f}
- Source Phases: {', '.join(self.candidate.source_phases)}
- Source Endgames: {', '.join(self.candidate.source_endgames)}

### Review Notes
"""
        for note in self.review_notes:
            md += f"- {note}\n"
        
        md += f"""
### Example Positions
"""
        for i, fen in enumerate(self.candidate.sample_fens[:3]):
            md += f"{i+1}. `{fen}`\n"
        
        md += f"""
### Code Template
```python
{self.code_template}
```

---
"""
        return md


def load_traces(trace_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load all trace files and extract stem cell data."""
    all_data = []
    
    for path in trace_paths:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_data.append(record)
                except json.JSONDecodeError:
                    continue
    
    return all_data


def extract_stem_cell_data(traces: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract stem cell observations from traces, grouped by cell_id."""
    cell_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for record in traces:
        # Look for stem cell data in various formats
        stem_info = record.get("stem_cells") or record.get("meta", {}).get("stem_cells")
        
        if stem_info:
            if isinstance(stem_info, dict):
                for cell_id, cell_record in stem_info.items():
                    cell_data[cell_id].append({
                        "fen": record.get("board_fen") or record.get("fen"),
                        "reward": record.get("reward_tick") or cell_record.get("reward"),
                        "features": cell_record.get("features"),
                        "phase": record.get("meta", {}).get("phase"),
                        "endgame": record.get("meta", {}).get("endgame"),
                    })
            elif isinstance(stem_info, list):
                for item in stem_info:
                    cell_id = item.get("cell_id")
                    if cell_id:
                        cell_data[cell_id].append({
                            "fen": record.get("board_fen"),
                            "reward": item.get("reward"),
                            "features": item.get("features"),
                            "phase": record.get("meta", {}).get("phase"),
                            "endgame": record.get("meta", {}).get("endgame"),
                        })
    
    return dict(cell_data)


def analyze_candidate(cell_id: str, observations: List[Dict[str, Any]]) -> Optional[StemCellCandidate]:
    """Analyze observations for a single stem cell."""
    if len(observations) < 10:
        return None
    
    # Collect features
    feature_samples = []
    rewards = []
    fens = []
    phases = set()
    endgames = set()
    
    for obs in observations:
        if obs.get("features"):
            feature_samples.append(obs["features"])
        if obs.get("reward") is not None:
            rewards.append(obs["reward"])
        if obs.get("fen"):
            fens.append(obs["fen"])
        if obs.get("phase"):
            phases.add(obs["phase"])
        if obs.get("endgame"):
            endgames.add(obs["endgame"])
    
    if not HAS_NUMPY or len(feature_samples) < 10:
        # Return basic stats without feature analysis
        return StemCellCandidate(
            cell_id=cell_id,
            pattern_signature=[],
            consistency_score=0.5,  # Unknown
            sample_count=len(observations),
            avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            source_phases=list(phases),
            source_endgames=list(endgames),
            sample_fens=fens[:10],
        )
    
    # Analyze feature consistency
    try:
        features = np.array(feature_samples)
        centroid = np.mean(features, axis=0)
        distances = np.linalg.norm(features - centroid, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        consistency = 1.0 - min(1.0, avg_distance / (max_distance + 1e-6))
        pattern_signature = centroid.tolist()
    except Exception:
        consistency = 0.5
        pattern_signature = []
    
    return StemCellCandidate(
        cell_id=cell_id,
        pattern_signature=pattern_signature,
        consistency_score=consistency,
        sample_count=len(observations),
        avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        source_phases=list(phases),
        source_endgames=list(endgames),
        sample_fens=fens[:10],
    )


def generate_sensor_name(candidate: StemCellCandidate) -> str:
    """Generate a proposed sensor name based on characteristics."""
    # Use cell_id as base
    base_name = candidate.cell_id.replace("stem_", "discovered_")
    
    # Add context from source
    if candidate.source_endgames:
        endgame = candidate.source_endgames[0].lower()
        if endgame not in base_name:
            base_name = f"{endgame}_{base_name}"
    
    return base_name


def infer_category(candidate: StemCellCandidate) -> str:
    """Infer the likely category for this sensor."""
    # Based on source and reward patterns
    endgames = [e.lower() for e in candidate.source_endgames]
    
    if any("krk" in e or "kqk" in e for e in endgames):
        return "GEOMETRIC"  # Likely mating patterns
    elif any("kpk" in e for e in endgames):
        return "POSITIONAL"  # Pawn-related
    elif candidate.avg_reward > 0.5:
        return "TACTICAL"  # High reward = tactical opportunity
    else:
        return "DYNAMIC"  # Default


def generate_code_template(candidate: StemCellCandidate, name: str) -> str:
    """Generate a code template for the promoted sensor."""
    category = infer_category(candidate)
    
    return f'''def compute_{name}(board: chess.Board, computed: Dict[str, float]) -> float:
    """
    Auto-discovered sensor from stem cell {candidate.cell_id}.
    
    Discovered in: {", ".join(candidate.source_endgames)}
    Pattern consistency: {candidate.consistency_score:.2f}
    Average reward correlation: {candidate.avg_reward:.3f}
    
    TODO: Review and refine this implementation based on pattern analysis.
    """
    # Pattern signature (centroid of discovered feature vectors):
    # {candidate.pattern_signature[:5]}...
    
    # Placeholder implementation - needs human refinement
    score = 0.0
    
    # Example logic based on discovered context
    # ... analyze board features that correlate with pattern ...
    
    return max(0.0, min(1.0, score))


# Registration (add to hub.py):
# hub.register(FeatureDefinition(
#     name="{name}",
#     category=FeatureCategory.{category},
#     compute_fn=compute_{name},
#     description="Auto-discovered: {candidate.cell_id}",
# ))
'''


def generate_review_notes(candidate: StemCellCandidate) -> List[str]:
    """Generate review notes for human evaluation."""
    notes = []
    
    if candidate.consistency_score > 0.8:
        notes.append("✅ High consistency - pattern is well-defined")
    elif candidate.consistency_score > 0.6:
        notes.append("⚠️ Moderate consistency - may need refinement")
    else:
        notes.append("❌ Low consistency - consider rejecting or more training")
    
    if candidate.sample_count > 100:
        notes.append(f"✅ Good sample size ({candidate.sample_count})")
    elif candidate.sample_count > 50:
        notes.append(f"⚠️ Moderate sample size ({candidate.sample_count})")
    else:
        notes.append(f"❌ Low sample size ({candidate.sample_count})")
    
    if candidate.avg_reward > 0.3:
        notes.append(f"✅ Strong reward correlation ({candidate.avg_reward:.3f})")
    elif candidate.avg_reward > 0.1:
        notes.append(f"⚠️ Moderate reward correlation ({candidate.avg_reward:.3f})")
    else:
        notes.append(f"❌ Weak reward correlation ({candidate.avg_reward:.3f})")
    
    if len(candidate.source_endgames) > 1:
        notes.append("✅ Pattern appears across multiple endgames - may be general")
    
    return notes


def create_proposal(candidate: StemCellCandidate) -> PromotionProposal:
    """Create a full promotion proposal for a candidate."""
    name = generate_sensor_name(candidate)
    category = infer_category(candidate)
    code = generate_code_template(candidate, name)
    notes = generate_review_notes(candidate)
    
    # Compute confidence score
    confidence = (
        candidate.consistency_score * 0.4 +
        min(1.0, candidate.sample_count / 100) * 0.3 +
        min(1.0, abs(candidate.avg_reward) / 0.5) * 0.3
    )
    
    description = f"Auto-discovered pattern from {candidate.cell_id}. "
    description += f"Found in {', '.join(candidate.source_endgames) or 'unknown'} positions. "
    if candidate.avg_reward > 0:
        description += f"Correlates with positive outcomes (avg reward: {candidate.avg_reward:.3f})."
    else:
        description += f"Correlates with negative outcomes (avg reward: {candidate.avg_reward:.3f})."
    
    return PromotionProposal(
        candidate=candidate,
        proposed_name=name,
        category=category,
        description=description,
        code_template=code,
        confidence=confidence,
        review_notes=notes,
    )


def main():
    parser = argparse.ArgumentParser(description="Stem Cell Promotion Pipeline")
    parser.add_argument("--traces", type=Path, nargs="+", required=True,
                       help="JSONL trace files to analyze")
    parser.add_argument("--output", type=Path, default=Path("proposals/sensors"),
                       help="Output directory for proposals")
    parser.add_argument("--min-consistency", type=float, default=0.5,
                       help="Minimum consistency score to consider")
    parser.add_argument("--min-samples", type=int, default=30,
                       help="Minimum samples to consider")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of top candidates to propose")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both",
                       help="Output format")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stem Cell Promotion Pipeline")
    print("=" * 60)
    
    # Expand glob patterns
    trace_files = []
    for pattern in args.traces:
        if "*" in str(pattern):
            trace_files.extend(Path(".").glob(str(pattern)))
        else:
            trace_files.append(pattern)
    
    print(f"Loading {len(trace_files)} trace files...")
    traces = load_traces(trace_files)
    print(f"Loaded {len(traces)} records")
    
    # Extract stem cell data
    print("\nExtracting stem cell data...")
    cell_data = extract_stem_cell_data(traces)
    print(f"Found {len(cell_data)} unique stem cells")
    
    if not cell_data:
        print("\nNo stem cell data found in traces.")
        print("Run training with --stem-cells flag to collect data.")
        return
    
    # Analyze candidates
    print("\nAnalyzing candidates...")
    candidates = []
    for cell_id, observations in cell_data.items():
        candidate = analyze_candidate(cell_id, observations)
        if candidate:
            if (candidate.consistency_score >= args.min_consistency and
                candidate.sample_count >= args.min_samples):
                candidates.append(candidate)
    
    print(f"Found {len(candidates)} viable candidates")
    
    if not candidates:
        print("\nNo candidates met the threshold criteria.")
        print(f"Criteria: consistency >= {args.min_consistency}, samples >= {args.min_samples}")
        return
    
    # Sort by score (consistency * sample_count * abs(reward))
    candidates.sort(
        key=lambda c: c.consistency_score * min(c.sample_count / 100, 1.0) * (1 + abs(c.avg_reward)),
        reverse=True
    )
    
    # Take top-k
    top_candidates = candidates[:args.top_k]
    
    # Generate proposals
    print(f"\nGenerating proposals for top {len(top_candidates)} candidates...")
    proposals = [create_proposal(c) for c in top_candidates]
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output results
    if args.format in ("json", "both"):
        json_path = args.output / f"proposals_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([p.to_dict() for p in proposals], f, indent=2)
        print(f"\nJSON proposals: {json_path}")
    
    if args.format in ("markdown", "both"):
        md_path = args.output / f"proposals_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write("# Stem Cell Promotion Proposals\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Analyzed: {len(trace_files)} trace files, {len(traces)} records\n")
            f.write(f"Candidates evaluated: {len(cell_data)}\n")
            f.write(f"Proposals generated: {len(proposals)}\n\n")
            f.write("---\n\n")
            for proposal in proposals:
                f.write(proposal.to_markdown())
        print(f"Markdown proposals: {md_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Promotion Pipeline Complete")
    print("=" * 60)
    print(f"\nTop {len(proposals)} proposals:")
    for i, p in enumerate(proposals, 1):
        status = "✅" if p.confidence > 0.7 else "⚠️" if p.confidence > 0.5 else "❓"
        print(f"  {i}. {status} {p.proposed_name} (confidence: {p.confidence:.1%})")
    
    print(f"\nReview proposals in: {args.output}")
    print("Manually review and add approved sensors to features/sensors_v2.py")


if __name__ == "__main__":
    main()

