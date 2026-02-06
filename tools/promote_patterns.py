#!/usr/bin/env python3
"""Pattern Promotion Pipeline for Stem Cell Discoveries.

Reads specialized stem cell patterns and converts them into permanent
graph nodes that can be loaded at runtime.

Usage:
    # Promote all ready patterns
    uv run python tools/promote_patterns.py
    
    # Promote from specific stem cell file
    uv run python tools/promote_patterns.py \
        --stem-cells weights/latest/stem_cells.json \
        --output-dir weights/promoted
    
    # Dry run (show what would be promoted)
    uv run python tools/promote_patterns.py --dry-run
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite.nodes.stem_cell import StemCellManager, StemCellState


@dataclass
class PromotedPattern:
    """A pattern promoted from stem cell to permanent detector."""
    
    pattern_id: str
    source_cell: str
    pattern_signature: List[float]
    threshold: float
    consistency: float
    sample_count: int
    avg_reward: float
    sample_fens: List[str]
    promoted_at: str
    detector_node_id: str
    action_node_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_spec(cls, spec: Dict[str, Any], pattern_id: str) -> "PromotedPattern":
        """Create from stem cell sensor spec."""
        meta = spec.get("metadata", {})
        return cls(
            pattern_id=pattern_id,
            source_cell=spec.get("source_cell", "unknown"),
            pattern_signature=spec.get("pattern_signature", []),
            threshold=spec.get("threshold", 0.6),
            consistency=spec.get("consistency", 0.0),
            sample_count=spec.get("sample_count", 0),
            avg_reward=meta.get("avg_reward", 0.0),
            sample_fens=meta.get("sample_fens", []),
            promoted_at=datetime.now().isoformat(),
            detector_node_id=f"Detector_{pattern_id}",
            action_node_id=f"Action_{pattern_id}",
        )


class PatternPromoter:
    """Promotes stem cell patterns to permanent graph nodes."""
    
    def __init__(
        self,
        stem_cells_path: Path,
        output_dir: Path,
        min_consistency: float = 0.6,
        verbose: bool = True,
    ):
        self.stem_cells_path = stem_cells_path
        self.output_dir = output_dir
        self.min_consistency = min_consistency
        self.verbose = verbose
        self.manifest_path = output_dir / "manifest.json"
        
    def load_stem_cells(self) -> Optional[StemCellManager]:
        """Load stem cell state from file."""
        if not self.stem_cells_path.exists():
            if self.verbose:
                print(f"No stem cells file found at {self.stem_cells_path}")
            return None
        
        try:
            return StemCellManager.load(self.stem_cells_path)
        except Exception as e:
            if self.verbose:
                print(f"Error loading stem cells: {e}")
            return None
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load or create promotion manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "version": "1.0",
            "patterns": {},
            "promoted_count": 0,
            "last_promotion": None,
        }
    
    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save promotion manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    
    def get_candidates(self, manager: StemCellManager) -> List[Tuple[str, Dict[str, Any]]]:
        """Get stem cells ready for promotion."""
        candidates = []
        
        for cell_id, cell in manager.cells.items():
            # Check if cell is specialized or candidate with enough consistency
            if cell.state == StemCellState.SPECIALIZED:
                # Already specialized - get the spec
                if cell.pattern_signature is not None:
                    spec = {
                        "sensor_id": f"sensor_{cell_id}",
                        "type": "pattern_detector",
                        "pattern_signature": cell.pattern_signature,
                        "threshold": cell.config.specialization_threshold,
                        "sample_count": len(cell.samples),
                        "consistency": cell.trust_score,
                        "source_cell": cell_id,
                        "metadata": {
                            "avg_reward": sum(s.reward for s in cell.samples) / max(1, len(cell.samples)),
                            "sample_fens": [s.fen for s in cell.samples[:10]],
                        },
                    }
                    if spec["consistency"] >= self.min_consistency:
                        candidates.append((cell_id, spec))
            
            elif cell.state == StemCellState.CANDIDATE:
                # Try to specialize
                spec = cell.specialize()
                if spec and spec.get("consistency", 0) >= self.min_consistency:
                    candidates.append((cell_id, spec))
        
        return candidates
    
    def promote_pattern(
        self,
        cell_id: str,
        spec: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> Optional[PromotedPattern]:
        """Promote a single pattern to permanent storage."""
        # Generate unique pattern ID
        pattern_num = manifest["promoted_count"] + 1
        pattern_id = f"pattern_{pattern_num:04d}"
        
        # Check if already promoted (by source cell)
        for existing_id, existing in manifest["patterns"].items():
            if existing.get("source_cell") == cell_id:
                if self.verbose:
                    print(f"  Skipping {cell_id}: already promoted as {existing_id}")
                return None
        
        # Create promoted pattern
        promoted = PromotedPattern.from_spec(spec, pattern_id)
        
        # Save individual pattern file
        pattern_path = self.output_dir / f"{pattern_id}.json"
        with open(pattern_path, "w") as f:
            json.dump(promoted.to_dict(), f, indent=2)
        
        # Update manifest
        manifest["patterns"][pattern_id] = {
            "source_cell": promoted.source_cell,
            "consistency": promoted.consistency,
            "sample_count": promoted.sample_count,
            "promoted_at": promoted.promoted_at,
            "file": pattern_path.name,
        }
        manifest["promoted_count"] = pattern_num
        manifest["last_promotion"] = promoted.promoted_at
        
        if self.verbose:
            print(f"  Promoted {cell_id} -> {pattern_id}")
            print(f"    Consistency: {promoted.consistency:.2f}")
            print(f"    Samples: {promoted.sample_count}")
            print(f"    Avg reward: {promoted.avg_reward:.2f}")
        
        return promoted
    
    def run(self, dry_run: bool = False) -> List[PromotedPattern]:
        """Run the promotion pipeline."""
        if self.verbose:
            print("=" * 60)
            print("Pattern Promotion Pipeline")
            print("=" * 60)
            print(f"Stem cells: {self.stem_cells_path}")
            print(f"Output dir: {self.output_dir}")
            print(f"Min consistency: {self.min_consistency}")
            print()
        
        # Load stem cells
        manager = self.load_stem_cells()
        if manager is None:
            return []
        
        if self.verbose:
            stats = manager.stats()
            print(f"Loaded stem cells: {stats['total_cells']} total")
            print(f"  By state: {stats['by_state']}")
            print()
        
        # Get promotion candidates
        candidates = self.get_candidates(manager)
        
        if self.verbose:
            print(f"Found {len(candidates)} candidates for promotion")
            print()
        
        if not candidates:
            return []
        
        if dry_run:
            if self.verbose:
                print("DRY RUN - would promote:")
                for cell_id, spec in candidates:
                    print(f"  {cell_id}: consistency={spec.get('consistency', 0):.2f}")
            return []
        
        # Load manifest
        manifest = self.load_manifest()
        
        # Promote each candidate
        promoted = []
        for cell_id, spec in candidates:
            result = self.promote_pattern(cell_id, spec, manifest)
            if result:
                promoted.append(result)
        
        # Save updated manifest
        self.save_manifest(manifest)
        
        if self.verbose:
            print()
            print(f"Promoted {len(promoted)} patterns")
            print(f"Total patterns in manifest: {manifest['promoted_count']}")
        
        return promoted


def main():
    parser = argparse.ArgumentParser(
        description="Promote stem cell patterns to permanent graph nodes"
    )
    parser.add_argument(
        "--stem-cells",
        type=str,
        default="weights/latest/stem_cells.json",
        help="Path to stem cells state file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/promoted",
        help="Directory to save promoted patterns",
    )
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.6,
        help="Minimum consistency threshold for promotion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be promoted without saving",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()
    
    promoter = PatternPromoter(
        stem_cells_path=Path(args.stem_cells),
        output_dir=Path(args.output_dir),
        min_consistency=args.min_consistency,
        verbose=not args.quiet,
    )
    
    promoted = promoter.run(dry_run=args.dry_run)
    
    # Print summary for script parsing
    print(f"\nPROMOTION_RESULT: promoted={len(promoted)}")


if __name__ == "__main__":
    main()

