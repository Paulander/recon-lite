"""Auto-Generated Detectors from Promoted Patterns.

Runtime loader that reads promoted patterns from weights/promoted/
and creates detector nodes using feature similarity matching.

Usage:
    from recon_lite_chess.patterns.auto_detectors import (
        load_promoted_patterns,
        create_auto_detector_nodes,
        inject_auto_detectors,
    )
    
    # Load patterns and inject into graph
    patterns = load_promoted_patterns()
    inject_auto_detectors(graph, patterns)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import chess

from .embeddings import encode_position

# Default paths
DEFAULT_PROMOTED_DIR = Path("weights/promoted")
DEFAULT_MANIFEST = DEFAULT_PROMOTED_DIR / "manifest.json"


@dataclass
class AutoDetector:
    """A detector created from a promoted pattern."""
    
    pattern_id: str
    pattern_signature: np.ndarray
    threshold: float
    consistency: float
    sample_count: int
    avg_reward: float
    detector_node_id: str
    action_node_id: str
    
    def match(self, board: chess.Board) -> float:
        """
        Check if board matches this pattern.
        
        Returns similarity score (0-1), or -1 if pattern has no signature.
        """
        if self.pattern_signature is None or len(self.pattern_signature) == 0:
            return -1.0
        
        # Encode current position
        embedding = encode_position(board)
        if embedding is None:
            return -1.0
        
        # Compute cosine similarity with pattern signature
        dot = np.dot(embedding.vector, self.pattern_signature)
        norm_board = np.linalg.norm(embedding.vector)
        norm_pattern = np.linalg.norm(self.pattern_signature)
        
        if norm_board == 0 or norm_pattern == 0:
            return 0.0
        
        similarity = dot / (norm_board * norm_pattern)
        return float(similarity)
    
    def detect(self, board: chess.Board) -> bool:
        """Return True if pattern is detected (similarity > threshold)."""
        return self.match(board) >= self.threshold
    
    def create_predicate(self) -> Callable:
        """Create a predicate function for use in graph nodes."""
        def predicate(env: Dict[str, Any]) -> bool:
            board = env.get("board")
            if board is None:
                return False
            return self.detect(board)
        return predicate


def load_pattern_file(path: Path) -> Optional[AutoDetector]:
    """Load a single pattern file into an AutoDetector."""
    try:
        with open(path) as f:
            data = json.load(f)
        
        signature = data.get("pattern_signature", [])
        if signature:
            signature = np.array(signature, dtype=np.float32)
        else:
            signature = np.array([], dtype=np.float32)
        
        return AutoDetector(
            pattern_id=data.get("pattern_id", "unknown"),
            pattern_signature=signature,
            threshold=data.get("threshold", 0.6),
            consistency=data.get("consistency", 0.0),
            sample_count=data.get("sample_count", 0),
            avg_reward=data.get("avg_reward", 0.0),
            detector_node_id=data.get("detector_node_id", f"Detector_{data.get('pattern_id', 'unknown')}"),
            action_node_id=data.get("action_node_id", f"Action_{data.get('pattern_id', 'unknown')}"),
        )
    except Exception as e:
        print(f"Warning: Could not load pattern from {path}: {e}")
        return None


def load_promoted_patterns(
    promoted_dir: Path = DEFAULT_PROMOTED_DIR,
) -> List[AutoDetector]:
    """
    Load all promoted patterns from the promoted directory.
    
    Returns list of AutoDetector objects.
    """
    detectors = []
    
    manifest_path = promoted_dir / "manifest.json"
    if not manifest_path.exists():
        return detectors
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load manifest: {e}")
        return detectors
    
    patterns = manifest.get("patterns", {})
    
    for pattern_id, pattern_info in patterns.items():
        pattern_file = pattern_info.get("file")
        if pattern_file:
            pattern_path = promoted_dir / pattern_file
            detector = load_pattern_file(pattern_path)
            if detector:
                detectors.append(detector)
    
    return detectors


def create_auto_detector_nodes(
    detectors: List[AutoDetector],
) -> List[Dict[str, Any]]:
    """
    Create node specifications for auto-detectors.
    
    Returns list of node specs that can be added to a graph.
    """
    from recon_lite.graph import Node, NodeType
    
    nodes = []
    
    for detector in detectors:
        # Create detector terminal node
        detector_node = Node(
            detector.detector_node_id,
            NodeType.TERMINAL,
            predicate=detector.create_predicate(),
            meta={
                "auto_generated": True,
                "pattern_id": detector.pattern_id,
                "consistency": detector.consistency,
                "fan_in_allowed": True,
            },
        )
        nodes.append({
            "node": detector_node,
            "type": "detector",
            "pattern_id": detector.pattern_id,
        })
        
        # Create action node (placeholder - actual move logic TBD)
        action_node = Node(
            detector.action_node_id,
            NodeType.SCRIPT,
            meta={
                "auto_generated": True,
                "pattern_id": detector.pattern_id,
                "layer": "tactical",
            },
        )
        nodes.append({
            "node": action_node,
            "type": "action",
            "pattern_id": detector.pattern_id,
        })
    
    return nodes


def inject_auto_detectors(
    graph,
    detectors: Optional[List[AutoDetector]] = None,
    connect_to: str = "GameRoot",
) -> int:
    """
    Inject auto-generated detectors into a graph.
    
    Args:
        graph: The ReCoN graph to modify
        detectors: List of detectors (loads from promoted/ if None)
        connect_to: Parent node to connect detectors under
        
    Returns:
        Number of detectors injected
    """
    from recon_lite.graph import LinkType
    
    if detectors is None:
        detectors = load_promoted_patterns()
    
    if not detectors:
        return 0
    
    injected = 0
    
    for detector in detectors:
        # Create detector node
        detector_node = type(graph.nodes.get(connect_to, None)).__class__(
            detector.detector_node_id,
            predicate=detector.create_predicate(),
            meta={
                "auto_generated": True,
                "pattern_id": detector.pattern_id,
                "consistency": detector.consistency,
                "fan_in_allowed": True,
            },
        ) if hasattr(graph, 'nodes') else None
        
        # Skip if we can't create nodes (graph structure unknown)
        if detector_node is None:
            continue
        
        try:
            # Add nodes
            graph.add_node(detector_node)
            
            # Connect to parent
            if connect_to in graph.nodes:
                graph.add_edge(connect_to, detector.detector_node_id, LinkType.SUB)
            
            injected += 1
        except Exception as e:
            print(f"Warning: Could not inject detector {detector.pattern_id}: {e}")
    
    return injected


def get_auto_detector_summary(
    promoted_dir: Path = DEFAULT_PROMOTED_DIR,
) -> Dict[str, Any]:
    """Get summary of available auto-detectors."""
    detectors = load_promoted_patterns(promoted_dir)
    
    return {
        "count": len(detectors),
        "detectors": [
            {
                "pattern_id": d.pattern_id,
                "consistency": d.consistency,
                "sample_count": d.sample_count,
                "avg_reward": d.avg_reward,
                "has_signature": len(d.pattern_signature) > 0,
            }
            for d in detectors
        ],
    }


if __name__ == "__main__":
    # Test loading
    summary = get_auto_detector_summary()
    print(f"Auto-detector summary:")
    print(f"  Total patterns: {summary['count']}")
    for d in summary['detectors']:
        print(f"  - {d['pattern_id']}: consistency={d['consistency']:.2f}, samples={d['sample_count']}")

