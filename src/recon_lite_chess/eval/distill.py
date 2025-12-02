"""
Distillation Module (M4 Stub)

This module provides the interface for training and using a distilled
evaluation function that learns from Stockfish evaluations.

Currently a stub - actual implementation will be added in a future milestone.

The idea is to:
1. Collect (position, stockfish_eval) pairs from traces
2. Train a lightweight neural network to predict the eval
3. Use the trained model for fast evaluation during gameplay

This allows us to get Stockfish-quality evaluations at heuristic speeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    validation_split: float = 0.1

    # Model architecture (placeholder)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1

    # Data parameters
    min_samples: int = 1000
    max_samples: int = 100000

    # Output
    model_path: Optional[Path] = None


@dataclass
class DistillationSample:
    """A single training sample for distillation."""

    fen: str
    stockfish_eval: float  # In pawn units
    heuristic_eval: float  # For comparison
    depth: int = 0  # Stockfish depth used
    meta: Dict[str, Any] = field(default_factory=dict)


class DistillationDataset:
    """
    Dataset for distillation training.

    Collects (position, eval) pairs from traces and prepares them for training.
    """

    def __init__(self, config: Optional[DistillationConfig] = None):
        self.config = config or DistillationConfig()
        self.samples: List[DistillationSample] = []

    def add_sample(
        self,
        fen: str,
        stockfish_eval: float,
        heuristic_eval: float = 0.0,
        depth: int = 0,
    ) -> None:
        """Add a training sample."""
        if len(self.samples) >= self.config.max_samples:
            return

        self.samples.append(
            DistillationSample(
                fen=fen,
                stockfish_eval=stockfish_eval,
                heuristic_eval=heuristic_eval,
                depth=depth,
            )
        )

    def load_from_traces(self, trace_paths: List[Path]) -> int:
        """
        Load samples from trace files.

        Args:
            trace_paths: List of JSONL trace file paths

        Returns:
            Number of samples loaded
        """
        # Stub implementation
        raise NotImplementedError(
            "Distillation dataset loading not yet implemented. "
            "This will be added in a future milestone."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_train_val_split(
        self,
    ) -> Tuple[List[DistillationSample], List[DistillationSample]]:
        """Split dataset into training and validation sets."""
        if not self.samples:
            return [], []

        split_idx = int(len(self.samples) * (1 - self.config.validation_split))
        return self.samples[:split_idx], self.samples[split_idx:]


class DistilledEvaluator:
    """
    Distilled evaluation model interface.

    This is a stub that will be implemented with actual ML model
    in a future milestone.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self._model = None  # Placeholder for actual model

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        raise NotImplementedError(
            "Distilled model loading not yet implemented. "
            "This will be added in a future milestone."
        )

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        raise NotImplementedError(
            "Distilled model saving not yet implemented. "
            "This will be added in a future milestone."
        )

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a position using the distilled model.

        Args:
            board: Chess board to evaluate

        Returns:
            Evaluation in pawn units (positive = white advantage)
        """
        raise NotImplementedError(
            "Distilled evaluation not yet implemented. "
            "Use heuristic or Stockfish evaluation instead."
        )

    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None


def train_distilled_eval(
    dataset: DistillationDataset,
    config: Optional[DistillationConfig] = None,
) -> DistilledEvaluator:
    """
    Train a distilled evaluation model.

    Args:
        dataset: Training dataset
        config: Training configuration

    Returns:
        Trained DistilledEvaluator

    Raises:
        NotImplementedError: This is a stub
    """
    raise NotImplementedError(
        "Distillation training not yet implemented. "
        "This will be added in a future milestone when GPU support is available."
    )


def collect_distillation_data(
    trace_paths: List[Path],
    stockfish_path: str,
    output_path: Path,
    depth: int = 10,
    max_samples: int = 10000,
) -> int:
    """
    Collect distillation data by evaluating positions from traces with Stockfish.

    Args:
        trace_paths: List of JSONL trace file paths
        stockfish_path: Path to Stockfish executable
        output_path: Output path for collected data
        depth: Stockfish search depth
        max_samples: Maximum number of samples to collect

    Returns:
        Number of samples collected

    Raises:
        NotImplementedError: This is a stub
    """
    raise NotImplementedError(
        "Distillation data collection not yet implemented. "
        "This will be added in a future milestone."
    )

