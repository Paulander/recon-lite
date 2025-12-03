"""
Distillation Module (M7)

This module provides the interface for training and using a distilled
evaluation function that learns from Stockfish evaluations.

The workflow is:
1. Collect (position, stockfish_eval) pairs with tools/collect_stockfish_evals.py
2. Train a lightweight neural network with tools/train_distilled_eval.py
3. Load the trained model here for fast evaluation during gameplay

This allows Stockfish-quality evaluations at heuristic speeds (<1ms per position).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

from .features import extract_features, features_to_tensor, FEATURE_COUNT


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    validation_split: float = 0.1

    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
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

    def load_from_jsonl(self, path: Path) -> int:
        """
        Load samples from a JSONL file.

        Args:
            path: Path to JSONL file (from collect_stockfish_evals.py)

        Returns:
            Number of samples loaded
        """
        import json
        
        count = 0
        with open(path) as f:
            for line in f:
                if len(self.samples) >= self.config.max_samples:
                    break
                    
                try:
                    data = json.loads(line)
                    self.add_sample(
                        fen=data["fen"],
                        stockfish_eval=data["stockfish_eval"],
                        depth=data.get("depth", 0),
                    )
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return count

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
    Distilled evaluation model that mimics Stockfish.

    Supports both PyTorch and sklearn backends.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self._model = None
        self._backend = None  # "pytorch" or "sklearn"
        self._scaler = None   # For sklearn
        self._normalizer = None  # For pytorch {"mean": ..., "std": ...}
        self._config = None
        
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Try PyTorch first
        try:
            import torch
            data = torch.load(path, map_location="cpu")
            
            if data.get("type") == "pytorch":
                self._load_pytorch(data)
                return
        except ImportError:
            pass
        except Exception:
            pass
        
        # Try sklearn
        try:
            import joblib
            data = joblib.load(path)
            
            if data.get("type") == "sklearn_mlp":
                self._load_sklearn(data)
                return
        except ImportError:
            pass
        except Exception:
            pass
        
        raise ValueError(f"Could not load model from {path}")

    def _load_pytorch(self, data: Dict[str, Any]) -> None:
        """Load PyTorch model."""
        import torch
        import torch.nn as nn
        
        config = data["model_config"]
        hidden_dims = config["hidden_dims"]
        input_dim = config["input_dim"]
        dropout = config.get("dropout", 0.1)
        
        # Rebuild model architecture
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        
        model = nn.Sequential(*layers)
        model.load_state_dict(data["model_state"])
        model.eval()
        
        self._model = model
        self._backend = "pytorch"
        self._normalizer = data["normalization"]
        self._config = config

    def _load_sklearn(self, data: Dict[str, Any]) -> None:
        """Load sklearn model."""
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._backend = "sklearn"
        self._config = data.get("config", {})

    def save(self, path: Path) -> None:
        """Save the trained model to disk."""
        if self._model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._backend == "pytorch":
            import torch
            torch.save({
                "type": "pytorch",
                "model_state": self._model.state_dict(),
                "model_config": self._config,
                "normalization": self._normalizer,
            }, path)
        elif self._backend == "sklearn":
            import joblib
            joblib.dump({
                "type": "sklearn_mlp",
                "model": self._model,
                "scaler": self._scaler,
                "config": self._config,
            }, path)

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a position using the distilled model.

        Args:
            board: Chess board to evaluate

        Returns:
            Evaluation in pawn units (positive = white advantage)
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Train a model with tools/train_distilled_eval.py "
                "and load it with load()."
            )
        
        # Extract features
        fv = extract_features(board)
        
        if self._backend == "pytorch":
            return self._evaluate_pytorch(fv.features)
        elif self._backend == "sklearn":
            return self._evaluate_sklearn(fv.features)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def _evaluate_pytorch(self, features: List[float]) -> float:
        """Evaluate using PyTorch model."""
        import torch
        import numpy as np
        
        X = np.array(features, dtype=np.float32)
        
        # Normalize
        mean = np.array(self._normalizer["mean"])
        std = np.array(self._normalizer["std"])
        X = (X - mean) / std
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.tensor(X).unsqueeze(0)
            pred = self._model(X_tensor)
            return float(pred.item())

    def _evaluate_sklearn(self, features: List[float]) -> float:
        """Evaluate using sklearn model."""
        import numpy as np
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        pred = self._model.predict(X_scaled)
        return float(pred[0])

    def evaluate_batch(self, boards: List[chess.Board]) -> List[float]:
        """Evaluate multiple positions at once (more efficient)."""
        if not boards:
            return []
        
        # Extract features for all boards
        features = [extract_features(board).features for board in boards]
        
        if self._backend == "pytorch":
            return self._evaluate_batch_pytorch(features)
        elif self._backend == "sklearn":
            return self._evaluate_batch_sklearn(features)
        else:
            return [self.evaluate(board) for board in boards]

    def _evaluate_batch_pytorch(self, features_list: List[List[float]]) -> List[float]:
        """Batch evaluation with PyTorch."""
        import torch
        import numpy as np
        
        X = np.array(features_list, dtype=np.float32)
        
        # Normalize
        mean = np.array(self._normalizer["mean"])
        std = np.array(self._normalizer["std"])
        X = (X - mean) / std
        
        with torch.no_grad():
            X_tensor = torch.tensor(X)
            preds = self._model(X_tensor)
            return preds.squeeze().tolist()

    def _evaluate_batch_sklearn(self, features_list: List[List[float]]) -> List[float]:
        """Batch evaluation with sklearn."""
        import numpy as np
        
        X = np.array(features_list)
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return preds.tolist()

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
    """
    config = config or DistillationConfig()
    
    if len(dataset) < config.min_samples:
        raise ValueError(
            f"Not enough training data: {len(dataset)} samples "
            f"(minimum: {config.min_samples}). "
            "Collect more data with tools/collect_stockfish_evals.py"
        )
    
    # This is a convenience wrapper - actual training should use
    # tools/train_distilled_eval.py for better control
    raise NotImplementedError(
        "Use tools/train_distilled_eval.py for training. "
        "This function is for API compatibility only."
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

    This is a convenience wrapper - prefer using tools/collect_stockfish_evals.py
    directly for better control over the collection process.

    Args:
        trace_paths: List of JSONL trace file paths
        stockfish_path: Path to Stockfish executable
        output_path: Output path for collected data
        depth: Stockfish search depth
        max_samples: Maximum number of samples to collect

    Returns:
        Number of samples collected
    """
    raise NotImplementedError(
        "Use tools/collect_stockfish_evals.py for data collection. "
        "This function is for API compatibility only."
    )
