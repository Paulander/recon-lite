"""MLP-based tactic pattern detector.

This module provides lightweight neural network classifiers for detecting
tactical patterns that are difficult to capture with heuristics alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import chess

from .features import extract_tactics_features, TacticsFeatureVector, FEATURE_COUNT


@dataclass
class DetectionResult:
    """Result of tactic detection."""
    detected: bool
    confidence: float
    tactic_type: str
    features_used: int = FEATURE_COUNT


# Global cache for loaded detectors
_DETECTOR_CACHE: Dict[str, "TacticMLPDetector"] = {}

# Default weights directory
WEIGHTS_DIR = Path("weights/tactics_mlp")


class TacticMLPDetector:
    """
    Lightweight MLP classifier for tactic pattern detection.
    
    Supports both sklearn and PyTorch backends:
    - sklearn: MLPClassifier, good for CPU-only training
    - pytorch: nn.Sequential, better for GPU training
    
    Example:
        detector = TacticMLPDetector("backRankMate")
        detector.load("weights/tactics_mlp/backRankMate_detector.pkl")
        detected, confidence = detector.detect(board)
    """
    
    def __init__(
        self,
        tactic_type: str,
        hidden_dims: List[int] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize detector.
        
        Args:
            tactic_type: Type of tactic to detect
            hidden_dims: Hidden layer dimensions (default: [64, 32])
            threshold: Detection threshold (default: 0.5)
        """
        self.tactic_type = tactic_type
        self.hidden_dims = hidden_dims or [64, 32]
        self.threshold = threshold
        
        self._model = None
        self._backend = None  # "sklearn" or "pytorch"
        self._scaler = None  # For sklearn feature normalization
        self._config = {}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def load(self, path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to model file (.pkl for sklearn, .pt for pytorch)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Try sklearn first (most common)
        if path.suffix == ".pkl":
            self._load_sklearn(path)
        elif path.suffix == ".pt":
            self._load_pytorch(path)
        else:
            # Try both
            try:
                self._load_sklearn(path)
            except Exception:
                self._load_pytorch(path)
    
    def _load_sklearn(self, path: Path) -> None:
        """Load sklearn model."""
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib required: pip install joblib")
        
        data = joblib.load(path)
        
        self._model = data["model"]
        self._scaler = data.get("scaler")
        self._config = data.get("config", {})
        self._backend = "sklearn"
        
        # Update threshold from saved config
        if "threshold" in self._config:
            self.threshold = self._config["threshold"]
    
    def _load_pytorch(self, path: Path) -> None:
        """Load PyTorch model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required: pip install torch")
        
        data = torch.load(path, map_location="cpu")
        
        # Rebuild model architecture
        config = data["model_config"]
        hidden_dims = config["hidden_dims"]
        input_dim = config.get("input_dim", FEATURE_COUNT)
        
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        model = nn.Sequential(*layers)
        model.load_state_dict(data["model_state"])
        model.eval()
        
        self._model = model
        self._backend = "pytorch"
        self._config = config
        
        if "threshold" in self._config:
            self.threshold = self._config["threshold"]
    
    def save(self, path: Path) -> None:
        """Save trained model to disk."""
        if self._model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._backend == "sklearn":
            import joblib
            joblib.dump({
                "model": self._model,
                "scaler": self._scaler,
                "config": {
                    "tactic_type": self.tactic_type,
                    "threshold": self.threshold,
                    "hidden_dims": self.hidden_dims,
                },
            }, path)
        elif self._backend == "pytorch":
            import torch
            torch.save({
                "model_state": self._model.state_dict(),
                "model_config": {
                    "tactic_type": self.tactic_type,
                    "threshold": self.threshold,
                    "hidden_dims": self.hidden_dims,
                    "input_dim": FEATURE_COUNT,
                },
            }, path)
    
    def detect(self, board: chess.Board) -> Tuple[bool, float]:
        """
        Detect if tactic pattern exists in position.
        
        Args:
            board: Chess board to analyze
            
        Returns:
            Tuple of (detected: bool, confidence: float)
        """
        if self._model is None:
            raise RuntimeError(f"No model loaded for {self.tactic_type}")
        
        # Extract features
        fv = extract_tactics_features(board, self.tactic_type)
        
        # Get prediction
        if self._backend == "sklearn":
            prob = self._predict_sklearn(fv.features)
        elif self._backend == "pytorch":
            prob = self._predict_pytorch(fv.features)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")
        
        return prob >= self.threshold, prob
    
    def _predict_sklearn(self, features: List[float]) -> float:
        """Predict with sklearn model."""
        import numpy as np
        
        X = np.array(features).reshape(1, -1)
        
        if self._scaler is not None:
            X = self._scaler.transform(X)
        
        # Use predict_proba if available
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X)
            return float(probs[0, 1])  # Probability of positive class
        else:
            return float(self._model.predict(X)[0])
    
    def _predict_pytorch(self, features: List[float]) -> float:
        """Predict with PyTorch model."""
        import torch
        import numpy as np
        
        X = torch.tensor(np.array(features, dtype=np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            prob = self._model(X)
            return float(prob.item())
    
    def train(
        self,
        X: List[List[float]],
        y: List[int],
        backend: str = "sklearn",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Train the detector on labeled data.
        
        Args:
            X: Feature vectors (list of feature lists)
            y: Labels (0 or 1)
            backend: "sklearn" or "pytorch"
            **kwargs: Additional training parameters
            
        Returns:
            Dict with training metrics
        """
        import numpy as np
        
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)
        
        if backend == "sklearn":
            return self._train_sklearn(X_arr, y_arr, **kwargs)
        elif backend == "pytorch":
            return self._train_pytorch(X_arr, y_arr, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _train_sklearn(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        **kwargs,
    ) -> Dict[str, float]:
        """Train with sklearn."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_val_scaled = self._scaler.transform(X_val)
        
        # Train model
        self._model = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_dims),
            activation="relu",
            solver="adam",
            max_iter=kwargs.get("max_iter", 500),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        
        self._model.fit(X_train_scaled, y_train)
        self._backend = "sklearn"
        
        # Evaluate
        y_pred = self._model.predict(X_val_scaled)
        y_prob = self._model.predict_proba(X_val_scaled)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        
        return metrics
    
    def _train_pytorch(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        **kwargs,
    ) -> Dict[str, float]:
        """Train with PyTorch."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Build model
        layers = []
        in_dim = FEATURE_COUNT
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self._model = nn.Sequential(*layers)
        self._backend = "pytorch"
        
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self._model.train()
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Evaluate
        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_tensor)
            preds_binary = (preds > 0.5).float()
            accuracy = (preds_binary == y_tensor).float().mean().item()
        
        return {
            "accuracy": accuracy,
            "final_loss": total_loss / len(loader),
            "epochs": epochs,
        }


def get_mlp_detector(tactic_type: str) -> Optional[TacticMLPDetector]:
    """
    Get an MLP detector for a tactic type, loading from cache if available.
    
    Args:
        tactic_type: Type of tactic (backRankMate, doubleCheck, smotheredMate)
        
    Returns:
        TacticMLPDetector if model exists, None otherwise
    """
    global _DETECTOR_CACHE
    
    # Check cache
    if tactic_type in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[tactic_type]
    
    # Try to load from default location
    model_path = WEIGHTS_DIR / f"{tactic_type}_detector.pkl"
    if not model_path.exists():
        model_path = WEIGHTS_DIR / f"{tactic_type}_detector.pt"
    
    if not model_path.exists():
        return None
    
    try:
        detector = TacticMLPDetector(tactic_type)
        detector.load(model_path)
        _DETECTOR_CACHE[tactic_type] = detector
        return detector
    except Exception as e:
        print(f"Warning: Failed to load MLP detector for {tactic_type}: {e}")
        return None


def clear_detector_cache() -> None:
    """Clear the detector cache."""
    global _DETECTOR_CACHE
    _DETECTOR_CACHE.clear()

