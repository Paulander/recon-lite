#!/usr/bin/env python3
"""Train distilled evaluation model (M7).

Trains a lightweight neural network to mimic Stockfish evaluations
using the collected evaluation data.

Usage:
    uv run python tools/train_distilled_eval.py --data data/distillation/evals.jsonl --out weights/distilled_eval.pt
    uv run python tools/train_distilled_eval.py --data data/distillation/evals.jsonl --epochs 200 --lr 0.001
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import chess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recon_lite_chess.eval.features import extract_features, FEATURE_COUNT, features_to_tensor


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    l2_reg: float = 0.0001


def load_training_data(data_path: Path) -> Tuple[List[List[float]], List[float]]:
    """Load and prepare training data from collected evals.
    
    Returns:
        features: List of feature vectors
        targets: List of evaluation targets
    """
    features = []
    targets = []
    
    with open(data_path) as f:
        for line in f:
            try:
                sample = json.loads(line)
                fen = sample["fen"]
                eval_score = sample["stockfish_eval"]
                
                # Extract features
                board = chess.Board(fen)
                fv = extract_features(board)
                
                features.append(fv.features)
                targets.append(eval_score)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                continue
    
    print(f"Loaded {len(features)} samples")
    return features, targets


def train_sklearn_model(
    features: List[List[float]],
    targets: List[float],
    config: TrainingConfig,
    output_path: Path,
) -> Dict[str, Any]:
    """Train using sklearn (fallback if PyTorch not available)."""
    try:
        import numpy as np
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib
    except ImportError:
        print("sklearn not available. Install with: uv pip install scikit-learn")
        return {"error": "sklearn not installed"}
    
    X = np.array(features)
    y = np.array(targets)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.validation_split, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
    
    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=config.hidden_dims,
        activation='relu',
        solver='adam',
        alpha=config.l2_reg,
        learning_rate_init=config.learning_rate,
        max_iter=config.epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=config.early_stopping_patience,
        verbose=True,
        random_state=42,
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    # Correlation
    train_corr = np.corrcoef(y_train, train_pred)[0, 1]
    val_corr = np.corrcoef(y_val, val_pred)[0, 1]
    
    print(f"\nTraining Results:")
    print(f"  Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}, Corr: {train_corr:.4f}")
    print(f"  Val MSE:   {val_mse:.4f}, R²: {val_r2:.4f}, Corr: {val_corr:.4f}")
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "type": "sklearn_mlp",
        "model": model,
        "scaler": scaler,
        "config": {
            "hidden_dims": config.hidden_dims,
            "feature_count": FEATURE_COUNT,
        },
        "metrics": {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_corr": train_corr,
            "val_corr": val_corr,
        }
    }
    
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to {output_path}")
    
    return model_data["metrics"]


def train_pytorch_model(
    features: List[List[float]],
    targets: List[float],
    config: TrainingConfig,
    output_path: Path,
) -> Dict[str, Any]:
    """Train using PyTorch (preferred if available)."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
    except ImportError:
        print("PyTorch not available, falling back to sklearn")
        return train_sklearn_model(features, targets, config, output_path)
    
    # Prepare data
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    
    # Split
    n_val = int(len(X) * config.validation_split)
    indices = torch.randperm(len(X))
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Normalize
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
    
    # Create model
    layers = []
    in_dim = FEATURE_COUNT
    for hidden_dim in config.hidden_dims:
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        ])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, 1))
    
    model = nn.Sequential(*layers)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).numpy()
        val_pred = model(X_val).numpy()
    
    y_train_np = y_train.numpy()
    y_val_np = y_val.numpy()
    
    train_mse = np.mean((train_pred - y_train_np) ** 2)
    val_mse = np.mean((val_pred - y_val_np) ** 2)
    train_corr = np.corrcoef(y_train_np.flatten(), train_pred.flatten())[0, 1]
    val_corr = np.corrcoef(y_val_np.flatten(), val_pred.flatten())[0, 1]
    
    print(f"\nTraining Results:")
    print(f"  Train MSE: {train_mse:.4f}, Corr: {train_corr:.4f}")
    print(f"  Val MSE:   {val_mse:.4f}, Corr: {val_corr:.4f}")
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "type": "pytorch",
        "model_state": model.state_dict(),
        "model_config": {
            "input_dim": FEATURE_COUNT,
            "hidden_dims": config.hidden_dims,
            "dropout": config.dropout,
        },
        "normalization": {
            "mean": X_mean.numpy(),
            "std": X_std.numpy(),
        },
        "metrics": {
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "train_corr": float(train_corr),
            "val_corr": float(val_corr),
        }
    }, output_path)
    
    print(f"\nModel saved to {output_path}")
    
    return {
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "train_corr": float(train_corr),
        "val_corr": float(val_corr),
    }


def main():
    parser = argparse.ArgumentParser(description="Train distilled evaluation model")
    parser.add_argument("--data", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--out", default="weights/distilled_eval.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=str, default="256,128", help="Hidden layer sizes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--backend", choices=["auto", "pytorch", "sklearn"], default="auto")
    args = parser.parse_args()
    
    # Parse hidden dims
    hidden_dims = tuple(int(x) for x in args.hidden.split(","))
    
    config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run tools/collect_stockfish_evals.py first to collect training data")
        sys.exit(1)
    
    features, targets = load_training_data(data_path)
    
    if len(features) < 100:
        print(f"Not enough training data ({len(features)} samples)")
        print("Collect more data with tools/collect_stockfish_evals.py")
        sys.exit(1)
    
    output_path = Path(args.out)
    
    # Train
    if args.backend == "sklearn":
        metrics = train_sklearn_model(features, targets, config, output_path)
    elif args.backend == "pytorch":
        metrics = train_pytorch_model(features, targets, config, output_path)
    else:
        # Auto: try PyTorch first, fall back to sklearn
        try:
            import torch
            metrics = train_pytorch_model(features, targets, config, output_path)
        except ImportError:
            metrics = train_sklearn_model(features, targets, config, output_path)
    
    # Summary
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    if "val_corr" in metrics:
        corr = metrics["val_corr"]
        if corr >= 0.85:
            print(f"✓ Correlation with Stockfish: {corr:.3f} (target: 0.85)")
        else:
            print(f"✗ Correlation with Stockfish: {corr:.3f} (target: 0.85)")
            print("  Consider collecting more training data or adjusting hyperparameters")


if __name__ == "__main__":
    main()

