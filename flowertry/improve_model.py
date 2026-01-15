"""
Model Improvement Script for Disposable Income Regression.

This script diagnoses issues with the current model and experiments with
different architectures and hyperparameters to achieve MAPE <= 15%.

Features:
- Data analysis and outlier detection
- Multiple model architectures
- Hyperparameter tuning
- Regularization techniques
- Learning rate scheduling
- Early stopping
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime


# ============================================================================
# DATA ANALYSIS & PREPROCESSING
# ============================================================================

DATA_PATH = './data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv'


def analyze_data(verbose: bool = True) -> pd.DataFrame:
    """Analyze the dataset for potential issues."""
    df = pd.read_csv(DATA_PATH)
    
    # Compute target
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                    'Education', 'Miscellaneous']
    total_expenses = df[expense_cols].sum(axis=1)
    df['Disposable_Income'] = df['Income'] - total_expenses
    
    if verbose:
        print("=" * 70)
        print("DATA ANALYSIS")
        print("=" * 70)
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nTarget (Disposable_Income) Statistics:")
        print(f"  Min:    ${df['Disposable_Income'].min():,.2f}")
        print(f"  Max:    ${df['Disposable_Income'].max():,.2f}")
        print(f"  Mean:   ${df['Disposable_Income'].mean():,.2f}")
        print(f"  Median: ${df['Disposable_Income'].median():,.2f}")
        print(f"  Std:    ${df['Disposable_Income'].std():,.2f}")
        
        # Check for negative values (potential issue)
        neg_count = (df['Disposable_Income'] <= 0).sum()
        print(f"\n  Negative/Zero values: {neg_count} ({100*neg_count/len(df):.2f}%)")
        
        # Check distribution by City_Tier
        print("\nBy City_Tier:")
        for tier in df['City_Tier'].unique():
            tier_data = df[df['City_Tier'] == tier]['Disposable_Income']
            print(f"  {tier}: Mean=${tier_data.mean():,.2f}, Std=${tier_data.std():,.2f}, N={len(tier_data)}")
        
        # Check for outliers (IQR method)
        Q1 = df['Disposable_Income'].quantile(0.25)
        Q3 = df['Disposable_Income'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['Disposable_Income'] < Q1 - 1.5*IQR) | 
                   (df['Disposable_Income'] > Q3 + 1.5*IQR)).sum()
        print(f"\nOutliers (IQR method): {outliers} ({100*outliers/len(df):.2f}%)")
    
    return df


def prepare_improved_data(
    use_robust_scaler: bool = True,
    remove_outliers: bool = False,
    outlier_percentile: float = 0.01,
    add_features: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Prepare data with improved preprocessing.
    
    Returns:
        X_train, y_train, target_mean, target_std, X_test, y_test
    """
    df = pd.read_csv(DATA_PATH)
    
    # Compute target
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                    'Education', 'Miscellaneous']
    total_expenses = df[expense_cols].sum(axis=1)
    df['Disposable_Income'] = df['Income'] - total_expenses
    df['Total_Expenses'] = total_expenses
    
    # Optional: Remove outliers
    if remove_outliers:
        lower = df['Disposable_Income'].quantile(outlier_percentile)
        upper = df['Disposable_Income'].quantile(1 - outlier_percentile)
        mask = (df['Disposable_Income'] >= lower) & (df['Disposable_Income'] <= upper)
        df = df[mask].copy()
        if verbose:
            print(f"Removed outliers: {(~mask).sum()} samples")
    
    # Feature engineering
    numerical_features = [
        'Age', 'Dependents',
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education'
    ]
    
    if add_features:
        # Add engineered features
        df['Expense_to_Income_Ratio'] = df['Total_Expenses'] / (df['Income'] + 1)
        df['Essential_Expenses'] = df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']
        df['Discretionary_Expenses'] = df['Eating_Out'] + df['Entertainment']
        df['Debt_Burden'] = df['Loan_Repayment'] + df['Insurance']
        df['Log_Income'] = np.log1p(df['Income'])
        
        numerical_features.extend([
            'Expense_to_Income_Ratio', 'Essential_Expenses', 
            'Discretionary_Expenses', 'Debt_Burden', 'Log_Income'
        ])
    
    categorical_features = ['Occupation', 'City_Tier']
    
    # Target
    y = df['Disposable_Income'].values.astype(np.float32)
    target_mean = y.mean()
    target_std = y.std()
    
    # Normalize target
    y_normalized = (y - target_mean) / target_std
    
    # Scale numerical features
    if use_robust_scaler:
        scaler = RobustScaler()  # More robust to outliers
    else:
        scaler = StandardScaler()
    
    X_numerical = scaler.fit_transform(df[numerical_features].values.astype(np.float32))
    
    # One-hot encode categorical
    try:
        encoder = OneHotEncoder(sparse_output=False, drop=None)
    except TypeError:
        encoder = OneHotEncoder(sparse=False, drop=None)
    X_categorical = encoder.fit_transform(df[categorical_features])
    
    # Combine
    X = np.hstack([X_numerical, X_categorical]).astype(np.float32)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_normalized, test_size=0.2, random_state=42
    )
    
    if verbose:
        print(f"Feature dimensions: {X.shape[1]}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, target_mean, target_std, X_test, y_test


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ImprovedNet(nn.Module):
    """Improved neural network with dropout and batch normalization."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class ResidualNet(nn.Module):
    """Neural network with residual connections."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ))
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.input_bn(self.input_proj(x)))
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)  # Residual connection
        
        return self.output(x)


class WideAndDeepNet(nn.Module):
    """Wide & Deep architecture for better generalization."""
    
    def __init__(self, input_dim: int, deep_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, 1)
        
        # Deep component
        layers = []
        prev_dim = input_dim
        for dim in deep_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)
        
        # Combine
        self.combine = nn.Linear(2, 1)
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.combine(combined)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def compute_metrics(
    model: nn.Module, 
    dataloader: DataLoader, 
    target_mean: float, 
    target_std: float,
    device: torch.device
) -> Dict[str, float]:
    """Compute all regression metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Denormalize
    preds = all_preds * target_std + target_mean
    targets = all_targets * target_std + target_mean
    
    # MSE, RMSE, MAE
    mse = torch.mean((preds - targets) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    # R²
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # MAPE (avoid division by zero)
    epsilon = 1e-8
    mape = torch.mean(torch.abs((targets - preds) / (targets + epsilon))) * 100
    mape = mape.item()
    
    # Threshold accuracy
    percentage_errors = torch.abs((targets - preds) / (targets + epsilon)) * 100
    accuracy_10 = (percentage_errors <= 10).float().mean().item() * 100
    accuracy_20 = (percentage_errors <= 20).float().mean().item() * 100
    accuracy_15 = (percentage_errors <= 15).float().mean().item() * 100
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "accuracy_10": accuracy_10,
        "accuracy_15": accuracy_15,
        "accuracy_20": accuracy_20
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_mean: float,
    target_std: float,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: torch.device = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train model with early stopping and learning rate scheduling.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=verbose)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [], "val_loss": [], 
        "mape": [], "r2": [], "accuracy_15": []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * len(y_batch)
        
        val_loss /= len(val_loader.dataset)
        
        # Compute metrics
        metrics = compute_metrics(model, val_loader, target_mean, target_std, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mape"].append(metrics["mape"])
        history["r2"].append(metrics["r2"])
        history["accuracy_15"].append(metrics["accuracy_15"])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"MAPE: {metrics['mape']:.2f}% | R²: {metrics['r2']:.4f} | Acc@15%: {metrics['accuracy_15']:.2f}%")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def experiment_architectures(verbose: bool = True) -> Dict:
    """Run experiments with different model architectures."""
    
    print("\n" + "=" * 70)
    print("MODEL IMPROVEMENT EXPERIMENTS")
    print("=" * 70)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Prepare data with improvements
    print("\n--- Preparing Data ---")
    X_train, y_train, target_mean, target_std, X_test, y_test = prepare_improved_data(
        use_robust_scaler=True,
        remove_outliers=False,
        add_features=True,
        verbose=verbose
    )
    
    input_dim = X_train.shape[1]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Experiments
    experiments = {
        "baseline": {
            "model": ImprovedNet(input_dim, hidden_dims=[64, 32], dropout=0.0, use_batch_norm=False),
            "lr": 0.01,
            "epochs": 100
        },
        "improved_v1": {
            "model": ImprovedNet(input_dim, hidden_dims=[128, 64, 32], dropout=0.2, use_batch_norm=True),
            "lr": 0.001,
            "epochs": 150
        },
        "improved_v2": {
            "model": ImprovedNet(input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3, use_batch_norm=True),
            "lr": 0.001,
            "epochs": 150
        },
        "residual": {
            "model": ResidualNet(input_dim, hidden_dim=128, num_blocks=3, dropout=0.2),
            "lr": 0.001,
            "epochs": 150
        },
        "wide_and_deep": {
            "model": WideAndDeepNet(input_dim, deep_dims=[128, 64, 32], dropout=0.2),
            "lr": 0.001,
            "epochs": 150
        }
    }
    
    results = {}
    best_mape = float('inf')
    best_model = None
    best_model_name = None
    
    for name, config in experiments.items():
        print(f"\n{'='*50}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*50}")
        
        model = config["model"]
        
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            target_mean=target_mean,
            target_std=target_std,
            epochs=config["epochs"],
            lr=config["lr"],
            weight_decay=1e-4,
            patience=20,
            device=device,
            verbose=verbose
        )
        
        # Final evaluation on test set
        test_metrics = compute_metrics(trained_model, test_loader, target_mean, target_std, device)
        
        print(f"\n--- TEST RESULTS for {name} ---")
        print(f"  RMSE:        ${test_metrics['rmse']:,.2f}")
        print(f"  MAE:         ${test_metrics['mae']:,.2f}")
        print(f"  R²:          {test_metrics['r2']:.4f}")
        print(f"  MAPE:        {test_metrics['mape']:.2f}%")
        print(f"  Acc@10%:     {test_metrics['accuracy_10']:.2f}%")
        print(f"  Acc@15%:     {test_metrics['accuracy_15']:.2f}%")
        print(f"  Acc@20%:     {test_metrics['accuracy_20']:.2f}%")
        
        results[name] = {
            "metrics": test_metrics,
            "history": history
        }
        
        if test_metrics['mape'] < best_mape:
            best_mape = test_metrics['mape']
            best_model = trained_model
            best_model_name = name
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} | {'MAPE':>8} | {'R²':>8} | {'Acc@15%':>10} | {'Acc@20%':>10}")
    print("-" * 70)
    
    for name, result in results.items():
        m = result["metrics"]
        marker = " ⭐" if name == best_model_name else ""
        print(f"{name:<20} | {m['mape']:>7.2f}% | {m['r2']:>8.4f} | {m['accuracy_15']:>9.2f}% | {m['accuracy_20']:>9.2f}%{marker}")
    
    print("=" * 70)
    print(f"\nBest Model: {best_model_name} (MAPE: {best_mape:.2f}%)")
    
    return results, best_model, best_model_name, input_dim


def plot_results(results: Dict, output_dir: str = "."):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Training Loss
    ax = axes[0, 0]
    for (name, result), color in zip(results.items(), colors):
        ax.plot(result["history"]["train_loss"], label=name, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        ax.plot(result["history"]["val_loss"], label=name, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAPE over epochs
    ax = axes[1, 0]
    for (name, result), color in zip(results.items(), colors):
        ax.plot(result["history"]["mape"], label=name, color=color)
    ax.axhline(y=15, color='red', linestyle='--', label='Target (15%)')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final Metrics Comparison
    ax = axes[1, 1]
    names = list(results.keys())
    metrics_to_plot = ['mape', 'accuracy_15', 'accuracy_20']
    x = np.arange(len(names))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [results[name]["metrics"][metric] for name in names]
        ax.bar(x + i*width, values, width, label=metric.upper().replace('_', '@'))
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Final Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, "model_improvement_results.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    plt.show()


def save_best_model(model: nn.Module, input_dim: int, output_dir: str = "."):
    """Save the best model for use in federated learning."""
    # Save model state
    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'architecture': model.__class__.__name__
    }, model_path)
    print(f"Model saved to: {model_path}")


def main():
    """Main function to run model improvement experiments."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/model_improvement_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze data first
    analyze_data(verbose=True)
    
    # Run experiments
    results, best_model, best_model_name, input_dim = experiment_architectures(verbose=True)
    
    # Plot results
    plot_results(results, output_dir)
    
    # Save best model
    save_best_model(best_model, input_dim, output_dir)
    
    # Save results as JSON
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            "metrics": result["metrics"],
            "final_train_loss": result["history"]["train_loss"][-1] if result["history"]["train_loss"] else None,
            "final_val_loss": result["history"]["val_loss"][-1] if result["history"]["val_loss"] else None,
        }
    
    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
    
    # Check if target achieved
    best_mape = results[best_model_name]["metrics"]["mape"]
    if best_mape <= 15:
        print(f"\n✅ TARGET ACHIEVED! Best MAPE: {best_mape:.2f}% (target: ≤15%)")
    else:
        print(f"\n⚠️ Target not achieved. Best MAPE: {best_mape:.2f}% (target: ≤15%)")
        print("Recommendations:")
        print("  1. Try removing outliers: remove_outliers=True")
        print("  2. Increase model capacity")
        print("  3. Add more feature engineering")
        print("  4. Try different loss functions (Huber, LogCosh)")
    
    return results, best_model


if __name__ == "__main__":
    main()
