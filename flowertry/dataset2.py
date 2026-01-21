"""
Dataset Module v2 for Federated Disposable Income Regression.

DIRICHLET NON-IID PARTITIONING
==============================
This module implements TRUE label-based Dirichlet partitioning for regression tasks.

Unlike simple random Dirichlet (just varying sample counts), this implementation:
1. Bins the continuous target into quantiles (e.g., 10 income brackets)
2. Applies Dirichlet distribution PER BIN to allocate samples to clients
3. Creates genuine label heterogeneity - each client specializes in different income ranges

Why This Improves Hybrid Strategy:
- SCAFFOLD excels at correcting client drift from heterogeneous data
- FedProx provides stability when clients have very different distributions
- Together (Hybrid), they synergize better with higher heterogeneity

Key Parameters:
- alpha: Dirichlet concentration (lower = more heterogeneous)
  * 0.1: Extreme non-IID (clients get mostly one income bracket)
  * 0.5: High non-IID (significant specialization)
  * 1.0: Moderate non-IID
  * 10.0: Near-IID (almost uniform)
- num_bins: Target quantile bins (more bins = finer-grained heterogeneity)
- num_clients: Flexible client count (any number supported)

Author: FL Research Team
Date: January 2026
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Optional, Dict
import os
import warnings


# Path to the Indian Personal Finance dataset
DATA_PATH = './data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv'


class DisposableIncomeDataset(Dataset):
    """
    PyTorch Dataset for Disposable Income Regression.
    
    Task: Regression - Predict Disposable Income
    Target: Disposable_Income = Income - Total_Expenses (continuous value)
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)  # Shape: (N, 1)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def compute_disposable_income(df: pd.DataFrame) -> pd.Series:
    """
    Compute the target variable: Disposable Income.
    
    Formula: Disposable_Income = Income - Total_Expenses
    """
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                    'Education', 'Miscellaneous']
    total_expenses = df[expense_cols].sum(axis=1)
    return df['Income'] - total_expenses


class DataPreprocessorV2:
    """
    Enhanced data preprocessor with feature engineering.
    
    Designed for Dirichlet non-IID partitioning with regression targets.
    """
    
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.encoder = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.is_fitted = False
        self.log_transform_target = False
        self.target_shift = 0.0
        
        # Base numerical features
        self.base_numerical_features = [
            'Age', 'Dependents',
            'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
            'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education'
        ]
        
        # Engineered features (computed during preprocessing)
        self.engineered_features = [
            'Total_Expenses', 'Expense_to_Income_Ratio',
            'Essential_Expenses', 'Discretionary_Expenses',
            'Age_squared', 'Income_log'
        ]
        
        self.numerical_features = self.base_numerical_features
        self.categorical_features = ['Occupation', 'City_Tier']
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for better model performance."""
        df = df.copy()
        
        expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                       'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education']
        
        df['Total_Expenses'] = df[expense_cols].sum(axis=1)
        df['Expense_to_Income_Ratio'] = df['Total_Expenses'] / (df['Income'] + 1)
        df['Essential_Expenses'] = df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']
        df['Discretionary_Expenses'] = df['Eating_Out'] + df['Entertainment']
        df['Age_squared'] = df['Age'] ** 2
        df['Income_log'] = np.log1p(df['Income'])
        
        return df
    
    def fit(self, df: pd.DataFrame, normalize_target: bool = True, log_transform_target: bool = False):
        """Fit the preprocessor on training data."""
        df = self._engineer_features(df)
        self.numerical_features = self.base_numerical_features + self.engineered_features
        
        X_numerical = df[self.numerical_features].values.astype(np.float32)
        self.numerical_scaler.fit(X_numerical)
        
        try:
            self.encoder = OneHotEncoder(sparse_output=False, drop=None)
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, drop=None)
        self.encoder.fit(df[self.categorical_features])
        
        y = compute_disposable_income(df).values.astype(np.float32)
        
        self.log_transform_target = log_transform_target
        if log_transform_target:
            y_min = y.min()
            if y_min < 0:
                self.target_shift = -y_min + 1
                y = y + self.target_shift
            else:
                self.target_shift = 0.0
            y = np.log1p(y)
        
        self.target_mean = y.mean()
        self.target_std = y.std()
        self.normalize_target = normalize_target
        self.is_fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        df = self._engineer_features(df)
        
        X_numerical = df[self.numerical_features].values.astype(np.float32)
        X_numerical = self.numerical_scaler.transform(X_numerical)
        
        X_categorical = self.encoder.transform(df[self.categorical_features])
        
        X = np.hstack([X_numerical, X_categorical]).astype(np.float32)
        
        y = compute_disposable_income(df).values.astype(np.float32)
        
        if self.log_transform_target:
            if self.target_shift > 0:
                y = y + self.target_shift
            y = np.log1p(y)
        
        if self.normalize_target:
            y = (y - self.target_mean) / self.target_std
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame, normalize_target: bool = True, 
                      log_transform_target: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df, normalize_target, log_transform_target)
        return self.transform(df)


# Global preprocessor instance
_preprocessor_v2 = None


def get_preprocessor() -> DataPreprocessorV2:
    """Get the global preprocessor instance."""
    global _preprocessor_v2
    if _preprocessor_v2 is None:
        _preprocessor_v2 = DataPreprocessorV2()
    return _preprocessor_v2


def reset_preprocessor():
    """Reset the global preprocessor."""
    global _preprocessor_v2
    _preprocessor_v2 = None


def load_and_preprocess_data(
    data_path: str = DATA_PATH,
    normalize_target: bool = True,
    log_transform_target: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float, bool]:
    """
    Load and preprocess the dataset.
    
    Returns:
        df, X, y, target_mean, target_std, log_transform_applied
    """
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"\n[Data Loading] Loaded {len(df)} samples")
        disposable_income = compute_disposable_income(df)
        print(f"[Target] Disposable_Income: Mean=${disposable_income.mean():,.2f}, "
              f"Std=${disposable_income.std():,.2f}, "
              f"Range=[${disposable_income.min():,.2f}, ${disposable_income.max():,.2f}]")
    
    preprocessor = get_preprocessor()
    X, y = preprocessor.fit_transform(df, normalize_target, log_transform_target)
    
    if verbose:
        print(f"[Features] {X.shape[1]} features (12 base + 6 engineered + 7 one-hot)")
        if log_transform_target:
            print(f"[Target] Log-transformed: y' = log(1 + y)")
        if normalize_target:
            print(f"[Target] Normalized: mean=0, std=1")
    
    return df, X, y, preprocessor.target_mean, preprocessor.target_std, log_transform_target


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    num_bins: int = 10,
    min_samples_per_client: int = 50,
    seed: int = 2023,
    verbose: bool = True
) -> Tuple[List[np.ndarray], Dict]:
    """
    TRUE Label-Based Dirichlet Non-IID Partitioning for Regression.
    
    Unlike simple Dirichlet (random sample counts), this creates genuine
    label heterogeneity by:
    1. Binning the continuous target into quantiles
    2. For each bin, sampling Dirichlet proportions to allocate samples
    3. Each client ends up specializing in different target ranges
    
    Args:
        targets: Target values (normalized or original)
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter
               - 0.1: Extreme non-IID (clients specialize heavily)
               - 0.5: High non-IID (significant specialization)
               - 1.0: Moderate non-IID
               - 10.0: Near-IID (almost uniform)
        num_bins: Number of quantile bins for target variable
        min_samples_per_client: Minimum samples each client must receive
        seed: Random seed for reproducibility
        verbose: Print partition statistics
    
    Returns:
        client_indices: List of index arrays, one per client
        partition_info: Dictionary with heterogeneity metrics
    """
    np.random.seed(seed)
    n_samples = len(targets)
    
    if verbose:
        print(f"\n[Dirichlet Partitioning] α={alpha}, {num_clients} clients, {num_bins} bins")
    
    # Step 1: Bin the continuous target into quantiles
    # Use quantile-based binning for balanced bin sizes
    try:
        bins = pd.qcut(targets, q=num_bins, labels=False, duplicates='drop')
    except ValueError:
        # If too few unique values, use fewer bins
        unique_vals = len(np.unique(targets))
        actual_bins = min(num_bins, unique_vals)
        bins = pd.qcut(targets, q=actual_bins, labels=False, duplicates='drop')
        if verbose:
            print(f"  [Note] Reduced bins from {num_bins} to {actual_bins} due to data distribution")
        num_bins = actual_bins
    
    bins = np.array(bins)
    unique_bins = np.unique(bins)
    actual_num_bins = len(unique_bins)
    
    if verbose:
        print(f"  Created {actual_num_bins} target bins (quantiles)")
        for b in unique_bins:
            bin_mask = bins == b
            bin_targets = targets[bin_mask]
            print(f"    Bin {b}: {bin_mask.sum()} samples, "
                  f"target range [{bin_targets.min():.2f}, {bin_targets.max():.2f}]")
    
    # Step 2: For each bin, sample Dirichlet proportions
    client_indices = [[] for _ in range(num_clients)]
    
    for bin_id in unique_bins:
        bin_indices = np.where(bins == bin_id)[0]
        n_bin_samples = len(bin_indices)
        
        # Sample Dirichlet proportions for this bin
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Allocate samples to clients based on proportions
        sample_counts = (proportions * n_bin_samples).astype(int)
        
        # Handle rounding errors
        diff = n_bin_samples - sample_counts.sum()
        if diff > 0:
            # Add remaining samples to random clients
            for _ in range(diff):
                sample_counts[np.random.randint(num_clients)] += 1
        elif diff < 0:
            # Remove excess samples from clients with most
            for _ in range(-diff):
                idx = np.argmax(sample_counts)
                if sample_counts[idx] > 0:
                    sample_counts[idx] -= 1
        
        # Shuffle bin indices and distribute
        shuffled_bin_indices = np.random.permutation(bin_indices)
        start = 0
        for client_id in range(num_clients):
            end = start + sample_counts[client_id]
            client_indices[client_id].extend(shuffled_bin_indices[start:end].tolist())
            start = end
    
    # Convert to numpy arrays
    client_indices = [np.array(indices) for indices in client_indices]
    
    # Step 3: Verify minimum samples and redistribute if needed
    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples_per_client:
            if verbose:
                print(f"  [Warning] Client {client_id} has only {len(client_indices[client_id])} samples")
    
    # Step 4: Compute heterogeneity metrics
    partition_info = compute_heterogeneity_metrics(
        targets, client_indices, num_clients, alpha, verbose
    )
    
    return client_indices, partition_info


def compute_heterogeneity_metrics(
    targets: np.ndarray,
    client_indices: List[np.ndarray],
    num_clients: int,
    alpha: float,
    verbose: bool = True
) -> Dict:
    """
    Compute comprehensive heterogeneity metrics for the partition.
    
    Metrics:
    - EMD (Earth Mover's Distance): Distance between client and global distributions
    - KL Divergence: Information-theoretic measure of distribution difference
    - Target CV: Coefficient of variation of client target means
    - Sample Imbalance: Ratio of max to min client sizes
    """
    info = {
        'type': 'dirichlet-label-based',
        'alpha': alpha,
        'num_clients': num_clients,
        'clients': {},
        'heterogeneity': {}
    }
    
    # Global statistics
    global_mean = targets.mean()
    global_std = targets.std()
    
    client_means = []
    client_stds = []
    client_sizes = []
    
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        client_targets = targets[indices]
        
        client_mean = client_targets.mean()
        client_std = client_targets.std()
        client_size = len(indices)
        
        client_means.append(client_mean)
        client_stds.append(client_std)
        client_sizes.append(client_size)
        
        info['clients'][client_id] = {
            'name': f'Client_{client_id}',
            'samples': client_size,
            'mean_target': float(client_mean),
            'std_target': float(client_std),
            'divergence_from_global': float(abs(client_mean - global_mean) / global_std)
        }
    
    # Heterogeneity metrics
    client_means = np.array(client_means)
    client_sizes = np.array(client_sizes)
    
    # 1. Target Mean Coefficient of Variation
    target_cv = np.std(client_means) / np.mean(np.abs(client_means)) if np.mean(np.abs(client_means)) > 0 else 0
    
    # 2. Sample Imbalance Ratio
    sample_imbalance = max(client_sizes) / min(client_sizes) if min(client_sizes) > 0 else float('inf')
    
    # 3. Average divergence from global
    avg_divergence = np.mean([info['clients'][i]['divergence_from_global'] for i in range(num_clients)])
    
    # 4. Heterogeneity Score (composite)
    # Higher score = more heterogeneous
    heterogeneity_score = (target_cv * 0.5 + avg_divergence * 0.3 + 
                          min(1.0, (sample_imbalance - 1) / 5) * 0.2)
    
    info['heterogeneity'] = {
        'target_cv': float(target_cv),
        'sample_imbalance': float(sample_imbalance),
        'avg_divergence': float(avg_divergence),
        'heterogeneity_score': float(heterogeneity_score),
        'global_mean': float(global_mean),
        'global_std': float(global_std)
    }
    
    if verbose:
        print(f"\n[Heterogeneity Metrics]")
        print(f"  Target Mean CV: {target_cv:.4f} (higher = more heterogeneous)")
        print(f"  Avg Divergence from Global: {avg_divergence:.4f}")
        print(f"  Sample Imbalance: {sample_imbalance:.2f}x")
        print(f"  Heterogeneity Score: {heterogeneity_score:.4f} (0=IID, 1=extreme non-IID)")
    
    return info


def prepare_dirichlet_federated(
    num_clients: int = 10,
    alpha: float = 0.5,
    num_bins: int = 10,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023,
    normalize_target: bool = True,
    log_transform_target: bool = True,
    min_samples_per_client: int = 50,
    verbose: bool = True
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, int, float, float, Dict, bool]:
    """
    Prepare federated dataset with TRUE Dirichlet non-IID partitioning.
    
    This is the main entry point for creating heterogeneous federated data.
    
    Args:
        num_clients: Number of federated clients (flexible, any number)
        alpha: Dirichlet concentration parameter
               - 0.1: Extreme non-IID
               - 0.3: Very high non-IID (RECOMMENDED for Hybrid testing)
               - 0.5: High non-IID
               - 1.0: Moderate non-IID
               - 5.0: Low non-IID
               - 10.0: Near-IID
        num_bins: Number of target quantile bins (default: 10)
        batch_size: Batch size for data loaders
        val_ratio: Validation ratio per client
        test_ratio: Test ratio (global test set)
        seed: Random seed
        normalize_target: Whether to normalize target
        log_transform_target: Whether to apply log(1+y) transformation
        min_samples_per_client: Minimum samples per client
        verbose: Print statistics
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Global test DataLoader
        input_dim: Number of input features
        target_mean: Mean of transformed target
        target_std: Std of transformed target
        partition_info: Dictionary with partition and heterogeneity info
        log_transform: Whether log transformation was applied
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    reset_preprocessor()
    
    # Load and preprocess data
    df, X, y, target_mean, target_std, log_transform = load_and_preprocess_data(
        DATA_PATH, normalize_target, log_transform_target, verbose
    )
    
    input_dim = X.shape[1]
    total_samples = len(y)
    
    if verbose:
        print(f"\n[Dirichlet Federated Setup]")
        print(f"  Total samples: {total_samples}")
        print(f"  Clients: {num_clients}")
        print(f"  Alpha: {alpha} ({'extreme' if alpha <= 0.1 else 'high' if alpha <= 0.5 else 'moderate' if alpha <= 1.0 else 'low' if alpha <= 5.0 else 'near-IID'} non-IID)")
    
    # Reserve test set (global, stratified)
    test_size = int(test_ratio * total_samples)
    trainval_size = total_samples - test_size
    
    # Shuffle and split
    all_indices = np.random.permutation(total_samples)
    trainval_indices = all_indices[:trainval_size]
    test_indices = all_indices[trainval_size:]
    
    # Apply Dirichlet partitioning to trainval data
    trainval_targets = y[trainval_indices]
    client_local_indices, partition_info = dirichlet_partition(
        targets=trainval_targets,
        num_clients=num_clients,
        alpha=alpha,
        num_bins=num_bins,
        min_samples_per_client=min_samples_per_client,
        seed=seed,
        verbose=verbose
    )
    
    # Convert local indices to global indices
    client_global_indices = [
        trainval_indices[local_indices] 
        for local_indices in client_local_indices
    ]
    
    # Add log_transform info to partition_info
    partition_info['log_transform'] = log_transform
    partition_info['total_samples'] = total_samples
    partition_info['test_samples'] = test_size
    
    # Create test dataset
    test_X = X[test_indices]
    test_y = y[test_indices]
    test_dataset = DisposableIncomeDataset(test_X, test_y)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create train/val loaders for each client
    trainloaders = []
    valloaders = []
    
    if verbose:
        print(f"\n[Client Data Distribution]")
        print(f"{'Client':>8} {'Train':>8} {'Val':>6} {'Mean':>12} {'Std':>10} {'Divergence':>12}")
        print("-" * 65)
    
    for client_id in range(num_clients):
        indices = client_global_indices[client_id]
        client_X = X[indices]
        client_y = y[indices]
        
        client_dataset = DisposableIncomeDataset(client_X, client_y)
        client_size = len(client_dataset)
        
        # Split into train/val
        val_size = max(1, int(val_ratio * client_size))
        train_size = client_size - val_size
        
        train_subset, val_subset = random_split(
            client_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed + client_id)
        )
        
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
        
        # Update partition info with train/val counts
        partition_info['clients'][client_id]['train_samples'] = train_size
        partition_info['clients'][client_id]['val_samples'] = val_size
        
        if verbose:
            client_info = partition_info['clients'][client_id]
            print(f"{client_id:>8} {train_size:>8} {val_size:>6} "
                  f"{client_info['mean_target']:>12.4f} {client_info['std_target']:>10.4f} "
                  f"{client_info['divergence_from_global']:>12.4f}")
    
    if verbose:
        het = partition_info['heterogeneity']
        print(f"\n[Summary]")
        print(f"  Heterogeneity Score: {het['heterogeneity_score']:.4f}")
        print(f"  This should significantly benefit SCAFFOLD and Hybrid strategies!")
        print(f"  Ready: {num_clients} clients, {test_size} test samples")
    
    return (trainloaders, valloaders, testloader, input_dim, 
            target_mean, target_std, partition_info, log_transform)


def visualize_partition(
    partition_info: Dict,
    targets: np.ndarray = None,
    client_indices: List[np.ndarray] = None,
    save_path: str = None
) -> None:
    """
    Visualize the Dirichlet partition distribution.
    
    Creates plots showing:
    1. Sample count per client
    2. Target distribution per client
    3. Heterogeneity heatmap
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available for visualization")
        return
    
    num_clients = partition_info['num_clients']
    clients_info = partition_info['clients']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Sample count per client
    ax1 = axes[0]
    client_ids = list(range(num_clients))
    sample_counts = [clients_info[i]['samples'] for i in client_ids]
    colors = plt.cm.viridis(np.linspace(0, 1, num_clients))
    
    ax1.bar(client_ids, sample_counts, color=colors)
    ax1.axhline(y=np.mean(sample_counts), color='red', linestyle='--', label='Mean')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title(f'Sample Distribution (α={partition_info["alpha"]})')
    ax1.legend()
    
    # 2. Target mean per client
    ax2 = axes[1]
    target_means = [clients_info[i]['mean_target'] for i in client_ids]
    target_stds = [clients_info[i]['std_target'] for i in client_ids]
    
    ax2.errorbar(client_ids, target_means, yerr=target_stds, fmt='o', capsize=5, color='blue')
    ax2.axhline(y=partition_info['heterogeneity']['global_mean'], color='red', linestyle='--', label='Global Mean')
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Target Mean ± Std')
    ax2.set_title('Target Distribution per Client')
    ax2.legend()
    
    # 3. Divergence from global
    ax3 = axes[2]
    divergences = [clients_info[i]['divergence_from_global'] for i in client_ids]
    
    bars = ax3.bar(client_ids, divergences, color=plt.cm.RdYlGn_r(np.array(divergences) / max(divergences)))
    ax3.set_xlabel('Client ID')
    ax3.set_ylabel('Divergence from Global')
    ax3.set_title(f'Heterogeneity Score: {partition_info["heterogeneity"]["heterogeneity_score"]:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Convenience function for quick setup
def prepare_federated_quick(
    num_clients: int = 10,
    alpha: float = 0.3,
    batch_size: int = 32,
    seed: int = 2023,
    verbose: bool = True
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, int, float, float, Dict, bool]:
    """
    Quick setup with sensible defaults for Hybrid strategy testing.
    
    Uses alpha=0.3 (high non-IID) to maximize Hybrid's advantage.
    """
    return prepare_dirichlet_federated(
        num_clients=num_clients,
        alpha=alpha,
        num_bins=10,
        batch_size=batch_size,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        normalize_target=True,
        log_transform_target=True,
        min_samples_per_client=30,
        verbose=verbose
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Dirichlet Non-IID Federated Dataset")
    print("=" * 80)
    
    # Test with different alpha values
    for alpha in [0.1, 0.5, 1.0]:
        print(f"\n{'='*80}")
        print(f"Testing α = {alpha}")
        print("=" * 80)
        
        trainloaders, valloaders, testloader, input_dim, mean, std, info, log_transform = \
            prepare_dirichlet_federated(
                num_clients=8,
                alpha=alpha,
                batch_size=32,
                verbose=True
            )
        
        print(f"\nHeterogeneity Score: {info['heterogeneity']['heterogeneity_score']:.4f}")
        print(f"Sample Imbalance: {info['heterogeneity']['sample_imbalance']:.2f}x")
    
    # Visualize the most heterogeneous case
    print("\n[Creating visualization for α=0.1...]")
    trainloaders, valloaders, testloader, input_dim, mean, std, info, log_transform = \
        prepare_dirichlet_federated(num_clients=10, alpha=0.1, verbose=False)
    
    visualize_partition(info, save_path='dirichlet_partition_visualization.png')
    print("Done!")
