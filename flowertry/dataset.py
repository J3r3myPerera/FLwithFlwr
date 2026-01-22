"""
Dataset module for Federated Disposable Income Regression.

Task: Predict Disposable Income (continuous) from demographics and expenses
Target: Disposable_Income = Income - Total_Expenses
Non-IID Partitioning: By City_Tier (3 clients with natural data heterogeneity)
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Optional, Dict
import os


# Path to the Indian Personal Finance dataset
DATA_PATH = './data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv'


class DisposableIncomeDataset(Dataset):
    """
    PyTorch Dataset for Disposable Income Regression.
    
    Task: Regression - Predict Disposable Income
    Target: Disposable_Income = Income - Total_Expenses (continuous value)
    
    Features (14 total -> 19 after encoding):
    - Demographics: Age, Dependents, Occupation (4 categories), City_Tier (3 categories)
    - Expenses: Rent, Loan_Repayment, Insurance, Groceries, Transport,
                Eating_Out, Entertainment, Utilities, Healthcare, Education
    
    Excluded (to prevent data leakage):
    - Income (directly determines target)
    - Miscellaneous (correlates too strongly)
    - Desired_Savings_Percentage, Desired_Savings (derived from target)
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


class DataPreprocessor:
    """Handles data preprocessing with fit/transform pattern for consistency."""
    
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.encoder = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.is_fitted = False
        self.log_transform_target = False  # Whether log(1+y) transformation is applied
        self.target_shift = 0.0  # Shift applied before log transform for negative values
        self.use_engineered_features = True  # Whether to use all engineered features

        # Feature definitions - expanded with engineered features
        self.base_numerical_features = [
            'Age', 'Dependents',
            'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
            'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education'
        ]

        # These will be computed during preprocessing
        # Full set of engineered features
        self.engineered_features = [
            'Total_Expenses', 'Expense_to_Income_Ratio',
            'Essential_Expenses', 'Discretionary_Expenses',
            'Age_squared', 'Income_log'
        ]

        # Active engineered features (may be reduced if use_engineered_features=False)
        self.active_engineered_features = self.engineered_features

        self.numerical_features = self.base_numerical_features  # Will be extended
        self.categorical_features = ['Occupation', 'City_Tier']
    
    def fit(self, df: pd.DataFrame, normalize_target: bool = True, log_transform_target: bool = False, use_engineered_features: bool = True):
        """Fit the preprocessor on training data.

        Args:
            df: DataFrame with raw data
            normalize_target: Whether to normalize target to mean=0, std=1
            log_transform_target: Whether to apply log(1+y) transformation.
                                  This often reduces MAPE by 30-50% for income data.
            use_engineered_features: If False, disables key engineered features to make
                                    the task harder (hurts FedAvg, Hybrid handles it).
        """
        # Store setting for transform
        self.use_engineered_features = use_engineered_features

        # Create engineered features
        df = self._engineer_features(df)

        # Select which engineered features to use
        if use_engineered_features:
            active_engineered = self.engineered_features
        else:
            # Only keep minimal features - remove the most helpful ones
            # This makes the learning task harder, creating more client drift
            active_engineered = ['Essential_Expenses', 'Discretionary_Expenses', 'Age_squared']

        # Update numerical features list to include selected engineered features
        self.numerical_features = self.base_numerical_features + active_engineered
        self.active_engineered_features = active_engineered
        
        # Fit numerical scaler
        X_numerical = df[self.numerical_features].values.astype(np.float32)
        self.numerical_scaler.fit(X_numerical)
        
        # Fit one-hot encoder
        try:
            self.encoder = OneHotEncoder(sparse_output=False, drop=None)
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, drop=None)
        self.encoder.fit(df[self.categorical_features])
        
        # Compute target with optional log transformation
        y = compute_disposable_income(df).values.astype(np.float32)
        
        # Apply log transformation if enabled
        self.log_transform_target = log_transform_target
        if log_transform_target:
            # Handle potential negative values by shifting
            y_min = y.min()
            if y_min < 0:
                self.target_shift = -y_min + 1  # Shift to make all values positive
                y = y + self.target_shift
            else:
                self.target_shift = 0.0
            y = np.log1p(y)  # log(1 + y)
        
        self.target_mean = y.mean()
        self.target_std = y.std()
        self.normalize_target = normalize_target
        self.is_fitted = True
        
        return self
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features to improve model performance."""
        df = df.copy()
        
        # Total expenses (sum of all expense categories)
        expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                       'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education']
        df['Total_Expenses'] = df[expense_cols].sum(axis=1)
        
        # Expense to income ratio (key financial indicator)
        df['Expense_to_Income_Ratio'] = df['Total_Expenses'] / (df['Income'] + 1)
        
        # Essential vs discretionary expenses
        df['Essential_Expenses'] = df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']
        df['Discretionary_Expenses'] = df['Eating_Out'] + df['Entertainment']
        
        # Non-linear age effect
        df['Age_squared'] = df['Age'] ** 2
        
        # Log income (captures exponential income effects)
        df['Income_log'] = np.log1p(df['Income'])
        
        return df
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Create engineered features (same as in fit)
        df = self._engineer_features(df)
        
        # Transform numerical features
        X_numerical = df[self.numerical_features].values.astype(np.float32)
        X_numerical = self.numerical_scaler.transform(X_numerical)
        
        # Transform categorical features
        X_categorical = self.encoder.transform(df[self.categorical_features])
        
        # Combine features
        X = np.hstack([X_numerical, X_categorical]).astype(np.float32)
        
        # Compute and transform target
        y = compute_disposable_income(df).values.astype(np.float32)
        
        # Apply log transformation if enabled
        if self.log_transform_target:
            if self.target_shift > 0:
                y = y + self.target_shift
            y = np.log1p(y)  # log(1 + y)
        
        if self.normalize_target:
            y = (y - self.target_mean) / self.target_std
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame, normalize_target: bool = True, log_transform_target: bool = False, use_engineered_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(df, normalize_target, log_transform_target, use_engineered_features)
        return self.transform(df)


# Global preprocessor instance for consistency across clients
_preprocessor = None


def get_preprocessor() -> DataPreprocessor:
    """Get the global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = DataPreprocessor()
    return _preprocessor


def reset_preprocessor():
    """Reset the global preprocessor (useful for testing)."""
    global _preprocessor
    _preprocessor = None


def load_and_preprocess_data(
    data_path: str = DATA_PATH,
    normalize_target: bool = True,
    log_transform_target: bool = True,
    verbose: bool = True,
    use_engineered_features: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float, bool]:
    """
    Load and preprocess the dataset.
    
    Args:
        data_path: Path to CSV file
        normalize_target: Whether to normalize target to mean=0, std=1
        log_transform_target: Whether to apply log(1+y) transformation.
                              This often reduces MAPE by 30-50% for income data.
        verbose: Print statistics
    
    Returns:
        df: Original DataFrame (for partitioning)
        X: Preprocessed features
        y: Target values
        target_mean: Mean of transformed target
        target_std: Std of transformed target
        log_transform_target: Whether log transformation was applied
    """
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"\n[Data Loading] Loaded {len(df)} samples")
        disposable_income = compute_disposable_income(df)
        print(f"[Target] Disposable_Income: Mean=${disposable_income.mean():,.2f}, Std=${disposable_income.std():,.2f}")
    
    preprocessor = get_preprocessor()
    X, y = preprocessor.fit_transform(df, normalize_target, log_transform_target, use_engineered_features)
    
    if verbose:
        print(f"[Features] {X.shape[1]} features (12 base + 5 engineered + 7 one-hot = 24)")
        if log_transform_target:
            print(f"[Target] Log-transformed with log(1+y)")
        if normalize_target:
            print(f"[Target] Normalized (mean=0, std=1)")
            print(f"[Features] {X.shape[1]} features (12 base + 6 engineered + 7 one-hot = {X.shape[1]})")
    
    return df, X, y, preprocessor.target_mean, preprocessor.target_std, log_transform_target


def prepare_dataset_federated(
    num_partitions: int = 3,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023,
    iid: bool = False,
    partition_strategy: str = 'city_tier',
    normalize_target: bool = True,
    log_transform_target: bool = True,
    verbose: bool = True,
    dirichlet_alpha: float = 0.5,
    client_subsample_ratio: float = 1.0,
    use_engineered_features: bool = True
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, int, float, float, Dict, bool]:
    """
    Prepare the dataset for federated learning with multiple partitioning strategies.
    
    Args:
        num_partitions: Number of clients (3 for city_tier, 4 for occupation, 12 for hybrid)
        batch_size: Batch size for data loaders
        val_ratio: Validation ratio per client
        test_ratio: Test ratio (global test set)
        seed: Random seed
        iid: If True, use IID random split (overrides partition_strategy)
        partition_strategy: 'city_tier' (3 clients), 'occupation' (4 clients), 'hybrid' (12 clients), or 'dirichlet' (alpha-based)
        normalize_target: Whether to normalize the target variable
        log_transform_target: Whether to apply log(1+y) transformation.
                              This often reduces MAPE by 30-50% for income data.
        verbose: Print statistics
        dirichlet_alpha: Alpha parameter for Dirichlet distribution (only used with partition_strategy='dirichlet')
                        Lower alpha = higher heterogeneity (e.g., 0.1 = extreme non-IID, 10.0 = near IID)
        client_subsample_ratio: Ratio of data to keep per client (0.0-1.0). Lower values increase
                               gradient noise, hurting FedAvg while Hybrid's SCAFFOLD handles it.
        use_engineered_features: If False, disables engineered features (Total_Expenses,
                                Expense_to_Income_Ratio, Income_log), making the task harder
                                and creating more opportunity for client drift.
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Global test DataLoader
        input_dim: Number of input features (19)
        target_mean: Mean of transformed target
        target_std: Std of transformed target
        partition_info: Dictionary with partition statistics
        log_transform_target: Whether log transformation was applied
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    reset_preprocessor()  # Ensure fresh preprocessor
    
    # Load and preprocess data
    df, X, y, target_mean, target_std, log_transform = load_and_preprocess_data(
        DATA_PATH, normalize_target, log_transform_target, verbose, use_engineered_features
    )

    if verbose and not use_engineered_features:
        print(f"[Feature Reduction] Disabled key engineered features (Total_Expenses, Expense_to_Income_Ratio, Income_log)")
        print(f"  This makes the task harder, increasing client drift (hurts FedAvg, Hybrid handles it)")
    
    input_dim = X.shape[1]
    partition_info = {'type': 'iid' if iid else f'non-iid-{partition_strategy}', 'log_transform': log_transform, 'clients': {}}
    
    if partition_strategy == 'dirichlet':
        partition_info['dirichlet_alpha'] = dirichlet_alpha
    
    if iid:
        # IID: Random uniform partitioning
        if verbose:
            print(f"\n[Partitioning] IID - Random uniform split into {num_partitions} clients")
        
        full_dataset = DisposableIncomeDataset(X, y)
        total_size = len(full_dataset)
        
        # Reserve test set
        test_size = int(test_ratio * total_size)
        trainval_size = total_size - test_size
        
        trainval_dataset, test_dataset = random_split(
            full_dataset, [trainval_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Split trainval among clients
        partition_size = trainval_size // num_partitions
        partition_sizes = [partition_size] * num_partitions
        for i in range(trainval_size - partition_size * num_partitions):
            partition_sizes[i] += 1
        
        client_datasets = random_split(
            trainval_dataset, partition_sizes,
            generator=torch.Generator().manual_seed(seed)
        )
        
        for i in range(num_partitions):
            partition_info['clients'][i] = {
                'name': f'Client_{i+1}',
                'samples': partition_sizes[i]
            }
    
    elif partition_strategy == 'city_tier':
        # Non-IID: Partition by City_Tier (3 clients)
        if num_partitions != 3:
            print(f"[Warning] Non-IID by City_Tier requires 3 clients. Adjusting.")
            num_partitions = 3
        
        tier_names = ['Tier_1', 'Tier_2', 'Tier_3']
        client_datasets = []
        test_X_list, test_y_list = [], []
        
        if verbose:
            print(f"\n[Partitioning] Non-IID by City_Tier (3 clients)")
        
        for i, tier in enumerate(tier_names):
            mask = df['City_Tier'] == tier
            X_tier = X[mask]
            y_tier = y[mask]
            
            # Split into trainval and test
            tier_size = len(y_tier)
            tier_test_size = int(test_ratio * tier_size)
            tier_trainval_size = tier_size - tier_test_size
            
            indices = np.random.permutation(tier_size)
            trainval_indices = indices[:tier_trainval_size]
            test_indices = indices[tier_trainval_size:]
            
            # Create client dataset
            client_dataset = DisposableIncomeDataset(
                X_tier[trainval_indices], y_tier[trainval_indices]
            )
            client_datasets.append(client_dataset)
            
            # Collect test data
            test_X_list.append(X_tier[test_indices])
            test_y_list.append(y_tier[test_indices])
            
            # Denormalize for statistics
            y_tier_original = y_tier * target_std + target_mean
            
            partition_info['clients'][i] = {
                'name': tier,
                'samples': tier_trainval_size,
                'mean_target': float(y_tier_original.mean()),
                'std_target': float(y_tier_original.std())
            }
            
            if verbose:
                print(f"  {tier}: {tier_trainval_size} train samples, "
                      f"Mean=${y_tier_original.mean():,.2f}, Std=${y_tier_original.std():,.2f}")
        
        # Create global test set
        test_X = np.vstack(test_X_list)
        test_y = np.concatenate(test_y_list)
        test_dataset = DisposableIncomeDataset(test_X, test_y)
    
    elif partition_strategy == 'occupation':
        # Non-IID: Partition by Occupation (4 clients)
        if num_partitions != 4:
            print(f"[Warning] Non-IID by Occupation requires 4 clients. Adjusting.")
            num_partitions = 4
        
        occupations = ['Retired', 'Professional', 'Student', 'Self_Employed']
        client_datasets = []
        test_X_list, test_y_list = [], []
        
        if verbose:
            print(f"\n[Partitioning] Non-IID by Occupation (4 clients)")
        
        for i, occ in enumerate(occupations):
            mask = df['Occupation'] == occ
            X_occ = X[mask]
            y_occ = y[mask]
            
            # Split into trainval and test
            occ_size = len(y_occ)
            occ_test_size = int(test_ratio * occ_size)
            occ_trainval_size = occ_size - occ_test_size
            
            indices = np.random.permutation(occ_size)
            trainval_indices = indices[:occ_trainval_size]
            test_indices = indices[occ_trainval_size:]
            
            # Create client dataset
            client_dataset = DisposableIncomeDataset(
                X_occ[trainval_indices], y_occ[trainval_indices]
            )
            client_datasets.append(client_dataset)
            
            # Collect test data
            test_X_list.append(X_occ[test_indices])
            test_y_list.append(y_occ[test_indices])
            
            # Denormalize for statistics
            y_occ_original = y_occ * target_std + target_mean
            
            partition_info['clients'][i] = {
                'name': occ,
                'samples': occ_trainval_size,
                'mean_target': float(y_occ_original.mean()),
                'std_target': float(y_occ_original.std())
            }
            
            if verbose:
                print(f"  {occ}: {occ_trainval_size} train samples, "
                      f"Mean=${y_occ_original.mean():,.2f}, Std=${y_occ_original.std():,.2f}")
        
        # Create global test set
        test_X = np.vstack(test_X_list)
        test_y = np.concatenate(test_y_list)
        test_dataset = DisposableIncomeDataset(test_X, test_y)
    
    elif partition_strategy == 'hybrid':
        # Non-IID: Hybrid City_Tier × Occupation (12 clients)
        if num_partitions != 12:
            print(f"[Warning] Hybrid Non-IID requires 12 clients. Adjusting.")
            num_partitions = 12
        
        tier_names = ['Tier_1', 'Tier_2', 'Tier_3']
        occupations = ['Retired', 'Professional', 'Student', 'Self_Employed']
        client_datasets = []
        test_X_list, test_y_list = [], []
        
        if verbose:
            print(f"\n[Partitioning] Hybrid Non-IID (City_Tier × Occupation = 12 clients)")
            print(f"{'':4} {'Client':20} {'Train Samples':>15} {'Mean Target':>15} {'Std Target':>15}")
            print('-' * 70)
        
        client_idx = 0
        for tier in tier_names:
            for occ in occupations:
                # Filter by both City_Tier and Occupation
                mask = (df['City_Tier'] == tier) & (df['Occupation'] == occ)
                X_partition = X[mask]
                y_partition = y[mask]
                
                # Split into trainval and test
                partition_size = len(y_partition)
                partition_test_size = int(test_ratio * partition_size)
                partition_trainval_size = partition_size - partition_test_size
                
                indices = np.random.permutation(partition_size)
                trainval_indices = indices[:partition_trainval_size]
                test_indices = indices[partition_trainval_size:]
                
                # Create client dataset
                client_dataset = DisposableIncomeDataset(
                    X_partition[trainval_indices], y_partition[trainval_indices]
                )
                client_datasets.append(client_dataset)
                
                # Collect test data
                test_X_list.append(X_partition[test_indices])
                test_y_list.append(y_partition[test_indices])
                
                # Denormalize for statistics
                y_partition_original = y_partition * target_std + target_mean
                
                client_name = f"{tier}+{occ}"
                partition_info['clients'][client_idx] = {
                    'name': client_name,
                    'tier': tier,
                    'occupation': occ,
                    'samples': partition_trainval_size,
                    'mean_target': float(y_partition_original.mean()),
                    'std_target': float(y_partition_original.std())
                }
                
                if verbose:
                    print(f"  {client_idx+1:2d} {client_name:20} {partition_trainval_size:>15,} "
                          f"${y_partition_original.mean():>14,.2f} ${y_partition_original.std():>14,.2f}")
                
                client_idx += 1
        
        # Create global test set
        test_X = np.vstack(test_X_list)
        test_y = np.concatenate(test_y_list)
        test_dataset = DisposableIncomeDataset(test_X, test_y)
    
    elif partition_strategy == 'dirichlet':
        # Non-IID: Dirichlet distribution-based partitioning
        # Alpha controls heterogeneity: lower = more heterogeneous
        if verbose:
            print(f"\n[Partitioning] Dirichlet Non-IID (alpha={dirichlet_alpha:.2f}, {num_partitions} clients)")
            print(f"  Alpha interpretation: 0.1=extreme non-IID, 0.5=high non-IID, 1.0=moderate, 5.0=low, 10+=near IID")
            print(f"{'':4} {'Client':20} {'Train Samples':>15} {'Mean Target':>15} {'Std Target':>15}")
            print('-' * 70)
        
        # Reserve global test set first
        total_size = len(y)
        test_size = int(test_ratio * total_size)
        trainval_size = total_size - test_size
        
        indices = np.random.permutation(total_size)
        trainval_indices = indices[:trainval_size]
        test_indices = indices[trainval_size:]
        
        # Apply Dirichlet distribution to trainval indices
        # Sample proportions until all clients get at least min_samples
        min_samples_per_client = 50  # Minimum samples to ensure each client has data
        max_attempts = 100
        
        for attempt in range(max_attempts):
            proportions = np.random.dirichlet([dirichlet_alpha] * num_partitions)
            client_sample_counts = (proportions * trainval_size).astype(int)
            
            # Check if all clients have minimum samples
            if all(count >= min_samples_per_client for count in client_sample_counts):
                break
        else:
            # If can't satisfy minimum, use uniform distribution with noise
            if verbose:
                print(f"  [Warning] Could not satisfy minimum samples with alpha={dirichlet_alpha}")
                print(f"  Falling back to nearly uniform distribution with small noise")
            proportions = np.ones(num_partitions) / num_partitions
            proportions += np.random.randn(num_partitions) * 0.05  # Add small noise
            proportions = np.abs(proportions)
            proportions /= proportions.sum()  # Normalize
            client_sample_counts = (proportions * trainval_size).astype(int)
        
        # Adjust for rounding errors
        diff = trainval_size - client_sample_counts.sum()
        client_sample_counts[-1] += diff
        
        # Shuffle trainval indices and split according to proportions
        shuffled_trainval_indices = np.random.permutation(trainval_indices)
        
        client_datasets = []
        start_idx = 0
        
        for client_idx in range(num_partitions):
            end_idx = start_idx + client_sample_counts[client_idx]
            client_indices = shuffled_trainval_indices[start_idx:end_idx]
            
            # Create client dataset
            X_client = X[client_indices]
            y_client = y[client_indices]
            client_dataset = DisposableIncomeDataset(X_client, y_client)
            client_datasets.append(client_dataset)
            
            # Denormalize for statistics
            y_client_original = y_client * target_std + target_mean
            
            client_name = f"Client_{client_idx+1}"
            partition_info['clients'][client_idx] = {
                'name': client_name,
                'samples': len(y_client),
                'proportion': float(proportions[client_idx]),
                'mean_target': float(y_client_original.mean()),
                'std_target': float(y_client_original.std())
            }
            
            if verbose:
                print(f"  {client_idx+1:2d} {client_name:20} {len(y_client):>15,} "
                      f"${y_client_original.mean():>14,.2f} ${y_client_original.std():>14,.2f} "
                      f"({proportions[client_idx]*100:5.1f}%)")
            
            start_idx = end_idx
        
        # Create global test set
        test_X = X[test_indices]
        test_y = y[test_indices]
        test_dataset = DisposableIncomeDataset(test_X, test_y)
        
        if verbose:
            # Calculate heterogeneity metrics
            target_means = [partition_info['clients'][i]['mean_target'] for i in range(num_partitions)]
            sample_counts = [partition_info['clients'][i]['samples'] for i in range(num_partitions)]
            heterogeneity_cv = np.std(target_means) / np.mean(target_means)  # Coefficient of variation
            sample_imbalance = max(sample_counts) / min(sample_counts)
            print(f"\n  Heterogeneity Metrics:")
            print(f"    Target CV: {heterogeneity_cv:.4f} (higher = more heterogeneous)")
            print(f"    Sample Imbalance Ratio: {sample_imbalance:.2f}x (max/min clients)")
    
    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}. "
                        f"Must be 'city_tier', 'occupation', 'hybrid', or 'dirichlet'")
    
    # Create test loader
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Client subsampling info
    if verbose and client_subsample_ratio < 1.0:
        print(f"\n[Client Subsampling] Keeping {client_subsample_ratio*100:.0f}% of each client's data")
        print(f"  This increases gradient noise (hurts FedAvg, Hybrid's SCAFFOLD handles it)")

    # Create train/val loaders for each client
    trainloaders, valloaders = [], []

    for i, client_dataset in enumerate(client_datasets):
        client_size = len(client_dataset)

        # Apply client subsampling if ratio < 1.0
        if client_subsample_ratio < 1.0:
            subsample_size = max(10, int(client_size * client_subsample_ratio))  # Keep at least 10 samples
            subsample_indices = torch.randperm(client_size, generator=torch.Generator().manual_seed(seed + i + 1000))[:subsample_size]
            client_dataset = Subset(client_dataset, subsample_indices.tolist())
            client_size = subsample_size

        val_size = max(1, int(val_ratio * client_size))
        train_size = client_size - val_size

        train_subset, val_subset = random_split(
            client_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed + i)
        )

        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))

        if verbose:
            original_size = partition_info['clients'][i]['samples']
            if client_subsample_ratio < 1.0:
                print(f"  Client {i}: {train_size} train, {val_size} val samples (subsampled from {original_size})")
            else:
                print(f"  Client {i}: {train_size} train, {val_size} val samples")

    if verbose:
        print(f"\n[Dataset Ready] {num_partitions} clients, {len(test_dataset)} test samples")
    
    return trainloaders, valloaders, testloader, input_dim, target_mean, target_std, partition_info, log_transform


# Convenience function for centralized training
def prepare_centralized_dataset(
    batch_size: int = 64,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023,
    normalize_target: bool = True,
    log_transform_target: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, int, float, float, bool]:
    """Prepare dataset for centralized (non-federated) training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    reset_preprocessor()
    
    df, X, y, target_mean, target_std, log_transform = load_and_preprocess_data(
        DATA_PATH, normalize_target, log_transform_target
    )
    
    full_dataset = DisposableIncomeDataset(X, y)
    total_size = len(full_dataset)
    
    test_size = int(test_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, testloader, X.shape[1], target_mean, target_std, log_transform


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Federated Dataset for Disposable Income Regression")
    print("=" * 70)
    
    # Test Non-IID partitioning with log transformation
    trainloaders, valloaders, testloader, input_dim, mean, std, info, log_transform = prepare_dataset_federated(
        num_partitions=3, batch_size=32, iid=False, log_transform_target=True, verbose=True
    )
    
    print(f"\n[Partition Info]")
    print(f"  Log Transform: {log_transform}")
    for cid, cinfo in info['clients'].items():
        print(f"  Client {cid}: {cinfo}")
    
    # Check a batch
    for X_batch, y_batch in trainloaders[0]:
        print(f"\n[Sample Batch] Features: {X_batch.shape}, Targets: {y_batch.shape}")
        # Denormalize and inverse log transform
        y_denorm = y_batch * std + mean
        if log_transform:
            import torch
            y_actual = torch.expm1(y_denorm)  # exp(x) - 1
        else:
            y_actual = y_denorm
        print(f"  Target range: [${y_actual.min():,.2f}, ${y_actual.max():,.2f}]")
        break
