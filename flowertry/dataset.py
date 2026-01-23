import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Tuple


class PersonalFinanceDataset(Dataset):
    """Dataset for Disposable Income regression task."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, target_scaler=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.target_scaler = target_scaler
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_and_preprocess_data(data_path: str, scale_target: bool = True) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, StandardScaler]:
    """
    Load and preprocess the Indian Personal Finance dataset.
    
    Args:
        data_path: Path to CSV file
        scale_target: If True, scale the target variable using StandardScaler
    
    Returns:
        features: Preprocessed feature array
        targets: Disposable_Income values (scaled if scale_target=True)
        preprocessor: Fitted preprocessor for features
        target_scaler: Fitted scaler for target (None if scale_target=False)
    """
    df = pd.read_csv(data_path)
    
    numerical_features = [
        'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
        'Utilities', 'Healthcare', 'Education'
    ]
    
    categorical_features = ['Occupation', 'City_Tier']
    
    X = df[numerical_features + categorical_features].copy()
    y = df['Disposable_Income'].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop=None, sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    target_scaler = None
    if scale_target:
        target_scaler = StandardScaler()
        y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_processed, y, preprocessor, target_scaler


def prepare_dataset_pure_partitioning(
    num_clients: int,
    batch_size: int,
    data_path: str = './data/IndianPersonalFinance/indianPersonalFinanceAndSpendingHabits.csv',
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, StandardScaler, dict]:
    """
    Pure tier partitioning for stratified client selection with FedAvg.
    
    Each client receives data from ONLY one City Tier (maximum heterogeneity).
    This creates the most challenging non-IID scenario for testing stratified
    client selection benefits.
    
    Client distribution:
        - Tier_1: 4 clients
        - Tier_2: 6 clients  
        - Tier_3: 4 clients
        - Total: 14 clients
    
    Args:
        num_clients: Total number of clients (should be 14 for default distribution)
        batch_size: Batch size for DataLoaders
        data_path: Path to the CSV dataset
        val_ratio: Validation split ratio per client
        test_ratio: Global test set ratio
        seed: Random seed for reproducibility
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Global test DataLoader
        target_scaler: StandardScaler fitted on targets for inverse transform
        client_strata: Dictionary mapping each tier to list of client IDs
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load and preprocess
    X, y, preprocessor, target_scaler = load_and_preprocess_data(data_path, scale_target=True)
    df = pd.read_csv(data_path)
    
    # Train/test split
    n_total = len(df)
    n_test = int(test_ratio * n_total)
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_val_indices = indices[n_test:]
    
    X_test, y_test = X[test_indices], y[test_indices]
    X_train_val, y_train_val = X[train_val_indices], y[train_val_indices]
    
    # Test loader
    test_dataset = PersonalFinanceDataset(X_test, y_test, target_scaler=target_scaler)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get tiers for train_val set
    df_train_val = df.iloc[train_val_indices]
    city_tiers = df_train_val['City_Tier'].values
    
    # Define client distribution for pure tier partitioning
    # Each tier gets a proportional number of clients based on data distribution
    clients_per_tier = {'Tier_1': 4, 'Tier_2': 6, 'Tier_3': 4}  # Total: 14
    tier_order = ['Tier_1', 'Tier_2', 'Tier_3']
    
    trainloaders, valloaders = [], []
    client_strata = {tier: [] for tier in tier_order}
    client_id = 0
    
    for tier in tier_order:
        # Get data for this tier only (pure partitioning - no mixing across tiers)
        tier_mask = (city_tiers == tier)
        tier_indices = np.where(tier_mask)[0]
        X_tier = X_train_val[tier_indices]
        y_tier = y_train_val[tier_indices]
        
        # Shuffle tier data before splitting among clients
        shuffle_idx = np.random.permutation(len(X_tier))
        X_tier, y_tier = X_tier[shuffle_idx], y_tier[shuffle_idx]
        
        # Divide tier data equally among clients assigned to this tier
        n_clients = clients_per_tier[tier]
        chunk_size = len(X_tier) // n_clients
        
        for i in range(n_clients):
            start = i * chunk_size
            end = start + chunk_size if i < n_clients - 1 else len(X_tier)
            
            X_client, y_client = X_tier[start:end], y_tier[start:end]
            
            # Train/val split
            n_val = int(val_ratio * len(X_client))
            
            train_ds = PersonalFinanceDataset(X_client[:-n_val], y_client[:-n_val])
            val_ds = PersonalFinanceDataset(X_client[-n_val:], y_client[-n_val:])
            
            trainloaders.append(DataLoader(train_ds, batch_size, shuffle=True))
            valloaders.append(DataLoader(val_ds, batch_size, shuffle=False))
            
            client_strata[tier].append(client_id)
            client_id += 1
    
    return trainloaders, valloaders, testloader, target_scaler, client_strata


# Alias for backward compatibility
prepare_dataset = prepare_dataset_pure_partitioning