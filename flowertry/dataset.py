import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Tuple
import os

class PersonalFinanceDataset(Dataset):
    """Dataset for Disposable Income regression task."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, target_scaler=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.target_scaler = target_scaler  # Store scaler for inverse transform if needed
    
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
    # Load data
    df = pd.read_csv(data_path)
    
    # Define features according to documentation
    numerical_features = [
        'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
        'Utilities', 'Healthcare', 'Education'
    ]
    
    categorical_features = ['Occupation', 'City_Tier']
    
    # Excluded features (to prevent data leakage)
    excluded_features = [
        'Income', 'Miscellaneous', 'Desired_Savings_Percentage',
        'Desired_Savings'
    ]
    
    # Extract features and target
    X = df[numerical_features + categorical_features].copy()
    y = df['Disposable_Income'].values
    
    # Create preprocessor for features
    # Note: Using drop=None to get 19 features total (12 numerical + 7 one-hot)
    # Occupation: 4 categories -> 4 features, City_Tier: 3 categories -> 3 features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop=None, sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)
    
    # Scale target if requested
    target_scaler = None
    if scale_target:
        target_scaler = StandardScaler()
        y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X_processed, y, preprocessor, target_scaler


def partition_by_city_tier(df: pd.DataFrame) -> dict:
    """
    Partition dataset by City_Tier for Non-IID distribution.
    
    Returns:
        Dictionary with City_Tier as keys and filtered DataFrames as values
    """
    partitions = {}
    for tier in ['Tier_1', 'Tier_2', 'Tier_3']:
        partitions[tier] = df[df['City_Tier'] == tier].copy()
    return partitions


def prepare_dataset(
    num_clients: int,
    batch_size: int,
    data_path: str = './data/IndianPersonalFinance/indianPersonalFinanceAndSpendingHabits.csv',
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    non_iid: bool = True,
    alpha: float = 0.1,
    seed: int = 2023
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Prepare dataset for federated learning with Non-IID partitioning.
    
    Args:
        num_clients: Number of federated clients (should be 3 for Non-IID by City_Tier)
        batch_size: Batch size for training
        data_path: Path to CSV file
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        non_iid: If True, partition by City_Tier (Non-IID), else random (IID)
        alpha: Heterogeneity parameter (0.0 = pure Non-IID, 1.0 = more IID-like)
               Lower alpha = more heterogeneous, Higher alpha = more homogeneous
               Typical values: 0.1 (highly heterogeneous), 0.5 (moderate), 1.0 (nearly IID)
        seed: Random seed for reproducibility
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Single test DataLoader
        target_scaler: StandardScaler fitted on target (for inverse transform in metrics)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load and preprocess data (scale target for better training)
    X, y, preprocessor, target_scaler = load_and_preprocess_data(data_path, scale_target=True)
    
    # Load original dataframe for partitioning
    df = pd.read_csv(data_path)
    
    # Split into train/test
    n_total = len(df)
    n_test = int(test_ratio * n_total)
    n_train_val = n_total - n_test
    
    # Create indices for train/val and test
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    test_indices = indices[:n_test]
    train_val_indices = indices[n_test:]
    
    # Split features and targets
    X_train_val = X[train_val_indices]
    y_train_val = y[train_val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Create test dataset (store target_scaler for inverse transform in metrics)
    test_dataset = PersonalFinanceDataset(X_test, y_test, target_scaler=target_scaler)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Store target_scaler in dataset for later use in metrics
    # We'll need to pass this to the model evaluation function
    
    # Partition training data
    trainloaders = []
    valloaders = []
    
    if non_iid:
        # Non-IID: Partition by City_Tier with alpha-controlled heterogeneity
        # Get City_Tier values for train_val samples
        df_train_val = df.iloc[train_val_indices].copy()
        city_tiers = df_train_val['City_Tier'].values
        
        tier_order = ['Tier_1', 'Tier_2', 'Tier_3']
        num_tiers = len(tier_order)
        
        # Ensure num_clients is divisible by number of tiers for balanced partitioning
        if num_clients % num_tiers != 0:
            old_num = num_clients
            num_clients = (num_clients // num_tiers) * num_tiers
            if num_clients < num_tiers:
                num_clients = num_tiers
            print(f"Warning: Adjusting num_clients from {old_num} to {num_clients} for balanced tier distribution.")
        
        clients_per_tier = num_clients // num_tiers
        print(f"Non-IID partitioning: {num_clients} clients ({clients_per_tier} clients per tier)")
        
        # Create base partitions by City_Tier
        tier_partitions = {}
        for tier in tier_order:
            tier_mask = (city_tiers == tier)
            tier_indices = np.where(tier_mask)[0]
            tier_partitions[tier] = {
                'indices': tier_indices,
                'X': X_train_val[tier_indices],
                'y': y_train_val[tier_indices]
            }
        
        # Apply alpha-based mixing for heterogeneity control
        # alpha=0.0: Pure Non-IID (each client gets only one tier)
        # alpha=1.0: More IID-like (data mixed across tiers)
        client_data = {i: {'indices': [], 'X': [], 'y': []} for i in range(num_clients)}
        
        for tier_idx, tier in enumerate(tier_order):
            tier_data = tier_partitions[tier]
            n_tier_samples = len(tier_data['indices'])
            
            # Clients assigned to this tier
            tier_client_start = tier_idx * clients_per_tier
            tier_client_end = tier_client_start + clients_per_tier
            tier_clients = list(range(tier_client_start, tier_client_end))
            
            # Calculate how much data stays with primary tier clients vs gets mixed
            # Primary tier clients get (1-alpha) of the data, others share alpha portion
            primary_ratio = 1.0 - alpha
            n_primary = int(primary_ratio * n_tier_samples)
            
            # Split primary data among clients in this tier
            primary_indices = tier_data['indices'][:n_primary]
            primary_X = tier_data['X'][:n_primary]
            primary_y = tier_data['y'][:n_primary]
            
            # Distribute primary data evenly among tier clients
            samples_per_tier_client = len(primary_indices) // clients_per_tier
            for local_idx, client_id in enumerate(tier_clients):
                start_idx = local_idx * samples_per_tier_client
                end_idx = start_idx + samples_per_tier_client if local_idx < clients_per_tier - 1 else len(primary_indices)
                client_data[client_id]['indices'].extend(primary_indices[start_idx:end_idx])
                client_data[client_id]['X'].extend(primary_X[start_idx:end_idx])
                client_data[client_id]['y'].extend(primary_y[start_idx:end_idx])
            
            # Distribute remaining data to clients from other tiers (mixing)
            if alpha > 0 and n_tier_samples > n_primary:
                remaining_indices = tier_data['indices'][n_primary:]
                remaining_X = tier_data['X'][n_primary:]
                remaining_y = tier_data['y'][n_primary:]
                
                # Shuffle for random distribution
                shuffle_idx = np.random.permutation(len(remaining_indices))
                remaining_indices = remaining_indices[shuffle_idx]
                remaining_X = remaining_X[shuffle_idx]
                remaining_y = remaining_y[shuffle_idx]
                
                # Get clients from other tiers
                other_clients = [c for c in range(num_clients) if c not in tier_clients]
                if len(other_clients) > 0:
                    n_remaining = len(remaining_indices)
                    samples_per_other = n_remaining // len(other_clients)
                    
                    for j, other_client in enumerate(other_clients):
                        start_idx = j * samples_per_other
                        end_idx = start_idx + samples_per_other if j < len(other_clients) - 1 else n_remaining
                        if start_idx < n_remaining:
                            client_data[other_client]['indices'].extend(remaining_indices[start_idx:end_idx])
                            client_data[other_client]['X'].extend(remaining_X[start_idx:end_idx])
                            client_data[other_client]['y'].extend(remaining_y[start_idx:end_idx])
        
        # Create datasets for each client
        for i in range(num_clients):
            if len(client_data[i]['indices']) == 0 or len(client_data[i]['X']) == 0:
                continue
                
            # Convert lists to arrays
            # X is a list of arrays, need to stack them
            X_list = client_data[i]['X']
            if isinstance(X_list[0], np.ndarray):
                X_client = np.vstack(X_list)
            else:
                X_client = np.array(X_list)
            
            y_client = np.array(client_data[i]['y'])
            
            # Shuffle client data
            shuffle_idx = np.random.permutation(len(X_client))
            X_client = X_client[shuffle_idx]
            y_client = y_client[shuffle_idx]
            
            # Split into train and validation
            n_client = len(X_client)
            n_val_client = int(val_ratio * n_client)
            n_train_client = n_client - n_val_client
            
            client_train_dataset = PersonalFinanceDataset(
                X_client[:n_train_client],
                y_client[:n_train_client]
            )
            client_val_dataset = PersonalFinanceDataset(
                X_client[n_train_client:],
                y_client[n_train_client:]
            )
            
            trainloaders.append(
                DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
            )
            valloaders.append(
                DataLoader(client_val_dataset, batch_size=batch_size, shuffle=False)
            )
            
            # Calculate tier distribution for this client
            client_tier_indices = client_data[i]['indices']
            if len(client_tier_indices) > 0:
                client_tiers = city_tiers[client_tier_indices]
                tier_counts = {tier: np.sum(client_tiers == tier) for tier in tier_order}
                tier_dist = {k: v/len(client_tiers)*100 for k, v in tier_counts.items()}
                print(f"Client {i}: {n_train_client} train, {n_val_client} val samples | Tier distribution: {tier_dist}")
    else:
        # IID: Random partitioning
        n_train_val = len(X_train_val)
        n_per_client = n_train_val // num_clients
        
        for i in range(num_clients):
            start_idx = i * n_per_client
            end_idx = (i + 1) * n_per_client if i < num_clients - 1 else n_train_val
            
            X_client = X_train_val[start_idx:end_idx]
            y_client = y_train_val[start_idx:end_idx]
            
            # Split into train and validation
            n_client = len(X_client)
            n_val_client = int(val_ratio * n_client)
            n_train_client = n_client - n_val_client
            
            client_train_dataset = PersonalFinanceDataset(
                X_client[:n_train_client],
                y_client[:n_train_client]
            )
            client_val_dataset = PersonalFinanceDataset(
                X_client[n_train_client:],
                y_client[n_train_client:]
            )
            
            trainloaders.append(
                DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
            )
            valloaders.append(
                DataLoader(client_val_dataset, batch_size=batch_size, shuffle=False)
            )
    
    return trainloaders, valloaders, testloader, target_scaler
