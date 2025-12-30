import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List

# Define the path to the dataset
DATA_PATH = './data/indianPersonalFinanceAndSpendingHabits/indianPersonalFinanceAndSpendingHabits.csv'


class PersonalFinanceDataset(Dataset):
    """
    PyTorch Dataset for Indian Personal Finance and Spending Habits.
    
    Task 1: Savings Potential Classification
    - Target: Desired_Savings_Percentage discretized into 3 classes
        - Low: <7%
        - Medium: 7-12%
        - High: >12%
    - Input Features: Income, Age, Dependents, Occupation, City_Tier, spending categories
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def discretize_savings(savings_percentage: pd.Series) -> np.ndarray:
    """
    Discretize Desired_Savings_Percentage into 3 classes:
    - 0 (Low): <7%
    - 1 (Medium): 7-12%
    - 2 (High): >12%
    """
    labels = np.zeros(len(savings_percentage), dtype=np.int64)
    labels[savings_percentage >= 7] = 1   # Medium
    labels[savings_percentage > 12] = 2   # High
    return labels


def load_and_preprocess_data(data_path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load and preprocess the Personal Finance dataset.
    
    Returns:
        features: Preprocessed feature matrix
        labels: Discretized class labels (0, 1, 2)
        num_features: Number of input features
        num_classes: Number of output classes (3)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Define feature columns for Task 1
    numerical_features = [
        'Income', 'Age', 'Dependents',
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
        'Education', 'Miscellaneous'
    ]
    
    categorical_features = ['Occupation', 'City_Tier']
    target_col = 'Desired_Savings_Percentage'
    
    # Extract numerical features
    X_numerical = df[numerical_features].values
    
    # Encode categorical features
    label_encoders = {}
    X_categorical_list = []
    for col in categorical_features:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        X_categorical_list.append(encoded.reshape(-1, 1))
        label_encoders[col] = le
    
    X_categorical = np.hstack(X_categorical_list)
    
    # Combine features
    X = np.hstack([X_numerical, X_categorical])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Discretize target
    y = discretize_savings(df[target_col])
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nGlobal Class Distribution:")
    class_names = ['Low (<7%)', 'Medium (7-12%)', 'High (>12%)']
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({class_names[cls]}): {count} samples ({100*count/len(y):.1f}%)")
    
    num_features = X.shape[1]
    num_classes = 3
    
    return X, y, num_features, num_classes


def partition_iid(labels: np.ndarray, num_clients: int, seed: int = 2023) -> List[List[int]]:
    """
    IID partitioning: Randomly shuffle and distribute data equally among clients.
    
    Args:
        labels: Array of labels
        num_clients: Number of clients
        seed: Random seed
    
    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(labels))
    partition_size = len(labels) // num_clients
    
    client_indices = []
    for i in range(num_clients):
        start = i * partition_size
        end = start + partition_size if i < num_clients - 1 else len(labels)
        client_indices.append(indices[start:end].tolist())
    
    return client_indices


def partition_dirichlet(
    labels: np.ndarray, 
    num_clients: int, 
    alpha: float = 0.5, 
    seed: int = 2023
) -> List[List[int]]:
    """
    Non-IID partitioning using Dirichlet distribution.
    
    Creates label skew where each client has unbalanced class distributions.
    Lower alpha = more heterogeneous (non-IID)
    Higher alpha = more homogeneous (closer to IID)
    
    Args:
        labels: Array of labels
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed
    
    Returns:
        List of index lists, one per client
    """
    np.random.seed(seed)
    
    num_classes = len(np.unique(labels))
    label_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    # Initialize empty client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples according to Dirichlet distribution
    for c in range(num_classes):
        class_indices = label_indices[c].copy()
        np.random.shuffle(class_indices)
        
        # Sample from Dirichlet distribution to get proportions
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Ensure minimum samples per client (at least 1 if possible)
        proportions = np.array([max(p, 0.001) for p in proportions])
        proportions = proportions / proportions.sum()
        
        # Split indices according to proportions
        splits = (proportions * len(class_indices)).astype(int)
        splits[-1] = len(class_indices) - splits[:-1].sum()  # Ensure all samples used
        
        idx_start = 0
        for client_id, num_samples in enumerate(splits):
            client_indices[client_id].extend(
                class_indices[idx_start:idx_start + num_samples].tolist()
            )
            idx_start += num_samples
    
    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices


def print_client_distribution(client_indices: List[List[int]], labels: np.ndarray, num_clients_to_show: int = 5):
    """Print class distribution for each client."""
    print(f"\nClient Data Distribution (showing first {num_clients_to_show} clients):")
    class_names = ['Low', 'Med', 'High']
    
    for i, indices in enumerate(client_indices[:num_clients_to_show]):
        client_labels = labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        
        dist_str = ", ".join([
            f"{class_names[c]}: {counts[list(unique).index(c)] if c in unique else 0}"
            for c in range(3)
        ])
        print(f"  Client {i}: {len(indices)} samples [{dist_str}]")
    
    if len(client_indices) > num_clients_to_show:
        print(f"  ... and {len(client_indices) - num_clients_to_show} more clients")


def prepare_dataset(
    num_partitions: int,
    batch_size: int,
    partition_type: str = "iid",
    dirichlet_alpha: float = 0.5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, int, int]:
    """
    Prepare the Personal Finance dataset for federated learning.
    
    Supports both IID and Non-IID (Dirichlet) partitioning.
    
    Args:
        num_partitions: Number of clients/partitions
        batch_size: Batch size for data loaders
        partition_type: "iid" or "dirichlet"
        dirichlet_alpha: Dirichlet concentration (lower = more non-IID)
        val_ratio: Fraction of each client's data for validation
        test_ratio: Fraction of total data reserved for global test set
        seed: Random seed for reproducibility
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Global test DataLoader
        num_features: Number of input features
        num_classes: Number of output classes
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load and preprocess data
    X, y, num_features, num_classes = load_and_preprocess_data()
    
    # Create full dataset
    full_dataset = PersonalFinanceDataset(X, y)
    total_size = len(full_dataset)
    
    # Split into train+val and test sets
    test_size = int(test_ratio * total_size)
    trainval_size = total_size - test_size
    
    # Create indices for train/val and test
    all_indices = np.random.permutation(total_size)
    trainval_indices = all_indices[:trainval_size]
    test_indices = all_indices[trainval_size:]
    
    # Create test loader
    test_dataset = Subset(full_dataset, test_indices.tolist())
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get labels for trainval data (for partitioning)
    trainval_labels = y[trainval_indices]
    
    # Partition trainval data among clients
    print(f"\nPartitioning Strategy: {partition_type.upper()}")
    if partition_type == "dirichlet":
        print(f"Dirichlet Alpha: {dirichlet_alpha}")
        client_local_indices = partition_dirichlet(
            trainval_labels, num_partitions, alpha=dirichlet_alpha, seed=seed
        )
    else:  # IID
        client_local_indices = partition_iid(trainval_labels, num_partitions, seed=seed)
    
    # Map local indices back to global indices
    client_global_indices = [
        [trainval_indices[i] for i in local_idx]
        for local_idx in client_local_indices
    ]
    
    # Print distribution
    print_client_distribution(client_local_indices, trainval_labels)
    
    # Create train and validation loaders for each client
    trainloaders = []
    valloaders = []
    
    for global_indices in client_global_indices:
        client_size = len(global_indices)
        val_size = max(1, int(val_ratio * client_size))
        train_size = client_size - val_size
        
        # Split client data into train and validation
        np.random.shuffle(global_indices)
        train_indices = global_indices[:train_size]
        val_indices = global_indices[train_size:]
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
    
    avg_samples = sum(len(idx) for idx in client_global_indices) / num_partitions
    print(f"\nDataset Summary:")
    print(f"  Total samples: {total_size}")
    print(f"  Test samples: {test_size}")
    print(f"  Train+Val samples: {trainval_size}")
    print(f"  Clients: {num_partitions}")
    print(f"  Avg samples per client: {avg_samples:.0f}")
    print(f"  Features: {num_features}, Classes: {num_classes}")
    
    return trainloaders, valloaders, testloader, num_features, num_classes
