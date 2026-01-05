import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


# Path to the Indian Personal Finance dataset
DATA_PATH = './data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv'


class PersonalFinanceDataset(Dataset):
    """
    PyTorch Dataset for Indian Personal Finance and Spending Habits.
    
    Task: Savings Potential Classification
    - Target: Desired_Savings_Percentage discretized into 3 classes
        - Class 0 (Low): < 7%
        - Class 1 (Medium): 7-12%
        - Class 2 (High): > 12%
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
    - 0 (Low): < 7%
    - 1 (Medium): 7-12%
    - 2 (High): > 12%
    """
    labels = np.zeros(len(savings_percentage), dtype=np.int64)
    labels[savings_percentage >= 7] = 1   # Medium
    labels[savings_percentage > 12] = 2   # High
    return labels


def load_and_preprocess_data(data_path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the Personal Finance dataset.
    
    Returns:
        features: Preprocessed feature matrix (N x 16)
        labels: Discretized class labels (0, 1, 2)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Define feature columns
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
    X_categorical_list = []
    for col in categorical_features:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        X_categorical_list.append(encoded.reshape(-1, 1))
    
    X_categorical = np.hstack(X_categorical_list)
    
    # Combine features (14 numerical + 2 categorical = 16 features)
    X = np.hstack([X_numerical, X_categorical])
    
    # Standardize all features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Discretize target into 3 classes
    y = discretize_savings(df[target_col])
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass Distribution:")
    class_names = ['Low (<7%)', 'Medium (7-12%)', 'High (>12%)']
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({class_names[cls]}): {count} samples ({100*count/len(y):.1f}%)")
    
    return X, y


def split_by_class(dataset: PersonalFinanceDataset, labels: np.ndarray):
    """
    Split dataset by class labels.
    
    Returns:
        class_datasets: List of datasets, one per class
    """
    class_datasets = {}
    for class_idx in np.unique(labels):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        class_features = dataset.features[class_indices]
        class_labels = dataset.labels[class_indices]
        class_datasets[class_idx] = PersonalFinanceDataset(
            class_features.numpy(), 
            class_labels.numpy()
        )
    return class_datasets


def create_non_iid_partitions(
    trainval_dataset: torch.utils.data.Subset,
    trainval_indices: np.ndarray,
    trainval_labels: np.ndarray,
    num_partitions: int,
    alpha: float = 0.5,
    seed: int = 2023
) -> Tuple[List[torch.utils.data.Subset], np.ndarray]:
    """
    Create non-IID partitions using Dirichlet distribution.
    
    Args:
        trainval_dataset: Training+validation dataset (Subset)
        trainval_indices: Indices of trainval samples in original dataset
        trainval_labels: Class labels for trainval samples
        num_partitions: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: Random seed
    
    Returns:
        Tuple of (client_datasets, proportions)
    """
    np.random.seed(seed)
    
    # Get unique classes
    num_classes = len(np.unique(trainval_labels))
    
    # Split data by class (using indices relative to trainval_dataset)
    class_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(trainval_labels):
        class_indices[int(label)].append(idx)  # idx is relative to trainval_dataset
    
    # Sample from Dirichlet distribution to get class proportions for each client
    # Shape: (num_partitions, num_classes)
    proportions = np.random.dirichlet([alpha] * num_classes, size=num_partitions)
    
    # Assign samples to clients based on proportions
    client_indices = [[] for _ in range(num_partitions)]
    
    for class_idx in range(num_classes):
        class_samples = class_indices[class_idx].copy()
        np.random.shuffle(class_samples)
        
        # Calculate how many samples each client should get for this class
        class_proportions = proportions[:, class_idx]
        class_proportions = class_proportions / class_proportions.sum()  # Normalize
        num_samples_per_client = (np.array(class_proportions) * len(class_samples)).astype(int)
        
        # Handle remainder
        remainder = len(class_samples) - num_samples_per_client.sum()
        if remainder > 0:
            num_samples_per_client[:remainder] += 1
        
        # Assign samples (indices are relative to trainval_dataset)
        start_idx = 0
        for client_idx in range(num_partitions):
            end_idx = start_idx + num_samples_per_client[client_idx]
            client_indices[client_idx].extend(class_samples[start_idx:end_idx])
            start_idx = end_idx
    
    # Create subsets (indices are relative to trainval_dataset)
    client_datasets = []
    for indices in client_indices:
        if len(indices) > 0:
            client_datasets.append(torch.utils.data.Subset(trainval_dataset, indices))
        else:
            # If a client has no samples, give it a small random subset
            random_indices = np.random.choice(len(trainval_dataset), size=min(10, len(trainval_dataset)), replace=False)
            client_datasets.append(torch.utils.data.Subset(trainval_dataset, random_indices))
    
    return client_datasets, proportions


def prepare_dataset(
    num_partitions: int,
    batch_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 2023,
    iid: bool = True,
    alpha: float = 0.5
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Prepare the Personal Finance dataset for federated learning.
    
    Args:
        num_partitions: Number of clients/partitions
        batch_size: Batch size for data loaders
        val_ratio: Fraction of each client's data for validation
        test_ratio: Fraction of total data reserved for global test set
        seed: Random seed for reproducibility
        iid: If True, use IID partitioning. If False, use non-IID (Dirichlet)
        alpha: Dirichlet concentration parameter for non-IID (lower = more heterogeneous)
              - alpha -> 0: extreme non-IID (each client has mostly one class)
              - alpha -> inf: approaches IID (uniform class distribution)
              - Typical values: 0.1 (very heterogeneous), 0.5 (moderate), 1.0 (mild)
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Global test DataLoader
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Create full dataset
    full_dataset = PersonalFinanceDataset(X, y)
    total_size = len(full_dataset)
    
    # Split into train+val and test sets
    test_size = int(test_ratio * total_size)
    trainval_size = total_size - test_size
    
    trainval_dataset, test_dataset = random_split(
        full_dataset, 
        [trainval_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create test loader
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get labels for trainval dataset
    trainval_indices = np.array(trainval_dataset.indices)  # Convert to numpy array for indexing
    trainval_labels = y[trainval_indices]
    
    # Split trainval into partitions for clients
    if iid:
        # IID partitioning: random split
        partition_size = trainval_size // num_partitions
        partition_sizes = [partition_size] * num_partitions
        
        # Handle remainder
        remainder = trainval_size - (partition_size * num_partitions)
        for i in range(remainder):
            partition_sizes[i] += 1
        
        client_datasets = random_split(
            trainval_dataset,
            partition_sizes,
            generator=torch.Generator().manual_seed(seed)
        )
        proportions = None
        print(f"\nData Distribution: IID (uniform random split)")
    else:
        # Non-IID partitioning: Dirichlet distribution
        client_datasets, proportions = create_non_iid_partitions(
            trainval_dataset,
            trainval_indices,
            trainval_labels,
            num_partitions,
            alpha=alpha,
            seed=seed
        )
        print(f"\nData Distribution: Non-IID (Dirichlet, alpha={alpha})")
        
        # Print class distribution for first few clients
        print("\nClass Distribution for First 5 Clients:")
        class_names = ['Low (<7%)', 'Medium (7-12%)', 'High (>12%)']
        for client_idx in range(min(5, num_partitions)):
            # Get indices relative to trainval_dataset
            client_subset_indices = np.array(client_datasets[client_idx].indices)  # Convert to numpy array
            # Map to original dataset indices
            original_indices = trainval_indices[client_subset_indices]
            # Get labels from original dataset
            client_labels = y[original_indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            total = len(client_labels)
            
            print(f"\n  Client {client_idx}:")
            for cls in range(len(class_names)):
                count = counts[unique == cls][0] if cls in unique else 0
                pct = (count / total * 100) if total > 0 else 0
                print(f"    {class_names[cls]}: {count} samples ({pct:.1f}%)")
    
    # Create train and validation loaders for each client
    trainloaders = []
    valloaders = []
    
    for client_dataset in client_datasets:
        client_size = len(client_dataset)
        val_size = max(1, int(val_ratio * client_size))
        train_size = client_size - val_size
        
        train_subset, val_subset = random_split(
            client_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
    
    print(f"\nDataset Prepared:")
    print(f"  Total samples: {total_size}")
    print(f"  Test samples: {test_size}")
    print(f"  Train+Val samples: {trainval_size}")
    print(f"  Number of clients: {num_partitions}")
    if iid:
        partition_size = trainval_size // num_partitions
        print(f"  Samples per client: ~{partition_size}")
    else:
        # Show sample size range for non-IID
        client_sizes = [len(ds) for ds in client_datasets]
        print(f"  Samples per client: min={min(client_sizes)}, max={max(client_sizes)}, avg={np.mean(client_sizes):.1f}")
    
    return trainloaders, valloaders, testloader
