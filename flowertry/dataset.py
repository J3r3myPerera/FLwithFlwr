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
    
    Task: Savings Potential Classification (4 Classes)
    - Target: Desired_Savings_Percentage discretized into 4 classes
        - Class 0 (Low Savers): < 6.5%
        - Class 1 (Lower-Middle Savers): 6.5-9%
        - Class 2 (Upper-Middle Savers): 9-13%
        - Class 3 (High Savers): > 13%
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def discretize_savings(savings_percentage: pd.Series, method: str = 'quantile') -> np.ndarray:
    """
    Discretize savings percentage into 4 classes.
    
    Methods:
    - 'fixed': Use data-driven thresholds (< 6.5%, 6.5-9%, 9-13%, > 13%)
    - 'quantile': Use quartile thresholds (25th/50th/75th percentiles)
    
    Args:
        savings_percentage: Series of savings percentages
        method: 'fixed' or 'quantile'
    
    Returns:
        labels: Class labels (0=Low, 1=Lower-Middle, 2=Upper-Middle, 3=High)
    """
    labels = np.zeros(len(savings_percentage), dtype=np.int64)
    
    if method == 'quantile':
        # Use quartiles for balanced 4 classes
        q25 = savings_percentage.quantile(0.25)
        q50 = savings_percentage.quantile(0.50)
        q75 = savings_percentage.quantile(0.75)
        labels[savings_percentage >= q25] = 1   # Lower-Middle
        labels[savings_percentage >= q50] = 2   # Upper-Middle
        labels[savings_percentage >= q75] = 3   # High
        print(f"  Quantile thresholds: Low < {q25:.2f}% < Lower-Middle < {q50:.2f}% < Upper-Middle < {q75:.2f}% < High")
    else:
        # Fixed thresholds based on data analysis (range 5-25%, median ~9%)
        labels[savings_percentage >= 6.5] = 1   # Lower-Middle (6.5-9%)
        labels[savings_percentage >= 9] = 2     # Upper-Middle (9-13%)
        labels[savings_percentage > 13] = 3     # High (>13%)
        print(f"  Fixed thresholds: Low < 6.5% < Lower-Middle < 9% < Upper-Middle < 13% < High")
    
    return labels


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that better capture savings potential.
    
    These engineered features have higher correlation with savings behavior.
    """
    df = df.copy()
    
    # Expense columns
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                    'Education', 'Miscellaneous']
    
    # Total expenses
    df['Total_Expenses'] = df[expense_cols].sum(axis=1)
    
    # Actual savings rate (derived from income and expenses)
    df['Actual_Savings_Rate'] = (df['Income'] - df['Total_Expenses']) / df['Income']
    
    # Expense-to-income ratio (key indicator)
    df['Expense_to_Income'] = df['Total_Expenses'] / df['Income']
    
    # Discretionary spending ratio
    df['Discretionary_Ratio'] = (df['Eating_Out'] + df['Entertainment'] + df['Miscellaneous']) / df['Income']
    
    # Essential spending ratio
    df['Essential_Ratio'] = (df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']) / df['Income']
    
    # Financial burden indicators
    df['Rent_Burden'] = df['Rent'] / df['Income']
    df['Debt_Burden'] = df['Loan_Repayment'] / df['Income']
    
    # Per-capita income (adjusted for dependents)
    df['Per_Capita_Income'] = df['Income'] / (1 + df['Dependents'])
    
    # Age-income interaction
    df['Age_Income_Ratio'] = df['Age'] / 100 * df['Income'] / df['Income'].mean()
    
    return df


def compute_class_weights(labels: np.ndarray, method: str = 'balanced') -> np.ndarray:
    """
    Compute class weights for handling imbalanced datasets.
    
    Args:
        labels: Array of class labels
        method: 'balanced' for inverse frequency, 'sqrt' for sqrt of inverse frequency,
                'middle_boost' to boost the harder middle classes (for 4-class)
    
    Returns:
        class_weights: Array of weights for each class
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique_classes)
    
    if method == 'balanced':
        # Weight inversely proportional to class frequency
        # w_i = n_samples / (n_classes * count_i)
        weights = n_samples / (n_classes * counts)
    elif method == 'sqrt':
        # Softer weighting: sqrt of balanced weights
        balanced_weights = n_samples / (n_classes * counts)
        weights = np.sqrt(balanced_weights)
    elif method == 'middle_boost':
        # Boost the middle classes (1 and 2) which are harder to classify
        # Start with balanced weights
        weights = n_samples / (n_classes * counts)
        # Increase weight for middle classes
        if len(weights) >= 4:
            weights[1] *= 2.0  # 2x weight for Lower-Middle class
            weights[2] *= 2.5  # 2.5x weight for Upper-Middle class (hardest)
        elif len(weights) >= 2:
            weights[1] *= 1.5  # 1.5x weight for Medium class (3-class case)
    else:
        # Uniform weights (no weighting)
        weights = np.ones(n_classes)
    
    print(f"\nClass Weights ({method}):")
    class_names = ['Low (<6.5%)', 'Lower-Middle (6.5-9%)', 'Upper-Middle (9-13%)', 'High (>13%)']
    for cls, weight in zip(unique_classes, weights):
        print(f"  Class {cls} ({class_names[cls]}): {weight:.4f}")
    
    return weights


def load_and_preprocess_data(data_path: str = DATA_PATH, use_engineered_features: bool = True, 
                              discretization_method: str = 'quantile') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the Personal Finance dataset with improved feature engineering.
    
    Args:
        data_path: Path to CSV file
        use_engineered_features: If True, add derived features (ratios, interactions)
        discretization_method: 'quantile' for balanced classes, 'fixed' for original thresholds
    
    Returns:
        features: Preprocessed feature matrix
        labels: Discretized class labels (0, 1, 2)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    print(f"\n[Data Loading] Loaded {len(df)} samples")
    
    # Apply feature engineering if enabled
    if use_engineered_features:
        df = engineer_features(df)
        print("[Feature Engineering] Added 9 derived features")
    
    # Define feature columns
    numerical_features = [
        'Income', 'Age', 'Dependents',
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
        'Education', 'Miscellaneous'
    ]
    
    # Add engineered features if enabled
    engineered_features = []
    if use_engineered_features:
        engineered_features = [
            'Total_Expenses', 'Actual_Savings_Rate', 'Expense_to_Income',
            'Discretionary_Ratio', 'Essential_Ratio', 'Rent_Burden', 
            'Debt_Burden', 'Per_Capita_Income', 'Age_Income_Ratio'
        ]
    
    all_numerical_features = numerical_features + engineered_features
    
    categorical_features = ['Occupation', 'City_Tier']
    target_col = 'Desired_Savings_Percentage'
    
    # Extract numerical features
    X_numerical = df[all_numerical_features].values
    
    # Handle any NaN/inf values from divisions
    X_numerical = np.nan_to_num(X_numerical, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode categorical features
    X_categorical_list = []
    for col in categorical_features:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        X_categorical_list.append(encoded.reshape(-1, 1))
    
    X_categorical = np.hstack(X_categorical_list)
    
    # Combine features
    X = np.hstack([X_numerical, X_categorical])
    
    print(f"[Features] Total features: {X.shape[1]} ({len(all_numerical_features)} numerical + {len(categorical_features)} categorical)")
    
    # Standardize all features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Discretize target into 4 classes using specified method
    y = discretize_savings(df[target_col], method=discretization_method)
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass Distribution:")
    class_names = ['Low', 'Lower-Middle', 'Upper-Middle', 'High']
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
    alpha: float = 0.5,
    use_class_weights: bool = False,
    class_weight_method: str = 'balanced',
    use_engineered_features: bool = True,
    discretization_method: str = 'quantile'
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, Optional[np.ndarray], int]:
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
        use_engineered_features: If True, add derived features
        discretization_method: 'quantile' for balanced classes or 'fixed' for original thresholds
    
    Returns:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        testloader: Global test DataLoader
        class_weights: Array of class weights (or None)
        input_dim: Number of input features
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(
        use_engineered_features=use_engineered_features,
        discretization_method=discretization_method
    )
    
    # Get input dimension
    input_dim = X.shape[1]
    print(f"\nFeatures: {input_dim} (engineered={use_engineered_features}, discretization={discretization_method})")
    
    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(y, method=class_weight_method)
    
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
        class_names = ['Low (<6.5%)', 'Lower-Middle (6.5-9%)', 'Upper-Middle (9-13%)', 'High (>13%)']
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
    print(f"  Input features: {input_dim}")
    if iid:
        partition_size = trainval_size // num_partitions
        print(f"  Samples per client: ~{partition_size}")
    else:
        # Show sample size range for non-IID
        client_sizes = [len(ds) for ds in client_datasets]
        print(f"  Samples per client: min={min(client_sizes)}, max={max(client_sizes)}, avg={np.mean(client_sizes):.1f}")
    
    return trainloaders, valloaders, testloader, class_weights, input_dim
