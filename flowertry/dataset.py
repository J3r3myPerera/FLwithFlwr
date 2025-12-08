import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path: str = './data'):
    """Load MNIST dataset with proper transformations."""
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    
    return trainset, testset


def iid_partition(dataset, num_partitions: int, seed: int = 2023):
    """
    IID partitioning: Randomly shuffle and divide data equally among clients.
    Each client gets a uniform random sample of the dataset.
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    # Split indices equally among partitions
    split_indices = np.array_split(indices, num_partitions)
    
    return [Subset(dataset, idx.tolist()) for idx in split_indices]


def dirichlet_partition(dataset, num_partitions: int, alpha: float = 0.5, 
                        num_classes: int = 10, seed: int = 2023):
    """
    Non-IID partitioning using Dirichlet distribution.
    
    Args:
        dataset: The dataset to partition
        num_partitions: Number of clients/partitions
        alpha: Concentration parameter for Dirichlet distribution.
               - alpha -> 0: Extremely non-IID (each client gets only 1-2 classes)
               - alpha -> inf: Approaches IID distribution
               - alpha = 0.1: Highly non-IID
               - alpha = 0.5: Moderately non-IID  
               - alpha = 1.0: Slightly non-IID
               - alpha = 10.0: Nearly IID
        num_classes: Number of classes in the dataset
        seed: Random seed for reproducibility
    
    Returns:
        List of Subset objects, one per client
    """
    np.random.seed(seed)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Group indices by class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_partitions)]
    
    # For each class, distribute samples using Dirichlet distribution
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        
        # Calculate number of samples per client for this class
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        # Split indices according to proportions
        split_indices = np.split(indices, proportions)
        
        for client_id, idx in enumerate(split_indices):
            client_indices[client_id].extend(idx.tolist())
    
    # Shuffle each client's data
    for client_id in range(num_partitions):
        np.random.shuffle(client_indices[client_id])
    
    return [Subset(dataset, indices) for indices in client_indices]


def label_skew_partition(dataset, num_partitions: int, labels_per_client: int = 2,
                         num_classes: int = 10, seed: int = 2023):
    """
    Non-IID partitioning where each client only has a limited number of classes.
    
    Args:
        dataset: The dataset to partition
        num_partitions: Number of clients/partitions  
        labels_per_client: Number of distinct labels each client will have (1-10 for MNIST)
                          - labels_per_client=1: Extreme non-IID (each client has only 1 class)
                          - labels_per_client=2: High non-IID
                          - labels_per_client=5: Moderate non-IID
                          - labels_per_client=10: IID-like
        num_classes: Total number of classes
        seed: Random seed for reproducibility
    
    Returns:
        List of Subset objects, one per client
    """
    np.random.seed(seed)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Group indices by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    
    # Track how many samples of each class have been assigned
    class_pointers = {c: 0 for c in range(num_classes)}
    
    # Assign labels to each client
    client_indices = []
    
    for client_id in range(num_partitions):
        # Assign labels_per_client classes to this client (cycling through)
        assigned_labels = [(client_id * labels_per_client + j) % num_classes 
                          for j in range(labels_per_client)]
        
        client_idx = []
        
        # Get samples from each assigned class
        samples_per_class = len(dataset) // (num_partitions * labels_per_client)
        
        for label in assigned_labels:
            start = class_pointers[label]
            end = min(start + samples_per_class, len(class_indices[label]))
            client_idx.extend(class_indices[label][start:end])
            class_pointers[label] = end
        
        np.random.shuffle(client_idx)
        client_indices.append(client_idx)
    
    return [Subset(dataset, indices) for indices in client_indices]


def quantity_skew_partition(dataset, num_partitions: int, min_samples: int = 100,
                           alpha: float = 0.5, seed: int = 2023):
    """
    Non-IID partitioning with quantity skew (unbalanced data sizes).
    Each client gets different amounts of data following Dirichlet distribution.
    
    Args:
        dataset: The dataset to partition
        num_partitions: Number of clients/partitions
        min_samples: Minimum samples per client
        alpha: Dirichlet concentration parameter for quantity distribution
        seed: Random seed
    
    Returns:
        List of Subset objects, one per client
    """
    np.random.seed(seed)
    
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    
    # Generate quantity proportions using Dirichlet
    proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
    
    # Ensure minimum samples per client
    min_total = min_samples * num_partitions
    if min_total > num_samples:
        min_samples = num_samples // num_partitions
    
    # Calculate samples per client
    remaining = num_samples - min_samples * num_partitions
    extra_samples = (proportions * remaining).astype(int)
    samples_per_client = min_samples + extra_samples
    
    # Adjust to ensure all samples are used
    diff = num_samples - samples_per_client.sum()
    samples_per_client[-1] += diff
    
    # Split indices
    client_indices = []
    start = 0
    for num in samples_per_client:
        client_indices.append(indices[start:start + num].tolist())
        start += num
    
    return [Subset(dataset, idx) for idx in client_indices]


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1,
                    partition_type: str = "dirichlet", alpha: float = 0.5,
                    labels_per_client: int = 2, num_classes: int = 10, seed: int = 2023):
    """
    Prepare federated datasets with configurable partitioning strategy.
    
    Args:
        num_partitions: Number of clients
        batch_size: Batch size for dataloaders
        val_ratio: Fraction of each client's data for validation
        partition_type: Type of partitioning
            - "iid": Uniform random split
            - "dirichlet": Dirichlet-based non-IID (recommended for realistic simulation)
            - "label_skew": Each client has limited classes
            - "quantity_skew": Unbalanced data sizes
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        labels_per_client: For label_skew partition, number of classes per client
        num_classes: Number of classes in dataset
        seed: Random seed for reproducibility
    
    Returns:
        trainloaders, valloaders, testloader
    """
    trainset, testset = get_mnist()
    
    # Choose partitioning strategy
    if partition_type == "iid":
        print("Using IID partitioning")
        trainsets = iid_partition(trainset, num_partitions, seed)
    elif partition_type == "dirichlet":
        print(f"Using Dirichlet non-IID partitioning (alpha={alpha})")
        trainsets = dirichlet_partition(trainset, num_partitions, alpha, num_classes, seed)
    elif partition_type == "label_skew":
        print(f"Using label-skew non-IID partitioning ({labels_per_client} labels/client)")
        trainsets = label_skew_partition(trainset, num_partitions, labels_per_client, num_classes, seed)
    elif partition_type == "quantity_skew":
        print(f"Using quantity-skew non-IID partitioning (alpha={alpha})")
        trainsets = quantity_skew_partition(trainset, num_partitions, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    # Print partition statistics
    print_partition_stats(trainsets, num_classes)
    
    # Create dataloaders with train + validation split
    trainloaders = []
    valloaders = []
    
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = max(1, int(val_ratio * num_total))  # At least 1 validation sample
        num_train = num_total - num_val
        
        # Get indices for this subset
        indices = list(range(num_total))
        np.random.shuffle(indices)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        # Create new subsets from the partition
        train_subset = Subset(trainset_, train_indices)
        val_subset = Subset(trainset_, val_indices)
        
        # num_workers=0 for macOS compatibility (avoids multiprocessing issues)
        trainloaders.append(DataLoader(train_subset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0))
        valloaders.append(DataLoader(val_subset, batch_size=batch_size, 
                                     shuffle=False, num_workers=0))
    
    testloader = DataLoader(testset, batch_size=128, num_workers=0)
    
    return trainloaders, valloaders, testloader


def print_partition_stats(partitions, num_classes: int = 10):
    """Print statistics about the data partitions for analysis."""
    print("\n" + "="*60)
    print("PARTITION STATISTICS")
    print("="*60)
    
    sizes = [len(p) for p in partitions]
    print(f"Number of clients: {len(partitions)}")
    print(f"Total samples: {sum(sizes)}")
    print(f"Samples per client - Min: {min(sizes)}, Max: {max(sizes)}, "
          f"Mean: {np.mean(sizes):.1f}, Std: {np.std(sizes):.1f}")
    
    # Calculate label distribution for first few clients
    print("\nLabel distribution (first 5 clients):")
    for i, partition in enumerate(partitions[:5]):
        # Get labels for this partition
        labels = []
        for idx in partition.indices:
            if hasattr(partition.dataset, 'targets'):
                labels.append(partition.dataset.targets[idx])
            else:
                labels.append(partition.dataset[idx][1])
        
        label_counts = np.bincount(labels, minlength=num_classes)
        dist_str = " ".join([f"{c}:{count}" for c, count in enumerate(label_counts) if count > 0])
        print(f"  Client {i}: {len(labels)} samples | {dist_str}")
    
    print("="*60 + "\n")
