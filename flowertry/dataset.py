import torch
import numpy as np
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST

def get_mnist(data_path: str = './data'):
    """Load MNIST dataset with standard normalization."""
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def partition_iid(dataset, num_clients: int, seed: int = 2023):
    """
    IID partitioning: randomly split dataset equally among clients.
    Each client gets approximately the same number of samples with
    similar class distributions.
    """
    num_images = len(dataset) // num_clients
    partition_len = [num_images] * num_clients
    
    return random_split(dataset, partition_len, torch.Generator().manual_seed(seed))


def partition_dirichlet(dataset, num_clients: int, alpha: float = 0.5, seed: int = 2023):
    """
    Non-IID partitioning using Dirichlet distribution.
    
    Alpha controls heterogeneity:
    - alpha -> 0: extreme non-IID (each client gets ~1 class)
    - alpha -> inf: approaches IID
    - alpha = 0.5: moderate non-IID (recommended starting point)
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients to partition data among
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: Random seed for reproducibility
    
    Returns:
        List of Subset objects, one per client
    """
    np.random.seed(seed)
    
    # Get all labels from the dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # Fallback for datasets without .targets attribute
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    # Sample from Dirichlet distribution for each class
    # This gives us the proportion of each class that goes to each client
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        # Dirichlet distribution gives proportions that sum to 1
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Shuffle class indices
        np.random.shuffle(class_indices[c])
        
        # Split indices according to proportions
        proportions = (proportions * len(class_indices[c])).astype(int)
        
        # Adjust for rounding errors
        proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()
        
        # Assign indices to clients
        idx = 0
        for client_id, num_samples in enumerate(proportions):
            client_indices[client_id].extend(
                class_indices[c][idx:idx + num_samples].tolist()
            )
            idx += num_samples
    
    # Create Subset for each client
    client_datasets = []
    for client_id in range(num_clients):
        # Shuffle each client's data
        np.random.shuffle(client_indices[client_id])
        client_datasets.append(Subset(dataset, client_indices[client_id]))
    
    return client_datasets


def get_client_class_distribution(client_datasets, num_classes: int = 10):
    """
    Get class distribution for each client (useful for visualization).
    
    Returns:
        dict: {client_id: {class_id: count}}
    """
    distributions = {}
    
    for client_id, subset in enumerate(client_datasets):
        class_counts = {c: 0 for c in range(num_classes)}
        
        # Get the original dataset and indices
        dataset = subset.dataset
        indices = subset.indices
        
        for idx in indices:
            if hasattr(dataset, 'targets'):
                label = dataset.targets[idx]
            else:
                label = dataset[idx][1]
            
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            class_counts[label] += 1
        
        distributions[client_id] = class_counts
    
    return distributions


def prepare_dataset(
    num_partitions: int,
    batch_size: int,
    val_ratio: float = 0.1,
    partition_type: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 2023
):
    """
    Prepare dataset with configurable partitioning strategy.
    
    Args:
        num_partitions: Number of clients/partitions
        batch_size: Batch size for data loaders
        val_ratio: Fraction of data to use for validation
        partition_type: "iid" or "dirichlet"
        dirichlet_alpha: Alpha parameter for Dirichlet distribution (only used if partition_type="dirichlet")
        seed: Random seed for reproducibility
    
    Returns:
        trainloaders, valloaders, testloader
    """
    trainset, testset = get_mnist()

    # Partition based on selected strategy
    if partition_type == "iid":
        print(f"Using IID partitioning")
        trainsets = partition_iid(trainset, num_partitions, seed)
    elif partition_type == "dirichlet":
        print(f"Using Dirichlet non-IID partitioning (alpha={dirichlet_alpha})")
        trainsets = partition_dirichlet(trainset, num_partitions, dirichlet_alpha, seed)
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}. Use 'iid' or 'dirichlet'")

    # Create dataloaders with train + validation support
    trainloaders = []
    valloaders = []
    
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], 
            torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=128)

    # Print partition statistics
    print(f"Created {len(trainloaders)} client partitions")
    if partition_type == "dirichlet":
        distributions = get_client_class_distribution(trainsets)
        # Show distribution for first few clients
        print("Sample class distributions (first 3 clients):")
        for cid in range(min(3, len(distributions))):
            dist = distributions[cid]
            print(f"  Client {cid}: {dict(dist)}")

    return trainloaders, valloaders, testloader
    


