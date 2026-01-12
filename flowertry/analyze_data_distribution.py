#!/usr/bin/env python3
"""
Standalone script to analyze dataset class distribution.

This script loads the dataset and analyzes class balance to diagnose
potential class imbalance issues that could be capping model accuracy.

Usage:
    python analyze_data_distribution.py
"""

import numpy as np
from dataset import load_and_preprocess_data, prepare_dataset
from diagnostics import analyze_class_distribution
import torch


def main():
    """Analyze class distribution in the dataset."""

    print("=" * 80)
    print("DATASET CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Load full dataset
    print("\n[1/4] Loading dataset...")
    X, y = load_and_preprocess_data()

    print(f"\nDataset loaded: {len(y)} total samples")

    # Analyze overall distribution
    print("\n[2/4] Analyzing overall class distribution...")
    overall_dist = analyze_class_distribution(y, "Full Dataset")

    # Prepare federated partitions
    print("\n[3/4] Analyzing federated learning partitions...")

    configs_to_test = [
        {'num_clients': 100, 'alpha': 0.1, 'name': 'Current (100 clients, alpha=0.1)'},
        {'num_clients': 20, 'alpha': 0.1, 'name': 'Recommended (20 clients, alpha=0.1)'},
        {'num_clients': 100, 'alpha': 0.5, 'name': 'Moderate heterogeneity (100 clients, alpha=0.5)'},
    ]

    for config in configs_to_test:
        print(f"\n{'-'*80}")
        print(f"Configuration: {config['name']}")
        print(f"{'-'*80}")

        trainloaders, _, testloader, _, input_dim = prepare_dataset(
            num_partitions=config['num_clients'],
            batch_size=32,
            iid=False,
            alpha=config['alpha']
        )
        
        print(f"Input dimension: {input_dim}")

        # Check samples per client
        client_sizes = [len(loader.dataset) for loader in trainloaders]
        avg_size = np.mean(client_sizes)
        min_size = np.min(client_sizes)
        max_size = np.max(client_sizes)

        print(f"\nClient data distribution:")
        print(f"  Average samples per client: {avg_size:.1f}")
        print(f"  Min samples: {min_size}")
        print(f"  Max samples: {max_size}")
        print(f"  Std deviation: {np.std(client_sizes):.1f}")

        # Check test set distribution
        test_labels = []
        for _, labels in testloader:
            test_labels.extend(labels.numpy())
        test_labels = np.array(test_labels)

        print(f"\nTest set size: {len(test_labels)} samples")

        # Analyze test distribution
        _ = analyze_class_distribution(test_labels, f"Test Set ({config['name']})")

        # Sample a few clients to see their distributions
        print(f"\nSample client class distributions (first 3 clients):")
        for i in range(min(3, len(trainloaders))):
            client_labels = []
            for _, labels in trainloaders[i]:
                client_labels.extend(labels.numpy())
            client_labels = np.array(client_labels)

            unique, counts = np.unique(client_labels, return_counts=True)
            dist_str = ", ".join([f"Class {c}: {cnt}" for c, cnt in zip(unique, counts)])
            print(f"  Client {i}: {len(client_labels)} samples ({dist_str})")

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    imbalance_ratio = overall_dist['imbalance_ratio']

    print(f"\n📊 Class Imbalance Ratio: {imbalance_ratio:.2f}x")

    if imbalance_ratio > 3.0:
        print("\n🔴 CRITICAL: Severe class imbalance detected!")
        print("\nThis is very likely why all models cap at ~60% accuracy.")
        print("The model is probably just predicting the majority class.\n")

        print("IMMEDIATE FIXES REQUIRED:\n")

        print("1. Add class weights to loss function:")
        print("   ```python")
        print("   # In model.py train(), train_fedprox(), train_scaffold()")
        print("   from sklearn.utils.class_weight import compute_class_weight")
        print("   class_weights = compute_class_weight('balanced',")
        print("                                         classes=np.unique(labels),")
        print("                                         y=labels)")
        print("   class_weights = torch.FloatTensor(class_weights).to(device)")
        print("   criterion = nn.CrossEntropyLoss(weight=class_weights)")
        print("   ```\n")

        print("2. OR rebalance classes using percentiles:")
        print("   ```python")
        print("   # In dataset.py discretize_savings()")
        print("   def discretize_savings(savings_percentage):")
        print("       low = savings_percentage.quantile(0.33)")
        print("       high = savings_percentage.quantile(0.67)")
        print("       labels = np.zeros(len(savings_percentage))")
        print("       labels[savings_percentage >= low] = 1")
        print("       labels[savings_percentage >= high] = 2")
        print("       return labels")
        print("   ```\n")

        print("Expected improvement: +10-20% accuracy\n")

    elif imbalance_ratio > 2.0:
        print("\n⚠️  Moderate class imbalance detected")
        print("\nThis could be contributing to the accuracy plateau.\n")
        print("RECOMMENDED FIX:")
        print("  - Add class weights to loss function (see above)")
        print("\nExpected improvement: +5-10% accuracy\n")

    else:
        print("\n✅ Classes are reasonably balanced")
        print("\nClass imbalance is NOT the primary issue.")
        print("Look for other causes:")
        print("  - Model architecture too simple")
        print("  - Insufficient training (too few rounds/epochs)")
        print("  - Learning rate issues")
        print("  - Batch size too large for client data size")

    # Check samples per client
    print("\n" + "=" * 80)
    print("CLIENT DATA SIZE ANALYSIS")
    print("=" * 80)

    print("\nWith current config (100 clients):")
    trainloaders, _, _, _, _ = prepare_dataset(100, 32, False, 0.1)
    client_sizes = [len(loader.dataset) for loader in trainloaders]
    avg = np.mean(client_sizes)

    print(f"  Average samples per client: {avg:.1f}")

    if avg < 20:
        print(f"\n🔴 CRITICAL: Too few samples per client!")
        print(f"  With {avg:.1f} samples and batch_size=32, each client has < 1 batch!")
        print(f"  This makes training extremely unstable.\n")
        print(f"IMMEDIATE FIX:")
        print(f"  Reduce number of clients to 20:")
        print(f"  ```yaml")
        print(f"  num_clients: 20  # Down from 100")
        print(f"  ```")
        print(f"  This would give ~{len(y)/20:.0f} samples per client")
        print(f"\n  Expected improvement: +10-15% accuracy\n")

    elif avg < 50:
        print(f"\n⚠️  Low samples per client")
        print(f"  Consider reducing to 20-50 clients for more stable training")

    else:
        print(f"\n✅ Sufficient samples per client")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
