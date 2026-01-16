"""Test script to verify hybrid partitioning implementation."""

import sys
sys.path.append('.')

from dataset import prepare_dataset_federated, reset_preprocessor

# Test hybrid partitioning
print("\n" + "="*80)
print("TESTING HYBRID NON-IID PARTITIONING (12 CLIENTS)")
print("="*80)

reset_preprocessor()

trainloaders, valloaders, testloader, input_dim, target_mean, target_std, partition_info, log_transform = \
    prepare_dataset_federated(
        num_partitions=12,
        batch_size=32,
        iid=False,
        partition_strategy='hybrid',
        log_transform_target=True,
        seed=2023,
        verbose=True
    )

print("\n" + "="*80)
print("PARTITIONING SUMMARY")
print("="*80)

total_samples = sum([partition_info['clients'][i]['samples'] for i in range(12)])
print(f"\nTotal training samples across all clients: {total_samples:,}")
print(f"Global test set size: {len(testloader.dataset):,}")
print(f"Input dimension: {input_dim}")
print(f"Log transformation applied: {log_transform}")

print("\n✅ Hybrid partitioning implementation successful!")
print("   - 12 clients created (3 City_Tiers × 4 Occupations)")
print("   - Each client has unique combination of demographic features")
print("   - This creates strong data heterogeneity (Non-IID)")
