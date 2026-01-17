"""
Test Dirichlet partitioning with different alpha values.
This script demonstrates how alpha controls heterogeneity.
"""

from dataset import prepare_dataset_federated, reset_preprocessor
import numpy as np

def test_dirichlet_partitioning():
    """Test Dirichlet partitioning with different alpha values."""
    
    print("="*80)
    print("TESTING DIRICHLET PARTITIONING")
    print("="*80)
    
    # Test different alpha values
    alphas = [0.1, 0.5, 1.0, 5.0, 10.0]
    num_clients = 10
    
    for alpha in alphas:
        print(f"\n{'='*80}")
        print(f"Testing with alpha = {alpha}")
        print(f"{'='*80}")
        
        reset_preprocessor()
        
        trainloaders, valloaders, testloader, input_dim, target_mean, target_std, partition_info, log_transform = \
            prepare_dataset_federated(
                num_partitions=num_clients,
                batch_size=32,
                iid=False,
                partition_strategy='dirichlet',
                dirichlet_alpha=alpha,
                log_transform_target=True,
                seed=2023,
                verbose=True
            )
        
        # Calculate heterogeneity statistics
        sample_counts = [partition_info['clients'][i]['samples'] for i in range(num_clients)]
        target_means = [partition_info['clients'][i]['mean_target'] for i in range(num_clients)]
        
        print(f"\n  Sample Distribution:")
        print(f"    Min samples: {min(sample_counts):,}")
        print(f"    Max samples: {max(sample_counts):,}")
        print(f"    Ratio (max/min): {max(sample_counts)/min(sample_counts):.2f}x")
        print(f"    CV of samples: {np.std(sample_counts)/np.mean(sample_counts):.4f}")
        
        print(f"\n  Target Distribution:")
        print(f"    Min mean: ${min(target_means):,.2f}")
        print(f"    Max mean: ${max(target_means):,.2f}")
        print(f"    CV of means: {np.std(target_means)/np.mean(target_means):.4f}")
        
        print(f"\n  Summary: With alpha={alpha}, heterogeneity is {'EXTREME' if alpha < 0.5 else 'HIGH' if alpha < 1.5 else 'MODERATE' if alpha < 5 else 'LOW'}")

if __name__ == '__main__':
    test_dirichlet_partitioning()
