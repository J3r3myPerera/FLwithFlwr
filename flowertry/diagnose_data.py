"""
Diagnostic script to check data distribution and configuration.
Run this to understand why FedAvg might outperform FedProx/SCAFFOLD.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
import numpy as np


@hydra.main(config_path="conf", config_name="base", version_base=None)
def diagnose(cfg: DictConfig):
    print("=" * 80)
    print("FEDERATED LEARNING DIAGNOSTIC")
    print("=" * 80)

    # 1. Check configuration parsing
    print("\n1. CONFIGURATION CHECK")
    print("-" * 80)
    print(f"IID setting: {cfg.iid}")
    print(f"Alpha value: {cfg.alpha}")
    print(f"Alpha type: {type(cfg.alpha)}")

    if not isinstance(cfg.alpha, (int, float)):
        print("❌ ERROR: Alpha is not a number! Check your YAML formatting.")
        print(f"   Current alpha: {cfg.alpha}")
        print("   Fix: Change 'alpha:\\n  0.3' to 'alpha: 0.3'")
        return
    else:
        print(f"✅ Alpha is correctly formatted as: {cfg.alpha}")

    # 2. Check strategy parameters
    print("\n2. STRATEGY PARAMETERS")
    print("-" * 80)
    print(f"FedProx fixed mu: {cfg.proximal_mu}")

    if cfg.get('adaptive_mu', {}).get('enabled', False):
        mode = cfg.adaptive_mu.get('mode', 'simple')
        print(f"Adaptive FedProx mode: {mode}")

        if mode == 'multi_signal':
            print(f"  Base mu: {cfg.multi_signal_mu.base_mu}")
            print(f"  Mu range: [{cfg.multi_signal_mu.mu_min}, {cfg.multi_signal_mu.mu_max}]")
            if cfg.multi_signal_mu.mu_max > 0.5:
                print(f"  ⚠️  WARNING: mu_max={cfg.multi_signal_mu.mu_max} is very high!")
                print("     High mu over-regularizes. Try mu_max=0.3 or 0.5")

    print(f"\nLocal epochs: {cfg.config_fit.local_epochs}")
    if cfg.config_fit.local_epochs > 15:
        print(f"  ⚠️  WARNING: {cfg.config_fit.local_epochs} local epochs is quite high!")
        print("     High local epochs can cause overfitting. Try 5-10 first.")

    # 3. Check data distribution
    print("\n3. DATA DISTRIBUTION ANALYSIS")
    print("-" * 80)

    trainloaders, valloaders, testloader, class_weights, input_dim = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        iid=cfg.get('iid', True),
        alpha=cfg.get('alpha', 0.5)
    )
    
    print(f"Input dimension: {input_dim}")

    # Analyze class distribution across clients
    print("\n4. CLIENT HETEROGENEITY ANALYSIS")
    print("-" * 80)

    client_class_distributions = []

    for i in range(min(10, len(trainloaders))):
        all_labels = []
        for _, labels in trainloaders[i]:
            all_labels.extend(labels.numpy().tolist())

        labels_arr = np.array(all_labels)
        unique, counts = np.unique(labels_arr, return_counts=True)

        # Create distribution array
        dist = np.zeros(cfg.num_classes)
        for cls, count in zip(unique, counts):
            dist[cls] = count

        # Normalize to percentages
        dist = dist / dist.sum() * 100
        client_class_distributions.append(dist)

        if i < 5:
            print(f"\nClient {i}:")
            for cls in range(cfg.num_classes):
                print(f"  Class {cls}: {dist[cls]:.1f}%")

    # Compute heterogeneity metrics
    client_class_distributions = np.array(client_class_distributions)

    # Standard deviation of class proportions across clients
    heterogeneity = np.mean(np.std(client_class_distributions, axis=0))

    print(f"\n5. HETEROGENEITY SCORE")
    print("-" * 80)
    print(f"Average std dev of class distributions: {heterogeneity:.2f}%")

    if heterogeneity < 10:
        print("📊 Data is quite HOMOGENEOUS (IID-like)")
        print("   → FedAvg should work well")
        print("   → FedProx/SCAFFOLD won't show much benefit")
    elif heterogeneity < 20:
        print("📊 Data is MODERATELY heterogeneous")
        print("   → FedProx should show some benefit")
        print("   → SCAFFOLD might show benefit with many local epochs")
    else:
        print("📊 Data is HIGHLY heterogeneous (non-IID)")
        print("   → FedProx should significantly outperform FedAvg")
        print("   → SCAFFOLD should show clear benefits")

    # 6. Recommendations
    print("\n6. RECOMMENDATIONS")
    print("-" * 80)

    if heterogeneity < 15 and cfg.alpha > 0.5:
        print("⚠️  Alpha={:.1f} produces low heterogeneity".format(cfg.alpha))
        print("   Try alpha=0.1 or 0.2 for more heterogeneity")

    if cfg.proximal_mu > 0.05:
        print(f"⚠️  proximal_mu={cfg.proximal_mu} might be too high")
        print("   Try proximal_mu=0.01 or 0.001")

    if cfg.config_fit.local_epochs > 15:
        print(f"⚠️  local_epochs={cfg.config_fit.local_epochs} is quite high")
        print("   Try local_epochs=5 or 10")

    adaptive_enabled = cfg.get('adaptive_mu', {}).get('enabled', False)
    if adaptive_enabled:
        mu_max = cfg.get('multi_signal_mu', {}).get('mu_max', 1.0)
        if mu_max > 0.5:
            print(f"⚠️  mu_max={mu_max} is very high for adaptive FedProx")
            print("   Try mu_max=0.3 or 0.5")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    diagnose()
