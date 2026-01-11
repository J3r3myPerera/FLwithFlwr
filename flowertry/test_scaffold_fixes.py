#!/usr/bin/env python3
"""
Quick test script to verify SCAFFOLD fixes are working correctly.

This script runs a short SCAFFOLD simulation and checks:
1. c_global_norm is being calculated and tracked
2. c_global_norm stays in reasonable range
3. SCAFFOLD runs without errors
4. Performance improves over rounds

Usage:
    python test_scaffold_fixes.py
"""

import sys
from pathlib import Path

# Add flowertry to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from flwr.common import ndarrays_to_parameters
import time

from model import Net, test
from dataset import load_datasets
from client import get_client_fn
from server import get_on_fit_config, get_evaluate_fn, get_initial_parameters
from scaffold_strategy import FedScaffoldStrategy


def quick_test():
    """Run a quick SCAFFOLD test to verify fixes."""

    print("=" * 70)
    print("SCAFFOLD Fix Verification Test")
    print("=" * 70)

    # Load minimal config
    cfg_dict = {
        'num_rounds': 10,  # Just 10 rounds for quick test
        'num_clients': 10,
        'num_clients_per_round_fit': 5,
        'dataset': {
            'name': 'custom_savings',
            'alpha': 0.1,  # High heterogeneity to test SCAFFOLD
        },
        'model': {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'local_epochs': 1,
            'batch_size': 32,
            'max_grad_norm': 1.0,
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    print(f"\n[1/5] Loading dataset...")
    trainloaders, validationloaders, testloader = load_datasets(
        num_clients=cfg.num_clients,
        batch_size=cfg.model.batch_size,
        alpha=cfg.dataset.alpha
    )
    print(f"✅ Dataset loaded: {cfg.num_clients} clients")

    print(f"\n[2/5] Creating SCAFFOLD strategy...")
    num_classes = 3  # savings classification

    # Create strategy
    strategy = FedScaffoldStrategy(
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_available_clients=cfg.num_clients,
        min_evaluate_clients=0,
        on_fit_config_fn=get_on_fit_config(cfg.model),
        evaluate_fn=get_evaluate_fn(num_classes, testloader),
        total_clients=cfg.num_clients,
        model=Net(num_classes),
        test_loader=testloader,
        criterion=torch.nn.CrossEntropyLoss(),
        config=OmegaConf.to_container(cfg.model),
        initial_parameters=get_initial_parameters(num_classes)
    )
    print(f"✅ SCAFFOLD strategy created")

    print(f"\n[3/5] Creating client function...")
    client_fn = get_client_fn(
        trainloaders=trainloaders,
        validationloaders=validationloaders,
        num_classes=num_classes,
        strategy=strategy
    )
    print(f"✅ Client function ready")

    print(f"\n[4/5] Running SCAFFOLD simulation...")
    print(f"    Rounds: {cfg.num_rounds}")
    print(f"    Clients per round: {cfg.num_clients_per_round_fit}")
    print(f"    Data heterogeneity (alpha): {cfg.dataset.alpha}")
    print()

    start_time = time.time()

    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 1.0, 'num_gpus': 0}
        )

        elapsed_time = time.time() - start_time
        print(f"\n✅ Simulation completed in {elapsed_time:.2f}s")

    except Exception as e:
        print(f"\n❌ Simulation failed with error:")
        print(f"    {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n[5/5] Analyzing results...")

    # Extract c_global_norm history
    c_global_norm_history = []
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        c_global_norm_history = history.metrics_distributed_fit.get('c_global_norm', [])

    # Extract accuracy history
    accuracy_history = []
    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        accuracy_history = history.metrics_centralized.get('accuracy', [])

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Check 1: c_global_norm tracking
    if c_global_norm_history:
        norms = [norm for _, norm in c_global_norm_history]
        print(f"\n✅ c_global_norm tracking: WORKING")
        print(f"    Initial c_global_norm: {norms[0]:.4f}")
        print(f"    Final c_global_norm:   {norms[-1]:.4f}")
        print(f"    Max c_global_norm:     {max(norms):.4f}")
        print(f"    Min c_global_norm:     {min(norms):.4f}")

        # Check if in reasonable range
        if max(norms) < 50:
            print(f"    ✅ c_global_norm in reasonable range (< 50)")
        else:
            print(f"    ⚠️  c_global_norm may be too large (> 50)")
            print(f"    → This might indicate gradient sign is wrong")
            print(f"    → Try flipping the sign in model.py:174")
    else:
        print(f"\n❌ c_global_norm tracking: NOT WORKING")
        print(f"    No c_global_norm data found in history")

    # Check 2: Accuracy improvement
    if accuracy_history:
        accuracies = [acc for _, acc in accuracy_history]
        print(f"\n✅ Accuracy tracking: WORKING")
        print(f"    Initial accuracy: {accuracies[0]*100:.2f}%")
        print(f"    Final accuracy:   {accuracies[-1]*100:.2f}%")
        print(f"    Improvement:      {(accuracies[-1] - accuracies[0])*100:.2f}%")

        # Check if improving
        if accuracies[-1] > accuracies[0]:
            print(f"    ✅ Accuracy is improving")
        else:
            print(f"    ⚠️  Accuracy not improving")

        # Check for degradation in later rounds
        mid_point = len(accuracies) // 2
        if mid_point > 0 and accuracies[-1] < accuracies[mid_point]:
            print(f"    ⚠️  Performance degrading in later rounds")
            print(f"    → Mid-round accuracy: {accuracies[mid_point]*100:.2f}%")
            print(f"    → This might indicate gradient sign is wrong")
    else:
        print(f"\n⚠️  Accuracy tracking: No data")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_checks_passed = True

    if c_global_norm_history:
        norms = [norm for _, norm in c_global_norm_history]
        if max(norms) < 50:
            print("✅ Fix #1 (epochs vs total_steps): LIKELY CORRECT")
            print("✅ Fix #2 (aggregation scaling): LIKELY CORRECT")
        else:
            print("⚠️  c_global_norm too large - check fixes are applied")
            all_checks_passed = False
    else:
        print("❌ c_global_norm not tracked - something wrong with strategy")
        all_checks_passed = False

    if accuracy_history:
        accuracies = [acc for _, acc in accuracy_history]
        mid_point = len(accuracies) // 2
        if mid_point > 0 and accuracies[-1] < accuracies[mid_point]:
            print("⚠️  Fix #3 (gradient sign): MAY NEED FLIPPING")
            print("   → Try changing model.py:174 from += to -=")
            all_checks_passed = False
        else:
            print("✅ Fix #3 (gradient sign): LIKELY CORRECT")

    if all_checks_passed:
        print("\n🎉 All checks passed! SCAFFOLD fixes appear to be working correctly.")
        print("\nNext step: Run full comparison to verify SCAFFOLD > FedProx > FedAvg")
        print("    python compare_strategies.py")
    else:
        print("\n⚠️  Some issues detected. See warnings above.")

    print("=" * 70)
    return all_checks_passed


if __name__ == '__main__':
    quick_test()
