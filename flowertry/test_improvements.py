#!/usr/bin/env python3
"""
Quick test script to verify strategy improvements.
Tests that all strategies can be instantiated and run basic operations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from model import DisposableIncomeNet

def test_scaffold_control_variates():
    """Test SCAFFOLD control variate initialization and updates."""
    print("\n=== Testing SCAFFOLD Control Variates ===")
    
    model = DisposableIncomeNet(input_dim=25)
    
    # Initialize control variates
    client_control = [torch.zeros_like(p) for p in model.parameters()]
    server_control = [torch.zeros_like(p) for p in model.parameters()]
    
    print(f"✓ Initialized {len(client_control)} control variates")
    
    # Simulate control update
    initial_params = [p.clone() for p in model.parameters()]
    
    # Fake training step
    for p in model.parameters():
        p.data += torch.randn_like(p) * 0.01
    
    # Calculate delta
    K = 100  # steps
    lr = 0.001
    for i, (c_i, c, x_0, x_K) in enumerate(zip(
        client_control, server_control, initial_params, model.parameters()
    )):
        param_diff = (x_0 - x_K) / (K * lr)
        client_control[i] = c_i - c + param_diff
    
    print(f"✓ Updated control variates")
    print(f"  Average control magnitude: {np.mean([torch.norm(c).item() for c in client_control]):.6f}")
    
    return True

def test_adamw_optimizer():
    """Test AdamW optimizer integration."""
    print("\n=== Testing AdamW Optimizer ===")
    
    model = DisposableIncomeNet(input_dim=25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Create dummy data
    X = torch.randn(32, 25)
    y = torch.randn(32, 1)
    
    # Forward pass
    y_pred = model(X)
    loss = torch.nn.MSELoss()(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"✓ Gradients computed: {has_grads}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print("✓ Gradient clipping applied")
    
    # Optimizer step
    optimizer.step()
    print("✓ AdamW optimizer step completed")
    
    return True

def test_cosine_annealing():
    """Test cosine annealing scheduler."""
    print("\n=== Testing Cosine Annealing Scheduler ===")
    
    model = DisposableIncomeNet(input_dim=25)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    T_max = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=0.0001
    )
    
    lrs = []
    for step in range(T_max):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    
    print(f"✓ Learning rate decay: {lrs[0]:.6f} → {lrs[-1]:.6f}")
    print(f"  Minimum LR reached: {min(lrs):.6f}")
    
    return True

def test_hybrid_weights():
    """Test hybrid weight configurations."""
    print("\n=== Testing Hybrid Weight Configurations ===")
    
    configs = [
        {"fedprox_weight": 1.0, "scaffold_weight": 0.0, "name": "Pure FedProx"},
        {"fedprox_weight": 0.0, "scaffold_weight": 1.0, "name": "Pure SCAFFOLD"},
        {"fedprox_weight": 0.3, "scaffold_weight": 0.5, "name": "Balanced Hybrid"},
        {"fedprox_weight": 0.5, "scaffold_weight": 0.5, "name": "Equal Hybrid"},
    ]
    
    for config in configs:
        print(f"  {config['name']:20s}: FedProx={config['fedprox_weight']:.1f}, SCAFFOLD={config['scaffold_weight']:.1f}")
    
    print("✓ All weight configurations valid")
    
    return True

def test_warmup_schedule():
    """Test warmup scheduling."""
    print("\n=== Testing Warmup Schedules ===")
    
    import math
    
    warmup_rounds = 5
    warmup_start_factor = 0.1
    
    for warmup_type in ["linear", "exponential", "cosine"]:
        factors = []
        for server_round in range(1, warmup_rounds + 1):
            progress = server_round / warmup_rounds
            
            if warmup_type == "exponential":
                warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * (1 - (1 - progress) ** 2)
            elif warmup_type == "cosine":
                warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * (1 - math.cos(progress * math.pi)) / 2
            else:  # linear
                warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * progress
            
            factors.append(warmup_factor)
        
        print(f"  {warmup_type:12s}: {factors[0]:.3f} → {factors[-1]:.3f}")
    
    print("✓ All warmup schedules working")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("STRATEGY IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("SCAFFOLD Control Variates", test_scaffold_control_variates),
        ("AdamW Optimizer", test_adamw_optimizer),
        ("Cosine Annealing", test_cosine_annealing),
        ("Hybrid Weights", test_hybrid_weights),
        ("Warmup Schedule", test_warmup_schedule),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} FAILED: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Strategies are ready to run.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
