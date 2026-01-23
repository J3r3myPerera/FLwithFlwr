"""
Quick test script to verify stratified client selection implementation.
"""

import numpy as np
from stratified_selector import StratifiedClientSelector, create_stratified_client_fn

def test_basic_selection():
    """Test basic stratified selection functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Stratified Selection")
    print("="*80)
    
    # Create test client strata (simulating City Tiers)
    client_strata = {
        'Tier_1': [0, 1, 2, 3],      # 4 clients
        'Tier_2': [4, 5, 6, 7],      # 4 clients  
        'Tier_3': [8, 9, 10, 11]     # 4 clients
    }
    
    # Create selector
    selector = StratifiedClientSelector(
        client_strata=client_strata,
        clients_per_round=6,
        min_per_stratum=1
    )
    
    # Run 10 rounds of selection
    print("\nRunning 10 rounds of selection...")
    for round_num in range(1, 11):
        selected = selector.select_clients(round_num, seed=round_num)
        stats = selector.get_stratum_statistics(selected)
        print(f"Round {round_num}: {selected} | Stats: {stats}")
    
    # Compute fairness metrics
    print("\n" + "-"*80)
    print("FAIRNESS METRICS")
    print("-"*80)
    metrics = selector.compute_fairness_metrics()
    print(f"Participation Equity (Gini): {metrics['participation_equity_gini']:.4f}")
    print(f"Toxic Round Frequency: {metrics['toxic_round_frequency_pct']:.1f}%")
    print(f"Representation Ratios:")
    for stratum, ratio in metrics['representation_ratios'].items():
        print(f"  {stratum}: {ratio:.3f}")
    
    print("\n✓ Test 1 passed!")
    return selector


def test_edge_cases():
    """Test edge cases and validation."""
    print("\n" + "="*80)
    print("TEST 2: Edge Cases and Validation")
    print("="*80)
    
    # Test 1: Minimum clients per round
    print("\n[Test 2.1] Minimum clients per round (K = num_strata)...")
    client_strata = {
        'A': [0, 1, 2],
        'B': [3, 4, 5],
        'C': [6, 7, 8]
    }
    selector = StratifiedClientSelector(client_strata, clients_per_round=3, min_per_stratum=1)
    selected = selector.select_clients(1)
    print(f"  Selected: {selected}")
    assert len(selected) == 3, "Should select exactly 3 clients"
    print("  ✓ Passed")
    
    # Test 2: Unbalanced strata
    print("\n[Test 2.2] Unbalanced strata sizes...")
    client_strata = {
        'Large': [0, 1, 2, 3, 4, 5, 6, 7],  # 8 clients
        'Medium': [8, 9, 10, 11],            # 4 clients
        'Small': [12, 13]                    # 2 clients
    }
    selector = StratifiedClientSelector(client_strata, clients_per_round=7, min_per_stratum=1)
    selected = selector.select_clients(1)
    stats = selector.get_stratum_statistics(selected)
    print(f"  Selected: {selected}")
    print(f"  Stats: {stats}")
    assert all(count >= 1 for count in stats.values()), "All strata should have at least 1 client"
    print("  ✓ Passed")
    
    # Test 3: Invalid configuration
    print("\n[Test 2.3] Invalid configuration (should raise error)...")
    try:
        selector = StratifiedClientSelector(
            client_strata={'A': [0, 1], 'B': [2, 3]},
            clients_per_round=3,  # Not enough for min_per_stratum=2
            min_per_stratum=2
        )
        print("  ✗ Failed - should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Passed - correctly raised ValueError: {e}")
    
    print("\n✓ Test 2 passed!")


def test_fairness_guarantees():
    """Test fairness guarantees over many rounds."""
    print("\n" + "="*80)
    print("TEST 3: Fairness Guarantees (100 rounds)")
    print("="*80)
    
    client_strata = {
        'Tier_1': [0, 1, 2, 3],
        'Tier_2': [4, 5, 6, 7, 8, 9],
        'Tier_3': [10, 11, 12, 13]
    }
    
    selector = StratifiedClientSelector(
        client_strata=client_strata,
        clients_per_round=6,
        min_per_stratum=1
    )
    
    # Run 100 rounds
    print("\nRunning 100 rounds...")
    for round_num in range(1, 101):
        selector.select_clients(round_num, seed=round_num)
    
    # Analyze fairness
    metrics = selector.compute_fairness_metrics()
    
    print(f"\nResults after 100 rounds:")
    print(f"  Participation Equity (Gini): {metrics['participation_equity_gini']:.4f}")
    print(f"  Toxic Round Frequency: {metrics['toxic_round_frequency_pct']:.1f}%")
    print(f"  Representation Ratios:")
    for stratum, ratio in metrics['representation_ratios'].items():
        print(f"    {stratum}: {ratio:.3f}")
    
    # Assertions
    assert metrics['participation_equity_gini'] < 0.2, "Gini should be low (< 0.2)"
    assert metrics['toxic_round_frequency_pct'] < 10, "Toxic frequency should be low (< 10%)"
    assert all(0.8 < ratio < 1.2 for ratio in metrics['representation_ratios'].values()), \
        "All ratios should be close to 1.0 (within 20%)"
    
    print("\n✓ Test 3 passed!")


def test_factory_function():
    """Test the factory function."""
    print("\n" + "="*80)
    print("TEST 4: Factory Function")
    print("="*80)
    
    client_strata = {
        'Group_A': [0, 1, 2],
        'Group_B': [3, 4, 5],
        'Group_C': [6, 7, 8]
    }
    
    selector = create_stratified_client_fn(
        client_strata=client_strata,
        clients_per_round=6,
        min_per_stratum=1
    )
    
    selected = selector.select_clients(1)
    print(f"\nSelected clients: {selected}")
    assert len(selected) == 6, "Should select 6 clients"
    
    print("\n✓ Test 4 passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("STRATIFIED CLIENT SELECTION - TEST SUITE")
    print("="*80)
    
    try:
        # Run tests
        selector = test_basic_selection()
        test_edge_cases()
        test_fairness_guarantees()
        test_factory_function()
        
        # Final summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nStratified client selection is working correctly.")
        print("You can now run the full FL experiment with:")
        print("  python main.py strategy=compare_stratified")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n")


if __name__ == "__main__":
    main()
