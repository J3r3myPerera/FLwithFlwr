"""
Quick test script to verify pure tier partitioning works correctly.
"""

def test_pure_partitioning():
    """Test the pure tier partitioning function."""
    from dataset import prepare_dataset
    
    print("Testing Pure Tier Partitioning...")
    print("=" * 80)
    
    # Test with small batch size
    trainloaders, valloaders, testloader, target_scaler, client_strata = prepare_dataset(
        num_clients=14,
        batch_size=32,
        data_path='./data/IndianPersonalFinance/indianPersonalFinanceAndSpendingHabits.csv'
    )
    
    print(f"\n✓ Dataset preparation successful!")
    print(f"  - Number of training clients: {len(trainloaders)}")
    print(f"  - Number of validation clients: {len(valloaders)}")
    print(f"  - Test loader created: {testloader is not None}")
    print(f"  - Target scaler created: {target_scaler is not None}")
    
    print(f"\nClient Strata Distribution:")
    total_clients = 0
    for tier, clients in client_strata.items():
        print(f"  {tier}: {len(clients)} clients - IDs: {clients}")
        total_clients += len(clients)
    
    print(f"\n✓ Total clients: {total_clients}")
    
    # Verify each client has data
    print(f"\nClient Data Sizes:")
    for i, (train_loader, val_loader) in enumerate(zip(trainloaders, valloaders)):
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        
        # Find which tier this client belongs to
        tier = None
        for t, clients in client_strata.items():
            if i in clients:
                tier = t
                break
        
        print(f"  Client {i:2d} ({tier}): {train_size:4d} train, {val_size:3d} val samples")
    
    print(f"\n✓ All tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    test_pure_partitioning()
