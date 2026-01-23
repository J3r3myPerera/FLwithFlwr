"""
Diagnostic script to verify stratified selection allocation logic.
"""
from stratified_selector import StratifiedClientSelector
import numpy as np

def test_allocation():
    """Test allocation with different configurations."""
    
    print("="*80)
    print("STRATIFIED SELECTION ALLOCATION DIAGNOSTIC")
    print("="*80)
    
    # Configuration from your experiment
    client_strata = {
        'Tier_1': [0, 1, 2, 3],      # 4 clients (28.6%)
        'Tier_2': [4, 5, 6, 7, 8, 9],# 6 clients (42.9%)
        'Tier_3': [10, 11, 12, 13]   # 4 clients (28.6%)
    }
    
    print("\nClient Distribution:")
    total = sum(len(clients) for clients in client_strata.values())
    for stratum, clients in client_strata.items():
        pct = len(clients) / total * 100
        print(f"  {stratum}: {len(clients)} clients ({pct:.1f}%)")
    
    print(f"\nTotal clients: {total}")
    
    # Test different configurations
    configs = [
        (8, 2, "Original (PROBLEMATIC)"),
        (8, 1, "Fixed: min=1"),
        (9, 2, "Alternative: k=9"),
    ]
    
    for k, min_per, desc in configs:
        print(f"\n{'-'*80}")
        print(f"Configuration: {desc}")
        print(f"  Clients per round (k): {k}")
        print(f"  Min per stratum: {min_per}")
        
        try:
            selector = StratifiedClientSelector(
                client_strata=client_strata,
                clients_per_round=k,
                min_per_stratum=min_per
            )
            
            print(f"\nBase Allocation:")
            for stratum, count in selector.base_allocation.items():
                expected_pct = len(client_strata[stratum]) / total * 100
                actual_pct = count / k * 100
                diff = actual_pct - expected_pct
                print(f"  {stratum}: {count} clients ({actual_pct:.1f}% vs {expected_pct:.1f}% expected, diff: {diff:+.1f}%)")
            
            # Test actual selection diversity
            print(f"\nTesting selection diversity (5 rounds):")
            selections = []
            for round_num in range(1, 6):
                selected = selector.select_clients(round_num, seed=round_num*42)
                selections.append(selected)
                
                # Count per stratum
                stratum_counts = {}
                for stratum, client_ids in client_strata.items():
                    count = sum(1 for c in selected if c in client_ids)
                    stratum_counts[stratum] = count
                
                print(f"  Round {round_num}: {stratum_counts} -> {selected}")
            
            # Check diversity
            unique_selections = len(set(tuple(sorted(s)) for s in selections))
            print(f"\n  Unique selection patterns: {unique_selections}/5")
            if unique_selections == 1:
                print("  ⚠️  WARNING: No diversity - same clients selected every round!")
            elif unique_selections < 3:
                print("  ⚠️  WARNING: Low diversity - limited variety in selections")
            else:
                print("  ✅ Good diversity - varied client selections")
                
        except ValueError as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("For best results with 14 clients (4-6-4 distribution):")
    print("  • Use min_clients_per_stratum=1 (allows proportional allocation)")
    print("  • Keep num_clients_per_round_fit=8")
    print("  • Expected allocation: 2-4-2 or 3-4-3 (varies by round)")
    print("\nOR:")
    print("  • Use num_clients_per_round_fit=9 with min_clients_per_stratum=2")
    print("  • Expected allocation: 2-5-2 or 3-5-2")
    print("="*80)

if __name__ == "__main__":
    test_allocation()
