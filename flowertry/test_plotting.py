#!/usr/bin/env python3
"""
Test script to verify history collection is working.
Creates a mock comparison_results.json with sample history data
and tests the plotting script.
"""

import json
import os
from pathlib import Path
import numpy as np

def create_test_data():
    """Create mock comparison results with history"""
    
    # Simulate 10 rounds of training
    num_rounds = 10
    rounds = list(range(1, num_rounds + 1))
    
    # Generate mock data for each strategy
    results = {}
    
    strategies = {
        'fedavg': {'base_r2': 0.85, 'base_mape': 8.0, 'color': '#1f77b4'},
        'fedprox': {'base_r2': 0.87, 'base_mape': 7.5, 'color': '#ff7f0e'},
        'scaffold': {'base_r2': 0.84, 'base_mape': 8.5, 'color': '#2ca02c'},
        'hybrid': {'base_r2': 0.90, 'base_mape': 6.5, 'color': '#d62728'}
    }
    
    for strategy, config in strategies.items():
        # Generate improving metrics over rounds
        r2_values = [config['base_r2'] * (0.7 + 0.3 * (r / num_rounds)) for r in rounds]
        r2_values = [min(0.95, r2 + np.random.normal(0, 0.01)) for r2 in r2_values]
        
        mape_values = [config['base_mape'] * (1.3 - 0.3 * (r / num_rounds)) for r in rounds]
        mape_values = [max(5.0, mape + np.random.normal(0, 0.3)) for mape in mape_values]
        
        loss_values = [(1 - r2) * 0.1 for r2 in r2_values]
        
        rmse_values = [4000 * (1 - r2) for r2 in r2_values]
        mae_values = [3000 * (1 - r2) for r2 in r2_values]
        
        acc10_values = [r2 * 90 + np.random.normal(0, 1) for r2 in r2_values]
        acc20_values = [r2 * 95 + np.random.normal(0, 1) for r2 in r2_values]
        
        results[strategy] = {
            'strategy': strategy,
            'final_metrics': {
                'loss': loss_values[-1],
                'rmse': rmse_values[-1],
                'mae': mae_values[-1],
                'r2': r2_values[-1],
                'mape': mape_values[-1],
                'accuracy_10': acc10_values[-1],
                'accuracy_20': acc20_values[-1]
            },
            'history': {
                'rounds': rounds,
                'loss': loss_values,
                'rmse': rmse_values,
                'mae': mae_values,
                'r2': r2_values,
                'mape': mape_values,
                'accuracy_10': acc10_values,
                'accuracy_20': acc20_values
            },
            'partition_info': {
                'type': 'non-iid-hybrid',
                'clients': {}
            }
        }
    
    return results


def main():
    """Create test data and save to outputs directory"""
    
    # Create test output directory
    test_dir = Path("./outputs/test_plotting")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧪 Creating test data...")
    results = create_test_data()
    
    # Save to JSON
    output_file = test_dir / "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Test data saved to: {output_file}")
    
    # Verify structure
    print("\n📊 Verifying structure:")
    for strategy, data in results.items():
        num_rounds = len(data['history']['rounds'])
        final_r2 = data['final_metrics']['r2']
        print(f"   {strategy}: {num_rounds} rounds, final R²={final_r2:.4f}")
    
    print("\n🚀 Now run the plotting script:")
    print(f"   python plot_round_progression.py --output-dir {test_dir}")
    print(f"   python plot_round_progression.py --output-dir {test_dir} --detailed --summary")


if __name__ == "__main__":
    main()
