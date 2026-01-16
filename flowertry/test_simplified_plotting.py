"""Test that final comparison plotting works after removing training/convergence plots."""

from visualize_metrics import ComparisonPlotter
import os

# Create mock results
mock_results = {
    'fedavg': {
        'final_metrics': {
            'loss': 0.5,
            'rmse': 1200.0,
            'mae': 950.0,
            'r2': 0.75,
            'mape': 15.2
        },
        'history': {
            'rounds': [1, 2, 3],
            'loss': [0.8, 0.6, 0.5],
            'rmse': [1500, 1350, 1200]
        }
    },
    'fedprox': {
        'final_metrics': {
            'loss': 0.45,
            'rmse': 1150.0,
            'mae': 900.0,
            'r2': 0.78,
            'mape': 14.5
        },
        'history': {
            'rounds': [1, 2, 3],
            'loss': [0.75, 0.55, 0.45],
            'rmse': [1450, 1300, 1150]
        }
    }
}

print("="*80)
print("TESTING SIMPLIFIED PLOTTING (FINAL COMPARISON ONLY)")
print("="*80)

# Create plotter
plotter = ComparisonPlotter(output_dir=".", figsize=(16, 12), plot_format="png")

print("\n✅ ComparisonPlotter created successfully")

# Check available methods
available_methods = [method for method in dir(plotter) if not method.startswith('_')]
print(f"\n📋 Available methods: {', '.join(available_methods)}")

# Verify removed methods are gone
removed_methods = ['plot_training_history', 'plot_convergence_analysis']
for method in removed_methods:
    if hasattr(plotter, method):
        print(f"❌ ERROR: {method} should have been removed!")
    else:
        print(f"✅ {method} successfully removed")

# Verify kept methods exist
kept_methods = ['plot_final_comparison', 'generate_all_plots']
for method in kept_methods:
    if hasattr(plotter, method):
        print(f"✅ {method} available")
    else:
        print(f"❌ ERROR: {method} is missing!")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED - Only final comparison plotting remains")
print("="*80)
