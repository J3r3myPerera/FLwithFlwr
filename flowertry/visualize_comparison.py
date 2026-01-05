"""
Visualize comparison results from federated learning strategy comparison.

Usage:
    python visualize_comparison.py <path_to_comparison_results.pkl>
"""

import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: str):
    """Load comparison results from pickle file."""
    with open(results_path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_comparison(results_data: dict, output_path: str = None):
    """Plot comparison of strategies."""
    results = results_data['results']
    config = results_data['config']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy progression
    for result in results:
        strategy = result['strategy']
        accuracies = result['accuracies']
        
        if accuracies:
            rounds = [rnd for rnd, _ in accuracies]
            accs = [acc * 100 for _, acc in accuracies]
            ax1.plot(rounds, accs, marker='o', label=strategy, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Progression by Strategy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Plot loss progression
    for result in results:
        strategy = result['strategy']
        losses = result['losses']
        
        if losses:
            rounds = [rnd for rnd, _ in losses]
            loss_vals = [loss for _, loss in losses]
            ax2.plot(rounds, loss_vals, marker='s', label=strategy, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Progression by Strategy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_detailed_summary(results_data: dict):
    """Print detailed summary of comparison."""
    results = results_data['results']
    config = results_data['config']
    
    print("=" * 80)
    print("DETAILED COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Rounds: {config['num_rounds']}")
    print(f"  Clients: {config['num_clients']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Local Epochs: {config['local_epochs']}")
    
    print(f"\n{'Strategy':<15} {'Final Acc':<15} {'Final Loss':<15} {'Time (s)':<15} {'Rounds':<10}")
    print("-" * 80)
    
    for result in results:
        strategy = result['strategy']
        final_acc = result['final_accuracy']
        final_loss = result['final_loss']
        elapsed_time = result.get('elapsed_time', 0)
        num_rounds = len(result['accuracies']) if result['accuracies'] else 0
        
        acc_str = f"{final_acc*100:.2f}%" if final_acc else "N/A"
        loss_str = f"{final_loss:.4f}" if final_loss else "N/A"
        time_str = f"{elapsed_time:.2f}" if elapsed_time else "N/A"
        
        print(f"{strategy:<15} {acc_str:<15} {loss_str:<15} {time_str:<15} {num_rounds:<10}")
    
    # Find best by accuracy
    valid_results = [r for r in results if r['final_accuracy'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['final_accuracy'])
        print(f"\n{'='*80}")
        print(f"BEST STRATEGY (by accuracy): {best['strategy']} ({best['final_accuracy']*100:.2f}%)")
        print(f"{'='*80}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_comparison.py <path_to_comparison_results.pkl>")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    if not Path(results_path).exists():
        print(f"Error: File not found: {results_path}")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from: {results_path}")
    results_data = load_results(results_path)
    
    # Print summary
    print_detailed_summary(results_data)
    
    # Plot comparison
    try:
        output_path = str(Path(results_path).parent / "comparison_plot.png")
        plot_comparison(results_data, output_path)
    except ImportError:
        print("\nNote: matplotlib not available. Skipping plot generation.")
        print("Install with: pip install matplotlib")


if __name__ == "__main__":
    main()

