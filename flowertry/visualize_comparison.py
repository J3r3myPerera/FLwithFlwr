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
    
    # Check if any strategy has mu_history (adaptive FedProx)
    has_adaptive_mu = any('mu_history' in result and result.get('mu_history') for result in results)
    
    if has_adaptive_mu:
        # Create figure with 3 subplots: accuracy, loss, and mu vs loss
        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
    else:
        # Create figure with 2 subplots: accuracy and loss
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax3 = None
    
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
    
    # Plot mu evolution vs loss (for adaptive strategies)
    if ax3 is not None:
        # Create twin axis for loss
        ax3_twin = ax3.twinx()
        
        # Use a color cycle for consistent coloring
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        color_idx = 0
        
        for result in results:
            strategy = result['strategy']
            mu_history = result.get('mu_history', [])
            losses = result.get('losses', [])
            
            if mu_history and losses:
                # Extract mu rounds and values
                mu_rounds = [rnd for rnd, _ in mu_history]
                mu_vals = [mu for _, mu in mu_history]
                
                # Match loss values to mu rounds (mu is recorded at evaluation time)
                # Loss rounds might be slightly different, so we'll align them
                loss_dict = {rnd: loss for rnd, loss in losses}
                
                # Create aligned data for plotting
                aligned_rounds = []
                aligned_mu = []
                aligned_loss = []
                
                for rnd, mu in mu_history:
                    if rnd in loss_dict:
                        aligned_rounds.append(rnd)
                        aligned_mu.append(mu)
                        aligned_loss.append(loss_dict[rnd])
                
                if aligned_rounds:
                    # Get color for this strategy
                    color = colors[color_idx % len(colors)]
                    color_idx += 1
                    
                    # Plot mu on left y-axis (solid line with circles)
                    line1 = ax3.plot(aligned_rounds, aligned_mu, marker='o', 
                                     label=f'{strategy} (μ)', color=color, 
                                     linewidth=2, markersize=6, linestyle='-')
                    
                    # Plot loss on right y-axis (dashed line with squares)
                    line2 = ax3_twin.plot(aligned_rounds, aligned_loss, marker='s', 
                                          label=f'{strategy} (Loss)', color=color, 
                                          linewidth=2, markersize=6, linestyle='--', alpha=0.7)
        
        # Set labels and titles
        ax3.set_xlabel('Round', fontsize=12)
        ax3.set_ylabel('Proximal Term (μ)', fontsize=12, fontweight='bold')
        ax3_twin.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax3.set_title('Adaptive μ vs Loss Evolution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        
        # Combine legends from both axes
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left', ncol=1)
        
        # Color the y-axis labels
        ax3.tick_params(axis='y', labelcolor='black')
        ax3_twin.tick_params(axis='y', labelcolor='black')
    
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
    
    # Check for adaptive mu config
    if 'adaptive_mu_config' in config and config['adaptive_mu_config']:
        adaptive_cfg = config['adaptive_mu_config']
        if adaptive_cfg.get('enabled', False):
            print(f"\nAdaptive μ Configuration:")
            print(f"  Initial μ: {adaptive_cfg.get('initial_mu', 'N/A')}")
            print(f"  μ Range: [{adaptive_cfg.get('mu_min', 'N/A')}, {adaptive_cfg.get('mu_max', 'N/A')}]")
            print(f"  Increase/Decrease Factors: {adaptive_cfg.get('increase_factor', 'N/A')}/{adaptive_cfg.get('decrease_factor', 'N/A')}")
            print(f"  Warmup Rounds: {adaptive_cfg.get('warmup_rounds', 'N/A')}")
    
    print(f"\n{'Strategy':<20} {'Final Acc':<15} {'Final Loss':<15} {'Time (s)':<12} {'Final μ':<12}")
    print("-" * 80)
    
    for result in results:
        strategy = result['strategy']
        final_acc = result['final_accuracy']
        final_loss = result['final_loss']
        elapsed_time = result.get('elapsed_time', 0)
        final_mu = result.get('final_mu', None)
        
        acc_str = f"{final_acc*100:.2f}%" if final_acc else "N/A"
        loss_str = f"{final_loss:.4f}" if final_loss else "N/A"
        time_str = f"{elapsed_time:.2f}" if elapsed_time else "N/A"
        mu_str = f"{final_mu:.4f}" if final_mu is not None else "-"
        
        print(f"{strategy:<20} {acc_str:<15} {loss_str:<15} {time_str:<12} {mu_str:<12}")
    
    # Print mu evolution for adaptive strategies
    for result in results:
        mu_history = result.get('mu_history', [])
        if mu_history:
            strategy = result['strategy']
            print(f"\n  μ Evolution for {strategy}:")
            # Show first 3, middle, and last 3 values
            if len(mu_history) <= 7:
                for rnd, mu in mu_history:
                    print(f"    Round {rnd}: μ = {mu:.4f}")
            else:
                for rnd, mu in mu_history[:3]:
                    print(f"    Round {rnd}: μ = {mu:.4f}")
                print(f"    ...")
                for rnd, mu in mu_history[-3:]:
                    print(f"    Round {rnd}: μ = {mu:.4f}")
    
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

