import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import os


# Strategy colors and markers for consistent styling
STRATEGY_STYLES = {
    'fedavg': {'color': '#1f77b4', 'marker': 'o', 'label': 'FedAvg'},
    'fedprox': {'color': '#ff7f0e', 'marker': 's', 'label': 'FedProx'},
    'scaffold': {'color': '#2ca02c', 'marker': '^', 'label': 'SCAFFOLD'},
}

PARTITION_STYLES = {
    'iid': {'linestyle': '-', 'label': 'IID'},
    'dirichlet': {'linestyle': '--', 'label': 'Non-IID'},
}


def plot_accuracy(
    history,
    config: Dict,
    save_path: str,
    show_plot: bool = True
):
    """
    Plot accuracy progression over federated learning rounds.
    
    Args:
        history: Flower History object with metrics
        config: Experiment configuration dict
        save_path: Directory to save the plot
        show_plot: Whether to display the plot
    """
    if not history.metrics_centralized:
        print("No centralized metrics found in history.")
        return
    
    accuracies = history.metrics_centralized.get('accuracy', [])
    if not accuracies:
        print("No accuracy metrics found.")
        return
    
    # Extract rounds and accuracy values
    rounds = [r for r, _ in accuracies]
    acc_values = [acc * 100 for _, acc in accuracies]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get style based on strategy
    strategy = config.get('strategy', 'fedavg')
    style = STRATEGY_STYLES.get(strategy, STRATEGY_STYLES['fedavg'])
    
    # Plot accuracy
    ax.plot(rounds, acc_values, 
            color=style['color'], 
            marker=style['marker'],
            linewidth=2, 
            markersize=8, 
            label='Test Accuracy')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    
    # Create title with config info
    strategy_name = strategy.upper()
    partition = config.get('partition_type', 'iid').upper()
    alpha = config.get('dirichlet_alpha', 'N/A')
    mu = config.get('fedprox_mu', 0)
    
    title = f"Federated Learning - {strategy_name}\n"
    title += f"Data: {partition}"
    if partition == "DIRICHLET":
        title += f" (α={alpha})"
    if strategy == "fedprox":
        title += f" | μ={mu}"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
    ax.set_ylim(0, 100)
    
    # Add final accuracy annotation
    final_acc = acc_values[-1]
    ax.annotate(
        f'Final: {final_acc:.1f}%',
        xy=(rounds[-1], final_acc),
        xytext=(rounds[-1] - 1, final_acc + 5),
        fontsize=11,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red'
    )
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plot_path = Path(save_path) / "accuracy_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def plot_comparison(
    results_paths: List[str],
    output_path: Optional[str] = None,
    show_plot: bool = True,
    title: Optional[str] = None
):
    """
    Compare accuracy across multiple experiments.
    
    Args:
        results_paths: List of paths to results.pkl files
        output_path: Where to save the comparison plot
        show_plot: Whether to display the plot
        title: Custom title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for path in results_paths:
        try:
            with open(path, 'rb') as f:
                results = pickle.load(f)
            
            history = results['history']
            config = results['config']
            
            accuracies = history.metrics_centralized.get('accuracy', [])
            if not accuracies:
                continue
            
            rounds = [r for r, _ in accuracies]
            acc_values = [acc * 100 for _, acc in accuracies]
            
            # Get styling
            strategy = config.get('strategy', 'fedavg')
            partition = config.get('partition_type', 'iid')
            alpha = config.get('dirichlet_alpha', 'N/A')
            
            style = STRATEGY_STYLES.get(strategy, {'color': 'gray', 'marker': 'x', 'label': strategy})
            
            # Create label
            label = f"{style['label']}"
            if partition == "dirichlet":
                label += f" (α={alpha})"
            else:
                label += " (IID)"
            
            ax.plot(
                rounds, acc_values,
                color=style['color'],
                marker=style['marker'],
                linewidth=2,
                markersize=8,
                label=label
            )
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    
    plot_title = title if title else 'FedAvg vs FedProx vs SCAFFOLD Comparison'
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def plot_full_comparison(
    results_dict: Dict[str, str],
    output_path: str = "full_comparison.png",
    show_plot: bool = True
):
    """
    Create a comprehensive comparison visualization with multiple subplots.
    
    Args:
        results_dict: Dict mapping experiment names to results.pkl paths
                     e.g., {'FedAvg-IID': 'path1', 'FedProx-NonIID': 'path2'}
        output_path: Where to save the plot
        show_plot: Whether to display
    """
    # Load all results
    all_results = {}
    for name, path in results_dict.items():
        try:
            with open(path, 'rb') as f:
                all_results[name] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    if not all_results:
        print("No results loaded!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Accuracy over rounds
    ax1 = axes[0]
    final_accs = {}
    
    for name, results in all_results.items():
        history = results['history']
        config = results['config']
        
        accuracies = history.metrics_centralized.get('accuracy', [])
        if not accuracies:
            continue
        
        rounds = [r for r, _ in accuracies]
        acc_values = [acc * 100 for _, acc in accuracies]
        
        strategy = config.get('strategy', 'fedavg')
        style = STRATEGY_STYLES.get(strategy, {'color': 'gray', 'marker': 'x'})
        
        ax1.plot(rounds, acc_values, 
                color=style['color'], 
                marker=style['marker'],
                linewidth=2, 
                markersize=6, 
                label=name)
        
        final_accs[name] = acc_values[-1]
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(0, 100)
    
    # Right plot: Final accuracy bar chart
    ax2 = axes[1]
    names = list(final_accs.keys())
    accs = list(final_accs.values())
    
    # Get colors based on strategy
    colors = []
    for name in names:
        if 'fedavg' in name.lower():
            colors.append(STRATEGY_STYLES['fedavg']['color'])
        elif 'fedprox' in name.lower():
            colors.append(STRATEGY_STYLES['fedprox']['color'])
        elif 'scaffold' in name.lower():
            colors.append(STRATEGY_STYLES['scaffold']['color'])
        else:
            colors.append('gray')
    
    bars = ax2.bar(range(len(names)), accs, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Final Accuracy (%)', fontsize=12)
    ax2.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax2.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFull comparison plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()
    
    # Print summary table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<30} {'Final Accuracy':>15}")
    print("-" * 60)
    for name, acc in sorted(final_accs.items(), key=lambda x: -x[1]):
        print(f"{name:<30} {acc:>14.2f}%")
    print("=" * 60)


def create_comparison_report(
    results_dict: Dict[str, str],
    output_dir: str = "comparison_results"
):
    """
    Create a comprehensive comparison report with multiple visualizations.
    
    Args:
        results_dict: Dict mapping experiment names to results.pkl paths
        output_dir: Directory to save all comparison outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_results = {}
    for name, path in results_dict.items():
        try:
            with open(path, 'rb') as f:
                all_results[name] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    if not all_results:
        print("No results loaded!")
        return
    
    # 1. Create main comparison plot
    plot_full_comparison(results_dict, 
                        output_path=os.path.join(output_dir, "full_comparison.png"),
                        show_plot=False)
    
    # 2. Create convergence comparison (accuracy over rounds)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for name, results in all_results.items():
        history = results['history']
        config = results['config']
        
        accuracies = history.metrics_centralized.get('accuracy', [])
        if not accuracies:
            continue
        
        rounds = [r for r, _ in accuracies]
        acc_values = [acc * 100 for _, acc in accuracies]
        
        strategy = config.get('strategy', 'fedavg')
        style = STRATEGY_STYLES.get(strategy, {'color': 'gray', 'marker': 'x'})
        
        ax.plot(rounds, acc_values,
               color=style['color'],
               marker=style['marker'],
               linewidth=2.5,
               markersize=8,
               label=name)
    
    ax.set_xlabel('Communication Round', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Convergence Comparison: FedAvg vs FedProx vs SCAFFOLD', 
                fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_comparison.png"), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Save summary to text file
    summary_path = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEDERATED LEARNING STRATEGY COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("FINAL ACCURACY RESULTS\n")
        f.write("-" * 40 + "\n")
        
        final_accs = []
        for name, results in all_results.items():
            config = results['config']
            history = results['history']
            accuracies = history.metrics_centralized.get('accuracy', [])
            
            if accuracies:
                final_acc = accuracies[-1][1] * 100
                final_accs.append((name, final_acc, config))
        
        # Sort by accuracy (descending)
        final_accs.sort(key=lambda x: -x[1])
        
        for name, acc, config in final_accs:
            f.write(f"\n{name}:\n")
            f.write(f"  Final Accuracy: {acc:.2f}%\n")
            f.write(f"  Strategy: {config.get('strategy', 'N/A')}\n")
            f.write(f"  Partition: {config.get('partition_type', 'N/A')}\n")
            if config.get('partition_type') == 'dirichlet':
                f.write(f"  Dirichlet Alpha: {config.get('dirichlet_alpha', 'N/A')}\n")
            if config.get('strategy') == 'fedprox':
                f.write(f"  FedProx Mu: {config.get('fedprox_mu', 'N/A')}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\nComparison report saved to: {output_dir}/")
    print(f"  - full_comparison.png")
    print(f"  - convergence_comparison.png")
    print(f"  - comparison_summary.txt")


def load_and_plot(results_path: str, show_plot: bool = True):
    """
    Load results from pickle file and plot accuracy.
    
    Args:
        results_path: Path to results.pkl file
        show_plot: Whether to display the plot
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    history = results['history']
    config = results['config']
    save_dir = str(Path(results_path).parent)
    
    plot_accuracy(history, config, save_dir, show_plot)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare" and len(sys.argv) > 2:
            # Compare multiple results
            results_paths = sys.argv[2:]
            plot_comparison(results_paths, output_path="comparison.png")
        else:
            # Load and plot single result
            results_path = sys.argv[1]
            load_and_plot(results_path)
    else:
        print("Usage:")
        print("  Single plot:    python visualize.py <path_to_results.pkl>")
        print("  Compare plots:  python visualize.py --compare <path1> <path2> <path3>")
        print("\nFor full comparison report in Python:")
        print("  from visualize import create_comparison_report")
        print("  create_comparison_report({")
        print("      'FedAvg-NonIID': 'path1/results.pkl',")
        print("      'FedProx-NonIID': 'path2/results.pkl',")
        print("      'SCAFFOLD-NonIID': 'path3/results.pkl'")
        print("  })")
