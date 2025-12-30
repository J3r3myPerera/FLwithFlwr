import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import List, Optional, Dict
import os


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
    
    # Plot accuracy
    ax.plot(rounds, acc_values, 'b-o', linewidth=2, markersize=8, label='Test Accuracy')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    
    # Create title with config info
    strategy = config.get('strategy', 'fedavg').upper()
    partition = config.get('partition_type', 'iid').upper()
    alpha = config.get('dirichlet_alpha', 'N/A')
    mu = config.get('fedprox_mu', 0)
    
    title = f"Federated Learning - {strategy}\n"
    title += f"Data: {partition}"
    if partition == "DIRICHLET":
        title += f" (α={alpha})"
    if strategy == "FEDPROX":
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
    show_plot: bool = True
):
    """
    Compare accuracy across multiple experiments.
    
    Args:
        results_paths: List of paths to results.pkl files
        output_path: Where to save the comparison plot
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, path in enumerate(results_paths):
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
            
            # Create label
            strategy = config.get('strategy', 'fedavg').upper()
            partition = config.get('partition_type', 'iid')
            alpha = config.get('dirichlet_alpha', 'N/A')
            
            label = f"{strategy} - {partition}"
            if partition == "dirichlet":
                label += f" (α={alpha})"
            
            ax.plot(
                rounds, acc_values,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linewidth=2,
                markersize=8,
                label=label
            )
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('FedAvg vs FedProx Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


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
        # Load and plot from provided path
        results_path = sys.argv[1]
        load_and_plot(results_path)
    else:
        print("Usage: python visualize.py <path_to_results.pkl>")
        print("\nTo compare multiple experiments:")
        print("  from visualize import plot_comparison")
        print("  plot_comparison(['path1/results.pkl', 'path2/results.pkl'])")
