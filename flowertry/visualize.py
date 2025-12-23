"""
Visualization script for comparing IID vs Non-IID federated learning experiments.

This script:
1. Loads results from multiple experiment runs
2. Plots training curves comparing all scenarios
3. Visualizes data distribution per client (for non-IID)
4. Generates comparison metrics

Usage:
    python visualize.py --results_dir outputs/
    
    Or specify individual result files:
    python visualize.py --results outputs/exp1/results.pkl outputs/exp2/results.pkl
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_mnist, partition_dirichlet, partition_iid, get_client_class_distribution


def load_results(results_path: str) -> Dict:
    """Load results from a pickle file."""
    with open(results_path, "rb") as f:
        return pickle.load(f)


def find_latest_results(outputs_dir: str, experiment_names: List[str]) -> Dict[str, str]:
    """
    Find the latest results.pkl for each experiment type.
    
    Args:
        outputs_dir: Base outputs directory
        experiment_names: List of experiment names to find
    
    Returns:
        Dict mapping experiment name to results path
    """
    outputs_path = Path(outputs_dir)
    results = {}
    
    # Walk through all subdirectories to find results.pkl files
    all_results = list(outputs_path.glob("**/results.pkl"))
    
    for results_file in all_results:
        try:
            data = load_results(str(results_file))
            exp_name = data.get('config', {}).get('experiment_name', 'unknown')
            
            # Keep the most recent result for each experiment type
            if exp_name in experiment_names:
                if exp_name not in results or results_file.stat().st_mtime > Path(results[exp_name]).stat().st_mtime:
                    results[exp_name] = str(results_file)
        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")
    
    return results


def plot_accuracy_comparison(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Plot accuracy curves for all experiments on the same graph.
    
    Args:
        results_dict: Dict mapping experiment name to results data
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 6))
    
    colors = {
        'iid_fedavg': 'green',
        'noniid_fedavg': 'red',
        'noniid_fedprox': 'blue'
    }
    
    labels = {
        'iid_fedavg': 'IID + FedAvg (Baseline)',
        'noniid_fedavg': 'Non-IID + FedAvg',
        'noniid_fedprox': 'Non-IID + FedProx'
    }
    
    for exp_name, data in results_dict.items():
        history = data['history']
        config = data.get('config', {})
        
        # Extract centralized accuracy
        if hasattr(history, 'metrics_centralized') and 'accuracy' in history.metrics_centralized:
            accuracies = history.metrics_centralized['accuracy']
            rounds = [r for r, _ in accuracies]
            accs = [a for _, a in accuracies]
            
            color = colors.get(exp_name, 'gray')
            label = labels.get(exp_name, exp_name)
            
            plt.plot(rounds, accs, marker='o', label=label, color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Federated Learning: IID vs Non-IID Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy comparison plot to: {save_path}")
    
    plt.show()


def plot_loss_comparison(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Plot loss curves for all experiments on the same graph.
    """
    plt.figure(figsize=(12, 6))
    
    colors = {
        'iid_fedavg': 'green',
        'noniid_fedavg': 'red',
        'noniid_fedprox': 'blue'
    }
    
    labels = {
        'iid_fedavg': 'IID + FedAvg (Baseline)',
        'noniid_fedavg': 'Non-IID + FedAvg',
        'noniid_fedprox': 'Non-IID + FedProx'
    }
    
    for exp_name, data in results_dict.items():
        history = data['history']
        
        # Extract centralized loss
        if hasattr(history, 'losses_centralized') and history.losses_centralized:
            losses = history.losses_centralized
            rounds = [r for r, _ in losses]
            loss_vals = [l for _, l in losses]
            
            color = colors.get(exp_name, 'gray')
            label = labels.get(exp_name, exp_name)
            
            plt.plot(rounds, loss_vals, marker='s', label=label, color=color, linewidth=2, markersize=6)
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Federated Learning: Loss Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss comparison plot to: {save_path}")
    
    plt.show()


def visualize_data_distribution(
    num_clients: int = 10,
    alpha: float = 0.5,
    num_classes: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize the data distribution for IID vs Non-IID partitioning.
    Shows how classes are distributed across clients.
    
    Args:
        num_clients: Number of clients to visualize
        alpha: Dirichlet alpha parameter
        num_classes: Number of classes in the dataset
        save_path: Path to save the figure
    """
    trainset, _ = get_mnist()
    
    # Create IID and Non-IID partitions
    iid_partitions = partition_iid(trainset, num_clients)
    noniid_partitions = partition_dirichlet(trainset, num_clients, alpha=alpha)
    
    # Get distributions
    iid_dist = get_client_class_distribution(iid_partitions, num_classes)
    noniid_dist = get_client_class_distribution(noniid_partitions, num_classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot IID distribution
    ax1 = axes[0]
    iid_matrix = np.zeros((num_clients, num_classes))
    for cid in range(num_clients):
        for cls in range(num_classes):
            iid_matrix[cid, cls] = iid_dist[cid][cls]
    
    im1 = ax1.imshow(iid_matrix, aspect='auto', cmap='Blues')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Client', fontsize=12)
    ax1.set_title('IID Partitioning\n(Uniform Distribution)', fontsize=14)
    ax1.set_xticks(range(num_classes))
    ax1.set_yticks(range(num_clients))
    plt.colorbar(im1, ax=ax1, label='Sample Count')
    
    # Plot Non-IID distribution
    ax2 = axes[1]
    noniid_matrix = np.zeros((num_clients, num_classes))
    for cid in range(num_clients):
        for cls in range(num_classes):
            noniid_matrix[cid, cls] = noniid_dist[cid][cls]
    
    im2 = ax2.imshow(noniid_matrix, aspect='auto', cmap='Reds')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Client', fontsize=12)
    ax2.set_title(f'Non-IID Partitioning (Dirichlet α={alpha})\n(Heterogeneous Distribution)', fontsize=14)
    ax2.set_xticks(range(num_classes))
    ax2.set_yticks(range(num_clients))
    plt.colorbar(im2, ax=ax2, label='Sample Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved data distribution plot to: {save_path}")
    
    plt.show()


def print_comparison_metrics(results_dict: Dict[str, Dict]):
    """
    Print comparison metrics for all experiments.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON METRICS")
    print("=" * 70)
    
    metrics = []
    
    for exp_name, data in results_dict.items():
        history = data['history']
        config = data.get('config', {})
        
        # Extract final metrics
        final_acc = None
        if hasattr(history, 'metrics_centralized') and 'accuracy' in history.metrics_centralized:
            accuracies = history.metrics_centralized['accuracy']
            if accuracies:
                final_acc = accuracies[-1][1]
        
        # Calculate convergence speed (rounds to reach 90% of final accuracy)
        convergence_round = None
        if final_acc and hasattr(history, 'metrics_centralized') and 'accuracy' in history.metrics_centralized:
            target = 0.9 * final_acc
            for r, acc in history.metrics_centralized['accuracy']:
                if acc >= target:
                    convergence_round = r
                    break
        
        metrics.append({
            'name': exp_name,
            'partition': config.get('partition_type', 'unknown'),
            'strategy': config.get('strategy', 'unknown'),
            'alpha': config.get('dirichlet_alpha'),
            'mu': config.get('fedprox_mu'),
            'final_accuracy': final_acc,
            'convergence_round': convergence_round
        })
    
    # Print table
    print(f"\n{'Experiment':<20} {'Partition':<12} {'Strategy':<10} {'α':<6} {'μ':<6} {'Final Acc':<12} {'Conv. Round':<12}")
    print("-" * 80)
    
    for m in metrics:
        alpha_str = f"{m['alpha']:.2f}" if m['alpha'] else "N/A"
        mu_str = f"{m['mu']:.2f}" if m['mu'] else "N/A"
        acc_str = f"{m['final_accuracy']:.4f}" if m['final_accuracy'] else "N/A"
        conv_str = str(m['convergence_round']) if m['convergence_round'] else "N/A"
        
        print(f"{m['name']:<20} {m['partition']:<12} {m['strategy']:<10} {alpha_str:<6} {mu_str:<6} {acc_str:<12} {conv_str:<12}")
    
    print("=" * 70)
    
    # Print analysis
    if len(metrics) >= 2:
        iid_acc = next((m['final_accuracy'] for m in metrics if m['partition'] == 'iid'), None)
        noniid_fedavg_acc = next((m['final_accuracy'] for m in metrics if m['partition'] == 'dirichlet' and m['strategy'] == 'fedavg'), None)
        noniid_fedprox_acc = next((m['final_accuracy'] for m in metrics if m['partition'] == 'dirichlet' and m['strategy'] == 'fedprox'), None)
        
        print("\nANALYSIS:")
        if iid_acc and noniid_fedavg_acc:
            drop = (iid_acc - noniid_fedavg_acc) * 100
            print(f"• Accuracy drop due to non-IID data: {drop:.2f}%")
        
        if noniid_fedavg_acc and noniid_fedprox_acc:
            recovery = (noniid_fedprox_acc - noniid_fedavg_acc) * 100
            print(f"• FedProx accuracy recovery: +{recovery:.2f}%")
        
        if iid_acc and noniid_fedprox_acc:
            remaining_gap = (iid_acc - noniid_fedprox_acc) * 100
            print(f"• Remaining gap from IID baseline: {remaining_gap:.2f}%")


def visualize_alpha_sensitivity(
    alphas: List[float] = [0.1, 0.3, 0.5, 1.0, 5.0],
    num_clients: int = 10,
    num_classes: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize how different Dirichlet alpha values affect data distribution.
    
    Lower alpha = more heterogeneous (non-IID)
    Higher alpha = more uniform (approaching IID)
    """
    trainset, _ = get_mnist()
    
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 6))
    
    for idx, alpha in enumerate(alphas):
        partitions = partition_dirichlet(trainset, num_clients, alpha=alpha)
        dist = get_client_class_distribution(partitions, num_classes)
        
        matrix = np.zeros((num_clients, num_classes))
        for cid in range(num_clients):
            for cls in range(num_classes):
                matrix[cid, cls] = dist[cid][cls]
        
        ax = axes[idx]
        im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        ax.set_xlabel('Class', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Client', fontsize=10)
        ax.set_title(f'α = {alpha}', fontsize=12)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_clients))
    
    fig.suptitle('Effect of Dirichlet Alpha on Data Heterogeneity\n(Lower α = More Non-IID)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved alpha sensitivity plot to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize FL experiment results')
    parser.add_argument('--results_dir', type=str, default='outputs',
                        help='Directory containing experiment outputs')
    parser.add_argument('--results', nargs='+', type=str,
                        help='Specific results.pkl files to compare')
    parser.add_argument('--save_dir', type=str, default='plots',
                        help='Directory to save plots')
    parser.add_argument('--show_distribution', action='store_true',
                        help='Show data distribution visualization')
    parser.add_argument('--show_alpha_sensitivity', action='store_true',
                        help='Show effect of different alpha values')
    parser.add_argument('--num_clients_viz', type=int, default=10,
                        help='Number of clients to visualize for distribution plots')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for distribution visualization')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load results
    results_dict = {}
    
    if args.results:
        # Load specific result files
        for results_path in args.results:
            data = load_results(results_path)
            exp_name = data.get('config', {}).get('experiment_name', Path(results_path).parent.name)
            results_dict[exp_name] = data
    else:
        # Find latest results for each experiment type
        experiment_names = ['iid_fedavg', 'noniid_fedavg', 'noniid_fedprox']
        found_results = find_latest_results(args.results_dir, experiment_names)
        
        for exp_name, results_path in found_results.items():
            print(f"Found {exp_name}: {results_path}")
            results_dict[exp_name] = load_results(results_path)
    
    if results_dict:
        # Plot comparisons
        print("\nGenerating comparison plots...")
        plot_accuracy_comparison(results_dict, save_path=str(save_dir / 'accuracy_comparison.png'))
        plot_loss_comparison(results_dict, save_path=str(save_dir / 'loss_comparison.png'))
        print_comparison_metrics(results_dict)
    else:
        print("No experiment results found. Run experiments first:")
        print("  python main.py --config-name=iid_fedavg")
        print("  python main.py --config-name=noniid_fedavg")
        print("  python main.py --config-name=noniid_fedprox")
    
    # Visualize data distribution
    if args.show_distribution:
        print("\nGenerating data distribution visualization...")
        visualize_data_distribution(
            num_clients=args.num_clients_viz,
            alpha=args.alpha,
            save_path=str(save_dir / 'data_distribution.png')
        )
    
    # Show alpha sensitivity
    if args.show_alpha_sensitivity:
        print("\nGenerating alpha sensitivity visualization...")
        visualize_alpha_sensitivity(
            num_clients=args.num_clients_viz,
            save_path=str(save_dir / 'alpha_sensitivity.png')
        )


if __name__ == "__main__":
    main()
