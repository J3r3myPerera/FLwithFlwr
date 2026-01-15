"""
Strategy Comparison Script for Federated Disposable Income Regression.

Compares FedAvg, FedProx, SCAFFOLD, and Hybrid strategies on Non-IID data.

Usage:
    python compare_strategies.py
    python compare_strategies.py --rounds 100 --iid
    python compare_strategies.py --fedprox-weight 0.5 --scaffold-weight 0.8
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from main import run_simulation, compare_strategies
from visualize_metrics import ComparisonPlotter, save_metrics_to_csv


def plot_comparison(results: Dict, output_dir: str = "."):
    """
    Plot comparison of strategies.
    
    Args:
        results: Dictionary with results for each strategy
        output_dir: Directory to save plots
    """
    strategies = list(results.keys())
    
    # Extract final metrics
    rmse_values = [results[s]["final_metrics"]["rmse"] for s in strategies]
    mae_values = [results[s]["final_metrics"]["mae"] for s in strategies]
    r2_values = [results[s]["final_metrics"]["r2"] for s in strategies]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE comparison
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    axes[0].bar(strategies, rmse_values, color=colors)
    axes[0].set_ylabel('RMSE ($)')
    axes[0].set_title('RMSE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 50, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # MAE comparison
    axes[1].bar(strategies, mae_values, color=colors)
    axes[1].set_ylabel('MAE ($)')
    axes[1].set_title('MAE Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(mae_values):
        axes[1].text(i, v + 50, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # R² comparison
    axes[2].bar(strategies, r2_values, color=colors)
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylim(0, 1)
    for i, v in enumerate(r2_values):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Plot saved to: {os.path.join(output_dir, 'strategy_comparison.png')}")


def main():
    parser = argparse.ArgumentParser(description="Compare FL strategies for regression")
    parser.add_argument("--rounds", type=int, default=50, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--mu", type=float, default=0.1, help="FedProx mu parameter")
    parser.add_argument("--scaffold-lr-correction", type=float, default=1.0, help="SCAFFOLD learning rate correction")
    parser.add_argument("--fedprox-weight", type=float, default=1.0, help="FedProx weight in hybrid (0-1)")
    parser.add_argument("--scaffold-weight", type=float, default=1.0, help="SCAFFOLD weight in hybrid (0-1)")
    parser.add_argument("--iid", action="store_true", help="Use IID partitioning")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("FEDERATED STRATEGY COMPARISON")
    print("Disposable Income Regression")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Rounds: {args.rounds}")
    print(f"  Clients: {args.clients}")
    print(f"  Local Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Mu (FedProx): {args.mu}")
    print(f"  SCAFFOLD LR Correction: {args.scaffold_lr_correction}")
    print(f"  FedProx Weight (Hybrid): {args.fedprox_weight}")
    print(f"  SCAFFOLD Weight (Hybrid): {args.scaffold_weight}")
    print(f"  IID: {args.iid}")
    print(f"  Seed: {args.seed}")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plotting configuration
    plotting_config = {
        "enabled": not args.no_plot,
        "show_plot": not args.no_plot,
        "save_plot": True,
        "format": "png",
        "figsize": [14, 10],
        "update_interval": 1
    }
    
    # Run comparison
    results = compare_strategies(
        strategies=["fedavg", "fedprox", "scaffold", "hybrid"],
        num_rounds=args.rounds,
        num_clients=args.clients,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mu=args.mu,
        scaffold_lr_correction=args.scaffold_lr_correction,
        fedprox_weight=args.fedprox_weight,
        scaffold_weight=args.scaffold_weight,
        iid=args.iid,
        seed=args.seed,
        verbose=True,
        output_dir=args.output_dir,
        plotting_config=plotting_config
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"comparison_{timestamp}.json")
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(output_file, "w") as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate comparison plots using the new plotter
    if not args.no_plot:
        try:
            comp_plotter = ComparisonPlotter(
                output_dir=args.output_dir,
                figsize=(16, 12),
                plot_format="png"
            )
            comp_plotter.generate_all_plots(results)
            
            # Also generate the simple bar plot
            plot_comparison(results, args.output_dir)
        except Exception as e:
            print(f"Could not create plot: {e}")
    
    # Save metrics to CSV
    save_metrics_to_csv(results, args.output_dir)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RANKINGS")
    print("=" * 70)
    
    # Sort by RMSE (lower is better)
    sorted_by_rmse = sorted(
        results.items(),
        key=lambda x: x[1]["final_metrics"]["rmse"]
    )
    
    print("\nBy RMSE (lower is better):")
    for i, (strategy, result) in enumerate(sorted_by_rmse, 1):
        metrics = result["final_metrics"]
        print(f"  {i}. {strategy}: RMSE=${metrics['rmse']:,.2f}, R²={metrics['r2']:.4f}")
    
    # Best strategy
    best_strategy = sorted_by_rmse[0][0]
    best_rmse = sorted_by_rmse[0][1]["final_metrics"]["rmse"]
    best_r2 = sorted_by_rmse[0][1]["final_metrics"]["r2"]
    
    print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
    print(f"   RMSE: ${best_rmse:,.2f}")
    print(f"   R²: {best_r2:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

