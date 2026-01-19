#!/usr/bin/env python3
"""
Plot round-by-round training progression for FL strategies.
Reads comparison_results.json and creates comprehensive visualizations.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def find_latest_output_dir():
    """Find the most recent output directory"""
    outputs_dir = Path("./outputs")
    
    if not outputs_dir.exists():
        return None
    
    # Get all date directories
    date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("202")]
    if not date_dirs:
        return None
    
    date_dirs.sort(reverse=True)
    
    for date_dir in date_dirs:
        # Get all time subdirectories
        time_dirs = [d for d in date_dir.iterdir() if d.is_dir()]
        if time_dirs:
            time_dirs.sort(reverse=True)
            return time_dirs[0]
    
    return None


def load_comparison_results(output_dir: Path) -> Dict:
    """Load comparison results JSON"""
    json_file = output_dir / "comparison_results.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"No comparison_results.json found in {output_dir}")
    
    with open(json_file, 'r') as f:
        return json.load(f)


def extract_histories(results: Dict) -> Dict[str, Dict]:
    """Extract round-by-round history for each strategy"""
    histories = {}
    
    for strategy, data in results.items():
        if isinstance(data, dict) and "history" in data:
            histories[strategy] = data["history"]
    
    return histories


def plot_training_progression(histories: Dict[str, Dict], output_dir: Path):
    """
    Create a 2x2 grid plot showing:
    1. Training Loss progression
    2. Validation Loss progression (if available)
    3. MAPE progression with target line
    4. Final Metrics comparison (bar chart)
    """
    
    # Setup style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    
    # Define colors for each strategy
    color_map = {
        'fedavg': '#1f77b4',      # Blue
        'fedprox': '#ff7f0e',     # Orange
        'scaffold': '#2ca02c',    # Green
        'hybrid': '#d62728',      # Red
        'baseline': '#9467bd',    # Purple
        'improved_v1': '#8c564b', # Brown
        'improved_v2': '#e377c2', # Pink
        'residual': '#7f7f7f',    # Gray
        'wide_and_deep': '#bcbd22' # Yellow-green
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Federated Learning Training Progression', fontsize=16, fontweight='bold')
    
    # 1. Training Loss (top-left) - using 'loss' from history
    ax_train = axes[0, 0]
    ax_train.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax_train.set_xlabel('Round', fontsize=12)
    ax_train.set_ylabel('Loss', fontsize=12)
    ax_train.grid(True, alpha=0.3)
    
    for strategy, history in histories.items():
        if 'loss' in history and len(history['loss']) > 0:
            rounds = history.get('rounds', list(range(1, len(history['loss']) + 1)))
            color = color_map.get(strategy.lower(), '#000000')
            ax_train.plot(rounds, history['loss'], 
                         marker='o', markersize=3, linewidth=2,
                         label=strategy, color=color, alpha=0.8)
    
    ax_train.legend(loc='best', fontsize=10)
    
    # 2. Validation Loss (top-right) - using R² as proxy (higher is better)
    ax_val = axes[0, 1]
    ax_val.set_title('Validation Performance (R² Score)', fontsize=14, fontweight='bold')
    ax_val.set_xlabel('Round', fontsize=12)
    ax_val.set_ylabel('R² Score', fontsize=12)
    ax_val.grid(True, alpha=0.3)
    
    for strategy, history in histories.items():
        if 'r2' in history and len(history['r2']) > 0:
            rounds = history.get('rounds', list(range(1, len(history['r2']) + 1)))
            color = color_map.get(strategy.lower(), '#000000')
            ax_val.plot(rounds, history['r2'], 
                       marker='s', markersize=3, linewidth=2,
                       label=strategy, color=color, alpha=0.8)
    
    ax_val.legend(loc='best', fontsize=10)
    
    # 3. MAPE Over Training (bottom-left)
    ax_mape = axes[1, 0]
    ax_mape.set_title('MAPE Over Training', fontsize=14, fontweight='bold')
    ax_mape.set_xlabel('Round', fontsize=12)
    ax_mape.set_ylabel('MAPE (%)', fontsize=12)
    ax_mape.grid(True, alpha=0.3)
    
    # Add target line at 15%
    if histories:
        max_rounds = max(len(h.get('mape', [])) for h in histories.values())
        if max_rounds > 0:
            ax_mape.axhline(y=15, color='red', linestyle='--', linewidth=2, 
                           label='Target (15%)', alpha=0.7)
    
    for strategy, history in histories.items():
        if 'mape' in history and len(history['mape']) > 0:
            rounds = history.get('rounds', list(range(1, len(history['mape']) + 1)))
            color = color_map.get(strategy.lower(), '#000000')
            ax_mape.plot(rounds, history['mape'], 
                        marker='^', markersize=3, linewidth=2,
                        label=strategy, color=color, alpha=0.8)
    
    ax_mape.legend(loc='best', fontsize=10)
    
    # 4. Final Metrics Comparison (bottom-right) - Bar chart
    ax_metrics = axes[1, 1]
    ax_metrics.set_title('Final Metrics Comparison', fontsize=14, fontweight='bold')
    ax_metrics.set_ylabel('Percentage (%)', fontsize=12)
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Extract final metrics for each strategy
    strategies_list = list(histories.keys())
    final_mape = []
    final_acc10 = []
    final_acc20 = []
    
    for strategy in strategies_list:
        history = histories[strategy]
        final_mape.append(history['mape'][-1] if 'mape' in history and len(history['mape']) > 0 else 0)
        final_acc10.append(history['accuracy_10'][-1] if 'accuracy_10' in history and len(history['accuracy_10']) > 0 else 0)
        final_acc20.append(history['accuracy_20'][-1] if 'accuracy_20' in history and len(history['accuracy_20']) > 0 else 0)
    
    x = np.arange(len(strategies_list))
    width = 0.25
    
    bars1 = ax_metrics.bar(x - width, final_mape, width, label='MAPE', color='#3498db')
    bars2 = ax_metrics.bar(x, final_acc10, width, label='ACCURACY@10%', color='#e74c3c')
    bars3 = ax_metrics.bar(x + width, final_acc20, width, label='ACCURACY@20%', color='#2ecc71')
    
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(strategies_list, rotation=45, ha='right')
    ax_metrics.legend(loc='best', fontsize=10)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax_metrics.annotate(f'{height:.1f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom',
                              fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "training_progression.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    plt.show()


def plot_detailed_metrics(histories: Dict[str, Dict], output_dir: Path):
    """
    Create detailed individual metric plots (6 subplots):
    - Loss, RMSE, MAE, R², MAPE, Accuracy
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Detailed Metrics Progression', fontsize=16, fontweight='bold')
    
    color_map = {
        'fedavg': '#1f77b4',
        'fedprox': '#ff7f0e',
        'scaffold': '#2ca02c',
        'hybrid': '#d62728',
    }
    
    metrics_config = [
        ('loss', 'Loss (MSE)', axes[0, 0]),
        ('rmse', 'RMSE ($)', axes[0, 1]),
        ('mae', 'MAE ($)', axes[1, 0]),
        ('r2', 'R² Score', axes[1, 1]),
        ('mape', 'MAPE (%)', axes[2, 0]),
        ('accuracy_10', 'Accuracy@10% (%)', axes[2, 1])
    ]
    
    for metric_name, ylabel, ax in metrics_config:
        ax.set_title(f'{ylabel} Progression', fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        for strategy, history in histories.items():
            if metric_name in history and len(history[metric_name]) > 0:
                rounds = history.get('rounds', list(range(1, len(history[metric_name]) + 1)))
                color = color_map.get(strategy.lower(), '#000000')
                ax.plot(rounds, history[metric_name], 
                       marker='o', markersize=2, linewidth=1.5,
                       label=strategy, color=color, alpha=0.8)
        
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "detailed_metrics_progression.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed plot saved to: {output_file}")
    
    plt.show()


def print_metrics_summary(histories: Dict[str, Dict], results: Dict):
    """Print a summary table of metrics"""
    
    print("\n" + "=" * 100)
    print("ROUND-BY-ROUND METRICS SUMMARY")
    print("=" * 100)
    
    for strategy, history in histories.items():
        print(f"\n📊 {strategy.upper()}")
        print("-" * 100)
        print(f"{'Round':>6} | {'Loss':>10} | {'RMSE':>10} | {'R²':>8} | {'MAPE':>7} | {'Acc@10%':>8} | {'Acc@20%':>8}")
        print("-" * 100)
        
        rounds = history.get('rounds', [])
        for i, round_num in enumerate(rounds):
            loss = history['loss'][i] if i < len(history.get('loss', [])) else 0
            rmse = history['rmse'][i] if i < len(history.get('rmse', [])) else 0
            r2 = history['r2'][i] if i < len(history.get('r2', [])) else 0
            mape = history['mape'][i] if i < len(history.get('mape', [])) else 0
            acc10 = history['accuracy_10'][i] if i < len(history.get('accuracy_10', [])) else 0
            acc20 = history['accuracy_20'][i] if i < len(history.get('accuracy_20', [])) else 0
            
            print(f"{round_num:>6} | {loss:>10.6f} | ${rmse:>9,.0f} | {r2:>8.4f} | {mape:>6.2f}% | {acc10:>7.2f}% | {acc20:>7.2f}%")
    
    print("\n" + "=" * 100)
    print("FINAL METRICS COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':>12} | {'Loss':>10} | {'RMSE':>10} | {'R²':>8} | {'MAPE':>7} | {'Acc@10%':>8} | {'Acc@20%':>8}")
    print("-" * 100)
    
    for strategy, data in results.items():
        if isinstance(data, dict) and 'final_metrics' in data:
            metrics = data['final_metrics']
            print(f"{strategy:>12} | {metrics.get('loss', 0):>10.6f} | ${metrics['rmse']:>9,.0f} | "
                  f"{metrics['r2']:>8.4f} | {metrics['mape']:>6.2f}% | "
                  f"{metrics['accuracy_10']:>7.2f}% | {metrics['accuracy_20']:>7.2f}%")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Plot FL round-by-round training progression')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Path to output directory (default: latest)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Generate detailed metrics plots')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Print metrics summary table')
    
    args = parser.parse_args()
    
    # Find output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = find_latest_output_dir()
        if output_dir is None:
            print("❌ No output directory found. Please specify with --output-dir")
            return
    
    print(f"\n📂 Using output directory: {output_dir}")
    
    # Load results
    try:
        results = load_comparison_results(output_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Extract histories
    histories = extract_histories(results)
    
    if not histories:
        print("❌ No history data found in comparison_results.json")
        print("   Make sure the simulation ran with compare_all=true")
        return
    
    print(f"\n✅ Found history data for {len(histories)} strategies:")
    for strategy, history in histories.items():
        num_rounds = len(history.get('rounds', []))
        print(f"   - {strategy}: {num_rounds} rounds")
    
    # Generate plots
    plot_training_progression(histories, output_dir)
    
    if args.detailed:
        plot_detailed_metrics(histories, output_dir)
    
    if args.summary:
        print_metrics_summary(histories, results)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
