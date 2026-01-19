#!/usr/bin/env python3
"""Create comparison plot similar to model_improvement_results.png for latest simulation."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(output_dir: str):
    """Load comparison results from JSON file."""
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_comparison(results: dict, output_dir: str):
    """Create 2x2 comparison plot similar to model_improvement_results.png."""
    
    # Extract strategies (exclude partition_info if present)
    strategies = {k: v for k, v in results.items() if isinstance(v, dict) and 'strategy' in v}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    names = list(strategies.keys())
    
    # Plot 1: R² Score Comparison
    ax = axes[0, 0]
    r2_values = [strategies[name]['final_metrics']['r2'] for name in names]
    bars = ax.bar(range(len(names)), r2_values, color=colors[:len(names)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.upper() for n in names], fontsize=11, fontweight='bold')
    ax.set_ylabel("R² Score", fontsize=12, fontweight='bold')
    ax.set_title("R² Score (Higher is Better)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(r2_values) * 0.98, max(r2_values) * 1.01])
    # Add value labels on bars
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: RMSE Comparison
    ax = axes[0, 1]
    rmse_values = [strategies[name]['final_metrics']['rmse'] for name in names]
    bars = ax.bar(range(len(names)), rmse_values, color=colors[:len(names)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.upper() for n in names], fontsize=11, fontweight='bold')
    ax.set_ylabel("RMSE", fontsize=12, fontweight='bold')
    ax.set_title("Root Mean Squared Error (Lower is Better)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: MAPE Comparison
    ax = axes[1, 0]
    mape_values = [strategies[name]['final_metrics']['mape'] for name in names]
    bars = ax.bar(range(len(names)), mape_values, color=colors[:len(names)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.upper() for n in names], fontsize=11, fontweight='bold')
    ax.set_ylabel("MAPE (%)", fontsize=12, fontweight='bold')
    ax.set_title("Mean Absolute Percentage Error (Lower is Better)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, mape_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Accuracy Metrics Comparison (Grouped Bar Chart)
    ax = axes[1, 1]
    x = np.arange(len(names))
    width = 0.35
    
    acc10_values = [strategies[name]['final_metrics']['accuracy_10'] for name in names]
    acc20_values = [strategies[name]['final_metrics']['accuracy_20'] for name in names]
    
    bars1 = ax.bar(x - width/2, acc10_values, width, label='Accuracy @ 10%', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, acc20_values, width, label='Accuracy @ 20%', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names], fontsize=11, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Prediction Accuracy (Higher is Better)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(min(acc10_values), min(acc20_values)) * 0.95, 100])
    
    # Find best strategy by R²
    best_strategy = max(strategies.items(), 
                       key=lambda x: x[1].get('final_metrics', {}).get('r2', 0))
    best_name = best_strategy[0].upper()
    best_r2 = best_strategy[1].get('final_metrics', {}).get('r2', 0)
    
    fig.suptitle(f'Federated Learning Strategy Comparison\nBest: {best_name} (R² = {best_r2:.4f})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save
    filepath = os.path.join(output_dir, "strategy_comparison_plot.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {filepath}")
    
    return filepath

if __name__ == "__main__":
    # Use latest simulation output
    output_dir = "/Users/dinukaperera/FLwithFlwr/flowertry/outputs/2026-01-19/01-35-08"
    
    print(f"Loading results from: {output_dir}")
    results = load_results(output_dir)
    
    print(f"Found strategies: {[k for k in results.keys() if isinstance(results.get(k), dict) and 'strategy' in results.get(k, {})]}")
    
    plot_comparison(results, output_dir)
    print("\n✅ Done!")
