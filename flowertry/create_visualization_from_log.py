"""
Create visualizations from FL simulation log file.
Parses the main2.log file and creates comparison plots similar to:
- final_comparison_YYYYMMDD_HHMMSS.png
- detailed_metrics_progression.png
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Log file path
LOG_PATH = '/Users/dinukaperera/FLwithFlwr/flowertry/outputs/2026-01-22/02-36-31/main.log'
OUTPUT_DIR = '/Users/dinukaperera/FLwithFlwr/flowertry/outputs/2026-01-22/02-36-31'


def parse_log_file(log_path: str) -> dict:
    """
    Parse the log file to extract metrics for each strategy.
    
    The log contains multiple FL runs (one per strategy).
    Each run has metrics_centralized with all metrics per round.
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all metrics_centralized blocks
    # Pattern: app_fit: metrics_centralized {...}
    pattern = r"app_fit: metrics_centralized \{([^}]+(?:\{[^}]*\}[^}]*)*)\}"
    
    # Alternative: Find the JSON-like blocks for each metric
    # Let's parse the structured data from the log
    
    strategies = ['fedavg', 'fedprox', 'scaffold', 'hybrid']
    results = {}
    
    # Split content by "FL starting" to get each run
    runs = content.split("FL starting")
    
    # Skip the first split (before first run)
    runs = runs[1:]
    
    for idx, run in enumerate(runs):
        if idx >= len(strategies):
            break
            
        strategy_name = strategies[idx]
        
        # Extract metrics from the run
        # Look for metrics_centralized block
        metrics_match = re.search(
            r"app_fit: metrics_centralized \{'rmse': \[(.*?)\], 'mae': \[(.*?)\], 'r2': \[(.*?)\], 'mape': \[(.*?)\], 'accuracy_10': \[(.*?)\], 'accuracy_20': \[(.*?)\]\}",
            run,
            re.DOTALL
        )
        
        if metrics_match:
            def parse_metric_tuples(s):
                """Parse tuples like (0, 12836.887) from string"""
                tuples = re.findall(r'\((\d+),\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\)', s)
                rounds = [int(t[0]) for t in tuples]
                values = [float(t[1]) for t in tuples]
                return rounds, values
            
            rounds_rmse, rmse = parse_metric_tuples(metrics_match.group(1))
            _, mae = parse_metric_tuples(metrics_match.group(2))
            _, r2 = parse_metric_tuples(metrics_match.group(3))
            _, mape = parse_metric_tuples(metrics_match.group(4))
            _, acc10 = parse_metric_tuples(metrics_match.group(5))
            _, acc20 = parse_metric_tuples(metrics_match.group(6))
            
            # Get final metrics (last round)
            results[strategy_name] = {
                'strategy': strategy_name,
                'final_metrics': {
                    'rmse': rmse[-1],
                    'mae': mae[-1],
                    'r2': r2[-1],
                    'mape': mape[-1],
                    'accuracy_10': acc10[-1],
                    'accuracy_20': acc20[-1]
                },
                'history': {
                    'rounds': rounds_rmse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'accuracy_10': acc10,
                    'accuracy_20': acc20
                }
            }
    
    return results


def create_final_comparison_plot(results: dict, output_dir: str):
    """
    Create a final comparison bar chart similar to final_comparison_YYYYMMDD_HHMMSS.png
    """
    strategies = list(results.keys())
    strategy_labels = [s.upper() for s in strategies]
    
    # Colors for each strategy
    colors = {
        'fedavg': '#3498db',      # Blue
        'fedprox': '#e74c3c',     # Red
        'scaffold': '#2ecc71',    # Green
        'hybrid': '#9b59b6'       # Purple
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Final FL Strategy Comparison - Dirichlet Non-IID (α=0.3)', fontsize=16, fontweight='bold')
    
    metrics = [
        ('rmse', 'RMSE ($)', 'lower'),
        ('mae', 'MAE ($)', 'lower'),
        ('r2', 'R² Score', 'higher'),
        ('mape', 'MAPE (%)', 'lower'),
        ('accuracy_10', 'Accuracy ±10%', 'higher'),
        ('accuracy_20', 'Accuracy ±20%', 'higher')
    ]
    
    for idx, (metric_key, metric_label, better) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        values = [results[s]['final_metrics'][metric_key] for s in strategies]
        bars = ax.bar(strategy_labels, values, color=[colors[s] for s in strategies], edgecolor='black', linewidth=1.2)
        
        # Highlight best
        if better == 'higher':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}' if metric_key in ['rmse', 'mae', 'r2'] else f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(output_dir, f'final_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] {save_path}")
    return save_path


def create_detailed_metrics_progression(results: dict, output_dir: str):
    """
    Create detailed metrics progression plot similar to detailed_metrics_progression.png
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics Progression - Dirichlet Non-IID (α=0.3, Heterogeneity=0.8762)', 
                 fontsize=14, fontweight='bold')
    
    # Style settings
    colors = {
        'fedavg': '#3498db',      # Blue
        'fedprox': '#e74c3c',     # Red
        'scaffold': '#2ecc71',    # Green
        'hybrid': '#9b59b6'       # Purple
    }
    
    linestyles = {
        'fedavg': '-',
        'fedprox': '--',
        'scaffold': '-.',
        'hybrid': '-'
    }
    
    linewidths = {
        'fedavg': 1.5,
        'fedprox': 1.5,
        'scaffold': 1.5,
        'hybrid': 2.5
    }
    
    metrics = [
        ('rmse', 'RMSE ($)', False),
        ('mae', 'MAE ($)', False),
        ('r2', 'R² Score', True),
        ('mape', 'MAPE (%)', False),
        ('accuracy_10', 'Accuracy ±10% (%)', True),
        ('accuracy_20', 'Accuracy ±20% (%)', True)
    ]
    
    for idx, (metric_key, metric_label, higher_better) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        for strategy_name, data in results.items():
            rounds = data['history']['rounds']
            values = data['history'][metric_key]
            
            ax.plot(rounds, values,
                   label=strategy_name.upper(),
                   color=colors[strategy_name],
                   linestyle=linestyles[strategy_name],
                   linewidth=linewidths[strategy_name],
                   marker='o' if strategy_name == 'hybrid' else None,
                   markersize=3,
                   alpha=0.9 if strategy_name == 'hybrid' else 0.7)
        
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Set appropriate y-axis limits
        if metric_key == 'r2':
            ax.set_ylim([-0.6, 1.0])
        elif metric_key in ['accuracy_10', 'accuracy_20']:
            ax.set_ylim([0, 100])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, 'detailed_metrics_progression.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] {save_path}")
    return save_path


def create_summary_table(results: dict, output_dir: str):
    """
    Create a summary table image showing final metrics.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Prepare table data
    strategies = ['fedavg', 'fedprox', 'scaffold', 'hybrid']
    columns = ['Strategy', 'RMSE ($)', 'MAE ($)', 'R²', 'MAPE (%)', 'Acc±10%', 'Acc±20%']
    
    table_data = []
    for s in strategies:
        if s in results:
            fm = results[s]['final_metrics']
            row = [
                s.upper(),
                f"${fm['rmse']:,.0f}",
                f"${fm['mae']:,.0f}",
                f"{fm['r2']:.4f}",
                f"{fm['mape']:.2f}%",
                f"{fm['accuracy_10']:.1f}%",
                f"{fm['accuracy_20']:.1f}%"
            ]
            table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(columns)
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Highlight best values
    # Find best for each metric (column)
    for col_idx in range(1, len(columns)):
        values = []
        for row_idx, s in enumerate(strategies):
            if s in results:
                fm = results[s]['final_metrics']
                metric_keys = ['rmse', 'mae', 'r2', 'mape', 'accuracy_10', 'accuracy_20']
                values.append(fm[metric_keys[col_idx - 1]])
            else:
                values.append(None)
        
        # Determine if higher or lower is better
        higher_better = col_idx in [3, 5, 6]  # r2, acc10, acc20
        
        valid_values = [v for v in values if v is not None]
        if valid_values:
            best_val = max(valid_values) if higher_better else min(valid_values)
            best_idx = values.index(best_val)
            table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')  # Light green
    
    # Color strategy column
    strategy_colors = {
        'FEDAVG': '#3498db',
        'FEDPROX': '#e74c3c',
        'SCAFFOLD': '#2ecc71',
        'HYBRID': '#9b59b6'
    }
    
    for row_idx, s in enumerate(strategies):
        if s in results:
            table[(row_idx + 1, 0)].set_facecolor(strategy_colors[s.upper()])
            table[(row_idx + 1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Final Metrics Summary - Dirichlet Non-IID (α=0.3)', fontsize=14, fontweight='bold', pad=20)
    
    save_path = os.path.join(output_dir, 'metrics_summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Saved] {save_path}")
    return save_path


def save_results_json(results: dict, output_dir: str):
    """Save results to JSON for later use."""
    save_path = os.path.join(output_dir, 'comparison_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {save_path}")
    return save_path


def main():
    print("=" * 70)
    print("Creating Visualizations from FL Simulation Log")
    print("=" * 70)
    
    # Parse log file
    print(f"\n[Parsing] {LOG_PATH}")
    results = parse_log_file(LOG_PATH)
    
    print(f"\n[Found] {len(results)} strategies:")
    for strategy_name, data in results.items():
        fm = data['final_metrics']
        print(f"  {strategy_name.upper()}: RMSE=${fm['rmse']:.0f}, R²={fm['r2']:.4f}, MAPE={fm['mape']:.2f}%")
    
    # Create visualizations
    print("\n[Creating Visualizations]")
    
    create_final_comparison_plot(results, OUTPUT_DIR)
    create_detailed_metrics_progression(results, OUTPUT_DIR)
    create_summary_table(results, OUTPUT_DIR)
    save_results_json(results, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("DONE! Visualizations saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
