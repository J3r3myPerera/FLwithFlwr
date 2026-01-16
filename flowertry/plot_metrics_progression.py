#!/usr/bin/env python3
"""
Plot training metrics progression from the most recent FL run.
Parses the log file and creates visualizations.
"""

import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_latest_output():
    """Find the most recent output directory"""
    outputs_dir = Path("./outputs")
    
    # Get all date directories
    date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("202")]
    date_dirs.sort(reverse=True)
    
    for date_dir in date_dirs:
        # Get all time subdirectories
        time_dirs = [d for d in date_dir.iterdir() if d.is_dir()]
        if time_dirs:
            time_dirs.sort(reverse=True)
            return time_dirs[0]
    
    return None

def parse_log_metrics(log_file):
    """Parse metrics from log file - handles multiple strategy runs"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Split log by FL runs (each run starts with "FL starting")
    runs = content.split('[INFO] - FL starting')
    
    all_metrics = {}
    
    for run_idx, run_content in enumerate(runs[1:], 1):  # Skip first split (before any FL run)
        metrics = {}
        
        # Find the metrics_centralized section for this run
        centralized_match = re.search(r"app_fit: metrics_centralized (.+?)(?=\[2026|\Z)", run_content, re.DOTALL)
        
        if centralized_match:
            metrics_str = centralized_match.group(1)
            
            # Extract each metric type
            metric_patterns = {
                'rmse': r"'rmse': \[(.*?)\]",
                'mae': r"'mae': \[(.*?)\]",
                'r2': r"'r2': \[(.*?)\]",
                'mape': r"'mape': \[(.*?)\]",
                'accuracy_10': r"'accuracy_10': \[(.*?)\]",
                'accuracy_20': r"'accuracy_20': \[(.*?)\]"
            }
            
            for metric_name, pattern in metric_patterns.items():
                match = re.search(pattern, metrics_str, re.DOTALL)
                if match:
                    # Parse tuples: (round, value)
                    tuples_str = match.group(1)
                    tuples = re.findall(r'\((\d+),\s*([-\d.]+)\)', tuples_str)
                    
                    rounds = [int(r) for r, _ in tuples]
                    values = [float(v) for _, v in tuples]
                    
                    metrics[metric_name] = {
                        'rounds': rounds,
                        'values': values
                    }
        
        # Extract distributed losses
        dist_loss_match = re.search(r"app_fit: losses_distributed \[(.*?)\]", run_content)
        if dist_loss_match:
            tuples_str = dist_loss_match.group(1)
            tuples = re.findall(r'\((\d+),\s*([-\d.]+)\)', tuples_str)
            
            rounds = [int(r) for r, _ in tuples]
            values = [float(v) for _, v in tuples]
            
            metrics['distributed_loss'] = {
                'rounds': rounds,
                'values': values
            }
        
        # Extract centralized losses
        cent_loss_match = re.search(r"app_fit: losses_centralized \[(.*?)\]", run_content)
        if cent_loss_match:
            tuples_str = cent_loss_match.group(1)
            tuples = re.findall(r'\((\d+),\s*([-\d.]+)\)', tuples_str)
            
            rounds = [int(r) for r, _ in tuples]
            values = [float(v) for _, v in tuples]
            
            metrics['centralized_loss'] = {
                'rounds': rounds,
                'values': values
            }
        
        all_metrics[f'run_{run_idx}'] = metrics
    
    return all_metrics

def load_comparison_results(output_dir):
    """Load comparison results JSON if available"""
    json_file = output_dir / "comparison_results.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    return None

def plot_metrics_progression(metrics, comparison_data, output_dir, strategy_name="Hybrid"):
    """Create comprehensive plots of metrics progression"""
    
    # Setup style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Loss Comparison (Distributed vs Centralized)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'distributed_loss' in metrics:
        ax1.plot(metrics['distributed_loss']['rounds'], 
                metrics['distributed_loss']['values'],
                'o-', color=colors[0], linewidth=2, markersize=5,
                label='Distributed (Client Training)', alpha=0.8)
    if 'centralized_loss' in metrics:
        ax1.plot(metrics['centralized_loss']['rounds'], 
                metrics['centralized_loss']['values'],
                's-', color=colors[1], linewidth=2, markersize=5,
                label='Centralized (Server Evaluation)', alpha=0.8)
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{strategy_name} - Training Loss Progression', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. MAPE Progression
    ax2 = fig.add_subplot(gs[0, 2])
    if 'mape' in metrics:
        rounds = metrics['mape']['rounds']
        values = metrics['mape']['values']
        ax2.plot(rounds, values, 'o-', color=colors[2], linewidth=2, markersize=4)
        ax2.axhline(y=values[-1], color='red', linestyle='--', alpha=0.5, 
                   label=f'Final: {values[-1]:.2f}%')
        ax2.fill_between(rounds, values, alpha=0.2, color=colors[2])
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax2.set_title('MAPE Progression', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. R² Score Progression
    ax3 = fig.add_subplot(gs[1, 0])
    if 'r2' in metrics:
        rounds = metrics['r2']['rounds']
        values = metrics['r2']['values']
        ax3.plot(rounds, values, 'o-', color=colors[3], linewidth=2, markersize=4)
        ax3.axhline(y=values[-1], color='red', linestyle='--', alpha=0.5,
                   label=f'Final: {values[-1]:.4f}')
        ax3.fill_between(rounds, values, alpha=0.2, color=colors[3])
    ax3.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('R² Score Progression', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. RMSE Progression
    ax4 = fig.add_subplot(gs[1, 1])
    if 'rmse' in metrics:
        rounds = metrics['rmse']['rounds']
        values = metrics['rmse']['values']
        ax4.plot(rounds, values, 'o-', color=colors[4], linewidth=2, markersize=4)
        ax4.axhline(y=values[-1], color='red', linestyle='--', alpha=0.5,
                   label=f'Final: ${values[-1]:,.0f}')
        ax4.fill_between(rounds, values, alpha=0.2, color=colors[4])
    ax4.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax4.set_ylabel('RMSE ($)', fontsize=11, fontweight='bold')
    ax4.set_title('RMSE Progression', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. MAE Progression
    ax5 = fig.add_subplot(gs[1, 2])
    if 'mae' in metrics:
        rounds = metrics['mae']['rounds']
        values = metrics['mae']['values']
        ax5.plot(rounds, values, 'o-', color=colors[0], linewidth=2, markersize=4)
        ax5.axhline(y=values[-1], color='red', linestyle='--', alpha=0.5,
                   label=f'Final: ${values[-1]:,.0f}')
        ax5.fill_between(rounds, values, alpha=0.2, color=colors[0])
    ax5.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax5.set_ylabel('MAE ($)', fontsize=11, fontweight='bold')
    ax5.set_title('MAE Progression', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Accuracy Metrics
    ax6 = fig.add_subplot(gs[2, 0])
    if 'accuracy_10' in metrics and 'accuracy_20' in metrics:
        rounds = metrics['accuracy_10']['rounds']
        acc10 = metrics['accuracy_10']['values']
        acc20 = metrics['accuracy_20']['values']
        ax6.plot(rounds, acc10, 'o-', color=colors[1], linewidth=2, 
                markersize=4, label='±10% Accuracy', alpha=0.8)
        ax6.plot(rounds, acc20, 's-', color=colors[2], linewidth=2, 
                markersize=4, label='±20% Accuracy', alpha=0.8)
        ax6.fill_between(rounds, acc10, alpha=0.15, color=colors[1])
        ax6.fill_between(rounds, acc20, alpha=0.15, color=colors[2])
    ax6.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 100])
    
    # 7. Metrics Summary Table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    
    # Create summary data
    summary_data = []
    summary_data.append(['Metric', 'Initial (Round 1)', 'Final (Round 30)', 'Change', '% Change'])
    
    for metric_name, display_name, fmt, is_better_lower in [
        ('mape', 'MAPE', '.2f%', True),
        ('r2', 'R² Score', '.4f', False),
        ('rmse', 'RMSE', ',.0f$', True),
        ('mae', 'MAE', ',.0f$', True),
        ('accuracy_10', 'Acc@10%', '.2f%', False),
    ]:
        if metric_name in metrics:
            initial = metrics[metric_name]['values'][1]  # Round 1 (skip round 0)
            final = metrics[metric_name]['values'][-1]
            change = final - initial
            pct_change = (change / initial * 100) if initial != 0 else 0
            
            # Format values
            if '$' in fmt:
                initial_str = f"${initial:,.0f}"
                final_str = f"${final:,.0f}"
                change_str = f"${abs(change):,.0f}"
            elif '%' in fmt:
                initial_str = f"{initial:.2f}%"
                final_str = f"{final:.2f}%"
                change_str = f"{abs(change):.2f}pp"
            else:
                initial_str = f"{initial:.4f}"
                final_str = f"{final:.4f}"
                change_str = f"{abs(change):.4f}"
            
            # Determine if improvement
            if is_better_lower:
                improved = change < 0
                arrow = '↓' if improved else '↑'
            else:
                improved = change > 0
                arrow = '↑' if improved else '↓'
            
            color_code = '✓' if improved else '✗'
            
            summary_data.append([
                display_name,
                initial_str,
                final_str,
                f"{arrow} {change_str}",
                f"{color_code} {abs(pct_change):.1f}%"
            ])
    
    # Create table
    table = ax7.table(cellText=summary_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
    
    # Add comparison with other strategies if available
    if comparison_data:
        comparison_text = "\n📊 Strategy Comparison (Final Metrics):\n"
        for strat_name, strat_data in comparison_data.items():
            if 'final_metrics' in strat_data:
                fm = strat_data['final_metrics']
                comparison_text += f"\n{strat_name.upper():12s}: MAPE={fm['mape']:5.2f}%  R²={fm['r2']:.4f}  RMSE=${fm['rmse']:,.0f}"
        
        fig.text(0.02, 0.02, comparison_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Federated Learning Metrics Progression - {strategy_name} Strategy', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_file = output_dir / f"metrics_progression_{strategy_name.lower()}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics progression plot: {output_file}")
    
    plt.show()

def main():
    """Main function"""
    print("=" * 70)
    print("Federated Learning Metrics Visualization")
    print("=" * 70)
    
    # Find latest output directory
    output_dir = find_latest_output()
    
    if not output_dir:
        print("❌ No output directory found!")
        return
    
    print(f"\n📁 Using output directory: {output_dir}")
    
    # Check for log file
    log_file = output_dir / "main.log"
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📄 Parsing log file: {log_file}")
    
    # Parse metrics (now returns dict of runs)
    all_metrics = parse_log_metrics(log_file)
    
    if not all_metrics:
        print("❌ No metrics found in log file!")
        return
    
    print(f"✓ Found {len(all_metrics)} strategy run(s)")
    
    # Load comparison data to match runs with strategy names
    comparison_data = load_comparison_results(output_dir)
    
    if comparison_data:
        print(f"✓ Loaded comparison data for {len(comparison_data)} strategies")
        strategy_names = list(comparison_data.keys())
    else:
        strategy_names = [f"Strategy {i}" for i in range(1, len(all_metrics) + 1)]
    
    # Create plots for each run
    for run_idx, (run_key, metrics) in enumerate(all_metrics.items(), 0):
        strategy_name = strategy_names[run_idx].capitalize() if run_idx < len(strategy_names) else f"Strategy {run_idx+1}"
        
        print(f"\n📊 Creating visualization for {strategy_name}...")
        
        for metric_name, data in metrics.items():
            if 'values' in data:
                print(f"  - {metric_name}: {len(data['values'])} data points")
        
        plot_metrics_progression(metrics, comparison_data, output_dir, strategy_name)
    
    print("\n" + "=" * 70)
    print("✓ All visualizations complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
