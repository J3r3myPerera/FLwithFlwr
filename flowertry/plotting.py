"""
Plotting utilities for comparing FedProx strategies.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any

def extract_metrics(history, metric_name='r2'):
    """
    Extract metrics from Flower history object.
    
    Args:
        history: Flower history object
        metric_name: Name of metric to extract ('r2', 'rmse', 'mae')
    
    Returns:
        rounds: List of round numbers
        values: List of metric values
    """
    rounds = []
    values = []
    
    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        if metric_name in history.metrics_centralized:
            for round_num, value in history.metrics_centralized[metric_name]:
                rounds.append(round_num)
                values.append(value)
    
    return rounds, values

def plot_comparison(all_results: Dict[str, Any], save_path: str, plot_config: Dict = None):
    """
    Create comparison plots for different FedProx strategies.
    
    Args:
        all_results: Dictionary containing results for each strategy
        save_path: Path to save plots
        plot_config: Configuration for plotting (figsize, dpi, etc.)
    """
    if plot_config is None:
        plot_config = {}
    
    save_plots = plot_config.get('save_plots', True)
    plot_format = plot_config.get('plot_format', 'png')
    figsize = plot_config.get('figsize', [12, 8])
    dpi = plot_config.get('dpi', 300)
    
    save_path = Path(save_path)
    
    # Extract data for each strategy
    strategies_data = {}
    for key, result in all_results.items():
        if 'history' in result:
            history = result['history']
            name = result.get('name', key)
            
            strategies_data[name] = {
                'r2_rounds': [],
                'r2_values': [],
                'rmse_rounds': [],
                'rmse_values': [],
                'mae_rounds': [],
                'mae_values': [],
                'loss_rounds': [],
                'loss_values': []
            }
            
            # Extract R²
            rounds, values = extract_metrics(history, 'r2')
            strategies_data[name]['r2_rounds'] = rounds
            strategies_data[name]['r2_values'] = values
            
            # Extract RMSE
            rounds, values = extract_metrics(history, 'rmse')
            strategies_data[name]['rmse_rounds'] = rounds
            strategies_data[name]['rmse_values'] = values
            
            # Extract MAE
            rounds, values = extract_metrics(history, 'mae')
            strategies_data[name]['mae_rounds'] = rounds
            strategies_data[name]['mae_values'] = values
            
            # Extract loss
            if hasattr(history, 'losses_centralized') and history.losses_centralized:
                for round_num, loss in history.losses_centralized:
                    strategies_data[name]['loss_rounds'].append(round_num)
                    strategies_data[name]['loss_values'].append(loss)
    
    if not strategies_data:
        print("Warning: No data to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    # Determine title based on number of strategies
    if len(strategies_data) == 3:
        title = 'FedProx Strategy Comparison: Base vs Static vs Adaptive Multi-Layer'
    else:
        title = 'FedProx Strategy Comparison: Base vs Multi-Layer'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Define colors and styles
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    linestyles = ['-', '--', '-.', ':']
    
    # Plot 1: R² Score
    ax1 = axes[0, 0]
    for idx, (name, data) in enumerate(strategies_data.items()):
        if data['r2_rounds']:
            ax1.plot(data['r2_rounds'], data['r2_values'], 
                    label=name, color=colors[idx % len(colors)], 
                    linestyle=linestyles[idx % len(linestyles)], linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² Score Over Rounds', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 2: RMSE
    ax2 = axes[0, 1]
    for idx, (name, data) in enumerate(strategies_data.items()):
        if data['rmse_rounds']:
            ax2.plot(data['rmse_rounds'], data['rmse_values'], 
                    label=name, color=colors[idx % len(colors)], 
                    linestyle=linestyles[idx % len(linestyles)], linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MAE
    ax3 = axes[1, 0]
    for idx, (name, data) in enumerate(strategies_data.items()):
        if data['mae_rounds']:
            ax3.plot(data['mae_rounds'], data['mae_values'], 
                    label=name, color=colors[idx % len(colors)], 
                    linestyle=linestyles[idx % len(linestyles)], linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('MAE', fontsize=11)
    ax3.set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss (use log scale for better comparison)
    ax4 = axes[1, 1]
    
    # First, collect all loss values to find global max for proper comparison
    all_loss_values = []
    for data in strategies_data.values():
        if data['loss_rounds']:
            all_loss_values.extend(data['loss_values'])
    
    global_max_loss = max(all_loss_values) if all_loss_values else 1.0
    
    for idx, (name, data) in enumerate(strategies_data.items()):
        if data['loss_rounds']:
            loss_values = np.array(data['loss_values'])
            # Use log scale for better visualization when losses vary widely
            # Also show raw values but use log scale on y-axis
            ax4.plot(data['loss_rounds'], loss_values, 
                    label=name, color=colors[idx % len(colors)], 
                    linestyle=linestyles[idx % len(linestyles)], linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('MSE Loss', fontsize=11)
    ax4.set_title('Training Loss (MSE)', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')  # Use log scale for better comparison
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        plot_path = save_path / f"comparison_plot.{plot_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Comparison plot saved to: {plot_path}")
    
    # Create summary statistics plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate final metrics
    final_metrics = {}
    for name, data in strategies_data.items():
        if data['r2_rounds']:
            final_metrics[name] = {
                'r2': data['r2_values'][-1] if data['r2_values'] else 0,
                'rmse': data['rmse_values'][-1] if data['rmse_values'] else 0,
                'mae': data['mae_values'][-1] if data['mae_values'] else 0
            }
    
    if final_metrics:
        names = list(final_metrics.keys())
        r2_values = [final_metrics[n]['r2'] for n in names]
        rmse_values = [final_metrics[n]['rmse'] for n in names]
        mae_values = [final_metrics[n]['mae'] for n in names]
        
        x = np.arange(len(names))
        width = 0.25
        
        # Normalize RMSE and MAE for visualization (divide by max)
        rmse_max = max(rmse_values) if rmse_values else 1
        mae_max = max(mae_values) if mae_values else 1
        rmse_norm = [v / rmse_max for v in rmse_values] if rmse_max > 0 else rmse_values
        mae_norm = [v / mae_max for v in mae_values] if mae_max > 0 else mae_values
        
        ax.bar(x - width, r2_values, width, label='R² Score', color='#3498db', alpha=0.8)
        ax.bar(x, rmse_norm, width, label='RMSE (normalized)', color='#e74c3c', alpha=0.8)
        ax.bar(x + width, mae_norm, width, label='MAE (normalized)', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (r2, rmse, mae) in enumerate(zip(r2_values, rmse_norm, mae_norm)):
            ax.text(i - width, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, rmse + 0.01, f'{rmse:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, mae + 0.01, f'{mae:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_plots:
            summary_path = save_path / f"summary_comparison.{plot_format}"
            plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
            print(f"  → Summary plot saved to: {summary_path}")
    
    plt.close('all')


def plot_adaptive_mu_evolution(mu_summary: Dict, save_path: str, plot_config: Dict = None):
    """
    Plot how adaptive mu values change over rounds.

    Args:
        mu_summary: Dictionary from AdaptiveMuController.get_mu_history_summary()
        save_path: Path to save plot
        plot_config: Configuration for plotting
    """
    if plot_config is None:
        plot_config = {}

    save_plots = plot_config.get('save_plots', True)
    plot_format = plot_config.get('plot_format', 'png')
    dpi = plot_config.get('dpi', 300)

    save_path = Path(save_path)

    history = mu_summary.get('history', [])
    if not history:
        print("Warning: No mu history to plot!")
        return

    # Extract data
    rounds = [h['round'] for h in history]
    schedule_factors = [h['schedule_factor'] for h in history]

    layer_names = ['input', 'hidden1', 'hidden2', 'output']
    layer_mus = {layer: [h.get(layer, 0) for h in history] for layer in layer_names}

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Adaptive Mu Values per Layer
    ax1 = axes[0]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    for idx, layer in enumerate(layer_names):
        ax1.plot(rounds, layer_mus[layer], label=layer.capitalize(),
                 color=colors[idx], linewidth=2, marker='o', markersize=4)

    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Adaptive Mu Value', fontsize=11)
    ax1.set_title('Adaptive Per-Layer Mu Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Schedule Factor
    ax2 = axes[1]
    ax2.plot(rounds, schedule_factors, color='#9b59b6', linewidth=2, marker='s', markersize=4)
    ax2.fill_between(rounds, schedule_factors, alpha=0.3, color='#9b59b6')

    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Schedule Factor', fontsize=11)
    ax2.set_title('Round-Based Schedule Factor', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plot_path = save_path / f"adaptive_mu_evolution.{plot_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        print(f"  -> Adaptive mu plot saved to: {plot_path}")

    plt.close()

    # Also plot divergence history if available
    div_history = mu_summary.get('divergence_history', [])
    if div_history:
        fig2, ax = plt.subplots(figsize=(10, 6))

        div_rounds = list(range(1, len(div_history) + 1))
        for idx, layer in enumerate(layer_names):
            layer_divs = [d.get(layer, 1.0) for d in div_history]
            ax.plot(div_rounds, layer_divs, label=layer.capitalize(),
                    color=colors[idx], linewidth=2, marker='^', markersize=4)

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0)')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Divergence Factor', fontsize=11)
        ax.set_title('Layer Divergence Factors Over Rounds', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            div_path = save_path / f"divergence_evolution.{plot_format}"
            plt.savefig(div_path, dpi=dpi, bbox_inches='tight')
            print(f"  -> Divergence plot saved to: {div_path}")

        plt.close()
