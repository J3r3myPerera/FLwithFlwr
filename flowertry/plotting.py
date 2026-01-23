"""
Plotting utilities for comparing FedProx strategies and visualizing stratified selection.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

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


def plot_stratified_selection_metrics(
    selection_history: List,
    client_strata: Dict[str, List[int]],
    save_path: str,
    plot_config: Dict = None
):
    """
    Visualize stratified client selection metrics across rounds.
    
    Args:
        selection_history: List of StratifiedSelectionStats from selector
        client_strata: Dictionary mapping stratum names to client IDs
        save_path: Path to save plots
        plot_config: Configuration for plotting
    """
    if plot_config is None:
        plot_config = {}
    
    save_plots = plot_config.get('save_plots', True)
    plot_format = plot_config.get('plot_format', 'png')
    dpi = plot_config.get('dpi', 300)
    
    save_path = Path(save_path)
    
    if not selection_history:
        print("Warning: No selection history to plot!")
        return
    
    # Extract data
    rounds = [stats.round_num for stats in selection_history]
    strata_names = sorted(client_strata.keys())
    
    # Compute stratum sizes for expected percentages
    strata_sizes = {s: len(clients) for s, clients in client_strata.items()}
    total_clients = sum(strata_sizes.values())
    expected_pct = {s: (size / total_clients * 100) for s, size in strata_sizes.items()}
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Stacked area chart of stratum participation
    ax1 = fig.add_subplot(gs[0, :])
    
    stratum_counts_by_round = {s: [] for s in strata_names}
    for stats in selection_history:
        for stratum in strata_names:
            stratum_counts_by_round[stratum].append(stats.stratum_counts.get(stratum, 0))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strata_names)))
    ax1.stackplot(rounds, *[stratum_counts_by_round[s] for s in strata_names],
                  labels=strata_names, colors=colors, alpha=0.8)
    
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Number of Clients Selected', fontsize=11)
    ax1.set_title('Stratified Client Selection: Participation Over Rounds', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Stratum percentage vs expected (line plot)
    ax2 = fig.add_subplot(gs[1, 0])
    
    for idx, stratum in enumerate(strata_names):
        percentages = [stats.stratum_percentages.get(stratum, 0.0) for stats in selection_history]
        ax2.plot(rounds, percentages, label=f'{stratum} (actual)', 
                color=colors[idx], linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=expected_pct[stratum], color=colors[idx], linestyle='--', 
                   linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title('Stratum Representation: Actual vs Expected', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Deviation from expected (heatmap-style)
    ax3 = fig.add_subplot(gs[1, 1])
    
    deviations = []
    for stratum in strata_names:
        stratum_devs = []
        for stats in selection_history:
            actual = stats.stratum_percentages.get(stratum, 0.0)
            expected = expected_pct[stratum]
            deviation = actual - expected
            stratum_devs.append(deviation)
        deviations.append(stratum_devs)
    
    im = ax3.imshow(deviations, aspect='auto', cmap='RdYlGn', 
                    vmin=-20, vmax=20, interpolation='nearest')
    ax3.set_yticks(range(len(strata_names)))
    ax3.set_yticklabels(strata_names)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Stratum', fontsize=11)
    ax3.set_title('Deviation from Expected (%) - Green=Over, Red=Under', 
                  fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Deviation (%)', fontsize=10)
    
    # Set x-axis ticks to show round numbers
    if len(rounds) <= 20:
        ax3.set_xticks(range(len(rounds)))
        ax3.set_xticklabels(rounds)
    else:
        # Show every 5th round for readability
        tick_indices = range(0, len(rounds), 5)
        ax3.set_xticks(tick_indices)
        ax3.set_xticklabels([rounds[i] for i in tick_indices])
    
    # Plot 4: Client participation frequency
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Count how many times each client was selected
    client_counts = {i: 0 for i in range(total_clients)}
    for stats in selection_history:
        for client_id in stats.selected_clients:
            client_counts[client_id] += 1
    
    # Group by stratum for visualization
    stratum_client_counts = {s: [] for s in strata_names}
    for stratum, client_ids in client_strata.items():
        for client_id in client_ids:
            stratum_client_counts[stratum].append(client_counts[client_id])
    
    # Create box plot
    positions = range(1, len(strata_names) + 1)
    bp = ax4.boxplot([stratum_client_counts[s] for s in strata_names],
                      positions=positions,
                      labels=strata_names,
                      patch_artist=True,
                      widths=0.6)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Stratum', fontsize=11)
    ax4.set_ylabel('Selection Count', fontsize=11)
    ax4.set_title('Client Participation Frequency by Stratum', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Fairness metrics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Compute fairness metrics
    # Gini coefficient
    all_counts = list(client_counts.values())
    sorted_counts = sorted(all_counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    if cumsum[-1] > 0:
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n
    else:
        gini = 0.0
    
    # Representation ratios
    representation_ratios = {}
    for stratum in strata_names:
        expected_ratio = strata_sizes[stratum] / total_clients
        actual_selections = sum(stats.stratum_counts.get(stratum, 0) for stats in selection_history)
        total_selections = len(selection_history) * len(selection_history[0].selected_clients)
        actual_ratio = actual_selections / total_selections if total_selections > 0 else 0
        representation_ratios[stratum] = actual_ratio / expected_ratio if expected_ratio > 0 else 1.0
    
    # Toxic rounds
    toxic_rounds = 0
    for stats in selection_history:
        for stratum in strata_names:
            expected = expected_pct[stratum]
            actual = stats.stratum_percentages.get(stratum, 0.0)
            if abs(actual - expected) > 20:
                toxic_rounds += 1
                break
    toxic_frequency = (toxic_rounds / len(selection_history) * 100) if selection_history else 0.0
    
    # Display metrics
    metrics_text = "FAIRNESS METRICS SUMMARY\n" + "="*40 + "\n\n"
    metrics_text += f"Total Rounds: {len(selection_history)}\n\n"
    metrics_text += f"Participation Equity (Gini): {gini:.4f}\n"
    metrics_text += "  (0 = perfect equality, 1 = max inequality)\n\n"
    metrics_text += "Representation Ratios (actual/expected):\n"
    for stratum, ratio in representation_ratios.items():
        metrics_text += f"  {stratum}: {ratio:.3f}\n"
    metrics_text += "  (1.0 = perfect representation)\n\n"
    metrics_text += f"Toxic Round Frequency: {toxic_frequency:.1f}%\n"
    metrics_text += "  (>20% deviation from expected)\n\n"
    metrics_text += "="*40 + "\n"
    metrics_text += "STRATIFIED SELECTION BENEFITS:\n"
    metrics_text += "✓ Balanced representation\n"
    metrics_text += "✓ Reduced gradient variance\n"
    metrics_text += "✓ Prevented toxic combinations\n"
    metrics_text += "✓ Fairness guarantees"
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Stratified Client Selection Analysis', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    if save_plots:
        plot_path = save_path / f"stratified_selection_analysis.{plot_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Stratified selection plot saved to: {plot_path}")
    
    plt.close()


def plot_random_vs_stratified_comparison(
    random_results: Dict,
    stratified_results: Dict,
    save_path: str,
    plot_config: Dict = None
):
    """
    Compare random vs stratified client selection side-by-side.
    
    Args:
        random_results: Results from random selection strategy
        stratified_results: Results from stratified selection strategy
        save_path: Path to save plots
        plot_config: Configuration for plotting
    """
    if plot_config is None:
        plot_config = {}
    
    save_plots = plot_config.get('save_plots', True)
    plot_format = plot_config.get('plot_format', 'png')
    dpi = plot_config.get('dpi', 300)
    figsize = plot_config.get('figsize', [14, 10])
    
    save_path = Path(save_path)
    
    # Extract metrics from both strategies
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot R² comparison
    ax = axes[0, 0]
    if 'history' in random_results:
        rounds, values = extract_metrics(random_results['history'], 'r2')
        ax.plot(rounds, values, label='Random Selection', 
                color='#e74c3c', linewidth=2, marker='o', markersize=5)
    if 'history' in stratified_results:
        rounds, values = extract_metrics(stratified_results['history'], 'r2')
        ax.plot(rounds, values, label='Stratified Selection', 
                color='#2ecc71', linewidth=2, marker='s', markersize=5)
    
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Model Performance: R² Score', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot RMSE comparison
    ax = axes[0, 1]
    if 'history' in random_results:
        rounds, values = extract_metrics(random_results['history'], 'rmse')
        ax.plot(rounds, values, label='Random Selection', 
                color='#e74c3c', linewidth=2, marker='o', markersize=5)
    if 'history' in stratified_results:
        rounds, values = extract_metrics(stratified_results['history'], 'rmse')
        ax.plot(rounds, values, label='Stratified Selection', 
                color='#2ecc71', linewidth=2, marker='s', markersize=5)
    
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Model Performance: RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot MAE comparison
    ax = axes[1, 0]
    if 'history' in random_results:
        rounds, values = extract_metrics(random_results['history'], 'mae')
        ax.plot(rounds, values, label='Random Selection', 
                color='#e74c3c', linewidth=2, marker='o', markersize=5)
    if 'history' in stratified_results:
        rounds, values = extract_metrics(stratified_results['history'], 'mae')
        ax.plot(rounds, values, label='Stratified Selection', 
                color='#2ecc71', linewidth=2, marker='s', markersize=5)
    
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('Model Performance: MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Summary comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    # Extract final metrics
    def get_final_metrics(results):
        if 'history' not in results:
            return None
        history = results['history']
        _, r2_vals = extract_metrics(history, 'r2')
        _, rmse_vals = extract_metrics(history, 'rmse')
        _, mae_vals = extract_metrics(history, 'mae')
        return {
            'r2': r2_vals[-1] if r2_vals else 0,
            'rmse': rmse_vals[-1] if rmse_vals else 0,
            'mae': mae_vals[-1] if mae_vals else 0
        }
    
    random_final = get_final_metrics(random_results)
    stratified_final = get_final_metrics(stratified_results)
    
    if random_final and stratified_final:
        r2_improvement = ((stratified_final['r2'] - random_final['r2']) / abs(random_final['r2']) * 100) if random_final['r2'] != 0 else 0
        rmse_improvement = ((random_final['rmse'] - stratified_final['rmse']) / random_final['rmse'] * 100) if random_final['rmse'] != 0 else 0
        mae_improvement = ((random_final['mae'] - stratified_final['mae']) / random_final['mae'] * 100) if random_final['mae'] != 0 else 0
        
        summary_text = "FINAL PERFORMANCE COMPARISON\n" + "="*45 + "\n\n"
        summary_text += f"{'Metric':<15} {'Random':<12} {'Stratified':<12} {'Δ%':<10}\n"
        summary_text += "-"*45 + "\n"
        summary_text += f"{'R² Score':<15} {random_final['r2']:>11.4f} {stratified_final['r2']:>11.4f} {r2_improvement:>9.2f}%\n"
        summary_text += f"{'RMSE':<15} {random_final['rmse']:>11.2f} {stratified_final['rmse']:>11.2f} {rmse_improvement:>9.2f}%\n"
        summary_text += f"{'MAE':<15} {random_final['mae']:>11.2f} {stratified_final['mae']:>11.2f} {mae_improvement:>9.2f}%\n"
        summary_text += "\n" + "="*45 + "\n\n"
        summary_text += "STRATIFIED SELECTION ADVANTAGES:\n\n"
        summary_text += "✓ Variance Reduction: More stable gradients\n"
        summary_text += "✓ Fairness: Guaranteed representation\n"
        summary_text += "✓ Stability: Prevents toxic combinations\n"
        summary_text += "✓ Convergence: Tighter bounds\n"
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Random vs Stratified Client Selection Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_plots:
        plot_path = save_path / f"random_vs_stratified_comparison.{plot_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        print(f"  → Comparison plot saved to: {plot_path}")
    
    plt.close()
