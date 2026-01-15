"""
Real-time Visualization Module for Federated Learning.

Provides live plotting of performance metrics during training and
comparison visualizations across different strategies.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class MetricsPlotter:
    """
    Real-time metrics plotter for federated learning experiments.
    
    Features:
    - Live updating plots during training
    - Multi-strategy comparison
    - Automatic saving to Hydra output directory
    """
    
    def __init__(
        self,
        output_dir: str = ".",
        figsize: Tuple[int, int] = (14, 10),
        show_plot: bool = True,
        save_plot: bool = True,
        plot_format: str = "png",
        update_interval: int = 1
    ):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size (width, height) in inches
            show_plot: Whether to display plot window
            save_plot: Whether to save plot to file
            plot_format: Output format (png, pdf, svg)
            update_interval: Update plot every N rounds
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.plot_format = plot_format
        self.update_interval = update_interval
        
        # Use non-interactive backend if not showing plot
        if not show_plot:
            matplotlib.use('Agg')
        
        # History storage for each strategy
        self.histories: Dict[str, Dict[str, List]] = {}
        
        # Figure and axes
        self.fig = None
        self.axes = None
        self._initialized = False
        
        # Style settings
        self.colors = {
            'fedavg': '#1f77b4',      # Blue
            'fedprox': '#ff7f0e',     # Orange
            'scaffold': '#2ca02c',    # Green
            'hybrid': '#d62728'       # Red
        }
        self.markers = {
            'fedavg': 'o',
            'fedprox': 's',
            'scaffold': '^',
            'hybrid': 'D'
        }
    
    def _init_plot(self):
        """Initialize the figure and axes."""
        if self._initialized:
            return
        
        # Create figure with 2x3 subplots for 6 metrics
        self.fig, self.axes = plt.subplots(2, 3, figsize=self.figsize)
        self.fig.suptitle('Federated Learning - Real-time Metrics', fontsize=14, fontweight='bold')
        
        # Configure each subplot
        titles = ['Training Loss', 'RMSE ($)', 'MAE ($)', 'R² Score', 'MAPE (%)', 'Accuracy within Threshold']
        ylabels = ['Loss', 'RMSE ($)', 'MAE ($)', 'R²', 'MAPE (%)', 'Accuracy (%)']
        
        for ax, title, ylabel in zip(self.axes.flat, titles, ylabels):
            ax.set_xlabel('Round')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        plt.tight_layout()
        self._initialized = True
        
        if self.show_plot:
            plt.ion()  # Interactive mode
            plt.show()
    
    def update(
        self,
        strategy: str,
        round_num: int,
        metrics: Dict[str, float]
    ):
        """Update the plot with new metrics for a strategy.
        
        Args:
            strategy: Strategy name
            round_num: Current round number
            metrics: Dictionary with loss, rmse, mae, r2, mape, accuracy_10, accuracy_20
        """
        # Initialize history for this strategy if needed
        if strategy not in self.histories:
            self.histories[strategy] = {
                'rounds': [],
                'loss': [],
                'rmse': [],
                'mae': [],
                'r2': [],
                'mape': [],
                'accuracy_10': [],
                'accuracy_20': []
            }
        
        # Add new data
        hist = self.histories[strategy]
        hist['rounds'].append(round_num)
        hist['loss'].append(metrics.get('loss', 0))
        hist['rmse'].append(metrics.get('rmse', 0))
        hist['mae'].append(metrics.get('mae', 0))
        hist['r2'].append(metrics.get('r2', 0))
        hist['mape'].append(metrics.get('mape', 0))
        hist['accuracy_10'].append(metrics.get('accuracy_10', 0))
        hist['accuracy_20'].append(metrics.get('accuracy_20', 0))
        
        # Only update plot at specified intervals
        if round_num % self.update_interval != 0:
            return
        
        # Initialize plot if needed
        self._init_plot()
        
        # Clear and redraw all axes
        metric_keys = ['loss', 'rmse', 'mae', 'r2', 'mape']
        
        # Plot first 5 metrics normally
        for ax, key in zip(self.axes.flat[:5], metric_keys):
            ax.clear()
            
            # Plot each strategy
            for strat, h in self.histories.items():
                if h['rounds']:
                    color = self.colors.get(strat, 'gray')
                    marker = self.markers.get(strat, 'o')
                    ax.plot(
                        h['rounds'], h[key],
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=1.5,
                        label=strat.upper(),
                        alpha=0.8
                    )
            
            # Set labels
            titles = {'loss': 'Training Loss', 'rmse': 'RMSE ($)', 'mae': 'MAE ($)', 'r2': 'R² Score', 'mape': 'MAPE (%)'}
            ylabels = {'loss': 'Loss', 'rmse': 'RMSE ($)', 'mae': 'MAE ($)', 'r2': 'R²', 'mape': 'MAPE (%)'}
            
            ax.set_xlabel('Round')
            ax.set_ylabel(ylabels[key])
            ax.set_title(titles[key])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        # Handle accuracy subplot (6th plot) - plot both thresholds
        ax_acc = self.axes.flat[5]
        ax_acc.clear()
        
        for strat, h in self.histories.items():
            if h['rounds']:
                color = self.colors.get(strat, 'gray')
                marker = self.markers.get(strat, 'o')
                # Plot 10% threshold accuracy
                ax_acc.plot(
                    h['rounds'], h['accuracy_10'],
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.5,
                    label=f"{strat.upper()} @10%",
                    alpha=0.8,
                    linestyle='-'
                )
                # Plot 20% threshold accuracy
                ax_acc.plot(
                    h['rounds'], h['accuracy_20'],
                    color=color,
                    marker=marker,
                    markersize=3,
                    linewidth=1.5,
                    label=f"{strat.upper()} @20%",
                    alpha=0.6,
                    linestyle='--'
                )
        
        ax_acc.set_xlabel('Round')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title('Accuracy within Threshold')
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc='best', fontsize=8)
        
        # Update figure
        self.fig.suptitle(
            f'Federated Learning - Round {round_num}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        if self.show_plot:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
    
    def save(self, filename: Optional[str] = None):
        """
        Save the current plot to file.
        
        Args:
            filename: Optional custom filename (without extension)
        """
        if not self.save_plot or self.fig is None:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_plot_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.{self.plot_format}")
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved to: {filepath}")
    
    def close(self):
        """Close the plot figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
            self._initialized = False


class ComparisonPlotter:
    """
    Create comparison plots for multiple federated learning strategies.
    """
    
    def __init__(
        self,
        output_dir: str = ".",
        figsize: Tuple[int, int] = (16, 12),
        plot_format: str = "png"
    ):
        """
        Initialize the comparison plotter.
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size (width, height)
            plot_format: Output format
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.plot_format = plot_format
        
        # Colors and markers for consistency
        self.colors = {
            'fedavg': '#1f77b4',
            'fedprox': '#ff7f0e',
            'scaffold': '#2ca02c',
            'hybrid': '#d62728'
        }
        self.markers = {
            'fedavg': 'o',
            'fedprox': 's',
            'scaffold': '^',
            'hybrid': 'D'
        }
    
    def plot_training_history(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history comparison for all strategies.
        
        Args:
            results: Dictionary with results for each strategy
            save_path: Optional path to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Strategy Comparison - Training History', fontsize=16, fontweight='bold')
        
        metric_configs = [
            ('loss', 'Training Loss', 'Loss'),
            ('rmse', 'RMSE Over Rounds', 'RMSE ($)'),
            ('mae', 'MAE Over Rounds', 'MAE ($)'),
            ('r2', 'R² Score Over Rounds', 'R²')
        ]
        
        for ax, (key, title, ylabel) in zip(axes.flat, metric_configs):
            for strategy, result in results.items():
                history = result.get('history', {})
                rounds = history.get('rounds', [])
                values = history.get(key, [])
                
                if rounds and values:
                    color = self.colors.get(strategy, 'gray')
                    marker = self.markers.get(strategy, 'o')
                    ax.plot(
                        rounds, values,
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=2,
                        label=strategy.upper(),
                        alpha=0.8
                    )
            
            ax.set_xlabel('Round', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Plot] Training history saved to: {save_path}")
        
        return fig
    
    def plot_final_comparison(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bar chart comparison of final metrics.
        
        Args:
            results: Dictionary with results for each strategy
            save_path: Optional path to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle('Strategy Comparison - Final Metrics', fontsize=16, fontweight='bold')
        
        strategies = list(results.keys())
        x = np.arange(len(strategies))
        width = 0.6
        
        metric_configs = [
            ('loss', 'Final Loss', 'Loss', False),
            ('rmse', 'Final RMSE', 'RMSE ($)', False),
            ('mae', 'Final MAE', 'MAE ($)', False),
            ('r2', 'Final R² Score', 'R²', True)  # Higher is better
        ]
        
        for ax, (key, title, ylabel, higher_better) in zip(axes, metric_configs):
            values = []
            colors = []
            
            for strategy in strategies:
                final_metrics = results[strategy].get('final_metrics', {})
                values.append(final_metrics.get(key, 0))
                colors.append(self.colors.get(strategy, 'gray'))
            
            bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black')
            
            # Highlight best performer
            if higher_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            
            ax.set_xlabel('Strategy', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([s.upper() for s in strategies], rotation=0)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f'{val:.4f}' if key == 'r2' else f'{val:,.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9
                )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Plot] Final comparison saved to: {save_path}")
        
        return fig
    
    def plot_convergence_analysis(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence analysis showing rate and stability.
        
        Args:
            results: Dictionary with results for each strategy
            save_path: Optional path to save the plot
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss convergence (log scale)
        ax1 = axes[0]
        for strategy, result in results.items():
            history = result.get('history', {})
            rounds = history.get('rounds', [])
            loss = history.get('loss', [])
            
            if rounds and loss:
                color = self.colors.get(strategy, 'gray')
                ax1.semilogy(rounds, loss, color=color, linewidth=2, label=strategy.upper())
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('Loss Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: R² improvement rate
        ax2 = axes[1]
        for strategy, result in results.items():
            history = result.get('history', {})
            rounds = history.get('rounds', [])
            r2 = history.get('r2', [])
            
            if len(r2) > 1:
                # Calculate improvement rate (diff between consecutive rounds)
                r2_diff = np.diff(r2)
                color = self.colors.get(strategy, 'gray')
                ax2.plot(rounds[1:], r2_diff, color=color, linewidth=2, label=strategy.upper(), alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('R² Improvement per Round')
        ax2.set_title('R² Improvement Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Final metrics radar chart (simplified as bar)
        ax3 = axes[2]
        
        # Normalize metrics for comparison
        metrics_to_compare = ['loss', 'rmse', 'mae']
        strategies = list(results.keys())
        
        # Get min and max for normalization
        all_values = {m: [] for m in metrics_to_compare}
        for result in results.values():
            for m in metrics_to_compare:
                all_values[m].append(result.get('final_metrics', {}).get(m, 0))
        
        # Calculate normalized scores (lower is better, so invert)
        scores = {}
        for strategy, result in results.items():
            final = result.get('final_metrics', {})
            score = 0
            for m in metrics_to_compare:
                vals = all_values[m]
                if max(vals) > min(vals):
                    # Normalize and invert (higher normalized = better)
                    normalized = 1 - (final.get(m, 0) - min(vals)) / (max(vals) - min(vals))
                    score += normalized
            # Add R² directly (higher is better)
            score += final.get('r2', 0)
            scores[strategy] = score / 4  # Average
        
        colors = [self.colors.get(s, 'gray') for s in strategies]
        bars = ax3.bar(strategies, [scores[s] for s in strategies], color=colors, alpha=0.8, edgecolor='black')
        
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Composite Score')
        ax3.set_title('Overall Performance Score')
        ax3.set_xticklabels([s.upper() for s in strategies])
        ax3.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Plot] Convergence analysis saved to: {save_path}")
        
        return fig
    
    def generate_all_plots(
        self,
        results: Dict[str, Dict],
        prefix: str = ""
    ):
        """
        Generate all comparison plots and save them.
        
        Args:
            results: Dictionary with results for each strategy
            prefix: Optional filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training history
        self.plot_training_history(
            results,
            os.path.join(self.output_dir, f"{prefix}training_history_{timestamp}.{self.plot_format}")
        )
        
        # Final comparison
        self.plot_final_comparison(
            results,
            os.path.join(self.output_dir, f"{prefix}final_comparison_{timestamp}.{self.plot_format}")
        )
        
        # Convergence analysis
        self.plot_convergence_analysis(
            results,
            os.path.join(self.output_dir, f"{prefix}convergence_analysis_{timestamp}.{self.plot_format}")
        )
        
        plt.close('all')


def save_metrics_to_csv(
    results: Dict[str, Dict],
    output_dir: str,
    filename: str = "metrics_history.csv"
):
    """
    Save metrics history to CSV file for later analysis.
    
    Args:
        results: Dictionary with results for each strategy
        output_dir: Directory to save the CSV
        filename: CSV filename
    """
    import csv
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['strategy', 'round', 'loss', 'rmse', 'mae', 'r2'])
        
        # Data
        for strategy, result in results.items():
            history = result.get('history', {})
            rounds = history.get('rounds', [])
            
            for i, round_num in enumerate(rounds):
                writer.writerow([
                    strategy,
                    round_num,
                    history.get('loss', [])[i] if i < len(history.get('loss', [])) else '',
                    history.get('rmse', [])[i] if i < len(history.get('rmse', [])) else '',
                    history.get('mae', [])[i] if i < len(history.get('mae', [])) else '',
                    history.get('r2', [])[i] if i < len(history.get('r2', [])) else ''
                ])
    
    print(f"[Data] Metrics saved to: {filepath}")


if __name__ == "__main__":
    # Demo/test
    print("Visualization module loaded successfully.")
    print("Available classes: MetricsPlotter, ComparisonPlotter")
    print("Available functions: save_metrics_to_csv")
