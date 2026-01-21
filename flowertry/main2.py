"""
Main Entry Point for Federated Learning with Dirichlet Non-IID Partitioning.

This module uses dataset2.py for TRUE label-based Dirichlet partitioning,
which creates genuine data heterogeneity to better showcase the advantages
of SCAFFOLD and Hybrid strategies.

Key Differences from main.py:
1. Uses dataset2.py (Dirichlet partitioning) instead of dataset.py
2. Flexible client count (not fixed to 3, 4, or 12)
3. Alpha parameter controls heterogeneity level
4. Better visualization of partition heterogeneity

Usage:
    python main2.py --config-name dirichlet
    python main2.py --config-name dirichlet dirichlet.alpha=0.1  # Extreme non-IID
    python main2.py --config-name dirichlet num_clients=15
    python main2.py --config-name dirichlet compare_all=true

Author: FL Research Team
Date: January 2026
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters
import hydra
from omegaconf import DictConfig, OmegaConf

# Use the new Dirichlet-based dataset module
from dataset2 import (
    prepare_dirichlet_federated, 
    reset_preprocessor,
    visualize_partition
)
from model import DisposableIncomeNet, get_parameters, set_parameters
from client import RegressionClient
from server import create_strategy, compute_regression_metrics
from visualize_metrics import ComparisonPlotter, save_metrics_to_csv


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_dirichlet_simulation(
    strategy_name: str,
    num_rounds: int = 30,
    num_clients: int = 10,
    alpha: float = 0.3,
    num_bins: int = 10,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    mu: float = 0.1,
    scaffold_lr_correction: float = 1.0,
    fedprox_weight: float = 1.0,
    scaffold_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    seed: int = 2023,
    verbose: bool = True,
    output_dir: str = ".",
    hybrid_config: Optional[Dict] = None,
    # Client sampling parameters
    fraction_fit: float = 0.6,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 4,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 4,
    # Visualization
    visualize_partition_plot: bool = True,
) -> Dict:
    """
    Run federated learning simulation with Dirichlet non-IID partitioning.
    
    Args:
        strategy_name: Strategy to use (fedavg, fedprox, scaffold, hybrid)
        num_rounds: Number of federated rounds
        num_clients: Number of clients (flexible, any number)
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        num_bins: Number of target bins for Dirichlet partitioning
        local_epochs: Local training epochs per round
        batch_size: Batch size
        learning_rate: Learning rate
        mu: FedProx proximal term weight
        scaffold_lr_correction: SCAFFOLD learning rate correction
        fedprox_weight: Weight for FedProx in hybrid
        scaffold_weight: Weight for SCAFFOLD in hybrid
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        verbose: Print progress
        output_dir: Directory for saving outputs
        hybrid_config: Optional hybrid-specific parameters
        fraction_fit: Fraction of clients for training
        fraction_evaluate: Fraction of clients for evaluation
        min_fit_clients: Minimum clients for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum available clients
        visualize_partition_plot: Whether to visualize and save partition plot
    
    Returns:
        Dictionary with training history and final metrics
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Federated Learning with DIRICHLET Non-IID - {strategy_name.upper()}")
        print(f"{'='*80}")
        print(f"  Alpha: {alpha} ({'extreme' if alpha <= 0.1 else 'very high' if alpha <= 0.3 else 'high' if alpha <= 0.5 else 'moderate' if alpha <= 1.0 else 'low'} non-IID)")
        print(f"  Clients: {num_clients}")
        print(f"  Bins: {num_bins}")
    
    # Prepare dataset with Dirichlet partitioning
    reset_preprocessor()
    (trainloaders, valloaders, testloader, input_dim, 
     target_mean, target_std, partition_info, log_transform) = prepare_dirichlet_federated(
        num_clients=num_clients,
        alpha=alpha,
        num_bins=num_bins,
        batch_size=batch_size,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        normalize_target=True,
        log_transform_target=True,
        min_samples_per_client=30,
        verbose=verbose
    )
    
    # Visualize partition if requested
    if visualize_partition_plot:
        partition_plot_path = os.path.join(output_dir, "dirichlet_partition.png")
        visualize_partition(partition_info, save_path=partition_plot_path)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if verbose:
        print(f"\n[Device] {device}")
        print(f"[Strategy] {strategy_name.upper()}")
        print(f"[Rounds] {num_rounds}")
        print(f"[Local Epochs] {local_epochs}")
        print(f"[Learning Rate] {learning_rate}")
        print(f"[Heterogeneity Score] {partition_info['heterogeneity']['heterogeneity_score']:.4f}")
        
        if strategy_name in ["fedprox", "hybrid"]:
            print(f"[Mu (FedProx)] {mu}")
        if strategy_name == "hybrid":
            print(f"[FedProx Weight] {fedprox_weight}")
            print(f"[SCAFFOLD Weight] {scaffold_weight}")
    
    # Create initial model and parameters
    initial_model = DisposableIncomeNet(input_dim=input_dim)
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
    
    # Config function for fit
    def fit_config(server_round: int) -> Dict:
        config = {
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "mu": mu,
            "scaffold_lr_correction": scaffold_lr_correction,
            "fedprox_weight": fedprox_weight,
            "scaffold_weight": scaffold_weight,
            "max_grad_norm": max_grad_norm,
            "server_round": server_round,
            "num_rounds": num_rounds
        }
        
        # Apply advanced parameters for hybrid strategy
        if strategy_name == "hybrid" and hybrid_config:
            config["dynamic_weights"] = hybrid_config.get("dynamic_weights", False)
            config["weight_increase_rate"] = hybrid_config.get("weight_increase_rate", 1.05)
            config["weight_max"] = hybrid_config.get("weight_max", 0.6)
            
            # Adaptive mu
            adaptive_mu = hybrid_config.get("adaptive_mu", False)
            if adaptive_mu and server_round > 1:
                mu_decay = hybrid_config.get("mu_decay", 0.98)
                mu_min = hybrid_config.get("mu_min", 0.001)
                config["mu"] = max(mu * (mu_decay ** (server_round - 1)), mu_min)
            
            # Warmup
            config["warmup_rounds"] = hybrid_config.get("warmup_rounds", 0)
            config["warmup_type"] = hybrid_config.get("warmup_type", "exponential")
            config["warmup_start_factor"] = hybrid_config.get("warmup_start_factor", 0.05)
            
            # Gradient clipping
            config["gradient_clip"] = hybrid_config.get("gradient_clip", True)
            config["gradient_clip_norm"] = hybrid_config.get("gradient_clip_norm", 1.0)
            
            # LR scheduling
            use_lr_scheduler = hybrid_config.get("use_lr_scheduler", False)
            if use_lr_scheduler:
                lr_warmup_rounds = hybrid_config.get("lr_warmup_rounds", 3)
                lr_decay_start = hybrid_config.get("lr_decay_start", 10)
                
                if server_round <= lr_warmup_rounds:
                    warmup_progress = server_round / lr_warmup_rounds
                    config["learning_rate"] = learning_rate * 0.1 + learning_rate * 0.9 * warmup_progress
                elif server_round >= lr_decay_start:
                    lr_decay = hybrid_config.get("lr_decay", 0.98)
                    lr_min = hybrid_config.get("lr_min", 0.0005)
                    decay_steps = server_round - lr_decay_start
                    config["learning_rate"] = max(learning_rate * (lr_decay ** decay_steps), lr_min)
        
        return config
    
    # Create strategy with smart client selection for hybrid
    strategy_kwargs = {
        "strategy_name": strategy_name,
        "testloader": testloader,
        "input_dim": input_dim,
        "target_mean": target_mean,
        "target_std": target_std,
        "num_clients": num_clients,
        "mu": mu,
        "scaffold_lr_correction": scaffold_lr_correction,
        "fedprox_weight": fedprox_weight,
        "scaffold_weight": scaffold_weight,
        "log_transform": log_transform,
        "initial_parameters": initial_parameters,
        "on_fit_config_fn": fit_config,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
    }
    
    # Add smart selection parameters only for hybrid
    if strategy_name == "hybrid" and hybrid_config:
        strategy_kwargs["use_smart_selection"] = hybrid_config.get("use_smart_selection", True)
        strategy_kwargs["selection_strategy"] = hybrid_config.get("selection_strategy", "power_of_choice")
    
    strategy = create_strategy(**strategy_kwargs)
    
    # Log configuration
    if verbose:
        print(f"\n[Client Sampling]")
        print(f"  Fraction Fit: {fraction_fit:.1%} ({int(num_clients * fraction_fit)} of {num_clients} clients)")
        print(f"  Min Fit Clients: {min_fit_clients}")
        
        if strategy_name == "hybrid" and hybrid_config:
            use_smart = hybrid_config.get("use_smart_selection", True)
            sel_strategy = hybrid_config.get("selection_strategy", "power_of_choice")
            if use_smart:
                print(f"\n[Intelligent Client Selection - ENABLED]")
                print(f"  Strategy: {sel_strategy}")
                if sel_strategy == "power_of_choice":
                    print(f"  Method: Sample {int(num_clients * fraction_fit * 2)} candidates, select top {int(num_clients * fraction_fit)}")
    
    # Client function
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return RegressionClient(
            cid=cid,
            trainloader=trainloaders[idx],
            valloader=valloaders[idx],
            input_dim=input_dim,
            target_mean=target_mean,
            target_std=target_std,
            log_transform=log_transform,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            strategy=strategy_name,
            mu=mu,
            scaffold_lr_correction=scaffold_lr_correction,
            fedprox_weight=fedprox_weight,
            scaffold_weight=scaffold_weight,
            device=device
        )
    
    # History tracking
    history = {
        "rounds": [],
        "loss": [],
        "rmse": [],
        "mae": [],
        "r2": [],
        "mape": [],
        "accuracy_10": [],
        "accuracy_20": []
    }
    
    # Run simulation
    if verbose:
        print(f"\n[Starting Training]")
        print(f"{'Round':>6} | {'Loss':>12} | {'RMSE':>12} | {'MAE':>12} | {'R²':>10}")
        print("-" * 70)
    
    result = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Extract history from result
    if hasattr(result, 'metrics_centralized') and result.metrics_centralized:
        metric_names = ['rmse', 'mae', 'r2', 'mape', 'accuracy_10', 'accuracy_20']
        
        for metric_name in metric_names:
            if metric_name in result.metrics_centralized:
                metric_list = result.metrics_centralized[metric_name]
                for round_num, value in metric_list:
                    if metric_name == 'rmse' and round_num not in history["rounds"]:
                        history["rounds"].append(round_num)
                        if hasattr(result, 'losses_centralized') and result.losses_centralized:
                            for loss_round, loss_value in result.losses_centralized:
                                if loss_round == round_num:
                                    history["loss"].append(float(loss_value))
                                    break
                            else:
                                history["loss"].append(0.0)
                        else:
                            history["loss"].append(0.0)
                    
                    if round_num in history["rounds"]:
                        idx = history["rounds"].index(round_num)
                        while len(history[metric_name]) <= idx:
                            history[metric_name].append(0.0)
                        history[metric_name][idx] = float(value)
        
        # Print progress summary
        if verbose and history["rounds"]:
            for i in range(0, len(history["rounds"]), max(1, len(history["rounds"]) // 5)):
                round_num = history["rounds"][i]
                loss = history["loss"][i] if i < len(history["loss"]) else 0
                rmse = history["rmse"][i] if i < len(history["rmse"]) else 0
                mae = history["mae"][i] if i < len(history["mae"]) else 0
                r2 = history["r2"][i] if i < len(history["r2"]) else 0
                print(f"{round_num:>6} | {loss:>12.6f} | {rmse:>12,.2f} | {mae:>12,.2f} | {r2:>10.4f}")
    
    # Final evaluation
    final_model = strategy.model
    final_metrics = compute_regression_metrics(
        final_model, testloader, target_std, target_mean, device, log_transform
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS - {strategy_name.upper()} (α={alpha})")
        print(f"{'='*80}")
        print(f"  RMSE:        ${final_metrics['rmse']:,.2f}")
        print(f"  MAE:         ${final_metrics['mae']:,.2f}")
        print(f"  R²:          {final_metrics['r2']:.4f}")
        print(f"  MAPE:        {final_metrics['mape']:.2f}%")
        print(f"  Accuracy±10%: {final_metrics['accuracy_10']:.2f}%")
        print(f"  Accuracy±20%: {final_metrics['accuracy_20']:.2f}%")
    
    return {
        "strategy": strategy_name,
        "alpha": alpha,
        "num_clients": num_clients,
        "heterogeneity_score": partition_info['heterogeneity']['heterogeneity_score'],
        "history": history,
        "final_metrics": final_metrics,
        "partition_info": partition_info
    }


def compare_strategies_dirichlet(
    strategies: List[str] = ["fedavg", "fedprox", "scaffold", "hybrid"],
    output_dir: str = ".",
    plotting_config: Optional[Dict] = None,
    strategy_configs: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    Compare multiple strategies with Dirichlet non-IID partitioning.
    
    The same heterogeneous partition is used across all strategies for
    fair comparison.
    """
    results = {}
    all_histories = {}
    
    print(f"\n{'#'*80}")
    print(f"# STRATEGY COMPARISON - DIRICHLET NON-IID")
    print(f"{'#'*80}")
    print(f"Strategies: {', '.join(s.upper() for s in strategies)}")
    print(f"Alpha: {kwargs.get('alpha', 0.3)} ({kwargs.get('num_clients', 10)} clients)")
    
    for strategy_name in strategies:
        # Get strategy-specific config
        strategy_cfg = {}
        if strategy_configs and strategy_name in strategy_configs:
            strategy_cfg = strategy_configs[strategy_name]
        
        # Merge with kwargs
        run_kwargs = {**kwargs}
        if 'lr' in strategy_cfg:
            run_kwargs['learning_rate'] = strategy_cfg['lr']
        if 'local_epochs' in strategy_cfg:
            run_kwargs['local_epochs'] = strategy_cfg['local_epochs']
        if 'max_grad_norm' in strategy_cfg:
            run_kwargs['max_grad_norm'] = strategy_cfg['max_grad_norm']
        
        # Run simulation
        result = run_dirichlet_simulation(
            strategy_name=strategy_name,
            output_dir=output_dir,
            **run_kwargs
        )
        
        results[strategy_name] = result
        all_histories[strategy_name] = result["history"]
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':>12} | {'RMSE':>12} | {'R²':>10} | {'MAPE':>10} | {'Acc±10%':>10}")
    print("-" * 70)
    
    for strategy_name in strategies:
        metrics = results[strategy_name]["final_metrics"]
        print(f"{strategy_name.upper():>12} | ${metrics['rmse']:>11,.2f} | "
              f"{metrics['r2']:>10.4f} | {metrics['mape']:>9.2f}% | "
              f"{metrics['accuracy_10']:>9.2f}%")
    
    # Identify best strategy
    best_rmse = min(results, key=lambda s: results[s]["final_metrics"]["rmse"])
    best_r2 = max(results, key=lambda s: results[s]["final_metrics"]["r2"])
    
    print(f"\n🏆 Best RMSE: {best_rmse.upper()} (${results[best_rmse]['final_metrics']['rmse']:,.2f})")
    print(f"🏆 Best R²: {best_r2.upper()} ({results[best_r2]['final_metrics']['r2']:.4f})")
    
    # Calculate improvement of Hybrid over SCAFFOLD
    if "hybrid" in results and "scaffold" in results:
        rmse_improvement = (results["scaffold"]["final_metrics"]["rmse"] - 
                          results["hybrid"]["final_metrics"]["rmse"])
        r2_improvement = (results["hybrid"]["final_metrics"]["r2"] - 
                         results["scaffold"]["final_metrics"]["r2"])
        print(f"\n📊 Hybrid vs SCAFFOLD:")
        print(f"   RMSE improvement: ${rmse_improvement:,.2f} ({rmse_improvement/results['scaffold']['final_metrics']['rmse']*100:.1f}%)")
        print(f"   R² improvement: {r2_improvement:.4f}")
    
    # Save results
    comparison_path = os.path.join(output_dir, "comparison_results.json")
    
    # Convert results to JSON-serializable format
    json_results = {}
    for name, result in results.items():
        json_results[name] = {
            "strategy": result["strategy"],
            "alpha": result["alpha"],
            "num_clients": result["num_clients"],
            "heterogeneity_score": result["heterogeneity_score"],
            "final_metrics": result["final_metrics"],
            "history": result["history"]
        }
    
    with open(comparison_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n[Saved] {comparison_path}")
    
    # Plot comparison
    if plotting_config and plotting_config.get("enabled", True):
        try:
            plotter = ComparisonPlotter(output_dir=output_dir)
            plotter.plot_comparison(
                all_histories,
                metrics=plotting_config.get("metrics_to_plot", ["rmse", "r2", "mape"]),
                save=plotting_config.get("save_plot", True),
                show=plotting_config.get("show_plot", False)
            )
        except Exception as e:
            print(f"[Warning] Could not create comparison plot: {e}")
    
    return results


@hydra.main(version_base=None, config_path="conf", config_name="dirichlet")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    # Get output directory (Hydra sets this)
    output_dir = os.getcwd()
    
    print(f"\n[Output Directory] {output_dir}")
    print(f"[Config] Using Dirichlet non-IID partitioning")
    
    # Extract parameters from config
    num_rounds = cfg.num_rounds
    num_clients = cfg.num_clients
    local_epochs = cfg.local_epochs
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    seed = cfg.seed
    
    # Dirichlet parameters
    alpha = cfg.dirichlet.alpha
    num_bins = cfg.dirichlet.num_bins
    
    # Strategy parameters
    mu = cfg.fedprox.mu
    scaffold_lr_correction = cfg.scaffold.learning_rate_correction
    fedprox_weight = cfg.hybrid.fedprox_weight
    scaffold_weight = cfg.hybrid.scaffold_weight
    
    # Hybrid config
    hybrid_config = OmegaConf.to_container(cfg.hybrid) if "hybrid" in cfg else None
    
    # Plotting config
    plotting_config = OmegaConf.to_container(cfg.plotting) if "plotting" in cfg else None
    
    # Client sampling
    client_sampling = cfg.get("client_sampling", {})
    fraction_fit = client_sampling.get("fraction_fit", 0.6)
    fraction_evaluate = client_sampling.get("fraction_evaluate", 0.5)
    min_fit_clients = client_sampling.get("min_fit_clients", 4)
    min_evaluate_clients = client_sampling.get("min_evaluate_clients", 2)
    min_available_clients = client_sampling.get("min_available_clients", 4)
    
    # Strategy configs
    strategy_configs = OmegaConf.to_container(cfg.strategy_configs) if "strategy_configs" in cfg else None
    
    # Print configuration
    if cfg.get("verbose", True):
        print(f"\n[Dirichlet Parameters]")
        print(f"  Alpha: {alpha}")
        print(f"  Num Bins: {num_bins}")
        print(f"\n[Training Parameters]")
        print(f"  Rounds: {num_rounds}")
        print(f"  Clients: {num_clients}")
        print(f"  Local Epochs: {local_epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"\n[Client Sampling]")
        print(f"  Fraction Fit: {fraction_fit:.1%}")
        print(f"  Min Fit Clients: {min_fit_clients}")
    
    # Common kwargs
    common_kwargs = {
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "alpha": alpha,
        "num_bins": num_bins,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mu": mu,
        "scaffold_lr_correction": scaffold_lr_correction,
        "fedprox_weight": fedprox_weight,
        "scaffold_weight": scaffold_weight,
        "seed": seed,
        "verbose": cfg.get("verbose", True),
        "output_dir": output_dir,
        "hybrid_config": hybrid_config,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "visualize_partition_plot": cfg.plotting.get("visualize_partition", True) if plotting_config else True,
    }
    
    # Run based on mode
    if cfg.get("compare_all", False):
        # Compare strategies
        strategies_to_compare = cfg.get("strategies", ["scaffold", "hybrid"])
        
        results = compare_strategies_dirichlet(
            strategies=strategies_to_compare,
            plotting_config=plotting_config,
            strategy_configs=strategy_configs,
            **common_kwargs
        )
    else:
        # Single strategy
        strategy_name = cfg.get("strategy", "hybrid")
        
        # Get strategy-specific config
        if strategy_configs and strategy_name in strategy_configs:
            strategy_cfg = strategy_configs[strategy_name]
            if 'lr' in strategy_cfg:
                common_kwargs['learning_rate'] = strategy_cfg['lr']
            if 'local_epochs' in strategy_cfg:
                common_kwargs['local_epochs'] = strategy_cfg['local_epochs']
            if 'max_grad_norm' in strategy_cfg:
                common_kwargs['max_grad_norm'] = strategy_cfg['max_grad_norm']
        
        result = run_dirichlet_simulation(
            strategy_name=strategy_name,
            **common_kwargs
        )
        
        # Save results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "strategy": result["strategy"],
                "alpha": result["alpha"],
                "num_clients": result["num_clients"],
                "heterogeneity_score": result["heterogeneity_score"],
                "final_metrics": result["final_metrics"],
                "history": result["history"]
            }, f, indent=2, default=str)
        print(f"\n[Saved] {results_path}")


if __name__ == "__main__":
    main()
