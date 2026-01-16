"""
Main entry point for Federated Disposable Income Regression.

Supports multiple strategies:
- FedAvg: Baseline federated averaging
- FedProx: With proximal term for Non-IID data
- SCAFFOLD: With control variates for variance reduction
- Hybrid: Combined FedProx + SCAFFOLD with configurable weights

Usage:
    python main.py --config-name base
    python main.py strategy=fedprox
    python main.py strategy=scaffold
    python main.py strategy=hybrid
    python main.py compare_all=true
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

from dataset import prepare_dataset_federated, reset_preprocessor
from model import DisposableIncomeNet, get_parameters, set_parameters
from client import RegressionClient, create_client
from server import create_strategy, compute_regression_metrics
from visualize_metrics import ComparisonPlotter, save_metrics_to_csv


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simulation(
    strategy_name: str,
    num_rounds: int = 50,
    num_clients: int = 3,
    local_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    mu: float = 0.1,
    scaffold_lr_correction: float = 1.0,
    fedprox_weight: float = 1.0,
    scaffold_weight: float = 1.0,
    iid: bool = False,
    partition_strategy: str = 'city_tier',
    max_grad_norm: float = 1.0,
    seed: int = 2023,
    verbose: bool = True,
    output_dir: str = ".",
    hybrid_config: Optional[Dict] = None,
    # Client sampling parameters for typical FL behavior
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
) -> Dict:
    """
    Run federated learning simulation.
    
    Args:
        strategy_name: Strategy to use (fedavg, fedprox, scaffold, hybrid)
        num_rounds: Number of federated rounds
        num_clients: Number of clients
        local_epochs: Local training epochs per round
        batch_size: Batch size
        learning_rate: Learning rate
        mu: FedProx proximal term weight
        scaffold_lr_correction: SCAFFOLD learning rate correction
        fedprox_weight: Weight for FedProx component in hybrid
        scaffold_weight: Weight for SCAFFOLD component in hybrid
        iid: If True, use IID partitioning (overrides partition_strategy)
        partition_strategy: 'city_tier' (3), 'occupation' (4), or 'hybrid' (12)
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        fraction_fit: Fraction of clients to sample for training (0.0-1.0)
        fraction_evaluate: Fraction of clients to sample for evaluation (0.0-1.0)
        min_fit_clients: Minimum number of clients required for training
        min_evaluate_clients: Minimum number of clients required for evaluation
        min_available_clients: Minimum available clients before training starts
        verbose: Print progress
        output_dir: Directory for saving outputs
        hybrid_config: Optional dict with hybrid-specific adaptive parameters
    
    Returns:
        Dictionary with training history and final metrics
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Federated Learning - {strategy_name.upper()}")
        print(f"{'='*70}")
    
    # Prepare dataset
    reset_preprocessor()
    trainloaders, valloaders, testloader, input_dim, target_mean, target_std, partition_info, log_transform = \
        prepare_dataset_federated(
            num_partitions=num_clients,
            batch_size=batch_size,
            iid=iid,
            partition_strategy=partition_strategy,
            log_transform_target=True,  # Enable log transformation for better MAPE
            seed=seed,
            verbose=verbose
        )
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    if verbose:
        print(f"\n[Device] {device}")
        print(f"[Strategy] {strategy_name}")
        print(f"[Rounds] {num_rounds}")
        print(f"[Local Epochs] {local_epochs}")
        print(f"[Learning Rate] {learning_rate}")
        if strategy_name in ["fedprox", "hybrid"]:
            print(f"[Mu (FedProx)] {mu}")
        if strategy_name == "hybrid":
            print(f"[FedProx Weight] {fedprox_weight}")
            print(f"[SCAFFOLD Weight] {scaffold_weight}")
        if strategy_name in ["scaffold", "hybrid"]:
            print(f"[SCAFFOLD LR Correction] {scaffold_lr_correction}")
    
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
            "server_round": server_round
        }
        
        # Apply advanced parameters for hybrid strategy
        if strategy_name == "hybrid" and hybrid_config:
            # Dynamic weight adjustment
            config["dynamic_weights"] = hybrid_config.get("dynamic_weights", False)
            config["weight_increase_rate"] = hybrid_config.get("weight_increase_rate", 1.05)
            config["weight_max"] = hybrid_config.get("weight_max", 0.6)
            
            # Adaptive mu with more sophisticated logic
            adaptive_mu = hybrid_config.get("adaptive_mu", False)
            if adaptive_mu and server_round > 1:
                mu_decay = hybrid_config.get("mu_decay", 0.98)
                mu_min = hybrid_config.get("mu_min", 0.001)
                config["mu"] = max(mu * (mu_decay ** (server_round - 1)), mu_min)
            
            # Advanced warmup parameters
            config["warmup_rounds"] = hybrid_config.get("warmup_rounds", 0)
            config["warmup_type"] = hybrid_config.get("warmup_type", "exponential")
            config["warmup_start_factor"] = hybrid_config.get("warmup_start_factor", 0.05)
            
            # Gradient clipping
            config["gradient_clip"] = hybrid_config.get("gradient_clip", False)
            config["gradient_clip_norm"] = hybrid_config.get("gradient_clip_norm", 1.0)
            
            # Momentum
            config["use_momentum"] = hybrid_config.get("use_momentum", False)
            config["momentum"] = hybrid_config.get("momentum", 0.9)
            
            # Advanced learning rate scheduling with warmup
            use_lr_scheduler = hybrid_config.get("use_lr_scheduler", False)
            if use_lr_scheduler:
                lr_warmup_rounds = hybrid_config.get("lr_warmup_rounds", 3)
                lr_decay_start = hybrid_config.get("lr_decay_start", 10)
                
                if server_round <= lr_warmup_rounds:
                    # LR warmup
                    warmup_progress = server_round / lr_warmup_rounds
                    config["learning_rate"] = learning_rate * 0.1 + learning_rate * 0.9 * warmup_progress
                elif server_round >= lr_decay_start:
                    # LR decay after warmup
                    lr_decay = hybrid_config.get("lr_decay", 0.98)
                    lr_min = hybrid_config.get("lr_min", 0.0005)
                    decay_steps = server_round - lr_decay_start
                    config["learning_rate"] = max(learning_rate * (lr_decay ** decay_steps), lr_min)
        
        return config
    
    # Create strategy
    strategy = create_strategy(
        strategy_name=strategy_name,
        testloader=testloader,
        input_dim=input_dim,
        target_mean=target_mean,
        target_std=target_std,
        num_clients=num_clients,
        mu=mu,
        scaffold_lr_correction=scaffold_lr_correction,
        fedprox_weight=fedprox_weight,
        scaffold_weight=scaffold_weight,
        log_transform=log_transform,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        # Client sampling parameters
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    
    # Log client sampling configuration
    if verbose:
        print(f"\n[Client Sampling]")
        print(f"  Fraction Fit: {fraction_fit:.1%} ({int(num_clients * fraction_fit)} of {num_clients} clients per round)")
        print(f"  Min Fit Clients: {min_fit_clients}")
        print(f"  Fraction Evaluate: {fraction_evaluate:.1%}")
        print(f"  Min Available Clients: {min_available_clients}")
    
    # Client function
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        client = RegressionClient(
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
        return client
    
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
        print(f"\n{'Round':>6} | {'Loss':>12} | {'RMSE':>12} | {'MAE':>12} | {'R²':>10}")
        print("-" * 70)
    
    result = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Extract history from result
    if result.metrics_centralized:
        for round_num, metrics in result.metrics_centralized.items():
            if isinstance(metrics, dict):
                # Get corresponding loss
                loss_value = result.losses_centralized.get(round_num, (0, 0))
                loss = loss_value[1] if isinstance(loss_value, tuple) else loss_value
                
                history["rounds"].append(round_num)
                history["loss"].append(loss)
                history["rmse"].append(metrics.get("rmse", 0))
                history["mae"].append(metrics.get("mae", 0))
                history["r2"].append(metrics.get("r2", 0))
                history["mape"].append(metrics.get("mape", 0))
                history["accuracy_10"].append(metrics.get("accuracy_10", 0))
                history["accuracy_20"].append(metrics.get("accuracy_20", 0))
                
                # Print verbose output
                if verbose:
                    print(f"{round_num:>6} | {loss:>12.6f} | {metrics.get('rmse', 0):>12,.2f} | {metrics.get('mae', 0):>12,.2f} | {metrics.get('r2', 0):>10.4f}")
                
                # Update real-time plotter if available
                if plotter is not None:
                    plotter.update(
                        strategy=strategy_name,
                        round_num=round_num,
                        metrics={
                            "loss": loss,
                            "rmse": metrics.get("rmse", 0),
                            "mae": metrics.get("mae", 0),
                            "r2": metrics.get("r2", 0),
                            "mape": metrics.get("mape", 0),
                            "accuracy_10": metrics.get("accuracy_10", 0),
                            "accuracy_20": metrics.get("accuracy_20", 0)
                        }
                    )
    
    # Final evaluation - use the model from strategy which has final parameters
    final_model = strategy.model  # Strategy maintains the model with final parameters
    final_metrics = compute_regression_metrics(
        final_model, testloader, target_std, target_mean, device, log_transform
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - {strategy_name.upper()}")
        print(f"{'='*70}")
        print(f"  Loss:        {final_metrics['loss']:.6f}")
        print(f"  RMSE:        ${final_metrics['rmse']:,.2f}")
        print(f"  MAE:         ${final_metrics['mae']:,.2f}")
        print(f"  R²:          {final_metrics['r2']:.4f}")
        print(f"  MAPE:        {final_metrics['mape']:.2f}%")
        print(f"  Accuracy@10%: {final_metrics['accuracy_10']:.2f}%")
        print(f"  Accuracy@20%: {final_metrics['accuracy_20']:.2f}%")
        print(f"{'='*70}")
    
    return {
        "strategy": strategy_name,
        "final_metrics": final_metrics,
        "history": history,
        "partition_info": partition_info
    }


def compare_strategies(
    strategies: List[str] = ["fedavg", "fedprox", "scaffold", "hybrid"],
    output_dir: str = ".",
    plotting_config: Optional[Dict] = None,
    strategy_configs: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    Compare multiple federated learning strategies.
    
    Args:
        strategies: List of strategy names to compare
        output_dir: Directory for saving outputs
        plotting_config: Configuration for plotting
        strategy_configs: Dict of strategy-specific configurations (lr, local_epochs, etc.)
        **kwargs: Additional arguments passed to run_simulation
    
    Returns:
        Dictionary with results for each strategy
    """
    results = {}
    
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON - Disposable Income Regression")
    print("=" * 70)
    
    for strategy in strategies:
        # Get strategy-specific config if available
        strategy_kwargs = kwargs.copy()
        if strategy_configs and strategy in strategy_configs:
            config = strategy_configs[strategy]
            print(f"\n[{strategy.upper()}] Using custom configuration:")
            print(f"  Learning Rate: {config.get('lr', kwargs.get('learning_rate'))}")
            print(f"  Local Epochs: {config.get('local_epochs', kwargs.get('local_epochs'))}")
            print(f"  Max Grad Norm: {config.get('max_grad_norm', 1.0)}")
            
            # Override with strategy-specific values
            if 'lr' in config:
                strategy_kwargs['learning_rate'] = config['lr']
            if 'local_epochs' in config:
                strategy_kwargs['local_epochs'] = config['local_epochs']
            if 'max_grad_norm' in config:
                strategy_kwargs['max_grad_norm'] = config['max_grad_norm']
        
        results[strategy] = run_simulation(
            strategy_name=strategy, 
            output_dir=output_dir,
            **strategy_kwargs
        )
    
    # Generate final comparison plot only
    comp_plotter = ComparisonPlotter(
        output_dir=output_dir,
        figsize=(16, 12),
        plot_format=plotting_config.get("format", "png") if plotting_config else "png"
    )
    comp_plotter.generate_all_plots(results)
    
    # Save metrics to CSV
    save_metrics_to_csv(results, output_dir)
    
    # Print comparison table
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Strategy':<12} | {'RMSE':>10} | {'MAE':>10} | {'R²':>8} | {'MAPE':>8} | {'Acc@10%':>8} | {'Acc@20%':>8}")
    print("-" * 100)
    
    for strategy, result in results.items():
        metrics = result["final_metrics"]
        print(f"{strategy:<12} | ${metrics['rmse']:>8,.0f} | ${metrics['mae']:>8,.0f} | {metrics['r2']:>8.4f} | {metrics['mape']:>7.2f}% | {metrics['accuracy_10']:>7.2f}% | {metrics['accuracy_20']:>7.2f}%")
    
    print("=" * 100)
    
    return results


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    print("\n" + "=" * 70)
    print("Federated Disposable Income Regression")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Extract strategy-specific parameters
    strategy = cfg.strategy
    
    # Get strategy-specific config if available
    strategy_config = None
    if "strategy_configs" in cfg and strategy in cfg.strategy_configs:
        strategy_config = cfg.strategy_configs[strategy]
        if cfg.get("verbose", True):
            print(f"\n[Strategy Config] Using custom config for {strategy}:")
            print(f"  Learning Rate: {strategy_config.lr}")
            print(f"  Local Epochs: {strategy_config.local_epochs}")
            print(f"  Max Grad Norm: {strategy_config.max_grad_norm}")
    
    # Use strategy-specific learning rate or default
    learning_rate = strategy_config.lr if strategy_config else cfg.learning_rate
    local_epochs = strategy_config.local_epochs if strategy_config else cfg.local_epochs
    max_grad_norm = strategy_config.max_grad_norm if strategy_config else 1.0
    
    # Get mu based on strategy
    if strategy == "fedprox":
        mu = cfg.fedprox.mu
    elif strategy == "hybrid":
        mu = cfg.hybrid.mu
    else:
        mu = cfg.get("fedprox", {}).get("mu", 0.1)
    
    # Get SCAFFOLD parameters
    scaffold_lr_correction = cfg.scaffold.learning_rate_correction
    
    # Get hybrid weights
    fedprox_weight = cfg.hybrid.fedprox_weight
    scaffold_weight = cfg.hybrid.scaffold_weight
    
    # Get hybrid advanced config
    hybrid_config = OmegaConf.to_container(cfg.hybrid) if "hybrid" in cfg else None
    
    # Get plotting configuration
    plotting_config = OmegaConf.to_container(cfg.plotting) if "plotting" in cfg else None
    
    # Get client sampling configuration (for typical FL behavior)
    client_sampling = cfg.get("client_sampling", {})
    fraction_fit = client_sampling.get("fraction_fit", 1.0)
    fraction_evaluate = client_sampling.get("fraction_evaluate", 1.0)
    min_fit_clients = client_sampling.get("min_fit_clients", 2)
    min_evaluate_clients = client_sampling.get("min_evaluate_clients", 2)
    min_available_clients = client_sampling.get("min_available_clients", 2)
    
    if cfg.get("verbose", True):
        print(f"\n[Client Sampling Configuration]")
        print(f"  Fraction Fit: {fraction_fit:.1%}")
        print(f"  Min Fit Clients: {min_fit_clients}")
        print(f"  Fraction Evaluate: {fraction_evaluate:.1%}")
        print(f"  Min Evaluate Clients: {min_evaluate_clients}")
        print(f"  Min Available Clients: {min_available_clients}")
    
    # Run based on mode
    if cfg.get("compare_all", False):
        # Compare strategies (use configured list or default to all)
        strategies_to_compare = cfg.get("strategies", ["fedavg", "fedprox", "scaffold", "hybrid"])
        
        # Get strategy_configs if available
        strategy_configs = OmegaConf.to_container(cfg.strategy_configs) if "strategy_configs" in cfg else None
        
        results = compare_strategies(
            strategies=strategies_to_compare,
            num_rounds=cfg.num_rounds,
            num_clients=cfg.num_clients,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            mu=mu,
            scaffold_lr_correction=scaffold_lr_correction,
            fedprox_weight=fedprox_weight,
            scaffold_weight=scaffold_weight,
            iid=cfg.get("iid", False),
            partition_strategy=cfg.get("partition_strategy", "city_tier"),
            seed=cfg.seed,
            verbose=cfg.get("verbose", True),
            output_dir=output_dir,
            plotting_config=plotting_config,
            hybrid_config=hybrid_config,
            strategy_configs=strategy_configs,
            # Client sampling parameters
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        
        # Save results
        with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
            # Convert numpy types for JSON serialization
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
            
            json.dump(convert(results), f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    
    else:
        # Run single strategy
        result = run_simulation(
            strategy_name=strategy,
            num_rounds=cfg.num_rounds,
            num_clients=cfg.num_clients,
            local_epochs=local_epochs,  # Use strategy-specific or default
            batch_size=cfg.batch_size,
            learning_rate=learning_rate,  # Use strategy-specific or default
            mu=mu,
            scaffold_lr_correction=scaffold_lr_correction,
            fedprox_weight=fedprox_weight,
            scaffold_weight=scaffold_weight,
            iid=cfg.get("iid", False),
            partition_strategy=cfg.get("partition_strategy", "city_tier"),
            max_grad_norm=max_grad_norm,  # Use strategy-specific or default
            seed=cfg.seed,
            verbose=cfg.get("verbose", True),
            output_dir=output_dir,
            hybrid_config=hybrid_config,
            # Client sampling parameters
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        
        # Save results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
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
            
            json.dump(convert(result), f, indent=2)
        
        # Save metrics to CSV
        save_metrics_to_csv({strategy: result}, output_dir)
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
