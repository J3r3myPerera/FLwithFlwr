"""
Main script for Federated Learning with Stratified Client Selection.
Simplified version focusing only on stratified selection functionality.
"""

import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from stratified_strategy import StratifiedFedAvg
from stratified_selector import create_stratified_client_fn
import flwr as fl
from plotting import (
    plot_comparison, 
    plot_stratified_selection_metrics,
    plot_random_vs_stratified_comparison
)
import random
import numpy as np
import torch


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility across the entire pipeline.
    
    This ensures:
    1. Model initialization is identical across runs
    2. Client selection (both random and stratified) is reproducible
    3. Training dynamics (SGD, dropout, etc.) are deterministic
    4. Fair comparison between strategies
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🌱 All random seeds set to: {seed}")

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse the config & get experiment for output directory
    print(OmegaConf.to_yaml(cfg))
    
    ## 1.1. Set global random seed for reproducibility
    global_seed = cfg.get('seed', 42)
    set_all_seeds(global_seed)
    print(f"\n{'='*80}")
    print(f"REPRODUCIBILITY MODE ENABLED")
    print(f"{'='*80}")
    print(f"Global seed: {global_seed}")
    print(f"This ensures identical results across runs with the same seed.")
    print(f"Both Random and Stratified selection will use the same starting conditions.")
    print(f"{'='*80}\n")

    ## 2. Prepare the datasets (Pure Tier Partitioning)
    print(f"\nData Partitioning: Pure Tier Partitioning (Maximum Heterogeneity)")
    print("Each client receives data from ONLY one City Tier")
    
    trainloaders, validationloaders, testloader, target_scaler, client_strata = prepare_dataset(
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        data_path=cfg.get('data_path', './data/IndianPersonalFinance/indianPersonalFinanceAndSpendingHabits.csv'),
        seed=global_seed  # Pass global seed to dataset preparation
    )
    
    # CRITICAL: Reset seeds after dataset preparation
    # Dataset prep internally calls np.random.seed() and torch.manual_seed()
    # which modifies the global random state. We need to reset it to ensure
    # consistent behavior for both Random and Stratified selection phases.
    print(f"🌱 Resetting seeds after dataset preparation (was modified internally)...")
    set_all_seeds(global_seed)
    
    print(f"Number of training clients: {len(trainloaders)}")
    print(f"Number of samples in first train loader: {len(trainloaders[0].dataset)}")
    print(f"\nClient Strata Configuration:")
    for stratum, clients in client_strata.items():
        print(f"  {stratum}: {len(clients)} clients (IDs: {clients})")

    ## 3. Define the clients
    input_dim = cfg.get('input_dim', 19)
    
    ## 4. Get strategy configuration
    strategy_name = cfg.get('strategy', 'compare_stratified').lower()
    
    ## 5. Run simulations
    all_results = {}
    save_path = HydraConfig.get().runtime.output_dir
    
    if strategy_name == 'compare_stratified':
        # Run comparison: Random vs Stratified Client Selection
        print("\n" + "="*80)
        print("RUNNING COMPARISON: Random vs Stratified Client Selection")
        print("="*80)
        
        # Get configurations
        fedprox_config = cfg.get('fedprox', None)
        
        # Calculate fractions
        fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
        fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0
        
        # ========== PHASE 1: Random Selection (Baseline) ==========
        print("\n" + "-"*80)
        print("PHASE 1: Running FedAvg with Random Client Selection (Baseline)")
        print("-"*80)
        
        # IMPORTANT: Reset seeds before random selection for fair comparison
        print(f"🌱 Resetting seeds to {global_seed} for Random Selection phase")
        set_all_seeds(global_seed)
        
        lr = fedprox_config.get('lr', cfg.config_fit.lr) if fedprox_config else cfg.config_fit.lr
        
        print(f"Configuration: Random Selection FedAvg")
        print(f"  Learning rate: {lr}")
        print(f"  Random seed: {global_seed} (ensures reproducibility)")
        
        client_fn_random = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_random = get_on_fit_config(cfg.config_fit, fedprox_config)
        
        strategy_random = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_random,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler)
        )
        
        history_random = fl.simulation.start_simulation(
            client_fn=client_fn_random,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_random,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )
        
        all_results['random_selection'] = {
            'history': history_random,
            'name': 'Random Selection',
            'config': OmegaConf.to_container(fedprox_config, resolve=True) if fedprox_config else {}
        }
        print(f"\n[OK] Random Selection completed!")
        
        # ========== PHASE 2: Stratified Selection ==========
        print("\n" + "-"*80)
        print("PHASE 2: Running FedAvg with Stratified Client Selection")
        print("-"*80)
        
        # IMPORTANT: Reset seeds before stratified selection for fair comparison
        # Both phases start with identical model initialization and conditions
        print(f"🌱 Resetting seeds to {global_seed} for Stratified Selection phase")
        set_all_seeds(global_seed)
        
        # Create stratified selector
        clients_per_round = cfg.num_clients_per_round_fit
        min_per_stratum = cfg.get('min_clients_per_stratum', 1)
        stratified_selector = create_stratified_client_fn(
            client_strata=client_strata,
            clients_per_round=clients_per_round,
            min_per_stratum=min_per_stratum
        )
        
        print(f"Configuration: Stratified Selection FedAvg")
        print(f"  Learning rate: {lr}")
        print(f"  Clients per round: {clients_per_round}")
        print(f"  Minimum per stratum: {min_per_stratum}")
        print(f"  Random seed: {global_seed} (same as Random for fair comparison)")
        
        client_fn_stratified = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_stratified = get_on_fit_config(cfg.config_fit, fedprox_config)
        
        strategy_stratified = StratifiedFedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_stratified,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            stratified_selector=stratified_selector
        )
        
        history_stratified = fl.simulation.start_simulation(
            client_fn=client_fn_stratified,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_stratified,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )
        
        all_results['stratified_selection'] = {
            'history': history_stratified,
            'name': 'Stratified Selection',
            'config': OmegaConf.to_container(fedprox_config, resolve=True) if fedprox_config else {},
            'selection_history': stratified_selector.get_selection_history(),
            'fairness_metrics': strategy_stratified.get_fairness_metrics()
        }
        print(f"\n[OK] Stratified Selection completed!")
        
        # Print final summary
        strategy_stratified.print_final_summary()
        
        # Generate plots
        print("\n" + "-"*80)
        print("GENERATING COMPARISON PLOTS")
        print("-"*80)
        
        # Standard comparison plot
        plot_comparison(all_results, save_path, cfg.get('plotting', {}))
        
        # Random vs Stratified specific comparison
        plot_random_vs_stratified_comparison(
            all_results['random_selection'],
            all_results['stratified_selection'],
            save_path,
            cfg.get('plotting', {})
        )
        
        # Stratified selection analysis
        plot_stratified_selection_metrics(
            stratified_selector.get_selection_history(),
            client_strata,
            save_path,
            cfg.get('plotting', {})
        )
        
        print("[OK] All plots saved!")

    elif strategy_name == 'stratified':
        # Run only stratified selection (no comparison)
        print("\n" + "="*80)
        print("RUNNING: Stratified Client Selection Only")
        print("="*80)
        
        # Reset seeds for reproducibility
        print(f"🌱 Resetting seeds to {global_seed} for Stratified Selection")
        set_all_seeds(global_seed)
        
        fedprox_config = cfg.get('fedprox', None)
        lr = fedprox_config.get('lr', cfg.config_fit.lr) if fedprox_config else cfg.config_fit.lr
        
        # Calculate fractions
        fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
        fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0
        
        # Create stratified selector
        clients_per_round = cfg.num_clients_per_round_fit
        min_per_stratum = cfg.get('min_clients_per_stratum', 1)
        stratified_selector = create_stratified_client_fn(
            client_strata=client_strata,
            clients_per_round=clients_per_round,
            min_per_stratum=min_per_stratum
        )
        
        print(f"Configuration:")
        print(f"  Learning rate: {lr}")
        print(f"  Clients per round: {clients_per_round}")
        print(f"  Minimum per stratum: {min_per_stratum}")
        print(f"  Random seed: {global_seed}")
        
        client_fn = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn = get_on_fit_config(cfg.config_fit, fedprox_config)
        
        strategy = StratifiedFedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            stratified_selector=stratified_selector
        )
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )
        
        all_results['stratified_selection'] = {
            'history': history,
            'name': 'Stratified Selection',
            'config': OmegaConf.to_container(cfg, resolve=True),
            'selection_history': stratified_selector.get_selection_history(),
            'fairness_metrics': strategy.get_fairness_metrics()
        }
        
        # Print final summary
        strategy.print_final_summary()
        
        # Generate plots
        print("\n" + "-"*80)
        print("GENERATING PLOTS")
        print("-"*80)
        
        plot_stratified_selection_metrics(
            stratified_selector.get_selection_history(),
            client_strata,
            save_path,
            cfg.get('plotting', {})
        )
        
        print("[OK] Plots saved!")
    
    else:
        print(f"\nError: Unknown strategy '{strategy_name}'")
        print("Available strategies: 'compare_stratified', 'stratified'")
        return

    ## 6. Save the results
    results_path = Path(save_path) / "results.pkl"
    
    results = {
        'all_results': all_results,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'strategy': strategy_name,
        'description': f'Personal Finance FL with Stratified Client Selection'
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nResults saved to: {results_path}")
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
