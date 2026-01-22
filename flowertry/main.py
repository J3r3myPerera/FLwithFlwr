import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import flwr as fl
from plotting import plot_comparison

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse the config & get experiment for output directory
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the datasets (only once, shared by both strategies)
    alpha = cfg.get('alpha', 0.1)  # Get alpha for heterogeneity control
    print(f"\nData Heterogeneity (alpha={alpha}): {'High heterogeneity' if alpha < 0.3 else 'Moderate heterogeneity' if alpha < 0.7 else 'Low heterogeneity (nearly IID)'}")
    
    trainloaders, validationloaders, testloader, target_scaler = prepare_dataset(
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        data_path=cfg.get('data_path', './data/IndianPersonalFinance/indianPersonalFinanceAndSpendingHabits.csv'),
        non_iid=cfg.get('non_iid', True),
        alpha=alpha
    )
    print(f"Number of training clients: {len(trainloaders)}")
    print(f"Number of samples in first train loader: {len(trainloaders[0].dataset)}")

    ## 3. Define the clients
    input_dim = cfg.get('input_dim', 19)
    
    ## 4. Get strategy configuration
    strategy_name = cfg.get('strategy', 'fedavg').lower()
    
    ## 5. Run simulations based on strategy
    all_results = {}
    save_path = HydraConfig.get().runtime.output_dir
    
    if strategy_name == 'compare':
        # Run both base FedProx and multi-layer FedProx
        print("\n" + "="*80)
        print("RUNNING COMPARISON: Base FedProx vs Multi-Layer FedProx")
        print("="*80)
        
        # Get configurations
        fedprox_base_config = cfg.get('fedprox_base', None)
        fedprox_multilayer_config = cfg.get('fedprox_multilayer', None)
        
        # Run Base FedProx
        print("\n" + "-"*80)
        print("PHASE 1: Running Base FedProx (suboptimal settings)")
        print("-"*80)
        base_name = fedprox_base_config.get('name', 'Base FedProx') if fedprox_base_config else 'Base FedProx'
        base_mu = fedprox_base_config.get('mu', 0.1) if fedprox_base_config else 0.1
        base_lr = fedprox_base_config.get('lr', cfg.config_fit.lr) if fedprox_base_config else cfg.config_fit.lr
        print(f"Configuration: {base_name}")
        print(f"  μ (proximal term): {base_mu}")
        print(f"  Learning rate: {base_lr}")
        
        client_fn_base = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_base = get_on_fit_config(cfg.config_fit, fedprox_base_config)
        
        # Calculate fraction_evaluate based on desired number of clients
        fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
        fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0
        
        print(f"  Client sampling: fraction_fit={fraction_fit:.2f} (min={cfg.num_clients_per_round_fit}), fraction_evaluate={fraction_evaluate:.2f} (min={cfg.num_clients_per_round_eval})")
        
        strategy_base = fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_base,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            proximal_mu=base_mu
        )
        
        history_base = fl.simulation.start_simulation(
            client_fn=client_fn_base,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_base,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )
        
        all_results['base_fedprox'] = {
            'history': history_base,
            'name': base_name,
            'config': OmegaConf.to_container(fedprox_base_config, resolve=True) if fedprox_base_config else {}
        }
        print(f"\n✓ Base FedProx completed!")
        
        # Run Multi-Layer FedProx
        print("\n" + "-"*80)
        print("PHASE 2: Running Multi-Layer FedProx (optimized settings)")
        print("-"*80)
        ml_name = fedprox_multilayer_config.get('name', 'Multi-Layer FedProx') if fedprox_multilayer_config else 'Multi-Layer FedProx'
        ml_mu = fedprox_multilayer_config.get('mu', 0.01) if fedprox_multilayer_config else 0.01
        ml_lr = fedprox_multilayer_config.get('lr', cfg.config_fit.lr) if fedprox_multilayer_config else cfg.config_fit.lr
        layer_mus = fedprox_multilayer_config.get('layer_mus', None) if fedprox_multilayer_config else None
        print(f"Configuration: {ml_name}")
        print(f"  Base μ (proximal term): {ml_mu}")
        print(f"  Learning rate: {ml_lr}")
        if layer_mus:
            print(f"  Layer-specific μ values: {dict(layer_mus)}")
        
        client_fn_ml = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_ml = get_on_fit_config(cfg.config_fit, fedprox_multilayer_config)
        
        # Calculate fraction_evaluate based on desired number of clients
        fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
        fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0
        
        print(f"  Client sampling: fraction_fit={fraction_fit:.2f} (min={cfg.num_clients_per_round_fit}), fraction_evaluate={fraction_evaluate:.2f} (min={cfg.num_clients_per_round_eval})")
        
        strategy_ml = fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_ml,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            proximal_mu=ml_mu  # Base mu for Flower's FedProx, layer-specific handled in training
        )
        
        history_ml = fl.simulation.start_simulation(
            client_fn=client_fn_ml,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_ml,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )
        
        all_results['multilayer_fedprox'] = {
            'history': history_ml,
            'name': ml_name,
            'config': OmegaConf.to_container(fedprox_multilayer_config, resolve=True) if fedprox_multilayer_config else {}
        }
        print(f"\n✓ Multi-Layer FedProx completed!")
        
        # Create comparison plots
        print("\n" + "-"*80)
        print("GENERATING COMPARISON PLOTS")
        print("-"*80)
        plot_comparison(all_results, save_path, cfg.get('plotting', {}))
        print("✓ Plots saved!")
        
    else:
        # Single strategy run (original behavior)
        fedprox_config = cfg.get('fedprox', None)
        on_fit_config_fn = get_on_fit_config(cfg.config_fit, fedprox_config)
        
        if strategy_name == 'fedprox':
            mu = fedprox_config.get('mu', 0.01) if fedprox_config else 0.01
            print(f"Using FedProx strategy with mu={mu}")
            
                # Calculate fractions based on desired number of clients
            fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0
            fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
            
            strategy = fl.server.strategy.FedProx(
                fraction_fit=fraction_fit,
                min_fit_clients=cfg.num_clients_per_round_fit,
                fraction_evaluate=fraction_evaluate,
                min_evaluate_clients=cfg.num_clients_per_round_eval,
                min_available_clients=cfg.num_clients,
                on_fit_config_fn=on_fit_config_fn,
                evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
                proximal_mu=mu
            )
        else:
            print(f"Using FedAvg strategy")
            # Calculate fractions based on desired number of clients
            fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0
            fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
            
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=fraction_fit,
                min_fit_clients=cfg.num_clients_per_round_fit,
                fraction_evaluate=fraction_evaluate,
                min_evaluate_clients=cfg.num_clients_per_round_eval,
                min_available_clients=cfg.num_clients,
                on_fit_config_fn=on_fit_config_fn,
                evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler)
            )
        
        client_fn = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )
        
        all_results['single'] = {
            'history': history,
            'name': strategy_name.upper(),
            'config': OmegaConf.to_container(cfg, resolve=True)
        }

    ## 6. Save the results
    results_path = Path(save_path) / "results.pkl"
    
    results = {
        'all_results': all_results,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'strategy': strategy_name,
        'description': f'Disposable Income Regression with Federated Learning'
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nResults saved to: {results_path}")
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
