import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_on_fit_config_adaptive, get_evaluate_fn
from strategy import AdaptiveFedProx
from adaptive_mu import AdaptiveMuController, AdaptiveMuConfig
import flwr as fl
from plotting import plot_comparison, plot_adaptive_mu_evolution

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

    elif strategy_name == 'compare_adaptive':
        # Run 3-way comparison: Base FedProx vs Static Multi-Layer vs Adaptive Multi-Layer
        print("\n" + "="*80)
        print("RUNNING 3-WAY COMPARISON: Base vs Static Multi-Layer vs Adaptive Multi-Layer")
        print("="*80)

        # Get configurations
        fedprox_base_config = cfg.get('fedprox_base', None)
        fedprox_multilayer_config = cfg.get('fedprox_multilayer', None)
        adaptive_config = cfg.get('fedprox_adaptive', None)

        # Calculate fractions
        fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
        fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0

        # ========== PHASE 1: Base FedProx ==========
        print("\n" + "-"*80)
        print("PHASE 1: Running Base FedProx (mu=1.0)")
        print("-"*80)
        base_name = fedprox_base_config.get('name', 'Base FedProx') if fedprox_base_config else 'Base FedProx'
        base_mu = fedprox_base_config.get('mu', 1.0) if fedprox_base_config else 1.0
        base_lr = fedprox_base_config.get('lr', cfg.config_fit.lr) if fedprox_base_config else cfg.config_fit.lr
        print(f"Configuration: {base_name}")
        print(f"  mu (proximal term): {base_mu}")
        print(f"  Learning rate: {base_lr}")

        client_fn_base = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_base = get_on_fit_config(cfg.config_fit, fedprox_base_config)

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
        print(f"\n[OK] Base FedProx completed!")

        # ========== PHASE 2: Static Multi-Layer FedProx ==========
        print("\n" + "-"*80)
        print("PHASE 2: Running Static Multi-Layer FedProx")
        print("-"*80)
        ml_name = fedprox_multilayer_config.get('name', 'Static Multi-Layer FedProx') if fedprox_multilayer_config else 'Static Multi-Layer FedProx'
        ml_mu = fedprox_multilayer_config.get('mu', 0.01) if fedprox_multilayer_config else 0.01
        ml_lr = fedprox_multilayer_config.get('lr', cfg.config_fit.lr) if fedprox_multilayer_config else cfg.config_fit.lr
        layer_mus = fedprox_multilayer_config.get('layer_mus', None) if fedprox_multilayer_config else None
        print(f"Configuration: {ml_name}")
        print(f"  Base mu: {ml_mu}")
        print(f"  Learning rate: {ml_lr}")
        if layer_mus:
            print(f"  Layer-specific mu values: {dict(layer_mus)}")

        client_fn_static = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_static = get_on_fit_config(cfg.config_fit, fedprox_multilayer_config)

        strategy_static = fl.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_static,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            proximal_mu=ml_mu
        )

        history_static = fl.simulation.start_simulation(
            client_fn=client_fn_static,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_static,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )

        all_results['static_multilayer'] = {
            'history': history_static,
            'name': ml_name,
            'config': OmegaConf.to_container(fedprox_multilayer_config, resolve=True) if fedprox_multilayer_config else {}
        }
        print(f"\n[OK] Static Multi-Layer FedProx completed!")

        # ========== PHASE 3: Adaptive Multi-Layer FedProx ==========
        print("\n" + "-"*80)
        print("PHASE 3: Running Adaptive Multi-Layer FedProx")
        print("-"*80)

        # Create adaptive mu controller
        base_layer_mus = dict(layer_mus) if layer_mus else {
            'input': 0.003, 'hidden1': 0.008, 'hidden2': 0.012, 'output': 0.025
        }

        adaptive_mu_config = AdaptiveMuConfig(
            base_layer_mus=base_layer_mus,
            schedule_type=adaptive_config.get('schedule_type', 'cosine') if adaptive_config else 'cosine',
            min_schedule_factor=adaptive_config.get('min_schedule_factor', 0.3) if adaptive_config else 0.3,
            warmup_rounds=adaptive_config.get('warmup_rounds', 3) if adaptive_config else 3,
            use_divergence_adaptation=adaptive_config.get('use_divergence', True) if adaptive_config else True,
            divergence_weight=adaptive_config.get('divergence_weight', 0.5) if adaptive_config else 0.5,
            min_mu=adaptive_config.get('min_mu', 0.001) if adaptive_config else 0.001,
            max_mu=adaptive_config.get('max_mu', 0.1) if adaptive_config else 0.1
        )

        adaptive_controller = AdaptiveMuController(adaptive_mu_config, cfg.num_rounds)

        adaptive_name = adaptive_config.get('name', 'Adaptive Multi-Layer FedProx') if adaptive_config else 'Adaptive Multi-Layer FedProx'
        print(f"Configuration: {adaptive_name}")
        print(f"  Schedule type: {adaptive_mu_config.schedule_type}")
        print(f"  Min schedule factor: {adaptive_mu_config.min_schedule_factor}")
        print(f"  Use divergence adaptation: {adaptive_mu_config.use_divergence_adaptation}")
        print(f"  Divergence weight: {adaptive_mu_config.divergence_weight}")
        print(f"  Base layer mus: {adaptive_mu_config.base_layer_mus}")

        client_fn_adaptive = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_adaptive = get_on_fit_config_adaptive(
            cfg.config_fit,
            fedprox_multilayer_config,
            adaptive_controller,
            cfg.num_rounds
        )

        strategy_adaptive = AdaptiveFedProx(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_adaptive,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            proximal_mu=ml_mu,
            adaptive_controller=adaptive_controller
        )

        history_adaptive = fl.simulation.start_simulation(
            client_fn=client_fn_adaptive,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_adaptive,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )

        all_results['adaptive_multilayer'] = {
            'history': history_adaptive,
            'name': adaptive_name,
            'config': OmegaConf.to_container(adaptive_config, resolve=True) if adaptive_config else {},
            'mu_history': adaptive_controller.get_mu_history_summary(),
            'divergence_history': strategy_adaptive.get_divergence_history()
        }
        print(f"\n[OK] Adaptive Multi-Layer FedProx completed!")

        # Generate plots
        print("\n" + "-"*80)
        print("GENERATING COMPARISON PLOTS")
        print("-"*80)
        plot_comparison(all_results, save_path, cfg.get('plotting', {}))
        plot_adaptive_mu_evolution(
            adaptive_controller.get_mu_history_summary(),
            save_path,
            cfg.get('plotting', {})
        )
        print("[OK] Plots saved!")

    elif strategy_name == 'compare_base_adaptive':
        # Run 2-way comparison: Base FedProx vs Adaptive Multi-Layer (skip Static)
        print("\n" + "="*80)
        print("RUNNING 2-WAY COMPARISON: Base FedProx vs Adaptive Multi-Layer")
        print("="*80)

        # Get configurations
        fedprox_base_config = cfg.get('fedprox_base', None)
        fedprox_multilayer_config = cfg.get('fedprox_multilayer', None)
        adaptive_config = cfg.get('fedprox_adaptive', None)

        # Calculate fractions
        fraction_evaluate = cfg.num_clients_per_round_eval / cfg.num_clients if cfg.num_clients > 0 else 1.0
        fraction_fit = cfg.num_clients_per_round_fit / cfg.num_clients if cfg.num_clients > 0 else 1.0

        # ========== PHASE 1: Base FedProx ==========
        print("\n" + "-"*80)
        print("PHASE 1: Running Base FedProx")
        print("-"*80)
        base_name = fedprox_base_config.get('name', 'Base FedProx') if fedprox_base_config else 'Base FedProx'
        base_mu = fedprox_base_config.get('mu', 0.1) if fedprox_base_config else 0.1
        base_lr = fedprox_base_config.get('lr', cfg.config_fit.lr) if fedprox_base_config else cfg.config_fit.lr
        print(f"Configuration: {base_name}")
        print(f"  mu (proximal term): {base_mu}")
        print(f"  Learning rate: {base_lr}")

        client_fn_base = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_base = get_on_fit_config(cfg.config_fit, fedprox_base_config)

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
        print(f"\n[OK] Base FedProx completed!")

        # ========== PHASE 2: Adaptive Multi-Layer FedProx ==========
        print("\n" + "-"*80)
        print("PHASE 2: Running Adaptive Multi-Layer FedProx")
        print("-"*80)

        # Get base layer mus from static multilayer config
        layer_mus = fedprox_multilayer_config.get('layer_mus', None) if fedprox_multilayer_config else None
        ml_mu = fedprox_multilayer_config.get('mu', 0.01) if fedprox_multilayer_config else 0.01

        # Create adaptive mu controller
        base_layer_mus = dict(layer_mus) if layer_mus else {
            'input': 0.003, 'hidden1': 0.008, 'hidden2': 0.012, 'output': 0.025
        }

        adaptive_mu_config = AdaptiveMuConfig(
            base_layer_mus=base_layer_mus,
            schedule_type=adaptive_config.get('schedule_type', 'cosine') if adaptive_config else 'cosine',
            min_schedule_factor=adaptive_config.get('min_schedule_factor', 0.3) if adaptive_config else 0.3,
            warmup_rounds=adaptive_config.get('warmup_rounds', 3) if adaptive_config else 3,
            use_divergence_adaptation=adaptive_config.get('use_divergence', True) if adaptive_config else True,
            divergence_weight=adaptive_config.get('divergence_weight', 0.5) if adaptive_config else 0.5,
            min_mu=adaptive_config.get('min_mu', 0.001) if adaptive_config else 0.001,
            max_mu=adaptive_config.get('max_mu', 0.1) if adaptive_config else 0.1
        )

        adaptive_controller = AdaptiveMuController(adaptive_mu_config, cfg.num_rounds)

        adaptive_name = adaptive_config.get('name', 'Adaptive Multi-Layer FedProx') if adaptive_config else 'Adaptive Multi-Layer FedProx'
        print(f"Configuration: {adaptive_name}")
        print(f"  Schedule type: {adaptive_mu_config.schedule_type}")
        print(f"  Min schedule factor: {adaptive_mu_config.min_schedule_factor}")
        print(f"  Use divergence adaptation: {adaptive_mu_config.use_divergence_adaptation}")
        print(f"  Divergence weight: {adaptive_mu_config.divergence_weight}")
        print(f"  Base layer mus: {adaptive_mu_config.base_layer_mus}")

        client_fn_adaptive = generate_client_fn(trainloaders, validationloaders, input_dim, target_scaler)
        on_fit_config_fn_adaptive = get_on_fit_config_adaptive(
            cfg.config_fit,
            fedprox_multilayer_config,
            adaptive_controller,
            cfg.num_rounds
        )

        strategy_adaptive = AdaptiveFedProx(
            fraction_fit=fraction_fit,
            min_fit_clients=cfg.num_clients_per_round_fit,
            fraction_evaluate=fraction_evaluate,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=on_fit_config_fn_adaptive,
            evaluate_fn=get_evaluate_fn(input_dim, testloader, target_scaler),
            proximal_mu=ml_mu,
            adaptive_controller=adaptive_controller
        )

        history_adaptive = fl.simulation.start_simulation(
            client_fn=client_fn_adaptive,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy_adaptive,
            client_resources={'num_cpus': 0.8, 'num_gpus': 0}
        )

        all_results['adaptive_multilayer'] = {
            'history': history_adaptive,
            'name': adaptive_name,
            'config': OmegaConf.to_container(adaptive_config, resolve=True) if adaptive_config else {},
            'mu_history': adaptive_controller.get_mu_history_summary(),
            'divergence_history': strategy_adaptive.get_divergence_history()
        }
        print(f"\n[OK] Adaptive Multi-Layer FedProx completed!")

        # Generate plots
        print("\n" + "-"*80)
        print("GENERATING COMPARISON PLOTS")
        print("-"*80)
        plot_comparison(all_results, save_path, cfg.get('plotting', {}))
        plot_adaptive_mu_evolution(
            adaptive_controller.get_mu_history_summary(),
            save_path,
            cfg.get('plotting', {})
        )
        print("[OK] Plots saved!")

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
