import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from cleint import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from visualize import plot_accuracy
import flwr as fl


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    Federated Learning for Savings Potential Classification
    
    Supports three aggregation strategies:
    - FedAvg (Federated Averaging) - baseline
    - FedProx (Federated Proximal) - handles client heterogeneity with proximal term
    - SCAFFOLD - handles client heterogeneity with control variates
    
    Data Partitioning:
    - IID: Uniform data distribution across clients
    - Non-IID (Dirichlet): Heterogeneous data with label skew
    """
    
    ## 1. Parse and display configuration
    print("=" * 70)
    print("FEDERATED LEARNING - SAVINGS POTENTIAL CLASSIFICATION")
    print("=" * 70)
    
    strategy_name = cfg.strategy.upper()
    print(f"\nStrategy: {strategy_name}")
    
    # Strategy-specific info
    if cfg.strategy == "fedprox":
        print(f"  └─ Proximal term (μ): {cfg.fedprox_mu}")
    elif cfg.strategy == "scaffold":
        print(f"  └─ Using control variates for variance reduction")
    
    print(f"\nData Partitioning: {cfg.partition_type.upper()}", end="")
    if cfg.partition_type == "dirichlet":
        print(f" (α={cfg.dirichlet_alpha})")
    else:
        print()
    
    print("\nFull Configuration:")
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the dataset with chosen partitioning
    print("=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)
    
    trainloaders, validationloaders, testloader, num_features, num_classes = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        partition_type=cfg.partition_type,
        dirichlet_alpha=cfg.dirichlet_alpha
    )
    
    print(f"\nClients ready: {len(trainloaders)}")
    print(f"Features: {num_features}, Classes: {num_classes}")

    ## 3. Define the client function (FedAvg, FedProx, or SCAFFOLD)
    client_fn = generate_client_fn(
        trainloaders=trainloaders,
        valloaders=validationloaders,
        num_features=num_features,
        num_classes=num_classes,
        strategy=cfg.strategy,
        fedprox_mu=cfg.fedprox_mu
    )

    ## 4. Define the server strategy (FedAvg aggregation)
    # Note: All three methods use FedAvg aggregation on the server
    # The difference is in the client-side training
    server_strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit, cfg.fedprox_mu, cfg.num_rounds),
        evaluate_fn=get_evaluate_fn(num_features, num_classes, testloader)
    )

    ## 5. Start the simulation
    print("\n" + "=" * 70)
    print("STARTING FEDERATED LEARNING SIMULATION")
    print("=" * 70)
    print(f"Strategy: {strategy_name}")
    print(f"Rounds: {cfg.num_rounds}")
    print(f"Clients per round: {cfg.num_clients_per_round_fit}")
    print(f"Local epochs: {cfg.config_fit.local_epochs}")
    print()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=server_strategy,
        client_resources={'num_cpus': 1.0, 'num_gpus': 0}
    )

    ## 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    config_dict = {
        'num_rounds': cfg.num_rounds,
        'num_clients': cfg.num_clients,
        'batch_size': cfg.batch_size,
        'partition_type': cfg.partition_type,
        'dirichlet_alpha': cfg.dirichlet_alpha,
        'strategy': cfg.strategy,
        'fedprox_mu': cfg.fedprox_mu if cfg.strategy == "fedprox" else None,
        'local_epochs': cfg.config_fit.local_epochs,
        'lr': cfg.config_fit.lr,
        'num_features': num_features,
        'num_classes': num_classes,
        'task': 'Savings Potential Classification'
    }

    results = {
        'history': history,
        'config': config_dict
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    ## 7. Print final results
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"\nStrategy: {strategy_name}")
    print(f"Data: {cfg.partition_type.upper()}", end="")
    if cfg.partition_type == "dirichlet":
        print(f" (α={cfg.dirichlet_alpha})")
    else:
        print()
    
    if history.metrics_centralized:
        accuracies = history.metrics_centralized.get('accuracy', [])
        if accuracies:
            print("\nAccuracy Progression:")
            for round_num, acc in accuracies:
                print(f"  Round {round_num}: {acc*100:.2f}%")
            
            final_acc = accuracies[-1][1] * 100
            print(f"\n{'='*40}")
            print(f"FINAL ACCURACY: {final_acc:.2f}%")
            print(f"{'='*40}")
    
    print(f"\nResults saved to: {results_path}")
    
    ## 8. Generate and save visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)
    
    try:
        plot_accuracy(history, config_dict, save_path, show_plot=False)
    except Exception as e:
        print(f"Could not generate plot: {e}")
        print("You can manually generate the plot later using:")
        print(f"  python visualize.py {results_path}")
    
    return history


if __name__ == "__main__":
    main()
