import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from cleint import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import flwr as fl


def get_client_resources(num_clients_per_round: int) -> dict:
    """
    Calculate optimal client resources for M1 Pro MacBook.
    
    M1 Pro has 8-10 CPU cores. We want to:
    - Leave some CPU headroom for system processes
    - Allow concurrent client training for better throughput
    
    Note: MPS (Apple Silicon GPU) is NOT used because it's incompatible with
    Ray's multiprocessing. CPU training on M1 Pro is still efficient due to
    unified memory and high single-core performance.
    """
    # Detect available CPU cores
    num_cpus = os.cpu_count() or 8  # Default to 8 if detection fails
    
    # Reserve some CPUs for system overhead and Ray coordination
    usable_cpus = max(num_cpus - 2, 4)  # Keep at least 4 CPUs available
    
    # Calculate CPUs per client to allow concurrent training
    # We want to run `num_clients_per_round` clients efficiently
    cpus_per_client = usable_cpus / num_clients_per_round
    
    # Ensure minimum viable CPU allocation per client
    cpus_per_client = max(cpus_per_client, 0.5)
    
    print(f"\n{'='*50}")
    print("HARDWARE CONFIGURATION")
    print(f"{'='*50}")
    print(f"Detected CPU cores: {num_cpus}")
    print(f"Usable CPUs for FL: {usable_cpus}")
    print(f"Clients per round: {num_clients_per_round}")
    print(f"CPUs per client: {cpus_per_client:.2f}")
    print(f"Device: CPU (MPS disabled - incompatible with Ray)")
    print(f"{'='*50}\n")
    
    return {
        'num_cpus': cpus_per_client,
        'num_gpus': 0.0
    }

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse the config & get experiment for output directory
    print("=" * 60)
    print("FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    # Extract experiment settings
    experiment_name = cfg.get("experiment_name", "unnamed_experiment")
    partition_type = cfg.get("partition_type", "iid")
    dirichlet_alpha = cfg.get("dirichlet_alpha", 0.5)
    strategy_name = cfg.get("strategy", "fedavg")
    fedprox_mu = cfg.get("fedprox_mu", 0.1)
    
    print(f"Experiment: {experiment_name}")
    print(f"Partition Type: {partition_type}")
    if partition_type == "dirichlet":
        print(f"Dirichlet Alpha: {dirichlet_alpha}")
    print(f"Strategy: {strategy_name}")
    if strategy_name == "fedprox":
        print(f"FedProx Mu: {fedprox_mu}")
    print("=" * 60)

    ## 2. Prepare the datasets with configurable partitioning
    trainloaders, validationloaders, testloader = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        partition_type=partition_type,
        dirichlet_alpha=dirichlet_alpha
    )
    print(f"Number of training clients: {len(trainloaders)}")
    print(f"Sample sizes per client (first 5): {[len(trainloaders[i].dataset) for i in range(min(5, len(trainloaders)))]}")

    ## 3. Define the clients based on strategy
    client_fn = generate_client_fn(
        trainloaders=trainloaders,
        valloaders=validationloaders,
        num_classes=cfg.num_classes,
        strategy=strategy_name,
        mu=fedprox_mu
    )

    ## 4. Define your strategy (server-side aggregation)
    # Note: Both FedAvg and FedProx use the same server-side aggregation
    # The difference is in the client-side local training (proximal term)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
    )

    ## 5. Start the simulation
    print("\nStarting Federated Learning Simulation...")
    
    # Get optimized client resources for M1 Pro hardware
    client_resources = get_client_resources(cfg.num_clients_per_round_fit)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=client_resources
    )

    ## 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    # Save comprehensive results for later analysis
    results = {
        'history': history,
        'config': {
            'experiment_name': experiment_name,
            'partition_type': partition_type,
            'dirichlet_alpha': dirichlet_alpha if partition_type == "dirichlet" else None,
            'strategy': strategy_name,
            'fedprox_mu': fedprox_mu if strategy_name == "fedprox" else None,
            'num_rounds': cfg.num_rounds,
            'num_clients': cfg.num_clients,
            'num_clients_per_round_fit': cfg.num_clients_per_round_fit,
        }
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print final results summary
    if history.metrics_centralized:
        final_accuracy = history.metrics_centralized.get('accuracy', [])
        if final_accuracy:
            print(f"\nFinal centralized accuracy: {final_accuracy[-1][1]:.4f}")
    
    print("=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()