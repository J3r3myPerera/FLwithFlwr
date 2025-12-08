import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn, get_strategy
import flwr as fl
import torch


def get_device_info():
    """Print device information for debugging."""
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        # MPS disabled due to Ray/Flower simulation compatibility issues
        return "CPU (MPS disabled for Ray compatibility)"


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse the config & print configuration
    print("\n" + "="*60)
    print("FEDERATED LEARNING CONFIGURATION")
    print("="*60)
    print(f"Device: {get_device_info()}")
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")

    ## 2. Prepare the datasets with non-IID partitioning
    trainloaders, validationloaders, testloader = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        partition_type=cfg.partition_type,
        alpha=cfg.dirichlet_alpha,
        labels_per_client=cfg.labels_per_client,
        num_classes=cfg.num_classes,
        seed=cfg.seed
    )
    print(f"Number of training clients: {len(trainloaders)}")
    print(f"Samples in first client's train loader: {len(trainloaders[0].dataset)}")

    ## 3. Define the clients
    client_fn = generate_client_fn(
        trainloaders, 
        validationloaders, 
        cfg.num_classes,
        use_fedprox=(cfg.strategy == "fedprox"),
        fedprox_mu=cfg.fedprox_mu
    )

    ## 4. Define your strategy (FedAvg or FedProx)
    strategy = get_strategy(
        cfg=cfg,
        testloader=testloader
    )

    ## 5. Start the simulation
    print(f"\nStarting FL simulation with {cfg.strategy.upper()} strategy...")
    print(f"Partition type: {cfg.partition_type}")
    if cfg.partition_type == "dirichlet":
        print(f"Dirichlet alpha: {cfg.dirichlet_alpha}")
    elif cfg.partition_type == "label_skew":
        print(f"Labels per client: {cfg.labels_per_client}")
    print(f"Number of rounds: {cfg.num_rounds}")
    print(f"Clients per round: {cfg.num_clients_per_round_fit}")
    print()

    # Optimized for Apple M1 Pro: allocate more CPUs per client with fewer total clients
    # M1 Pro has 8 performance + 2 efficiency cores = 10 cores total
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0}
    )

    ## 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        'history': history,
        'config': OmegaConf.to_container(cfg),
        'partition_type': cfg.partition_type,
        'strategy': cfg.strategy,
        'dirichlet_alpha': cfg.dirichlet_alpha if cfg.partition_type == "dirichlet" else None
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if history.metrics_centralized:
        for round_num, (_, metrics) in enumerate(history.metrics_centralized.get('accuracy', [])):
            if round_num % 5 == 0 or round_num == len(history.metrics_centralized.get('accuracy', [])) - 1:
                print(f"Round {round_num}: Accuracy = {metrics:.4f}")
    print(f"\nResults saved to: {results_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
