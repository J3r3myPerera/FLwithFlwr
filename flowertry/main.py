import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from cleint import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import flwr as fl


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    Federated Learning for Savings Potential Classification
    
    Task: Classify users into savings potential categories:
    - Low (<7% savings)
    - Medium (7-12% savings)
    - High (>12% savings)
    
    Dataset: Indian Personal Finance and Spending Habits
    Model: Multi-Layer Perceptron (MLP)
    """
    
    ## 1. Parse and display configuration
    print("=" * 60)
    print("FEDERATED LEARNING - SAVINGS POTENTIAL CLASSIFICATION")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the dataset
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)
    
    trainloaders, validationloaders, testloader = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        iid=cfg.get('iid', True),
        alpha=cfg.get('alpha', 0.5)
    )
    
    print(f"\nNumber of clients: {len(trainloaders)}")
    print(f"Samples in first client's training set: {len(trainloaders[0].dataset)}")

    ## 3. Define the client function
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define the server strategy (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
    )

    ## 5. Start the simulation
    print("\n" + "=" * 60)
    print("STARTING FEDERATED LEARNING SIMULATION")
    print("=" * 60)
    print(f"Rounds: {cfg.num_rounds}")
    print(f"Clients per round (fit): {cfg.num_clients_per_round_fit}")
    print(f"Clients per round (eval): {cfg.num_clients_per_round_eval}")
    print()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1.0, 'num_gpus': 0}
    )

    ## 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        'history': history,
        'config': {
            'num_rounds': cfg.num_rounds,
            'num_clients': cfg.num_clients,
            'batch_size': cfg.batch_size,
            'num_classes': cfg.num_classes,
            'lr': cfg.config_fit.lr,
            'local_epochs': cfg.config_fit.local_epochs,
            'task': 'Savings Potential Classification'
        }
    }

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    ## 7. Print final results
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    
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


if __name__ == "__main__":
    main()
