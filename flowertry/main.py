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
    ## 1. Parse the config & get experoment for output directory
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the datasets
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)
    print(f"Number of training clients: {len(trainloaders)}")
    print(f"Number of length of one train loader: {len(trainloaders[0].dataset)}")

    ## 3. Define the clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
    )

    ## 5. Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 0.8, 'num_gpus': 0}
    )

    ## 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {'history': history, 'anything else': 'you want to save'}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()