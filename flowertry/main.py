import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from cleint import generate_client_fn

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
if __name__ == "__main__":
    main()