import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse the config & get experoment for output directory
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()