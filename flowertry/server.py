from omegaconf import DictConfig
from model import Net, test
import torch
from collections import OrderedDict
import flwr as fl


def get_device():
    """
    Get the best available device for Flower simulation.
    
    Note: MPS (Apple Silicon) is disabled because Ray's multiprocessing
    (used by Flower simulation) causes segmentation faults with MPS.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_on_fit_config(config_fit: DictConfig, use_fedprox: bool = False, fedprox_mu: float = 0.01):
    """
    Generate the fit configuration function for clients.
    
    Args:
        config_fit: Training configuration (lr, momentum, epochs)
        use_fedprox: Whether to use FedProx regularization
        fedprox_mu: FedProx proximal term coefficient
    """
    def fit_config_fn(server_round: int):
        config = {
            'lr': config_fit.lr,
            'momentum': config_fit.momentum,
            'local_epochs': config_fit.local_epochs,
            'use_fedprox': use_fedprox,
            'fedprox_mu': fedprox_mu,
            'server_round': server_round
        }
        return config

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """
    Generate the centralized evaluation function for the server.
    
    Args:
        num_classes: Number of output classes
        testloader: DataLoader for test data
    """
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = get_device()

        parameters_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in parameters_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)
        
        print(f"Server Round {server_round}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return loss, {"accuracy": accuracy}

    return evaluate_fn


def get_strategy(cfg: DictConfig, testloader):
    """
    Create the FL strategy based on configuration.
    
    Supports:
    - FedAvg: Standard federated averaging
    - FedProx: FedAvg with proximal regularization (better for non-IID data)
    
    Args:
        cfg: Full configuration object
        testloader: Test data loader for centralized evaluation
    
    Returns:
        Flower strategy instance
    """
    use_fedprox = (cfg.strategy == "fedprox")
    
    # Common strategy parameters
    common_params = {
        "fraction_fit": 0.0,  # Use min_fit_clients instead for exact control
        "min_fit_clients": cfg.num_clients_per_round_fit,
        "min_evaluate_clients": cfg.num_clients_per_round_eval,
        "min_available_clients": cfg.num_clients,
        "on_fit_config_fn": get_on_fit_config(
            cfg.config_fit, 
            use_fedprox=use_fedprox, 
            fedprox_mu=cfg.fedprox_mu
        ),
        "evaluate_fn": get_evaluate_fn(cfg.num_classes, testloader),
    }
    
    if cfg.strategy == "fedprox":
        print(f"Using FedProx strategy with mu={cfg.fedprox_mu}")
        # FedProx uses the same aggregation as FedAvg, the difference is in client training
        # The proximal term is added during client-side training
        strategy = fl.server.strategy.FedProx(
            proximal_mu=cfg.fedprox_mu,
            **common_params
        )
    else:
        print("Using FedAvg strategy")
        strategy = fl.server.strategy.FedAvg(**common_params)
    
    return strategy
