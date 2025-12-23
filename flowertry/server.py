from omegaconf import DictConfig
from model import Net, test
import torch
from collections import OrderedDict


def get_device():
    """
    Get the best available device for evaluation.
    
    Note: MPS (Apple Silicon GPU) is NOT used because it's incompatible
    with Ray's multiprocessing used by Flower simulation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_on_fit_config(config_fit: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            'lr': config_fit.lr,
            'momentum': config_fit.momentum,
            'local_epochs': config_fit.local_epochs
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Server-side model evaluation using best available device."""
    
    def evaluate_fn(server_round: int, parameters, config):
        model = Net(num_classes)
        device = get_device()

        parameters_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in parameters_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}

    return evaluate_fn