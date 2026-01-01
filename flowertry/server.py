from collections import OrderedDict
import numpy as np
import torch
from omegaconf import DictConfig
from model import MLP, test


def set_parameters(model, parameters):
    """Load model parameters from numpy arrays into PyTorch model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict()
    for k, v in params_dict:
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v.copy())
        else:
            state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


def get_on_fit_config(config_fit: DictConfig, fedprox_mu: float = 0.0, total_rounds: int = 25):
    """
    Create a function that returns training configuration for each round.
    
    Now passes fedprox_mu to clients for proper FedProx operation.
    """
    def fit_config_fn(server_round: int):
        # Learning rate decay: reduce by 5% every 5 rounds
        lr_decay = 0.95 ** (server_round // 5)
        current_lr = config_fit.lr * lr_decay
        
        # Get weight_decay from config or use default
        weight_decay = getattr(config_fit, 'weight_decay', 0.0001)
        
        return {
            'lr': current_lr,
            'momentum': config_fit.momentum,
            'local_epochs': config_fit.local_epochs,
            'weight_decay': weight_decay,
            'server_round': server_round,
            'fedprox_mu': fedprox_mu  # Pass mu to FedProx clients
        }

    return fit_config_fn


def get_evaluate_fn(num_features: int, num_classes: int, testloader):
    """
    Create a function for server-side model evaluation.
    """
    def evaluate_fn(server_round: int, parameters, config):
        model = MLP(num_features=num_features, num_classes=num_classes)
        device = torch.device("cpu")

        set_parameters(model, parameters)

        loss, accuracy = test(model, testloader, device)
        
        print(f"Round {server_round}: Test Accuracy = {accuracy*100:.2f}%")
        
        return loss, {"accuracy": accuracy}

    return evaluate_fn
