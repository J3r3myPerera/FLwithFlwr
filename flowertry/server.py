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
        # Handle both arrays and scalars properly
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v.copy())
        else:
            state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


def get_on_fit_config(config_fit: DictConfig):
    """
    Create a function that returns training configuration for each round.
    """
    def fit_config_fn(server_round: int):
        return {
            'lr': config_fit.lr,
            'momentum': config_fit.momentum,
            'local_epochs': config_fit.local_epochs
        }

    return fit_config_fn


def get_evaluate_fn(num_features: int, num_classes: int, testloader):
    """
    Create a function for server-side model evaluation.
    
    Args:
        num_features: Number of input features for the MLP
        num_classes: Number of output classes
        testloader: Global test DataLoader
    
    Returns:
        evaluate_fn: Function that evaluates the global model
    """
    def evaluate_fn(server_round: int, parameters, config):
        # Create model with correct dimensions
        model = MLP(num_features=num_features, num_classes=num_classes)
        
        # Use CPU for evaluation
        device = torch.device("cpu")

        # Load parameters into model
        set_parameters(model, parameters)

        # Evaluate on test set
        loss, accuracy = test(model, testloader, device)
        
        print(f"Round {server_round}: Test Accuracy = {accuracy*100:.2f}%")
        
        return loss, {"accuracy": accuracy}

    return evaluate_fn
