from omegaconf import DictConfig
from model import Net, test
import torch
from collections import OrderedDict


def get_on_fit_config(config_fit: DictConfig):
    """
    Create a function that returns training configuration for each round.
    
    Args:
        config_fit: Configuration containing lr, momentum, local_epochs
    
    Returns:
        fit_config_fn: Function that returns config dict for each round
    """
    def fit_config_fn(server_round: int):
        return {
            'lr': config_fit.lr,
            'momentum': config_fit.momentum,
            'local_epochs': config_fit.local_epochs
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """
    Create a function for server-side model evaluation.
    
    Args:
        num_classes: Number of output classes (3 for savings classification)
        testloader: Global test DataLoader
    
    Returns:
        evaluate_fn: Function that evaluates the global model on test set
    """
    def evaluate_fn(server_round: int, parameters, config):
        # Create model
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load parameters into model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate on test set
        loss, accuracy = test(model, testloader, device)
        
        # Print progress
        print(f"Round {server_round}: Test Accuracy = {accuracy*100:.2f}%")
        
        return loss, {"accuracy": accuracy}

    return evaluate_fn
