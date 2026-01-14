from omegaconf import DictConfig
from model import Net, test, DEFAULT_INPUT_DIM
import torch
from collections import OrderedDict
from flwr.common import ndarrays_to_parameters


def get_initial_parameters(num_classes: int = 3, input_dim: int = DEFAULT_INPUT_DIM):
    """
    Create initial parameters for the model.

    Args:
        num_classes: Number of output classes (default: 3)
        input_dim: Number of input features (default: DEFAULT_INPUT_DIM)

    Returns:
        Parameters: Initial model parameters as Flower Parameters object
    """
    model = Net(num_classes, input_dim=input_dim)
    # Get model parameters as numpy arrays
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(params)


def get_on_fit_config(config_fit: DictConfig, proximal_mu: float = None, max_grad_norm: float = None):
    """
    Create a function that returns training configuration for each round.
    
    Args:
        config_fit: Configuration containing lr, momentum, local_epochs, max_grad_norm
        proximal_mu: Optional fixed proximal mu value for FedProx
        max_grad_norm: Maximum gradient norm for clipping (defaults to config_fit.max_grad_norm or 1.0)
    
    Returns:
        fit_config_fn: Function that returns config dict for each round
    """
    # Get max_grad_norm from config if not explicitly provided
    if max_grad_norm is None:
        max_grad_norm = config_fit.get('max_grad_norm', 1.0)
    
    def fit_config_fn(server_round: int):
        config = {
            'lr': config_fit.lr,
            'momentum': config_fit.momentum,
            'local_epochs': config_fit.local_epochs,
            'max_grad_norm': max_grad_norm
        }
        # Include proximal_mu if provided (for FedProx)
        if proximal_mu is not None:
            config['proximal_mu'] = proximal_mu
        return config

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader, class_weights=None, input_dim: int = DEFAULT_INPUT_DIM):
    """
    Create a function for server-side model evaluation.

    Args:
        num_classes: Number of output classes (3 for savings classification)
        testloader: Global test DataLoader
        class_weights: Optional class weights for weighted loss
        input_dim: Number of input features

    Returns:
        evaluate_fn: Function that evaluates the global model on test set
    """
    def evaluate_fn(server_round: int, parameters, config):
        # Create model with correct input dimension
        model = Net(num_classes, input_dim=input_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters are already ndarrays (list of numpy arrays) when passed from FedAvg/FedProx
        # No need to convert
        ndarrays = parameters

        # Load parameters into model
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=torch.float32) for k, v in zip(model.state_dict().keys(), ndarrays)}
        )
        model.load_state_dict(state_dict, strict=True)

        # Evaluate on test set
        loss, accuracy = test(model, testloader, device, class_weights)
        
        # Print progress
        print(f"Round {server_round}: Test Accuracy = {accuracy*100:.2f}%")
        
        return loss, {"accuracy": accuracy}

    return evaluate_fn
