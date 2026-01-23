"""
Server configuration for Stratified Client Selection.
Simplified version without adaptive mu functionality.
"""

from omegaconf import DictConfig
from model import DisposableIncomeModel, test
import torch
from collections import OrderedDict


def get_on_fit_config(config_fit: DictConfig, fedprox_config: DictConfig = None):
    """
    Create fit configuration function for training.
    
    Args:
        config_fit: Base training configuration (lr, momentum, local_epochs)
        fedprox_config: FedProx configuration (mu, lr override)
    
    Returns:
        Function that generates per-round fit configuration
    """
    def fit_config_fn(server_round: int):
        # Use strategy-specific learning rate if provided, otherwise use default
        if fedprox_config is not None and 'lr' in fedprox_config:
            lr = fedprox_config.get('lr')
        else:
            lr = config_fit.lr
        
        config = {
            'lr': lr,
            'momentum': config_fit.get('momentum', 0.0),
            'local_epochs': config_fit.local_epochs
        }
        
        # Add FedProx mu parameter if configured
        if fedprox_config is not None:
            config['mu'] = fedprox_config.get('mu', 0.0)
        else:
            config['mu'] = 0.0  # Default to FedAvg (no proximal term)
        
        return config
    
    return fit_config_fn


def get_evaluate_fn(input_dim: int, testloader, target_scaler=None):
    """
    Return evaluation function for server-side evaluation.
    Evaluates on global test set and returns regression metrics.
    
    Args:
        input_dim: Input dimension for the model
        testloader: DataLoader for test set
        target_scaler: StandardScaler for target (to compute metrics in original scale)
    
    Returns:
        Evaluation function that takes (server_round, parameters, config)
    """
    def evaluate_fn(server_round: int, parameters, config):
        model = DisposableIncomeModel(input_dim)
        
        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        # Set model parameters if provided
        if parameters is not None and len(parameters) > 0:
            try:
                parameters_dict = zip(model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.Tensor(v) for k, v in parameters_dict})
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(f"Warning: Failed to load parameters: {e}. Using random initialization.")
        # If parameters are None or empty, model uses default initialization

        # Evaluate on test set (with target_scaler for original scale metrics)
        mse, rmse, mae, r2 = test(model, testloader, device, target_scaler=target_scaler)
        
        return mse, {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    return evaluate_fn
