from omegaconf import DictConfig
from model import DisposableIncomeModel, test
import torch
from collections import OrderedDict
from typing import Optional

from adaptive_mu import AdaptiveMuController


def get_on_fit_config(config_fit: DictConfig, fedprox_config: DictConfig = None):
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
            
            # Add layer-specific mu values if provided (for multi-layer FedProx)
            layer_mus = fedprox_config.get('layer_mus', None)
            if layer_mus is not None:
                # Convert OmegaConf DictConfig to regular dict if needed
                if hasattr(layer_mus, '_content'):
                    config['layer_mus'] = dict(layer_mus)
                else:
                    config['layer_mus'] = layer_mus
        else:
            config['mu'] = 0.0  # Default to FedAvg (no proximal term)
        
        return config
    return fit_config_fn


def get_on_fit_config_adaptive(
    config_fit: DictConfig,
    fedprox_config: DictConfig,
    adaptive_controller: Optional[AdaptiveMuController] = None,
    total_rounds: int = 20
):
    """
    Create fit config function with adaptive mu support.

    Args:
        config_fit: Base training configuration
        fedprox_config: FedProx configuration including base layer_mus
        adaptive_controller: Optional adaptive mu controller for dynamic mu values
        total_rounds: Total number of training rounds

    Returns:
        Function that generates per-round fit configuration with adaptive mu
    """
    def fit_config_fn(server_round: int):
        # Use strategy-specific learning rate if provided
        if fedprox_config is not None and 'lr' in fedprox_config:
            lr = fedprox_config.get('lr')
        else:
            lr = config_fit.lr

        config = {
            'lr': lr,
            'momentum': config_fit.get('momentum', 0.0),
            'local_epochs': config_fit.local_epochs,
            'current_round': server_round,
            'total_rounds': total_rounds,
        }

        # Add FedProx mu parameters
        if fedprox_config is not None:
            config['mu'] = fedprox_config.get('mu', 0.0)

            # Get layer-specific mu values
            if adaptive_controller is not None:
                # Use ADAPTIVE layer mus computed by controller
                adaptive_layer_mus = adaptive_controller.compute_round_layer_mus(server_round)
                config['layer_mus'] = adaptive_layer_mus
                config['compute_divergence'] = True  # Request divergence metrics from clients

                # Log adaptive mu values
                print(f"  Round {server_round} adaptive mu: {adaptive_layer_mus}")
            else:
                # Use STATIC layer mus from config
                layer_mus = fedprox_config.get('layer_mus', None)
                if layer_mus is not None:
                    if hasattr(layer_mus, '_content'):
                        config['layer_mus'] = dict(layer_mus)
                    else:
                        config['layer_mus'] = layer_mus
                config['compute_divergence'] = False
        else:
            config['mu'] = 0.0
            config['compute_divergence'] = False

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
