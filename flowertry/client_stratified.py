"""
Simplified Flower client for Stratified Client Selection.
Removes adaptive mu and layer-specific mu functionality.
"""

from collections import OrderedDict
from typing import Dict
import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import DisposableIncomeModel, train, test


class FlowerClient(fl.client.NumPyClient):
    """Simplified Flower client for personal finance regression."""
    
    def __init__(self, trainloader, valloader, input_dim: int = 19) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        
        self.model = DisposableIncomeModel(input_dim)

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Store global parameters for FedProx
        self.global_parameters = None
        
        # Store target_scaler for inverse transform in evaluation
        self.target_scaler = None

    def set_parameters(self, parameters):
        """
        Set model parameters from numpy arrays.
        Properly handles BatchNorm buffers (running_mean, running_var, etc.)
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        
        model_state = self.model.state_dict()
        
        for k, v in params_dict:
            # Convert numpy array to torch tensor, preserving shape and dtype
            if isinstance(v, np.ndarray):
                v_copy = np.asarray(v).copy()
                tensor = torch.from_numpy(v_copy)
            else:
                tensor = torch.tensor(v)
            
            # Get expected shape and dtype from model
            expected_shape = model_state[k].shape
            expected_dtype = model_state[k].dtype
            
            # Handle shape mismatches
            if tensor.shape != expected_shape:
                expected_numel = 1
                for dim in expected_shape:
                    expected_numel *= dim
                
                if tensor.numel() == 0 or expected_numel == 0:
                    tensor = torch.zeros(expected_shape, dtype=expected_dtype)
                elif tensor.numel() == expected_numel:
                    tensor = tensor.reshape(expected_shape)
                else:
                    tensor = torch.zeros(expected_shape, dtype=expected_dtype)
            
            # Ensure correct dtype
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            
            state_dict[k] = tensor
        
        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        
        # Store global parameters for FedProx
        self.global_parameters = parameters

    def get_parameters(self, config: Dict[str, Scalar]):
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train the model on local data."""
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config.get('momentum', 0.0)
        epochs = config['local_epochs']

        # Get FedProx mu parameter (default 0.0 for FedAvg)
        mu = config.get('mu', 0.0)

        # Use SGD optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training with FedProx support
        train(
            self.model,
            self.trainloader,
            optim,
            epochs,
            self.device,
            mu=mu,
            global_params=self.global_parameters,
            layer_mus=None  # No layer-specific mu in simplified version
        )

        # Return updated parameters and number of samples
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on local validation data."""
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)

        # Use target_scaler if available for metrics in original scale
        mse, rmse, mae, r2 = test(self.model, self.valloader, self.device, target_scaler=self.target_scaler)
        
        # Return loss (MSE), number of samples, and metrics dictionary
        return float(mse), len(self.valloader.dataset), {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }


def generate_client_fn(trainloaders, valloaders, input_dim: int = 19, target_scaler=None):
    """
    Generate a client function for Flower simulation.
    
    Args:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        input_dim: Input dimension for the model
        target_scaler: StandardScaler for target variable (for inverse transform)
    
    Returns:
        Client function that creates a FlowerClient for a given client ID
    """
    def client_fn(cid: str):
        client = FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            input_dim=input_dim
        )
        # Store target_scaler for use in evaluation
        client.target_scaler = target_scaler
        return client
    
    return client_fn
