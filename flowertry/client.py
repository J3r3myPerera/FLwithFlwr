from collections import OrderedDict
from typing import Dict
import torch
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import DisposableIncomeModel, train, test
from adaptive_mu import compute_layer_divergences

class FlowerClient(fl.client.NumPyClient):
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
                # Use torch.from_numpy for better shape preservation
                # Copy to avoid issues with non-contiguous arrays
                v_copy = np.asarray(v).copy()
                tensor = torch.from_numpy(v_copy)
            else:
                # Fallback for other types
                tensor = torch.tensor(v)
            
            # Get expected shape and dtype from model
            expected_shape = model_state[k].shape
            expected_dtype = model_state[k].dtype
            
            # Handle shape mismatches
            if tensor.shape != expected_shape:
                # Calculate expected number of elements
                expected_numel = 1
                for dim in expected_shape:
                    expected_numel *= dim
                
                if tensor.numel() == 0 or expected_numel == 0:
                    # If either is empty, create a tensor with expected shape
                    tensor = torch.zeros(expected_shape, dtype=expected_dtype)
                elif tensor.numel() == expected_numel:
                    # Try to reshape if sizes match
                    tensor = tensor.reshape(expected_shape)
                else:
                    # Size mismatch - use zeros
                    tensor = torch.zeros(expected_shape, dtype=expected_dtype)
            
            # Ensure correct dtype
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            
            state_dict[k] = tensor
        
        # Load state dict - use strict=False to handle any edge cases gracefully
        self.model.load_state_dict(state_dict, strict=False)
        
        # Store global parameters for FedProx
        self.global_parameters = parameters

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Copy the parameters sent by the server to the local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config.get('momentum', 0.0)
        epochs = config['local_epochs']

        # Get FedProx mu parameter (default 0.0 for FedAvg)
        mu = config.get('mu', 0.0)

        # Get layer-specific mu values if provided (for multi-layer FedProx)
        layer_mus = config.get('layer_mus', None)
        if layer_mus is not None:
            # Convert to dict if it's not already
            if isinstance(layer_mus, dict):
                layer_mus = layer_mus
            else:
                layer_mus = None  # Invalid format, use base mu

        # Check if server requests divergence computation (for adaptive mu)
        compute_divergence = config.get('compute_divergence', False)

        # Store pre-training parameters for divergence computation
        if compute_divergence and self.global_parameters is not None:
            pre_training_params = [p.clone().detach().cpu() for p in self.model.parameters()]

        # Use SGD optimizer (can be changed to Adam if needed)
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Do local training with FedProx support (base or multi-layer)
        train(
            self.model,
            self.trainloader,
            optim,
            epochs,
            self.device,
            mu=mu,
            global_params=self.global_parameters,
            layer_mus=layer_mus
        )

        # Compute layer divergences after training (for adaptive mu feedback)
        fit_metrics = {}
        if compute_divergence and self.global_parameters is not None:
            try:
                post_training_params = [p.detach().cpu() for p in self.model.parameters()]
                param_names = [name for name, _ in self.model.named_parameters()]

                divergences = compute_layer_divergences(
                    post_training_params,
                    pre_training_params,
                    param_names
                )

                # Pack divergences into metrics (prefix with 'div_' for easy extraction)
                for layer_name, div_value in divergences.items():
                    fit_metrics[f'div_{layer_name}'] = float(div_value)
            except Exception as e:
                # If divergence computation fails, continue without it
                pass

        # Return updated parameters, number of samples, and metrics
        return self.get_parameters({}), len(self.trainloader.dataset), fit_metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
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
