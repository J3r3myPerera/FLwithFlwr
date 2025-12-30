from collections import OrderedDict
from typing import Dict
import copy
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from flwr.common.typing import Scalar, NDArrays

from model import MLP, train, test


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


def get_parameters(model):
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    """
    Standard Flower client using FedAvg (Federated Averaging).
    
    Used for Savings Potential Classification (3-class).
    """
    
    def __init__(self, trainloader, valloader, num_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters to send to server."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Train the model on local data using standard FedAvg."""
        set_parameters(self.model, parameters)

        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on local validation data."""
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


class FedProxClient(fl.client.NumPyClient):
    """
    Flower client using FedProx algorithm.
    
    FedProx adds a proximal term to the local objective:
    L_local = L_original + (mu/2) * ||w - w_global||^2
    
    This prevents client drift in heterogeneous (non-IID) settings
    by penalizing deviations from the global model.
    
    Args:
        mu: Proximal term weight (higher = stronger regularization)
    """
    
    def __init__(
        self, 
        trainloader, 
        valloader, 
        num_features: int, 
        num_classes: int = 3,
        mu: float = 0.1
    ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")
        self.mu = mu  # Proximal term weight

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters to send to server."""
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """Train the model using FedProx (with proximal term)."""
        set_parameters(self.model, parameters)
        
        # Store global model parameters for proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]

        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)

        # Train with proximal term
        self._train_with_proximal(lr, momentum, epochs, global_params)

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def _train_with_proximal(self, lr: float, momentum: float, epochs: int, global_params: list):
        """
        Train with FedProx proximal term.
        
        Loss = CrossEntropy + (mu/2) * ||w - w_global||^2
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.model.train()
        self.model.to(self.device)
        
        for _ in range(epochs):
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Original loss
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Add proximal term: (mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_params):
                    proximal_term += ((local_param - global_param.to(self.device)) ** 2).sum()
                
                loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on local validation data."""
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


def generate_client_fn(
    trainloaders, 
    valloaders, 
    num_features: int, 
    num_classes: int = 3,
    strategy: str = "fedavg",
    fedprox_mu: float = 0.1
):
    """
    Generate a client function for Flower simulation.
    
    Args:
        trainloaders: List of training DataLoaders (one per client)
        valloaders: List of validation DataLoaders (one per client)
        num_features: Number of input features for the MLP
        num_classes: Number of output classes
        strategy: "fedavg" or "fedprox"
        fedprox_mu: Proximal term weight for FedProx
    
    Returns:
        client_fn: Function that creates FlowerClient or FedProxClient instances
    """
    def client_fn(cid: str):
        if strategy == "fedprox":
            return FedProxClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_features=num_features,
                num_classes=num_classes,
                mu=fedprox_mu
            )
        else:  # Default: FedAvg
            return FlowerClient(
                trainloader=trainloaders[int(cid)],
                valloader=valloaders[int(cid)],
                num_features=num_features,
                num_classes=num_classes
            )

    return client_fn
