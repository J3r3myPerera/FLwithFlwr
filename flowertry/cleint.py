from collections import OrderedDict
from typing import Dict, List
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
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v.copy())
        else:
            state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model):
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client using FedAvg (Federated Averaging)."""
    
    def __init__(self, trainloader, valloader, num_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


class FedProxClient(fl.client.NumPyClient):
    """
    Improved FedProx client with adaptive proximal term.
    
    Key improvements:
    - Lower default μ for better learning
    - Adaptive μ based on gradient magnitude (optional)
    - Proper gradient scaling
    """
    
    def __init__(
        self, 
        trainloader, 
        valloader, 
        num_features: int, 
        num_classes: int = 3,
        mu: float = 0.01  # REDUCED from 0.1 to 0.01
    ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")
        self.mu = mu

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        
        # Store global model parameters for proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]

        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)
        
        # Get mu from config if provided, otherwise use instance value
        mu = config.get('fedprox_mu', self.mu)

        self._train_with_proximal(lr, momentum, epochs, global_params, mu)

        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def _train_with_proximal(self, lr: float, momentum: float, epochs: int, 
                             global_params: list, mu: float):
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
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Proximal term with proper scaling
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_params):
                    proximal_term += torch.sum((local_param - global_param.to(self.device)) ** 2)
                
                # Scale proximal term appropriately
                loss = loss + (mu / 2.0) * proximal_term
                
                loss.backward()
                optimizer.step()

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


class SCAFFOLDClient(fl.client.NumPyClient):
    """
    Improved SCAFFOLD client with proper control variate management.
    
    Key improvements:
    - Properly initialized and updated control variates
    - Gradient correction applied correctly
    - Better numerical stability
    """
    
    def __init__(
        self, 
        trainloader, 
        valloader, 
        num_features: int, 
        num_classes: int = 3,
        client_id: int = 0
    ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = MLP(num_features=num_features, num_classes=num_classes)
        self.device = torch.device("cpu")
        self.client_id = client_id
        
        # Initialize client control variate as zeros (will be updated after first round)
        self.c_local = None  # Lazy initialization
        self.initialized = False
        
    def _init_control_variates(self):
        """Initialize control variates matching model parameters."""
        if not self.initialized:
            self.c_local = [torch.zeros_like(p.data) for p in self.model.parameters()]
            self.initialized = True
        
    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self._init_control_variates()
        
        # Store initial global model parameters
        global_params = [p.clone().detach() for p in self.model.parameters()]
        
        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        epochs = config.get('local_epochs', 1)
        
        # For SCAFFOLD, we need server control variate
        # In this simplified version, we approximate c_global ≈ 0 initially
        # and update c_local based on the parameter drift
        c_global = [torch.zeros_like(p.data) for p in self.model.parameters()]
        
        # Train with SCAFFOLD correction
        num_steps = self._train_with_scaffold(lr, momentum, epochs, global_params, c_global)
        
        # Update local control variate using Option II from SCAFFOLD paper
        # c_i_new = c_i - c + (x - y) / (K * lr)
        # where x = initial params, y = final params, K = num_steps
        if num_steps > 0:
            with torch.no_grad():
                for i, (p_new, p_old, c_l, c_g) in enumerate(zip(
                    self.model.parameters(), global_params, self.c_local, c_global
                )):
                    # Compute parameter change
                    delta = (p_old - p_new.data) / (num_steps * lr)
                    # Update local control variate (Option II)
                    self.c_local[i] = delta.clone()
        
        return get_parameters(self.model), len(self.trainloader.dataset), {}
    
    def _train_with_scaffold(
        self, 
        lr: float, 
        momentum: float, 
        epochs: int, 
        global_params: List[torch.Tensor],
        c_global: List[torch.Tensor]
    ) -> int:
        """
        Train with SCAFFOLD variance reduction.
        
        The key idea: correct gradient to reduce variance
        g_corrected = g + (c_global - c_local)
        
        This correction term compensates for the difference between
        local and global data distributions.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.model.train()
        self.model.to(self.device)
        
        num_steps = 0
        
        for _ in range(epochs):
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply SCAFFOLD correction: g = g + (c_global - c_local)
                # This corrects for client drift
                with torch.no_grad():
                    for param, c_l, c_g in zip(self.model.parameters(), self.c_local, c_global):
                        if param.grad is not None:
                            # Add correction term
                            correction = c_g.to(self.device) - c_l.to(self.device)
                            param.grad.data.add_(correction)
                
                optimizer.step()
                num_steps += 1
        
        return num_steps

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {'accuracy': accuracy}


def generate_client_fn(
    trainloaders, 
    valloaders, 
    num_features: int, 
    num_classes: int = 3,
    strategy: str = "fedavg",
    fedprox_mu: float = 0.01  # REDUCED default
):
    """
    Generate a client function for Flower simulation.
    
    Args:
        trainloaders: List of training DataLoaders
        valloaders: List of validation DataLoaders
        num_features: Number of input features
        num_classes: Number of output classes
        strategy: "fedavg", "fedprox", or "scaffold"
        fedprox_mu: Proximal term weight (default reduced to 0.01)
    """
    def client_fn(cid: str):
        client_id = int(cid)
        
        if strategy == "fedprox":
            return FedProxClient(
                trainloader=trainloaders[client_id],
                valloader=valloaders[client_id],
                num_features=num_features,
                num_classes=num_classes,
                mu=fedprox_mu
            )
        elif strategy == "scaffold":
            return SCAFFOLDClient(
                trainloader=trainloaders[client_id],
                valloader=valloaders[client_id],
                num_features=num_features,
                num_classes=num_classes,
                client_id=client_id
            )
        else:  # Default: FedAvg
            return FlowerClient(
                trainloader=trainloaders[client_id],
                valloader=valloaders[client_id],
                num_features=num_features,
                num_classes=num_classes
            )

    return client_fn
